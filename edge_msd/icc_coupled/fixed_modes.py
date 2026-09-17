"""Nine validation-only constant ICC modes, then two held-out test episodes."""

import argparse
import csv
import hashlib
import json
from pathlib import Path
from time import perf_counter

import numpy as np
import torch

from edge_msd.config import Settings
from edge_msd.icc_coupled.codecs import load_codec
from edge_msd.icc_coupled.environment import CoupledSimulator, TelemetrySettings
from edge_msd.icc_coupled.ppo import MODES, interval_reward
from edge_msd.icc_paper import load_scenario
from edge_msd.network import Network
from edge_msd.placement import place_core


class ConstantMode:
    """Use only cumulative accounting; never archive or inspect source labels."""

    def __init__(self, index):
        self.index = index
        self.previous = None
        self.initial = None
        self.raw_return = 0.0
        self.discounted_return = 0.0
        self.intervals = []
        self.calls = 0

    def _close(self, accounting):
        reward = interval_reward(self.previous, accounting)
        elapsed = self.previous["time_ms"]-self.initial["time_ms"]
        self.raw_return += reward
        self.discounted_return += .995**elapsed*reward
        self.intervals.append({"before":self.previous, "after":accounting,
                               "reward":reward, "weight":.995**elapsed})

    def act(self, observation, training_context=None):
        accounting = dict(observation["accounting"])
        if self.previous is not None:
            self._close(accounting)
        else:
            self.initial = accounting
        self.previous = accounting
        self.calls += 1
        return MODES[self.index]

    def finish(self, metrics):
        self._close(dict(metrics["accounting"]))
        assert abs(self.raw_return-interval_reward(self.initial, metrics["accounting"])) < 1e-8
        mc = 0.0
        for interval in reversed(self.intervals):
            elapsed = interval["after"]["time_ms"]-interval["before"]["time_ms"]
            mc = interval["reward"] + .995**elapsed*mc
        assert abs(mc-self.discounted_return) < 1e-8


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path,
                        default=Path("runs/icc_coupled_fixed_modes_v2_discounted"),
                        help="Fresh result directory; existing nonempty directories are rejected")
    args = parser.parse_args(argv)
    torch.set_num_threads(1)
    output = args.output
    if output.exists() and any(output.iterdir()):
        raise RuntimeError("Choose a fresh output directory; do not rerun the bounded sweep")
    output.mkdir(parents=True, exist_ok=True)
    dataset = Path("data/icc_paper/scenario_2026.json")
    codec_path = Path("runs/icc_coupled_models/importance.pt")
    codec = load_codec(codec_path).eval().requires_grad_(False)
    encoder_hash = codec.encoding_hash()
    scenario = load_scenario(dataset, .4)
    settings = Settings(duration_ms=270, slot_ms=1, seed=8400, ec_admission_window_ms=1)
    counts = place_core(scenario, Network(scenario), settings).counts
    telemetry = TelemetrySettings(arrival_ms=120, control_ms=1, report_ms=100,
                                  report_bps=64000, dynamic=True)
    metadata = {
        "validation_seeds": [8400], "held_out_test_seeds": [8500, 8501],
        "selection_metric": "highest physical-unit discounted return, then lowest mode index",
        "gamma":.995, "discount_time_unit":"physical ms, interval start relative to episode start",
        "episodes_budget": 11, "load": .4, "arrival_ms": 120, "drain_ms": 150,
        "control_ms": 1, "ec_admission_window_ms": 1, "alpha": settings.alpha,
        "report_ms": 100, "report_bps": 64000, "modes": MODES,
        "codec": str(codec_path), "codec_encoding_hash": encoder_hash,
        "codec_checkpoint_sha256": hashlib.sha256(codec_path.read_bytes()).hexdigest(),
        "environment_sha256": hashlib.sha256(Path("edge_msd/icc_coupled/environment.py").read_bytes()).hexdigest(),
        "dataset_sha256": hashlib.sha256(dataset.read_bytes()).hexdigest(),
        "forecast_bin_alignment": "fixed absolute sample bins",
        "frozen_codec": True, "training_context_used": False,
        "raw_return": "-delta_cost/100 -2*delta_deadline_failures -.01*delta_bytes/32",
        "raw_return_initial_offset": "Initial core deployment predates first action, matching PPO accounting.",
        "core_counts": {f"{service}@{node}":count for (service,node),count in counts.items() if count},
    }
    (output/"metadata.json").write_text(json.dumps(metadata, indent=2)+"\n")
    rows, episodes = [], []

    def run_one(seed, index, split):
        policy = ConstantMode(index)
        control = Settings(duration_ms=270, slot_ms=1, seed=seed, ec_admission_window_ms=1)
        started = perf_counter()
        sim = CoupledSimulator(scenario, control, telemetry, codec, core_counts=counts, policy=policy)
        result = sim.run()
        elapsed = perf_counter()-started
        assert codec.encoding_hash() == encoder_hash
        row = {"split":split, "seed":seed, "mode":index,
               "eta_multiplier":MODES[index][0], "parallelism_cap":MODES[index][1],
               "raw_return":policy.raw_return, "undiscounted_return":policy.raw_return,
               "discounted_return":policy.discounted_return, **result["overall"],
               "total_cost":result["costs"]["total"], "report_bytes":result["report_bytes"],
               "report_bps":result["report_bps"], "wall_seconds":elapsed,
               "policy_decisions":policy.calls}
        rows.append(row)
        episodes.append({"row":row, "result":result, "reward_intervals":policy.intervals})
        (output/"episodes.json").write_text(json.dumps(episodes, indent=2)+"\n")
        with (output/"episodes.csv").open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(row))
            writer.writeheader()
            writer.writerows(rows)
        print(f"{split} seed={seed} mode={index} {MODES[index]} "
              f"discounted={policy.discounted_return:.3f} total={policy.raw_return:.3f} "
              f"SLA={row['ontime_rate']:.5f} "
              f"cost={row['total_cost']:.1f} wall={elapsed:.2f}s", flush=True)

    for index in range(len(MODES)):
        run_one(8400, index, "validation")
    assert len({(item["result"]["arrival_sha256"], item["result"]["resource_sha256"])
                for item in episodes}) == 1
    selected = max(rows, key=lambda row:(row["discounted_return"], -row["mode"]))["mode"]
    selection = {"selected_mode":selected, "selected_action":MODES[selected],
                 "selection_seed":8400, "selection_uses_test_data":False,
                 "selection_metric":"physical-unit discounted return, gamma=.995 per ms",
                 "selected_validation":next(row for row in rows if row["mode"] == selected)}
    (output/"selection.json").write_text(json.dumps(selection, indent=2)+"\n")
    for seed in (8500, 8501):
        run_one(seed, selected, "test")
    tests = [row for row in rows if row["split"] == "test"]
    summary = {**selection, "validation_episodes":9, "test_episodes":2,
               "test_mean_raw_return":float(np.mean([row["raw_return"] for row in tests])),
               "test_mean_discounted_return":float(np.mean([row["discounted_return"] for row in tests])),
               "test_mean_ontime_rate":float(np.mean([row["ontime_rate"] for row in tests])),
               "test_mean_total_cost":float(np.mean([row["total_cost"] for row in tests])),
               "test_rows":tests}
    (output/"summary.json").write_text(json.dumps(summary, indent=2)+"\n")
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()

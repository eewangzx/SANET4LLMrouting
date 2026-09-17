"""Predictive X -> Z -> future training, with joint PPO and importance ablations."""

import argparse
import csv
import hashlib
import json
from dataclasses import asdict, replace
from pathlib import Path
from time import perf_counter

import numpy as np
import torch
from torch import nn
from torch.distributions import Categorical

from .environment import Config
from .experiment import dump
from .learning import (
    Policy,
    advantages,
    collect_pretraining,
    public_features,
    report_inputs,
    tensors,
)
from .predictive import PredictiveCodec, PredictiveNet, ReportingEnv

SPECS = {
    "dense_joint": dict(selection="dense", dimension=4, keep=3, joint=True),
    "if_joint": dict(selection="importance", dimension=8, keep=3, joint=True),
    "if_frozen": dict(selection="importance", dimension=8, keep=3, joint=False),
    "random_joint": dict(selection="random", dimension=8, keep=3, joint=True),
    "stats16": dict(selection="stats", dimension=16, keep=3, joint=False),
    "adaptive_joint": dict(selection="adaptive", dimension=8, keep=3, joint=True),
}


def make_model(name, cfg, seed):
    torch.manual_seed(seed)
    spec = {k: v for k, v in SPECS[name].items() if k != "joint"}
    model = PredictiveNet(cfg.history, cfg.horizon, **spec)
    # All methods start with exactly the same actor and critic parameters.
    torch.manual_seed(seed + 100)
    for module in (model.actor, model.critic):
        for layer in module.modules():
            if isinstance(layer, nn.Linear):
                layer.reset_parameters()
    return model


def parameter_hash(model):
    h = hashlib.sha256()
    for key, value in sorted(model.state_dict().items()):
        h.update(key.encode())
        h.update(value.detach().cpu().numpy().tobytes())
    return h.hexdigest()


def validation_prediction(model, valid):
    x, y = valid
    with torch.no_grad():
        z, scores, mask = model.representation(x)
        result = {
            "forecast_mse": float((model.forecast(z) - y).square().mean()),
            "persistence_mse": float((x[:, -1, [1, 5, 6]][:, None] - y).square().mean()),
            "soft_count": float(scores.sum(-1).mean()),
        }
        if model.selection in ("importance", "random", "adaptive"):
            hard = (mask > 0.5).int()
            patterns = (hard * 2 ** torch.arange(model.latent_dim)).sum(-1)
            result.update(selection_frequencies=hard.float().mean(0).tolist(),
                          unique_masks=int(patterns.unique().numel()))
        if model.selection == "adaptive":
            deployed_counts = model.budget_logits(x).argmax(-1) + 2
            deployed_z = model.encode(x, deployed_counts)
            result["deployed_forecast_mse"] = float((model.forecast(deployed_z) - y).square().mean())
            result["deployed_mean_keep"] = float((deployed_counts - 1).float().mean())
            for keep in (1, 2, 3):
                zz = model.encode(x, torch.full((len(x),), keep + 1))
                result[f"forecast_mse_k{keep}"] = float((model.forecast(zz) - y).square().mean())
        return result


def prediction_pretrain(model, training, valid, epochs, seed, output):
    torch.manual_seed(seed + 200)
    x, y = training
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    logs = [{"epoch": 0, **validation_prediction(model, valid)}]
    for epoch in range(1, epochs + 1):
        model.train()
        losses = []
        for index in torch.randperm(len(x)).split(128):
            counts = torch.randint(2, 5, (len(index),)) if model.selection == "adaptive" else None
            z, scores, _ = model.representation(x[index], counts)
            prediction = (model.forecast(z) - y[index]).square().mean()
            loss = prediction + 0.001 * model.bandwidth_penalty(scores, None if counts is None else counts - 1)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            losses.append(float(prediction.detach()))
        model.eval()
        logs.append({"epoch": epoch, "train_forecast_mse": float(np.mean(losses)),
                     **validation_prediction(model, valid)})
        if epoch % 10 == 0:
            print(f"prediction epoch={epoch}/{epochs} val_mse={logs[-1]['forecast_mse']:.5f}", flush=True)
    dump(output / "pretraining.json", logs)
    return logs


def evaluate_policy(model, dataset, cfg, seed, wire_price=1.0):
    model.eval()
    codec = PredictiveCodec(model)
    env = ReportingEnv(dataset, cfg, seed, codec, "periodic", wire_price=wire_price)
    policy = Policy(model, codec)
    reward = 0.0
    while env.tick < env.total_steps:
        action = policy.select(env.observe())
        _, r, _, _ = env.step(action)
        reward += r
    result = env.results()
    result["reward_per_arrival"] = reward / max(1, result["arrivals"])
    result["mean_packet_bytes"] = float(np.mean(codec.sizes))
    result["min_packet_bytes"], result["max_packet_bytes"] = min(codec.sizes), max(codec.sizes)
    result["mean_selected_dimensions"] = float(np.mean(codec.keeps)) if codec.keeps else model.latent_dim
    for keep in (1, 2, 3):
        result[f"budget_k{keep}_fraction"] = float(np.mean(np.asarray(codec.keeps) == keep)) if codec.keeps else 0.0
    return result


def rollout_validation(model, dataset, cfg, wire_price):
    rows = []
    for period, seed in ((100.0, 6000), (200.0, 6001)):
        result = evaluate_policy(model, dataset, replace(cfg, report_period_ms=period), seed, wire_price)
        rows.append({"period_ms": period, "seed": seed, **result})
    return {"success": float(np.mean([r["success_rate"] for r in rows])),
            "reward": float(np.mean([r["reward_per_arrival"] for r in rows])), "episodes": rows}


def collect_rollout(env, model, codec, teacher=False):
    rows = []
    while env.tick < env.total_steps:
        obs = env.observe()
        z, ages, present, k = report_inputs(obs, codec)
        public = public_features(obs)
        windows, targets = env.training_windows()
        budget_windows = np.zeros_like(windows)
        budget_actions = np.zeros(3, np.int64)
        budget_logprobs = np.zeros(3, np.float32)
        budget_present = np.zeros(3, bool)
        if len(codec.events) > 3:
            raise AssertionError("More than one report per node at a decision instant")
        for n, event in enumerate(codec.events):
            budget_windows[n], budget_actions[n] = event["window"], event["action"]
            budget_logprobs[n], budget_present[n] = event["logprob"], True
        codec.events.clear()
        with torch.no_grad():
            logits, value = model.forward_z(
                *(torch.as_tensor(a)[None] for a in (z, ages, present, public)), obs["slot_ms"])
            dist = Categorical(logits=logits[0])
            action = int(dist.sample()) if obs["request"] is not None else 0
            if teacher and obs["request"] is not None:
                # A causal public-information prior: local node, cheapest model
                # satisfying the same public quality eligibility as every actor.
                score = 10 * public[:, 13] - public[:, 8]
                score[logits[0].numpy() < -1e8] = -np.inf
                action = int(score.argmax())
            logprob = float(dist.log_prob(torch.tensor(action)))
        _, reward, done, info = env.step(action if obs["request"] is not None else None)
        rows.append(dict(windows=windows, targets=targets, ages=ages, present=present, k=k,
                         public=public, action=action, logprob=logprob, value=float(value),
                         reward=reward / 10, done=done, elapsed=info["elapsed_ms"],
                         mask=obs["request"] is not None, budget_windows=budget_windows,
                         budget_actions=budget_actions, budget_logprobs=budget_logprobs,
                         budget_present=budget_present))
    return rows


def warm_start(model, dataset, cfg, epochs, seed, output, wire_price):
    """Initialize routing from a public-only feasible rule, equally for all methods.

    Encoder/filter remain fixed. This is supervised initialization, not a claimed
    PPO gain. It avoids spending most requests exploring obviously costly models.
    """
    torch.manual_seed(seed + 250)
    rows = []
    for trajectory, period in ((3000, 100.0), (3001, 200.0)):
        codec = PredictiveCodec(model, stochastic=True)
        env = ReportingEnv(dataset, replace(cfg, report_period_ms=period), trajectory,
                           codec, "periodic", wire_price=wire_price)
        rows.extend(collect_rollout(env, model, codec, teacher=True))
    data = tensors(rows)
    active = data["mask"]
    indices = torch.arange(len(rows))[active]
    # Only the actor is optimized here, so predictive representation is unchanged.
    optimizer = torch.optim.Adam(model.actor.parameters(), lr=1e-3)
    with torch.no_grad():
        z = model.encode(data["windows"], data["k"]).detach()
    logs = []
    for epoch in range(1, epochs + 1):
        losses = []
        for idx in indices[torch.randperm(len(indices))].split(128):
            logits, _ = model.forward_z(z[idx], *(data[k][idx] for k in
                                                 ("ages", "present", "public")), cfg.slot_ms)
            loss = nn.functional.cross_entropy(logits, data["action"][idx])
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            losses.append(float(loss.detach()))
        logs.append({"epoch": epoch, "cross_entropy": float(np.mean(losses))})
    dump(output / "warm_start.json", {"epochs": epochs, "examples": len(indices),
         "seeds": [3000, 3001], "teacher": "local node + cheapest publicly quality-eligible model",
         "encoder_frozen": True, "logs": logs})


def distribution_diagnostics(model, data, slot_ms):
    logratios = []
    with torch.no_grad():
        for index in torch.arange(len(data["action"])).split(512):
            logits, _ = model(*(data[k][index] for k in
                                ("windows", "k", "ages", "present", "public")), slot_ms)
            lr = Categorical(logits=logits).log_prob(data["action"][index]) - data["logprob"][index]
            logratios.append(lr[data["mask"][index]])
    lr = torch.cat(logratios)
    result = {"approx_kl": float((lr.exp() - 1 - lr).mean()),
            "clip_fraction": float(((lr.exp() - 1).abs() > 0.2).float().mean()),
            "max_logprob_difference": float(lr.abs().max())}
    if model.selection == "adaptive":
        with torch.no_grad():
            active = data["budget_present"]
            d = Categorical(logits=model.budget_logits(data["budget_windows"][active]))
            lr = d.log_prob(data["budget_actions"][active]) - data["budget_logprobs"][active]
            result["budget_kl"] = float((lr.exp() - 1 - lr).mean())
            result["budget_logprob_difference"] = float(lr.abs().max())
    return result


def joint_ppo(model, name, dataset, cfg, valid, updates, seed, output, val_every=5, wire_price=1.0):
    torch.manual_seed(seed + 300)
    joint = SPECS[name]["joint"]
    model.encoder.requires_grad_(joint and model.mode != "stats")
    model.importance.requires_grad_(joint and model.selection in ("importance", "adaptive"))
    model.budget_actor.requires_grad_(model.selection == "adaptive")
    groups = [
        {"params": list(model.actor.parameters()), "lr": 1e-4},
        {"params": list(model.critic.parameters()), "lr": 3e-4},
        {"params": list(model.predictor.parameters()), "lr": 3e-4},
    ]
    representation_parameters = [p for component in (model.encoder, model.importance)
                                 for p in component.parameters() if p.requires_grad]
    if representation_parameters:
        groups.append({"params": representation_parameters, "lr": 5e-5})
    if model.selection == "adaptive":
        groups.append({"params": list(model.budget_actor.parameters()), "lr": 3e-4})
    optimizer = torch.optim.Adam(groups)
    logs, validation = [], []
    initial_hash = parameter_hash(model)
    best_score, best_update = -np.inf, 0

    def checkpoint(update):
        nonlocal best_score, best_update
        pred = validation_prediction(model, valid)
        score = rollout_validation(model, dataset, cfg, wire_price)
        row = {"update": update, **pred, "validation_success": score["success"],
               "validation_reward": score["reward"], "episodes": score["episodes"]}
        validation.append(row)
        saved = {"model": model.state_dict(), "method": name, "config": asdict(cfg),
                 "train_seed": seed, "update": update, "spec": SPECS[name], "wire_price": wire_price}
        torch.save(saved, output / "last.pt")
        torch.save(saved, output / f"checkpoint_{update:03d}.pt")
        if score["reward"] > best_score:
            best_score, best_update = score["reward"], update
            torch.save(saved, output / "best.pt")
        dump(output / "validation.json", validation)
        print(f"{name} seed={seed} update={update} validation_SLA={100*score['success']:.2f}% "
              f"forecast_mse={pred['forecast_mse']:.5f}", flush=True)

    checkpoint(0)
    for update in range(1, updates + 1):
        model.eval()
        rows, metrics = [], []
        for episode in range(2):
            sequence = 2 * (update - 1) + episode
            period = (50.0, 100.0, 200.0)[sequence % 3]
            env = ReportingEnv(dataset, replace(cfg, report_period_ms=period), 4000 + sequence,
                               PredictiveCodec(model, stochastic=True), "periodic", wire_price=wire_price)
            sample = collect_rollout(env, model, env.codec)
            rows.extend(sample)
            metrics.append({"period_ms": period, "success": env.results()["success_rate"],
                            "reward": sum(r["reward"] for r in sample) * 10 /
                                      max(1, env.results()["arrivals"])})
        data = tensors(rows)
        before = distribution_diagnostics(model, data, cfg.slot_ms)
        if max(before["max_logprob_difference"], before.get("budget_logprob_difference", 0)) > 1e-4:
            raise AssertionError(f"Wire/recomputed policy mismatch: {before}")
        adv, returns = advantages(rows, cfg.slot_ms, gamma=0.995, lam=0.98)
        adv = torch.as_tensor(adv, dtype=torch.float32)
        returns = torch.as_tensor(returns, dtype=torch.float32)
        mask = data["mask"]
        adv = (adv - adv[mask].mean()) / adv[mask].std(unbiased=False).clamp_min(1e-6)
        steps = []
        model.train()
        for epoch in range(4):
            for index in torch.randperm(len(rows)).split(256):
                logits, value = model(*(data[k][index] for k in
                                       ("windows", "k", "ages", "present", "public")), cfg.slot_ms)
                dist = Categorical(logits=logits)
                ratio = (dist.log_prob(data["action"][index]) - data["logprob"][index]).exp()
                active = mask[index]
                policy = (-torch.minimum(ratio * adv[index], ratio.clamp(0.8, 1.2) * adv[index])
                          [active].mean()) if active.any() else value.sum() * 0
                entropy = dist.entropy()[active].mean() if active.any() else value.sum() * 0
                value_loss = 0.5 * (value - returns[index]).square().mean()
                z, scores, _ = model.representation(data["windows"][index], data["k"][index])
                valid_reports = data["present"][index]
                prediction = (model.forecast(z) - data["targets"][index]).square().mean((-1, -2))
                prediction_loss = (prediction * valid_reports).sum() / valid_reports.sum().clamp_min(1)
                counts = (data["k"][index][valid_reports > 0] - 1).clamp(1, 3) if model.selection == "adaptive" else None
                rate_loss = model.bandwidth_penalty(scores[valid_reports > 0], counts) if valid_reports.any() else z.sum() * 0
                budget_policy, budget_entropy = z.sum() * 0, z.sum() * 0
                budget_active = data["budget_present"][index]
                if model.selection == "adaptive" and budget_active.any():
                    budget_dist = Categorical(logits=model.budget_logits(data["budget_windows"][index]))
                    br = (budget_dist.log_prob(data["budget_actions"][index]) - data["budget_logprobs"][index]).exp()
                    ba = adv[index, None].expand_as(br)
                    budget_policy = -torch.minimum(br * ba, br.clamp(0.8, 1.2) * ba)[budget_active].mean()
                    budget_entropy = budget_dist.entropy()[budget_active].mean()
                loss = (policy + budget_policy + value_loss - 0.01 * (entropy + budget_entropy)
                        + 0.5 * prediction_loss + 0.001 * rate_loss)
                if not torch.isfinite(loss):
                    raise FloatingPointError("Non-finite PPO loss")
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                for group in groups:
                    nn.utils.clip_grad_norm_(group["params"], 0.5)
                optimizer.step()
                steps.append([float(t.detach()) for t in
                              (policy, value_loss, entropy, prediction_loss, rate_loss, budget_policy, budget_entropy)])
            after = distribution_diagnostics(model, data, cfg.slot_ms)
            if max(after["approx_kl"], after.get("budget_kl", 0)) > 0.02:
                break
        means = np.mean(steps, axis=0)
        record = {"update": update, "train_success": float(np.mean([m["success"] for m in metrics])),
                  "train_reward": float(np.mean([m["reward"] for m in metrics])),
                  "policy_loss": float(means[0]), "value_loss": float(means[1]),
                  "entropy": float(means[2]), "prediction_loss": float(means[3]),
                  "bandwidth_penalty": float(means[4]), "ppo_epochs": epoch + 1,
                  "budget_policy_loss": float(means[5]), "budget_entropy": float(means[6]),
                  "initial_max_logprob_difference": before["max_logprob_difference"],
                  **after, "episodes": metrics}
        logs.append(record)
        dump(output / "ppo.json", logs)
        if update % val_every == 0 or update == updates:
            model.eval()
            checkpoint(update)
    if not all(torch.isfinite(p).all() for p in model.parameters()):
        raise FloatingPointError("Non-finite trained parameter")
    result = {"initial_parameter_hash": initial_hash, "final_parameter_hash": parameter_hash(model),
              "best_update": best_update, "best_validation_reward": best_score,
              "updates": updates, "joint": joint,
              "convergence_note": "Empirical curves and last checkpoints; no global convergence proof."}
    dump(output / "training_summary.json", result)
    return result


def prepare(args):
    args.output.mkdir(parents=True, exist_ok=True)
    cfg = Config(arrival_steps=args.train_steps, horizon=60)
    training = collect_pretraining(args.dataset, cfg, range(1000, 1006))
    valid = collect_pretraining(args.dataset, cfg, (2000, 2001))
    np.savez_compressed(args.output / "histories.npz", train_x=training[0], train_y=training[1],
                        valid_x=valid[0], valid_y=valid[1])
    dump(args.output / "data_metadata.json", {"config": asdict(cfg),
         "dataset_hash": hashlib.sha256(args.dataset.read_bytes()).hexdigest(),
         "training_seeds": list(range(1000, 1006)), "validation_seeds": [2000, 2001],
         "train_windows": len(training[0]), "validation_windows": len(valid[0])})


def train(args):
    cfg = Config(arrival_steps=args.train_steps, horizon=60)
    metadata = json.loads((args.data / "data_metadata.json").read_text())
    if metadata["config"] != asdict(cfg):
        raise ValueError("Prepared data configuration differs")
    if metadata["dataset_hash"] != hashlib.sha256(args.dataset.read_bytes()).hexdigest():
        raise ValueError("Prepared dataset differs")
    data = np.load(args.data / "histories.npz")
    training = tuple(torch.as_tensor(data[k]) for k in ("train_x", "train_y"))
    valid = tuple(torch.as_tensor(data[k]) for k in ("valid_x", "valid_y"))
    for name in args.methods:
        output = args.output / f"seed{args.train_seed}" / name
        output.mkdir(parents=True, exist_ok=False)
        start = perf_counter()
        model = make_model(name, cfg, args.train_seed)
        dump(output / "metadata.json", {"method": name, "spec": SPECS[name], "config": asdict(cfg),
             "train_seed": args.train_seed, "epochs": args.epochs, "updates": args.updates,
             "wire_price_per_32B": args.wire_price,
             "gamma_per_5ms": 0.995, "gae_lambda_per_5ms": 0.98,
             "packet_bytes": 16 + 4 * model.wire_floats, "data_metadata": metadata,
             "ppo_training_seeds": [4000, 4000 + 2 * args.updates - 1],
             "rollout_validation_seeds": [6000, 6001], "warm_start_epochs": args.warm_epochs,
             "actor_input": "received Z, age, missing flag and public request/model metadata",
             "forecast_targets": "future compute and two outgoing link availability trajectories",
             "loss": "PPO + 0.5*prediction_MSE + 0.001*exp(sum(soft_scores)-k); no reconstruction or IB",
             "update_protocol": "weights fixed for complete episodes; gradients use source windows of received reports",
             "training_communication": "offline centralized simulator training; gradients/model synchronization are not charged to inference reporting"})
        prediction_pretrain(model, training, valid, args.epochs, args.train_seed, output)
        if args.warm_epochs:
            warm_start(model, args.dataset, cfg, args.warm_epochs, args.train_seed, output, args.wire_price)
        summary = joint_ppo(model, name, args.dataset, cfg, valid, args.updates, args.train_seed,
                            output, args.val_every, args.wire_price)
        summary["seconds"] = perf_counter() - start
        dump(output / "training_summary.json", summary)
        print(f"completed {name} seed={args.train_seed} seconds={summary['seconds']:.1f}", flush=True)


def evaluate(args):
    args.output.mkdir(parents=True, exist_ok=True)
    rows = []
    for seed in args.train_seeds:
        for name in args.methods:
            for which in args.checkpoints:
                path = args.models / f"seed{seed}" / name / f"{which}.pt"
                saved = torch.load(path, map_location="cpu", weights_only=True)
                cfg = Config(**saved["config"])
                model = make_model(name, cfg, seed)
                model.load_state_dict(saved["model"])
                for test_seed in args.eval_seeds:
                    for period in args.periods:
                        cfg_eval = replace(cfg, arrival_steps=args.eval_steps, report_period_ms=period)
                        r = evaluate_policy(model, args.dataset, cfg_eval, test_seed, saved["wire_price"])
                        rows.append({"method": name, "train_seed": seed, "test_seed": test_seed,
                                     "checkpoint": which, "update": saved["update"],
                                     "period_ms": period, "packet_bytes": 16 + 4 * model.wire_floats, **r})
                        with (args.output / "episodes.csv").open("w", newline="") as f:
                            writer = csv.DictWriter(f, fieldnames=list(rows[0]))
                            writer.writeheader()
                            writer.writerows(rows)
                print(f"evaluated {name} train_seed={seed} checkpoint={which}", flush=True)
    for test_seed in args.eval_seeds:
        hashes = {r["trace_hash"] for r in rows if r["test_seed"] == test_seed}
        if len(hashes) != 1:
            raise AssertionError("Unmatched exogenous test traces")
    dump(args.output / "metadata.json", {"train_seeds": args.train_seeds, "test_seeds": args.eval_seeds,
         "periods_ms": args.periods, "eval_steps": args.eval_steps, "checkpoints": args.checkpoints,
         "dataset_hash": hashlib.sha256(args.dataset.read_bytes()).hexdigest(),
         "paired_trace_hash_check": "passed", "selection": "best selected only by validation reward; last also evaluated"})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", required=True, choices=("prepare", "train", "evaluate"))
    parser.add_argument("--dataset", type=Path, default=Path("data/icc_paper/scenario_2026.json"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--data", type=Path, default=Path("runs/icc_predictive_data"))
    parser.add_argument("--models", type=Path, default=Path("runs/icc_predictive_models"))
    parser.add_argument("--methods", nargs="+", choices=list(SPECS), default=list(SPECS))
    parser.add_argument("--train-seed", type=int, default=7)
    parser.add_argument("--train-seeds", nargs="+", type=int, default=[7, 17])
    parser.add_argument("--eval-seeds", nargs="+", type=int, default=[251, 252, 253])
    parser.add_argument("--periods", nargs="+", type=float, default=[50, 100, 200])
    parser.add_argument("--train-steps", type=int, default=120)
    parser.add_argument("--eval-steps", type=int, default=400)
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--updates", type=int, default=60)
    parser.add_argument("--val-every", type=int, default=5)
    parser.add_argument("--wire-price", type=float, default=1.0)
    parser.add_argument("--warm-epochs", type=int, default=0)
    parser.add_argument("--checkpoints", nargs="+", default=["best", "last"])
    args = parser.parse_args()
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    getattr(__import__(__name__, fromlist=[args.stage]), args.stage)(args)


if __name__ == "__main__":
    main()

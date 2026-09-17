"""Check the project runtime and bundled predictive checkpoints without training."""

import argparse
import importlib
import importlib.metadata
import json
import sys
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--with-sanet', action='store_true')
    args = parser.parse_args()
    if sys.version_info < (3, 11):
        raise SystemExit('Python 3.11+ is required; Python 3.12 is recommended.')
    modules = {'numpy': 'numpy', 'scipy': 'scipy', 'networkx': 'networkx',
               'torch': 'torch', 'matplotlib': 'matplotlib', 'pytest': 'pytest'}
    if args.with_sanet:
        modules.update(tqdm='tqdm', tensorboard='tensorboard')
    versions, errors = {}, []
    for package, module in modules.items():
        try:
            importlib.import_module(module)
            versions[package] = importlib.metadata.version(package)
        except Exception as exc:
            errors.append(f'{package}: {exc}')
    print(json.dumps({'python': sys.version, 'packages': versions}, indent=2))
    if errors:
        raise SystemExit('\n'.join(errors))
    import numpy as np
    import torch
    from scipy.optimize import milp

    from edge_msd.icc_coupled.codecs import load_codec
    from edge_msd.icc_coupled.ppo import HierarchicalPPO
    from edge_msd.icc_paper import load_scenario

    torch.set_num_threads(1)
    assert callable(milp)
    array = np.ones((2, 3), dtype=np.float32)
    assert np.array_equal(torch.from_numpy(array).numpy(), array)
    root = Path(__file__).resolve().parents[1]
    scenario = load_scenario(root / 'data/icc_paper/scenario_2026.json', .4)
    print(f'ICC scenario: {len(scenario.nodes)} nodes, {len(scenario.tasks)} task types')
    for name in ('raw', 'stats16', 'stats4', 'dense4', 'importance', 'ae4'):
        path = root / 'runs/icc_coupled_models' / f'{name}.pt'
        if path.exists():
            codec = load_codec(path)
            payload = codec.encode_wire(np.ones((8, 16), np.float32))
            forecast = codec.forecast_wire(payload)
            assert forecast.shape == (61, 14) and np.isfinite(forecast).all()
            print(f'Codec {name}: {payload.nbytes} payload bytes, forecast OK')
    for mode in ('frozen', 'joint'):
        path = root / f'runs/icc_coupled_v2_ppo_{mode}_seed7/last.pt'
        if path.exists():
            policy = HierarchicalPPO.load(path, resume_optimizer=True)
            print(f'PPO {mode}: update {policy.update_number}, optimizer/RNG checkpoint readable')
    print('Environment and checkpoint checks passed; no training started.')


if __name__ == '__main__':
    main()

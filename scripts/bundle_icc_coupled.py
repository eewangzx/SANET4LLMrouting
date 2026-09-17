"""Bundle the ICC implementation, selected evidence and resumable checkpoints."""

import hashlib
import json
import subprocess
import zipfile
from pathlib import Path


def main():
    root = Path(__file__).resolve().parents[1]
    selected = set()
    for directory in ('edge_msd', 'examples', 'data/icc_paper'):
        selected.update(p for p in (root / directory).rglob('*')
                        if p.is_file() and p.suffix in ('.py', '.json'))
    selected.update(root / name for name in ('pyproject.toml', 'README_ICC_COUPLED.md'))
    selected.update((root / 'tests').glob('test_*.py'))
    selected.update((root / 'docs').glob('*.md'))
    selected.update((root / 'scripts').glob('*.py'))
    for name in ('icc_coupled_models', 'icc_coupled_v2_evaluation',
                 'icc_coupled_v2_ppo_joint_seed7', 'icc_coupled_v2_ppo_frozen_seed7',
                 'icc_coupled_fixed_modes_v2_discounted', 'icc_coupled_v2_summary'):
        selected.update(p for p in (root / 'runs' / name).rglob('*') if p.is_file())
    for pattern in ('icc_coupled_v2*.log', 'icc_coupled_v2_manifest.json',
                    'icc_coupled_v2_latent_sensitivity.*',
                    'icc_coupled_final_tests.log', 'icc_coupled_restored_equivalence_v1.json',
                    'icc_coupled_fast_controller_v2_benchmark.json',
                    'icc_coupled_forecast_integration_audit.json'):
        selected.update((root / 'runs').glob(pattern))
    for name in ('LICENSE', '.gitignore'):
        if (root / name).is_file():
            selected.add(root / name)
    selected = sorted(p for p in selected if p.is_file() and not p.name.endswith('.tmp'))
    manifest = {
        'base_commit': subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=root, text=True).strip(),
        'working_tree_changes_included': True,
        'main_entry': 'README_ICC_COUPLED.md',
        'files': {str(p.relative_to(root)): hashlib.sha256(p.read_bytes()).hexdigest() for p in selected},
    }
    destination = root / 'deliverables' / 'icc_coupled_v2.zip'
    destination.parent.mkdir(exist_ok=True)
    with zipfile.ZipFile(destination, 'w', compression=zipfile.ZIP_DEFLATED) as archive:
        for path in selected:
            archive.write(path, Path('icc_coupled') / path.relative_to(root))
        archive.writestr('icc_coupled/README.md', (root / 'README_ICC_COUPLED.md').read_text())
        archive.writestr('icc_coupled/BUNDLE_MANIFEST.json', json.dumps(manifest, indent=2) + '\n')
    print(destination, f'{destination.stat().st_size / 1048576:.2f} MiB')


if __name__ == '__main__':
    main()

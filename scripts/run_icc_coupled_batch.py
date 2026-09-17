"""Run/resume matched training seeds on allocated local/HPC CPU workers.

Use the machine's existing batch allocation; this script assumes no particular
scheduler, account, partition or remote credentials. It never submits remote jobs.
"""

import argparse
import os
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--python', default=sys.executable)
    parser.add_argument('--workers', type=int, default=2)
    parser.add_argument('--seeds', type=int, nargs='+', default=[7, 8, 9])
    parser.add_argument('--updates', type=int, default=120)
    parser.add_argument('--load', type=float, default=.4)
    parser.add_argument('--prefix', default='runs/icc_coupled_paper')
    parser.add_argument('--dry-run', action='store_true')
    args = parser.parse_args()
    if args.workers < 1 or args.updates < 1 or len(set(args.seeds)) != len(args.seeds):
        parser.error('workers/updates must be positive and seeds must be unique')
    root = Path(__file__).resolve().parents[1]
    jobs = []
    for seed in args.seeds:
        for mode in ('frozen', 'joint'):
            out = Path(f'{args.prefix}_{mode}_seed{seed}')
            cmd = [args.python, '-u', '-m', 'edge_msd.icc_coupled.ppo_experiment',
                   '--load', str(args.load), '--seed', str(seed), '--updates', str(args.updates),
                   '--episodes-per-update', '2', '--validation-every', '12',
                   '--train-seed-base', '100000', '--output', str(out)]
            if mode == 'joint':
                cmd.append('--joint')
            checkpoint = root / out / 'last.pt'
            if checkpoint.exists():
                cmd += ['--resume', str(checkpoint)]
            elif (root / out).exists() and any((root / out).iterdir()):
                parser.error(f'{out} is nonempty without last.pt; inspect it before reusing')
            jobs.append((out, cmd))
    if args.dry_run:
        import shlex
        for _, cmd in jobs:
            print(shlex.join(cmd))
        return
    env = os.environ.copy()
    env.update(OMP_NUM_THREADS='1', MKL_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1')

    def run(job):
        out, cmd = job
        (root / out).parent.mkdir(parents=True, exist_ok=True)
        with (root / Path(f'{out}.log')).open('a') as log:
            result = subprocess.run(cmd, cwd=root, env=env, stdout=log, stderr=subprocess.STDOUT)
        return str(out), result.returncode

    failed = []
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = [pool.submit(run, job) for job in jobs]
        for future in as_completed(futures):
            name, status = future.result()
            print(f'{name}: exit={status}', flush=True)
            if status:
                failed.append(name)
    if failed:
        raise SystemExit(f'Failed jobs (inspect their logs): {failed}')


if __name__ == '__main__':
    main()

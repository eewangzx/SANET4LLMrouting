"""Launch independent predictive compression followed by frozen-codec DQN on HPC4."""
import argparse
from datetime import datetime
import hashlib
import json
from pathlib import Path
import shlex
import shutil
import subprocess


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--reference',type=Path,required=True)
    p.add_argument('--source-archive',type=Path,required=True)
    p.add_argument('--git-commit',required=True)
    p.add_argument('--case',choices=['driving','azure'],required=True)
    p.add_argument('--forecast-epochs',type=int,default=60)
    p.add_argument('--require-trained-rl',action='store_true')
    args=p.parse_args()
    repo=Path('/home/eewangzx/icc_sanet_project')
    python='/home/eewangzx/anaconda3/envs/fnochannel/bin/python'
    label='decoded_trained' if args.require_trained_rl else 'decoded_separate'
    root=repo/'runs'/f'routing_{args.case}_{label}_{datetime.now():%Y%m%d_%H%M%S}'
    root.mkdir(parents=True)
    source=root/'source';source.mkdir()
    subprocess.run(['tar','-xf',str(args.source_archive),'-C',str(source)],check=True)
    reference=root/'reference';reference.mkdir()
    original=json.loads((args.reference/'experiment.json').read_text())
    for name in ('experiment.json','scenario.json','fixed_placement.json','summary.json'):
        shutil.copy2(args.reference/name,reference/name)
    for suffix in ('npz','json'):
        asset=args.reference/f'azure_trace.{suffix}'
        if asset.exists():shutil.copy2(asset,reference/asset.name)
    for method in ('proposed','greedy','random','shortest_queue'):
        out=reference/method;out.mkdir()
        for seed in original['test_seeds']:
            shutil.copy2(args.reference/method/f'test_{seed}.json',out/f'test_{seed}.json')
    settings={**original,'git_commit':args.git_commit,'methods':['decoded_separate'],
        'algorithm':'dqn','purpose':'independent predictive compression, then frozen decoded-forecast-state DQN',
        'end_to_end_forecast':False,'forecast_epochs':args.forecast_epochs,'episodes':48,'updates_per_episode':1000,
        'state_representation':'full 61x14 age-aligned receiver forecast per node instead of Z',
        'codec_initialization':'random importance codec; independent supervised prediction training only',
        'rl_initialization':'fresh 64-wide Q; same predictive cost prior and bounded residual',
        'codec_frozen_during_rl':True,'reference_run':str(args.reference),
        'initial_policy_eligible_for_selection':not args.require_trained_rl,
        'source_archive_sha256':hashlib.sha256(args.source_archive.read_bytes()).hexdigest()}
    # Joint initialization belongs to the reference proposed, never this baseline.
    for key in ('joint_init_sha256','initial_checkpoint_reference','initialization'):
        settings.pop(key,None)
    (root/'experiment.json').write_text(json.dumps(settings,indent=2)+'\n')
    forecast=[python,'-u','scripts/train_separate_forecast.py','--reference',str(reference),
              '--output',str(root/'forecast_codec'),'--epochs',str(args.forecast_epochs)]
    command=[python,'-u','scripts/run_joint_routing.py','--bench','decoded_separate',
        '--algorithm','dqn','--output',str(root/'decoded_separate'),
        '--placement',str(reference/'fixed_placement.json'),'--dataset',str(reference/'scenario.json'),
        '--codec-init',str(root/'forecast_codec/importance.pt'),'--fresh-q',
        '--episodes','48','--updates-per-episode','1000','--epsilon-start','0.05',
        '--train-seed-base',str(original['train_seeds'][0]),
        '--validation-seeds',','.join(map(str,original['validation_seeds'])),
        '--test-seeds',','.join(map(str,original['test_seeds'])),
        '--validate-every','8','--dynamics-profile',original['dynamics_profile']]
    if args.require_trained_rl:command+=['--exclude-initial-selection']
    for flag,key in [('load','load'),('warmup-ms','warmup_ms'),('arrival-ms','arrival_ms'),
                     ('drain-ms','drain_ms'),('report-ms','report_ms'),('report-bps','report_bps')]:
        command+=['--'+flag,str(original[key])]
    if original['dynamics_profile']=='azure':
        command+=['--azure-trace',str(reference/'azure_trace.npz'),
                  '--azure-case',original.get('azure_case','busy_transitions')]
    script=root/'run.sbatch'
    script.write_text('#!/bin/bash\nset -e\ncd '+shlex.quote(str(source))+'\n'+
                      shlex.join(forecast)+'\n'+shlex.join(command)+'\n')
    options=['sbatch','--parsable','--account=khaledgroup',f'--job-name=icc_decoded_{args.case}',
             '--cpus-per-task=4','--mem=32G','--time=02:00:00','--partition=gpu-l20',
             '--qos=l20_qos','--gres=gpu:1','--output='+str(root/'stdout.log'),
             '--error='+str(root/'stdout.log'),str(script)]
    job_id=subprocess.check_output(options,text=True).strip()
    (root/'jobs.json').write_text(json.dumps({'slurm_job_id':job_id,'forecast_command':forecast,
                                             'rl_command':command},indent=2)+'\n')
    print(json.dumps({'root':str(root),'slurm_job_id':job_id},indent=2))


if __name__=='__main__':main()

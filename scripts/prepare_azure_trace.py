"""Aggregate the official ordered Code trace; normalize using training days only."""
import argparse
import csv
from datetime import date
import hashlib
import json
from pathlib import Path

import numpy as np


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--csv',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    args=p.parse_args()
    if args.csv.stat().st_size!=691989454:
        raise ValueError('The official Code CSV is incomplete or has changed; check download')
    seconds_per_bin=5
    bins=7*86400//seconds_per_bin
    counts=np.zeros(bins,np.int64);tokens=np.zeros(bins,np.int64)
    epoch=date(2024,5,10);days={}
    rows=0;previous=-1;first=None;last=None
    with args.csv.open(newline='') as f:
        reader=csv.DictReader(f)
        for row in reader:
            timestamp=row['TIMESTAMP']
            day=timestamp[:10]
            if day not in days:days[day]=(date.fromisoformat(day)-epoch).days
            second=days[day]*86400+int(timestamp[11:13])*3600+int(timestamp[14:16])*60+int(timestamp[17:19])
            if not 0<=second<7*86400 or second<previous:
                raise ValueError('Timestamp range/order does not match Code one-week trace')
            index=second//seconds_per_bin
            counts[index]+=1;tokens[index]+=int(row['ContextTokens'])
            previous=second;rows+=1;last=timestamp
            if first is None:first=timestamp
    if rows!=16803695:raise ValueError('Unexpected Code row count')
    signals=np.stack((counts,tokens),axis=1).astype(np.float64)
    train_end=4*86400//seconds_per_bin
    val_end=5*86400//seconds_per_bin
    lower=np.percentile(signals[:train_end],5,axis=0)
    upper=np.percentile(signals[:train_end],95,axis=0)
    pressure=np.clip((signals-lower)/np.maximum(upper-lower,1.),0,1).astype(np.float32)
    digest=hashlib.sha256()
    with args.csv.open('rb') as f:
        for chunk in iter(lambda:f.read(1024*1024),b''):digest.update(chunk)
    metadata={
        'source_url':'https://github.com/Azure/AzurePublicDataset/releases/download/dataset-llm-2024/AzureLLMInferenceTrace_code_1week.csv',
        'source_sha256':digest.hexdigest(),'rows':rows,'first_utc':first,'last_utc':last,
        'real_seconds_per_bin':seconds_per_bin,'sim_ms_per_bin':5.,
        'time_acceleration':1000,'pressure_fields':['request_count','context_tokens'],
        'generated_tokens_used':False,'normalization':'clip((x-train-p5)/(train-p95-train-p5),0,1)',
        'normalization_lower':lower.tolist(),'normalization_upper':upper.tolist(),
        'ranges':{'train':[0,train_end],'validation':[train_end,val_end],'test':[val_end,bins]},
        'chronological_days':'first four days train, day five validation, last two days test',
        'resource_mapping':'background pressure reduces light availability; endpoint request pressure reduces link rate',
        'simulation_label':'Azure trace-driven ICC driving simulation; resources and links are modeled, not measured',
        'correlation':{},
    }
    for width in (1,2,12):
        grouped=signals[:len(signals)//width*width].reshape(-1,width,2).sum(1)
        metadata['correlation'][str(width*seconds_per_bin)+'s']={
            name:float(np.corrcoef(grouped[:-1,i],grouped[1:,i])[0,1])
            for i,name in enumerate(metadata['pressure_fields'])}
    args.output.parent.mkdir(parents=True,exist_ok=True)
    np.savez_compressed(args.output,pressure=pressure,request_count=counts,context_tokens=tokens,
                        metadata=np.asarray(json.dumps(metadata)))
    args.output.with_suffix('.json').write_text(json.dumps(metadata,indent=2)+'\n')
    print(json.dumps(metadata,indent=2),flush=True)


if __name__=='__main__':main()

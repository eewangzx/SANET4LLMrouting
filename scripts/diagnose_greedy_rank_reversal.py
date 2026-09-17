"""Count stale-state Greedy rank reversals without changing its decisions."""
from pathlib import Path
import importlib.util
import json
import sys

root=Path(sys.argv[1]).resolve();seed=int(sys.argv[2])
spec=importlib.util.spec_from_file_location('routing_runner',root/'source/scripts/run_joint_routing.py')
runner=importlib.util.module_from_spec(spec);spec.loader.exec_module(runner)
original=runner.DynamicRoutingEnvironment.greedy_node
rows=[]

def greedy_node(self):
    selected=original(self);stage=self._routable_stage();service=self.scenario.services[stage.service]
    predicted={};future={};ready={}
    for node in self.legal_nodes(stage):
        i=self.resource.node_index[node]
        current=1. if service.kind=='core' else float(self.forecast[i,0,self.resource.service_index[stage.service]])
        predicted[node]=max(self.now,self.observed_network.data_ready(stage,node))+(
            self.backlog(stage,node)+1)*service.mean_processing_ms/max(.15,current)
        ready[node]=self.network.data_ready(stage,node)
        realized=1. if service.kind=='core' else float(self.resource.current(ready[node])[
            i,self.resource.service_index[stage.service]])
        future[node]=max(self.now,ready[node])+(
            self.backlog(stage,node)+1)*service.mean_processing_ms/max(.15,realized)
    best=min(predicted,key=lambda n:(predicted[n],n));actual=min(future,key=lambda n:(future[n],n))
    assert selected==best
    values=sorted(future.values());rank=1+sum(v<future[selected]-1e-9 for v in values)
    report=self.received[self.resource.node_index[selected]]
    rows.append({'request_id':stage.request.id,'service':stage.service,'kind':service.kind,
        'now_ms':self.now,'selected':selected,'actual_best':actual,'legal_nodes':len(future),
        'future_rank':rank,'becomes_worst':rank==len(future),'rank_reversed':actual!=selected,
        'selected_ready_delay_ms':ready[selected]-self.now,
        'selected_report_age_ms':None if report is None else self.now-report[0],
        'predicted_score':predicted[selected]-self.now,'future_score':future[selected]-self.now})
    return selected

runner.DynamicRoutingEnvironment.greedy_node=greedy_node
command=json.loads((root/'jobs.json').read_text())['greedy']['command']
out=root/'rank_reversal'/str(seed)
command[command.index('--output')+1]=str(out)
command[command.index('--test-seeds')+1]=str(seed)
sys.argv=['run_joint_routing.py',*command[3:]];runner.main()
replayed=json.loads((out/f'test_{seed}.json').read_text());reference=json.loads((root/'greedy'/f'test_{seed}.json').read_text())
for key in ('arrival_sha256','resource_sha256','arrivals','ontime','pending','requests'):
    if replayed[key]!=reference[key]:raise ValueError(f'diagnostic changed {key}')
def summary(items):
    return {'decisions':len(items),'rank_reversed':sum(r['rank_reversed'] for r in items),
            'became_worst':sum(r['becomes_worst'] for r in items),
            'rank_reversal_fraction':sum(r['rank_reversed'] for r in items)/len(items),
            'became_worst_fraction':sum(r['becomes_worst'] for r in items)/len(items),
            'mean_report_age_ms':sum(r['selected_report_age_ms'] for r in items)/len(items),
            'mean_ready_delay_ms':sum(r['selected_ready_delay_ms'] for r in items)/len(items)}
light=[r for r in rows if r['kind']=='light'];audio=[r for r in light if r['service']=='audio_preprocess']
result={'method':'greedy','seed':seed,'replay_identical':True,
        'definition':'decision-time queues held fixed; compare stale reported score with physical future link/service trace at candidate input-ready time',
        'light_stages':summary(light),'audio_preprocess':summary(audio)}
(root/'rank_reversal').mkdir(exist_ok=True);(root/'rank_reversal'/f'summary_{seed}.json').write_text(json.dumps(result,indent=2)+'\n')
(root/'rank_reversal'/f'rows_{seed}.json').write_text(json.dumps(rows,indent=2)+'\n')
print(json.dumps(result,indent=2))

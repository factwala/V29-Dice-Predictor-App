#!/usr/bin/env python3
"""V28 guarded pair-level calibration; strictly lagged gating on prior expert vs base outcomes.
Candidate sets fixed before this run: 4 alternative forecasters, 2 windows, 2 gates.
"""
from pathlib import Path
import json,csv,hashlib
import numpy as np
import v28_research as v28
import v27_research as v27
import v20_research as v20
from v15_research import score
P=Path(__file__).resolve().parent
ADVISORS=('pair_residual_w100_low','pair_residual_w200_low','face_residual_w100_low','contextsum_w200_low')
POLICIES=('win2','win3')
WINDOWS=(32,96)

def past_roll(arr,wind):
    return v28.prior_roll(np.asarray(arr),wind)

def make_gate(d,base,advisor,wind,policy):
    y=d['y'];n=len(y);base_r,bhit,_=score(base,y,d['tie']);ar,ahit,_=score(advisor,y,d['tie'])
    delta=ahit.astype(int)-bhit.astype(int)
    win=past_roll((delta==1).astype(float),wind);lose=past_roll((delta==-1).astype(float),wind)
    diff=past_roll((ar[:,0]!=base_r[:,0]).astype(float),wind)
    if policy=='win2':
        gate=(win>=2)&(win>lose)&(diff>=3)
    else:
        gate=(win>=3)&(win>=lose+2)&(diff>=5)
    # skip warmup 40; otherwise there are too few past examples to estimate utility
    gate[:40]=False
    out=base.copy();out[gate]=advisor[gate]
    return out,gate

def main():
    path=P/'throws(2).txt';assert hashlib.sha256(path.read_bytes()).hexdigest()==v20.V.SHA
    rows=v20.V.load(path);d,models=v27.build_experts(rows);d['rows']=rows;base=models['V26_255']
    scored=[];out={'V26_REFERENCE':base}
    scored.append(v28.summarize('V26_REFERENCE',base,base,d))
    for name in ADVISORS:
        alt=v28.estimate(d,base,next(s for s in v28.CANDIDATES if s[0]==name))
        for window in WINDOWS:
            for policy in POLICIES:
                nm=f'guard_{name}_w{window}_{policy}'
                p,gate=make_gate(d,base,alt,window,policy)
                rec=v28.summarize(nm,p,base,d);rec['gate_activated']=int(gate.sum());out[nm]=p;scored.append(rec)
    selected=max(scored[1:],key=lambda z:(z['first504'],z['validation_351_504'],-z['avg_logloss']))
    hist=max(scored[1:],key=lambda z:(z['hits'],z['later_505_672'],-z['avg_logloss']))
    with (P/'guarded_results.csv').open('w',newline='') as f:
        fields=list(scored[1]);w=csv.DictWriter(f,fieldnames=fields);w.writeheader();w.writerows(scored[1:])
    (P/'guarded_selection.json').write_text(json.dumps({'selection_first504':selected,'historical':hist,'baseline':scored[0],'candidate_count':len(scored)-1},indent=2,ensure_ascii=False))
    print(json.dumps({'selection_first504':selected,'historical':hist,'baseline':scored[0],'candidate_count':len(scored)-1},indent=2,ensure_ascii=False))
if __name__=='__main__':main()

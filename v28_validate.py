#!/usr/bin/env python3
"""Replay and prefix test for V28 guarded forecasts."""
from pathlib import Path
import json,hashlib,csv
import numpy as np
import v28_guarded as gd
import v28_research as vr
import v27_research as v27
import v20_research as v20
from v15_research import score
P=Path(__file__).resolve().parent
SPEC=next(x for x in vr.CANDIDATES if x[0]=='face_residual_w100_low')

def run_model(rows):
    d,models=v27.build_experts(rows);d['rows']=rows
    base=models['V26_255'];adv=vr.estimate(d,base,SPEC)
    out,g=gd.make_gate(d,base,adv,32,'win2')
    return d,base,out,g

def main():
    orig=P/'throws(2).txt';sha=hashlib.sha256(orig.read_bytes()).hexdigest();assert sha==v20.V.SHA
    rows=v20.V.load(orig);assert len(rows)==672
    d,base,v28,g=run_model(rows);rb,hb,lb=score(base,d['y'],d['tie']);r,h,loss=score(v28,d['y'],d['tie'])
    assert int(h.sum())==255 and int(hb.sum())==255
    assert int(sum(set(rb[t])!=set(r[t]) for t in range(len(r))))<=int(g.sum())
    # Test that editing the current result without changing history cannot change its prequential forecast.
    checks=[1,50,201,451,672]
    for throw in checks:
        mutated=list(rows[:throw]);a,b,c=mutated[-1];mutated[-1]=(3-a,7-b,c)
        dd,bb,pp,gg=run_model(mutated)
        assert np.allclose(pp[-1],v28[throw-1],atol=1e-11),f'leak at {throw}'
    with (P/'LIVE_673_693.csv').open(newline='') as f:
        live=[tuple(map(int,r[1:])) for r in csv.reader(f)]
    assert len(live)==21
    allrows=rows+live;dd,bb,pp,gg=run_model(allrows)
    _,hk,_=score(pp,dd['y'],dd['tie']);_,hkbase,_=score(bb,dd['y'],dd['tie'])
    assert np.allclose(pp[:672],v28)
    # Alternative independent explicit per-row gate replay with past outcomes.
    dadv=vr.estimate(d,base,SPEC);ra,had,_=score(dadv,d['y'],d['tie'])
    independ=base.copy()
    for t in range(len(rows)):
        win=slice(max(0,t-32),t)
        wins=int(np.sum(had[win]&~hb[win]));loses=int(np.sum(~had[win]&hb[win]))
        ndis=int(np.sum(ra[win,0]!=rb[win,0]))
        if t>=40 and wins>=2 and wins>loses and ndis>=3:
            independ[t]=dadv[t]
    assert np.allclose(independ,v28)
    report={'original_sha256':sha,'throw_count':672,'V26_top5_hits':int(hb.sum()),'V28_top5_hits':int(h.sum()),
      'new_hits':int(np.sum(h&~hb)),'lost_hits':int(np.sum(hb&~h)),
      'base_logloss':float(lb.mean()),'V28_logloss':float(loss.mean()),
      'gate_active':int(g.sum()),'top5_sets_changed':int(sum(set(rb[t])!=set(r[t]) for t in range(len(r)))),
      'prefix_tests_passed':checks,'streaming_gate_replay':True,
      'known_followup':{'n':21,'V26':int(hkbase[672:].sum()),'V28':int(hk[672:].sum()),'previously_seen':True}}
    (P/'validation_v28.json').write_text(json.dumps(report,indent=2,ensure_ascii=False))
    print(json.dumps(report,indent=2,ensure_ascii=False))
if __name__=='__main__':main()

#!/usr/bin/env python3
"""V22 locked-candidate repeat, rolling-prefix replay, pair-entrant audit."""
import csv,json,hashlib,math
from pathlib import Path
import numpy as np
from scipy.stats import binomtest
import v22_research as mod
import v20_research as v
from v15_research import score
R=Path(__file__).resolve().parent

SPECS={
 'V22_249_retrospective':{'expert':'face25','w':24,'freqw':100,'utility_threshold':0,'margin':1,'mix':.5},
 'V22_247_early_selected':{'expert':'face25','w':24,'freqw':100,'utility_threshold':-.04,'margin':1,'mix':.5}
}

def predict(rows,spec):
    d,experts,base,_=mod.get_base(rows)
    n=len(rows); y=d['y'];tie=d['tie'];q=experts[spec['expert']]
    rb,hb,_=score(base,y,tie);rq,hq,_=score(q,y,tie)
    actual=np.eye(21)[y]; mix=spec['mix']; qpred=np.array(base,copy=True)
    on=np.zeros(n,bool);util=np.zeros(n,float);gg=np.zeros(n,int);ll=np.zeros(n,int)
    # Separate streaming implementation: at each t, history is only [0,t) (never row t).
    for t in range(n):
        a=max(0,t-spec['w']);start=max(0,t-spec['freqw'])
        gw=sum(bool(hq[k] and not hb[k]) for k in range(a,t))
        gl=sum(bool(hb[k] and not hq[k]) for k in range(a,t))
        prior=(actual[start:t].sum(axis=0)+20*v.V.FAIR)/(t-start+20)
        added=set(map(int,rq[t]))-set(map(int,rb[t]));removed=set(map(int,rb[t]))-set(map(int,rq[t]))
        utility=sum(prior[x] for x in added)-sum(prior[x] for x in removed)
        gg[t]=gw;ll[t]=gl;util[t]=utility
        on[t]=bool(added) and utility >=spec['utility_threshold'] and gw-gl>=spec['margin'] and gw+gl>=2
        if on[t]:qpred[t]=(1-mix)*base[t]+mix*q[t]
    return d,base,qpred,on,util,gg,ll

def csv_save(path,records):
    with open(path,'w',encoding='utf-8',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(records[0]));w.writeheader();w.writerows(records)

if __name__=='__main__':
    master=R/'throws(2).txt';assert hashlib.sha256(master.read_bytes()).hexdigest()==v.V.SHA
    rows=v.V.load(master);assert len(rows)==672
    d,b,p,on,ut,gw,gl=predict(rows,SPECS['V22_249_retrospective'])
    r21,h21,_=score(b,d['y'],d['tie']);r22,h22,_=score(p,d['y'],d['tie'])
    assert h21.sum()==247 and h22.sum()==249
    extra=np.flatnonzero(h22&~h21)+1; lost=np.flatnonzero(h21&~h22)+1
    assert not len(extra)<0
    print('HITS',h21.sum(),h22.sum(),'WIN',extra.tolist(),'LOSS',lost.tolist(), 'ACTIVE',int(on.sum()),flush=True)
    records=[]
    for i,z in enumerate(rows):
        def fmt(ids):return ';'.join(','.join(map(str,v.V.PAIRS[j])) for j in ids)
        records.append({'throw':i+1,'original_triple':','.join(map(str,z)), 'actual_pair':','.join(map(str,v.V.pair(z))),
          'V21_top5':fmt(r21[i]),'V21_hit':int(h21[i]),'V22_top5':fmt(r22[i]),'V22_hit':int(h22[i]),
          'pair_entry_gate':int(on[i]),'pair_entry_prior_advantage':round(float(ut[i]),6),
          'past_24_expert_win':int(gw[i]),'past_24_expert_loss':int(gl[i]),
          'new_hit':int(h22[i]&~h21[i]),'lost_hit':int(h21[i]&~h22[i])})
    csv_save(R/'audit_672.csv',records)
    checks=[]
    for pos in (1,50,101,251,351,451,504,600,672):
        altered=list(rows);old=altered[pos-1];altered[pos-1]=(3-old[0],7-old[1],old[2])
        d2,b2,p2,*_=predict(altered,SPECS['V22_249_retrospective'])
        ri=score(p2,d2['y'],d2['tie'])[0]; ref=r22[:pos]
        ok=bool(np.array_equal(ref,ri[:pos]))
        if not ok: raise AssertionError(f'Prefix leakage t={pos}')
        checks.append({'throw':pos,'prefix_unchanged':ok})
    with (R/'LIVE_673_693.csv').open() as f:
        live=[tuple(map(int,row[1:])) for row in csv.reader(f)]
    assert len(live)==21
    dl,bl,pl,onl,*_=predict(rows+live,SPECS['V22_249_retrospective'])
    hbl=score(bl,dl['y'],dl['tie'])[1];hpl=score(pl,dl['y'],dl['tie'])[1]
    ra=score(pl,dl['y'],dl['tie'])[0]
    assert np.array_equal(ra[:672],r22)
    sec= {'historical':{'V21':int(h21.sum()),'V22':int(h22.sum()), 'delta':int(h22.sum()-h21.sum()),'new_hit_throws':extra.tolist(),'lost_hit_throws':lost.tolist(),
          'gate_on':int(on.sum()),'different_set_count':int(sum(set(r22[i])!=set(r21[i]) for i in range(672))),
          'sign_test_p':float(binomtest(len(extra),len(extra)+len(lost),.5).pvalue) if len(extra)+len(lost) else 1.},
          'folds':{name:{'V21':int(h21[a:c].sum()),'V22':int(h22[a:c].sum())} for name,(a,c) in mod.FOLDS.items()},
          'checks':checks,'known_673_693':{'V21':int(hbl[672:].sum()),'V22':int(hpl[672:].sum()),'n':21,'not_blind':True}}
    print(json.dumps(sec,indent=2)); (R/'validation.json').write_text(json.dumps(sec,ensure_ascii=False,indent=2))

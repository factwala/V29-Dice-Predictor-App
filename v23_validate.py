#!/usr/bin/env python3
"""Independent streaming verification for a LOCKED V23 candidate (no answer access at time t)."""
import csv, json, hashlib
from pathlib import Path
import numpy as np
from scipy.stats import binomtest
import v22_validate as old
import v22_research as older
import v20_research as v
from v15_research import score
ROOT=Path(__file__).resolve().parent
SELECTED={'expert':'face25','lookback':15,'metric':'freq','threshold':.03,'evidence_w':24,'minwin':0,'minn':2,'alpha':.25}

def predict(rows, spec=SELECTED):
    d, _, base, on, *_=old.predict(rows, old.SPECS['V22_249_retrospective'])
    _, ex, _, _=older.get_base(rows)
    y,tie=d['y'],d['tie'];n=len(rows)
    alt=ex[spec['expert']]
    rb,hb,_=score(base,y,tie); rq,hq,_=score(alt,y,tie)
    truth=np.eye(21)[y]
    pred=base.copy()
    active=np.zeros(n,dtype=bool)
    utility=np.zeros(n,float)
    evidence=np.zeros((n,2),int)
    # Independent loop intentionally uses rows ending at t-1 only for *all* gate features.
    for t in range(n):
        start=max(0,t-spec['evidence_w']);fst=max(0,t-spec['lookback'])
        wins=sum(int(hq[j] and not hb[j]) for j in range(start,t))
        losses=sum(int(hb[j] and not hq[j]) for j in range(start,t))
        evidence[t]=(wins,losses)
        freq=(truth[fst:t].sum(axis=0)+20*v.V.FAIR)/(t-fst+20)
        added=set(map(int,rq[t]))-set(map(int,rb[t]));removed=set(map(int,rb[t]))-set(map(int,rq[t]))
        utility[t]=sum(freq[j] for j in added)-sum(freq[j] for j in removed)
        # each mix decided without referencing y[t]
        active[t]=bool(added) and utility[t]>=spec['threshold'] and wins-losses>=spec['minwin'] and wins+losses>=spec['minn']
        if active[t]:pred[t]=(1-spec['alpha'])*base[t]+spec['alpha']*alt[t]
    return d,base,pred,active,utility,evidence

def fmt(rank):return ';'.join(','.join(map(str,v.V.PAIRS[j])) for j in rank)

def write_csv(path,records):
    with open(path,'w',newline='',encoding='utf-8') as f:
        w=csv.DictWriter(f,fieldnames=list(records[0]));w.writeheader();w.writerows(records)

if __name__=='__main__':
    master=ROOT/'throws(2).txt';assert hashlib.sha256(master.read_bytes()).hexdigest()==v.V.SHA
    rows=v.V.load(master);assert len(rows)==672
    d,p22,p23,gate,util,ev=predict(rows)
    rank22,h22,_=score(p22,d['y'],d['tie']);rank23,h23,_=score(p23,d['y'],d['tie'])
    assert int(h22.sum())==249 and int(h23.sum())==250
    got=int(gate.sum());setchange=int(sum(set(rank22[i])!=set(rank23[i]) for i in range(len(rows))))
    gains=list(map(int,np.flatnonzero(h23&~h22)+1));losses=list(map(int,np.flatnonzero(h22&~h23)+1))
    records=[]
    for i,z in enumerate(rows):
        records.append({'throw':i+1,'original_triple':','.join(map(str,z)), 'actual_pair':','.join(map(str,v.V.pair(z))),
            'V22_top5':fmt(rank22[i]),'V22_hit':int(h22[i]),'V23_top5':fmt(rank23[i]),'V23_hit':int(h23[i]),
            'gate_active':int(gate[i]),'top5_set_changed':int(set(rank22[i])!=set(rank23[i])),
            'prior_entry_minus_exit':round(float(util[i]),7),'previous_wins':int(ev[i,0]),'previous_losses':int(ev[i,1]),
            'new_hit':int(h23[i]&~h22[i]),'lost_hit':int(h22[i]&~h23[i])})
    write_csv(ROOT/'audit_672.csv',records)
    checks=[]
    for pos in (1,50,101,200,251,322,351,451,501,504,600,672):
        altered=list(rows);oldrow=altered[pos-1];altered[pos-1]=(3-oldrow[0],7-oldrow[1],oldrow[2])
        da,pa,qa,*_=predict(altered)
        rqa=score(qa,da['y'],da['tie'])[0]
        ok=np.array_equal(rqa[:pos],rank23[:pos]);assert ok,(pos,'leakage')
        checks.append(pos)
    live=[]
    with (ROOT/'LIVE_673_693.csv').open(newline='') as f:
        live=[tuple(map(int,line[1:])) for line in csv.reader(f)]
    assert len(live)==21
    dl,pl,ql,*_=predict(rows+live)
    rbl,hbl,_=score(pl,dl['y'],dl['tie']);rql,hql,_=score(ql,dl['y'],dl['tie'])
    assert np.array_equal(rank23,rql[:672])
    results={'base_V22':int(h22.sum()),'V23':int(h23.sum()),'accuracy':round(float(h23.mean()*100),5),
       'new_hits':gains,'lost_hits':losses,'gate_active':got,'set_changed':setchange,
       'folds':{name:{'V22':int(h22[a:b].sum()),'V23':int(h23[a:b].sum())} for name,(a,b) in older.FOLDS.items()},
       'known_673_693':{'V22':int(hbl[672:].sum()),'V23':int(hql[672:].sum()),'n':21,'blind':False},
       'prefix_checks':checks,
       'paired_sign_p':float(binomtest(len(gains),len(gains)+len(losses),.5).pvalue) if gains or losses else 1.0,
       'master_sha256':v.V.SHA,'fixed_spec':SELECTED}
    print(json.dumps(results,indent=2,ensure_ascii=False));(ROOT/'validation.json').write_text(json.dumps(results,ensure_ascii=False,indent=2))

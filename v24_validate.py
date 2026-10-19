#!/usr/bin/env python3
"""Independent sequential V24 reproduction and prefix-no-leakage tests.
It does not choose hyperparameters; selection is fixed separately in research.py.
"""
from pathlib import Path
import csv, json, hashlib
import numpy as np
from scipy.stats import binomtest
import v23_validate as v23
import v22_research as v22
import v20_research as v20
from v15_research import score
ROOT=Path(__file__).resolve().parent
SPECS={
 'V24_253_retrospective':{'expert':'markov','lookback':100,'evidence_window':48,'risk_z':0.0,'confidence_bonus':0.1,'threshold':-0.02,'alpha':0.5},
 'V24_252_selected_first504':{'expert':'recent25','lookback':50,'evidence_window':24,'risk_z':0.5,'confidence_bonus':0.0,'threshold':0.0,'alpha':0.75},
 'V24_243_selected_first350':{'expert':'face25','lookback':15,'evidence_window':24,'risk_z':0.0,'confidence_bonus':0.0,'threshold':0.02,'alpha':0.75},
}

def predict(rows,spec):
    d,_,base,*_=v23.predict(rows)
    d2,experts,*_=v22.get_base(rows)
    assert np.array_equal(d['y'],d2['y'])
    n=len(rows);y=d['y']; tie=d['tie']; q=experts[spec['expert']]
    rb,hb,_=score(base,y,tie); rq,hq,_=score(q,y,tie)
    past=np.eye(21)[y]
    final=base.copy();on=np.zeros(n,bool)
    ed=np.zeros(n,float); uncertainty=np.zeros(n,float); pri_adv=np.zeros(n,float)
    won=np.zeros(n,int);lost=np.zeros(n,int)
    for t in range(n):
        add=set(map(int,rq[t]))-set(map(int,rb[t]))
        dropped=set(map(int,rb[t]))-set(map(int,rq[t]))
        if not add: continue
        left=max(0,t-spec['lookback'])
        cnt=past[left:t].sum(axis=0)
        N=t-left+20
        p=(cnt+20*v20.V.FAIR)/N
        sdev=np.sqrt(np.maximum(p*(1-p),1e-12)/(N+1))
        ed[t]=sum(p[j] for j in add)-sum(p[j] for j in dropped)
        uncertainty[t]=np.sqrt(sum(sdev[j]**2 for j in add|dropped))
        evleft=max(0,t-spec['evidence_window'])
        wins=sum(bool(hq[k] and not hb[k]) for k in range(evleft,t))
        losses=sum(bool(hb[k] and not hq[k]) for k in range(evleft,t))
        won[t]=wins;lost[t]=losses
        adv=(wins-losses)/(wins+losses+4)
        pri_adv[t]=adv
        net=ed[t]-spec['risk_z']*uncertainty[t]+spec['confidence_bonus']*adv
        on[t]=((wins+losses)>=2 and adv>=-0.1 and net>=spec['threshold'])
        if on[t]:final[t]=(1-spec['alpha'])*base[t]+spec['alpha']*q[t]
    return d,base,final,on,ed,uncertainty,pri_adv,won,lost

def format_pairs(ids):
    return ';'.join(','.join(map(str,v20.V.PAIRS[j])) for j in ids)

def write_audit(rows,spec,output):
    d,b,p,on,edge,uncertainty,adv,wins,losses=predict(rows,spec)
    ra,ha,_=score(b,d['y'],d['tie']);rb,hb,_=score(p,d['y'],d['tie'])
    assert len(rows)==672 and ha.sum()==250
    new=np.flatnonzero(hb&~ha)+1;lost=np.flatnonzero(ha&~hb)+1
    keys=['throw','original_triple','actual_pair','V23_top5','V23_hit','V24_top5','V24_hit',
          'gate_active','set_changed','prior_pair_gain','risk_uncertainty','past_expert_advantage','past_wins','past_losses','new_hit','lost_hit']
    with open(output,'w',newline='',encoding='utf8') as f:
        writer=csv.DictWriter(f,fieldnames=keys);writer.writeheader()
        for i,z in enumerate(rows):
            writer.writerow({'throw':i+1,'original_triple':','.join(map(str,z)),
            'actual_pair':','.join(map(str,v20.V.pair(z))),
            'V23_top5':format_pairs(ra[i]),'V23_hit':int(ha[i]),
            'V24_top5':format_pairs(rb[i]),'V24_hit':int(hb[i]),
            'gate_active':int(on[i]),'set_changed':int(set(ra[i])!=set(rb[i])),
            'prior_pair_gain':round(float(edge[i]),7),'risk_uncertainty':round(float(uncertainty[i]),7),
            'past_expert_advantage':round(float(adv[i]),7), 'past_wins':int(wins[i]),'past_losses':int(losses[i]),
            'new_hit':int(hb[i]&~ha[i]),'lost_hit':int(ha[i]&~hb[i])})
    return {'V23':int(ha.sum()),'V24':int(hb.sum()),'gates':int(on.sum()),
          'sets_changed':int(sum(set(ra[i])!=set(rb[i]) for i in range(len(rows)))),
          'new_hits':new.tolist(),'lost_hits':lost.tolist(),
          'periods':{k:{'V23':int(ha[a:c].sum()),'V24':int(hb[a:c].sum())}
            for k,(a,c) in {'warmup_1_100':(0,100),'early_101_350':(100,350),
                             'validation_351_504':(350,504),'later_505_672':(504,672)}.items()},
          'paired_sign_p':float(binomtest(len(new),len(new)+len(lost),.5).pvalue) if len(new)+len(lost) else 1.0}

if __name__=='__main__':
    path=ROOT/'throws(2).txt';assert hashlib.sha256(path.read_bytes()).hexdigest()==v20.V.SHA
    rows=v20.V.load(path);assert len(rows)==672
    summaries={}
    for name,spec in SPECS.items():
        filename=ROOT/('audit_672.csv' if name=='V24_253_retrospective' else name+'_audit_672.csv')
        summary=write_audit(rows,spec,filename)
        summaries[name]=summary
        print(name,summary,flush=True)
    # Test independent locked spec: modifying throw t or any suffix cannot change predicted rank at t or earlier.
    d,b,q,*_=predict(rows,SPECS['V24_253_retrospective'])
    original_ranks=score(q,d['y'],d['tie'])[0]
    checks=[]
    for ix in (1,50,72,101,176,251,322,351,451,501,504,600,672):
        mutated=list(rows);a,b,c=mutated[ix-1];mutated[ix-1]=(3-a,7-b,c)
        dd,bb,qq,*_=predict(mutated,SPECS['V24_253_retrospective'])
        rnew=score(qq,dd['y'],dd['tie'])[0]
        assert np.array_equal(rnew[:ix],original_ranks[:ix]),f'future leakage at {ix}'
        checks.append(ix)
    with (ROOT/'LIVE_673_693.csv').open(newline='') as f:
        live=[tuple(map(int,fields[1:])) for fields in csv.reader(f)]
    assert len(live)==21
    for name,spec in SPECS.items():
        da,bb,pp,*_=predict(rows+live,spec)
        r,h,_=score(pp,da['y'],da['tie'])
        assert r.shape==(693,5)
        summaries[name]['known_673_693']={'hits':int(h[672:].sum()),'n':21,'already_known_not_blind':True}
    summaries['prefix_checks']=checks
    summaries['master_sha256']=v20.V.SHA
    summaries['specs']=SPECS
    (ROOT/'validation.json').write_text(json.dumps(summaries,ensure_ascii=False,indent=2))
    print('PREFIX TESTS:',len(checks),'ALL PASS; known result comparison complete',flush=True)

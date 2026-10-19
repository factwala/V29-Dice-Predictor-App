#!/usr/bin/env python3
"""Streaming implementation of fixed V25 risk calibration; no test-target outcomes used in forecast.
Reproduces research arrays without calling candidate selection on the target.
"""
from pathlib import Path
import sys,csv,json,hashlib,math
import numpy as np
from scipy.stats import binomtest
R=Path(__file__).resolve().parent
sys.path.insert(0,str(R))
import v24_validate as v24
import v22_research as v22
import v20_research as v20
from v15_research import score
import v25_research as v25
SPECS={
 'V25_historical_255':{'expert':'six_bayes','fw':50,'ew':96,'risk':0.0,'history_bonus':0.3,'threshold':-0.03,'margin':-1,'alpha':0.25},
 'V25_selected_254':{'expert':'recent25','fw':200,'ew':48,'risk':0.0,'history_bonus':0.0,'threshold':0.0,'margin':-1,'alpha':0.25},
 'V25_early_250':{'expert':'face25','fw':25,'ew':48,'risk':0.5,'history_bonus':0.0,'threshold':0.0,'margin':-1,'alpha':0.5}
}

def forecast(rows,spec):
    d,base23,base,oldgate,*_=v24.predict(rows,v24.SPECS['V24_253_retrospective'])
    _,ex,_,_=v22.get_base(rows)
    y,tie=d['y'],d['tie'];n=len(rows)
    q=ex[spec['expert']]
    rb,hb,_=score(base,y,tie);rq,hq,_=score(q,y,tie)
    final=base.copy();activated=np.zeros(n,bool);edge=np.zeros(n,float);unc=np.zeros(n,float);advance=np.zeros(n,float)
    wins=np.zeros(n,int);losses=np.zeros(n,int)
    actual=np.eye(21)[y]
    for t in range(n):
        entrant=set(map(int,rq[t]))-set(map(int,rb[t]));drop=set(map(int,rb[t]))-set(map(int,rq[t]))
        if not entrant:continue
        fw=spec['fw'];start=max(0,t-fw)
        p=(actual[start:t].sum(axis=0)+25*v20.V.FAIR)/(t-start+25)
        delta_pair=np.zeros(21);delta_pair[list(entrant)]=1.;delta_pair[list(drop)]=-1.
        edge[t]=np.sum(delta_pair*p)
        unc[t]=math.sqrt(sum(float(p[j])*(1-float(p[j])) for j in entrant|drop)/(t-start+26))
        ew=spec['ew'];a=max(0,t-ew)
        w=sum(bool(hq[i] and not hb[i]) for i in range(a,t))
        l=sum(bool(hb[i] and not hq[i]) for i in range(a,t))
        wins[t]=w;losses[t]=l
        adv=(w-l)/(w+l+6);advance[t]=adv
        net=edge[t]-spec['risk']*unc[t]+spec['history_bonus']*adv
        approved=(w+l>=2) and (w-l>=spec['margin']) and (net>=spec['threshold'])
        if approved:
            activated[t]=True
            final[t]=(1-spec['alpha'])*base[t]+spec['alpha']*q[t]
    return d,base,final,activated,edge,unc,advance,wins,losses

def fmt(rank):return ';'.join(','.join(map(str,v20.V.PAIRS[int(j)])) for j in rank)

def audit(rows,spec,filename=None):
    d,base,q,on,edge,un,adv,wins,losses=forecast(rows,spec)
    y,tie=d['y'],d['tie']
    rb,hb,lb=score(base,y,tie); rq,hq,lq=score(q,y,tie)
    gains=np.flatnonzero(hq&~hb)+1;lost=np.flatnonzero(hb&~hq)+1
    periods={k:{'V24':int(hb[a:b].sum()),'V25':int(hq[a:b].sum())} for k,(a,b) in v25.FOLDS.items()}
    changed=[set(rb[i])!=set(rq[i]) for i in range(len(rows))]
    if filename:
        keys=['throw','original_triple','actual_pair','V24_top5','V24_hit','V25_top5','V25_hit','gate_active',
              'set_changed','prior_expected_gain','uncertainty','past_advantage','past_wins','past_losses','new_hit','lost_hit']
        with open(filename,'w',newline='',encoding='utf-8') as f:
            writer=csv.DictWriter(f,fieldnames=keys);writer.writeheader()
            for i,z in enumerate(rows):
                writer.writerow({'throw':i+1,'original_triple':','.join(map(str,z)),
                  'actual_pair':','.join(map(str,v20.V.pair(z))), 'V24_top5':fmt(rb[i]),'V24_hit':int(hb[i]),
                  'V25_top5':fmt(rq[i]),'V25_hit':int(hq[i]),'gate_active':int(on[i]),'set_changed':int(changed[i]),
                  'prior_expected_gain':round(float(edge[i]),7),'uncertainty':round(float(un[i]),7),
                  'past_advantage':round(float(adv[i]),7),'past_wins':int(wins[i]),'past_losses':int(losses[i]),
                  'new_hit':int(hq[i] and not hb[i]),'lost_hit':int(hb[i] and not hq[i])})
    return {'V24_hits':int(hb.sum()),'V25_hits':int(hq.sum()),'accuracy':round(100*float(hq.mean()),5),
            'gates':int(on.sum()),'sets_changed':int(sum(changed)), 'new_hits':gains.tolist(),'lost_hits':lost.tolist(),
            'periods':periods,'paired_sign_p':float(binomtest(len(gains),len(gains)+len(lost),.5).pvalue) if len(gains)+len(lost) else 1.0}

if __name__=='__main__':
    master=R/'throws(2).txt';assert hashlib.sha256(master.read_bytes()).hexdigest()==v20.V.SHA
    rows=v20.V.load(master);assert len(rows)==672
    summary={}
    summary['master_sha256']=v20.V.SHA
    for name,spec in SPECS.items():
        fname=R/('audit_672.csv' if name=='V25_historical_255' else name+'_audit.csv')
        summary[name]=audit(rows,spec,fname)
        summary[name]['spec']=spec
        print(name,summary[name],flush=True)
    # Test 1: independent reference scores against a separately-generated batch result file
    selection=json.loads((R/'selection.json').read_text())
    assert summary['V25_historical_255']['V25_hits']==selection['highest_historical']['hits']
    assert summary['V25_selected_254']['V25_hits']==selection['selected_first504']['hits']
    assert summary['V25_early_250']['V25_hits']==selection['selected_first350']['hits']
    summary['independent_batch_vs_streaming']='PASS'
    # Test 2: current truth and suffix do not influence the forecast for that index.
    # Prefix-based test avoids recomputing the entire future for each edited row.
    checks=[]
    full=forecast(rows,SPECS['V25_historical_255'])
    do,_,pfull,*_=full
    ranks_full=score(pfull,do['y'],do['tie'])[0]
    for ix in (1,50,101,251,451,672):
        prefix=list(rows[:ix]);a,b,c=prefix[-1];prefix[-1]=(3-a,7-b,c)
        d,_,pred,*_=forecast(prefix,SPECS['V25_historical_255'])
        r=score(pred,d['y'],d['tie'])[0]
        assert np.array_equal(r,ranks_full[:ix]),f'future leakage at {ix}'
        checks.append(ix)
    summary['prefix_checks']=checks
    with (R/'LIVE_673_693.csv').open(newline='') as f:
        live=[tuple(map(int,line[1:])) for line in csv.reader(f)]
    assert len(live)==21
    for name,spec in SPECS.items():
        d,b,p,*_=forecast(rows+live,spec)
        _,h,_=score(p,d['y'],d['tie'])
        summary[name]['known_673_693']={'hits':int(h[672:].sum()),'n':21,'already_known_not_blind':True}
    (R/'validation.json').write_text(json.dumps(summary,indent=2,ensure_ascii=False))
    print('validation completed:',len(checks),'prefix checks PASS',flush=True)

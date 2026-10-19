#!/usr/bin/env python3
"""V22: Pair-entry utility and gate sensitivity; STRICTLY PREQUENTIAL.
All features at time t use rows[:t]; parameter selection is retrospective on reused file.
"""
from pathlib import Path
import hashlib,csv,json,math
import numpy as np
import v20_research as v
import v21_research as v21
from v15_research import score
from scipy.stats import binomtest
R=Path(__file__).resolve().parent

FOLDS={'warmup_1_100':(0,100),'early_101_350':(100,350),'validation_351_504':(350,504),'later_505_672':(504,672)}

def past_roll(x,w):
    x=np.asarray(x,float)
    if x.ndim==1: x=x[:,None]
    n=len(x);c=np.concatenate((np.zeros((1,x.shape[1])),np.cumsum(x,axis=0)),axis=0)
    t=np.arange(n); return c[t]-c[np.maximum(0,t-w)]

def rolling_pair_probs(actual, w, prior=20):
    n=len(actual);count=past_roll(actual,w)
    return (count+prior*v.V.FAIR)/(np.minimum(np.arange(n),w)[:,None]+prior)

def entry_utility(p,q,prior,tie):
    # Determines which pair(s) an alternative forecast would ADD/DROP BEFORE observing this throw.
    ixp=rank(p,tie);ixq=rank(q,tie)
    mp=np.zeros_like(p,dtype=float);mq=np.zeros_like(q,dtype=float)
    n=len(p);mp[np.arange(n)[:,None],ixp]=1;mq[np.arange(n)[:,None],ixq]=1
    add=(mq>0)&(mp==0);drop=(mp>0)&(mq==0)
    size=add.sum(axis=1)
    return ((add*prior).sum(axis=1)-(drop*prior).sum(axis=1)),size,add,drop

def rank(p,tie):
    # hashes are stable; use same score() tie handling as historical implementations.
    n=len(p); dummy=np.zeros(n,dtype=int)
    return score(p,dummy,tie)[0]

def get_base(rows):
    d,experts,p19,q16,v20,on,h=v21.build_base(rows)
    y=d['y']; tie=d['tie']; n=len(y)
    alt=experts['change_bayes_60']
    hit20=score(v20,y,tie)[1]; hitalt=score(alt,y,tie)[1]
    won=past_roll((hitalt & ~hit20).astype(float),24).reshape(-1)
    lost=past_roll((hit20 & ~hitalt).astype(float),24).reshape(-1)
    active=((won-lost)>=0)&((won+lost)>=4)
    p21=(1-.25*active[:,None])*v20+.25*active[:,None]*alt
    return d,experts,p21,v20

def features(rows):
    d,ex,p,b=get_base(rows); y=d['y']; tie=d['tie'];n=len(y)
    actual=np.eye(21)[y]
    short={w:rolling_pair_probs(actual,w) for w in (15,25,50,100,200)}
    # rolling forecast residual from *prior* outcomes (current truth never enters current row)
    residual={w:past_roll(actual-p,w)/(np.minimum(np.arange(n),w)[:,None]+40) for w in (25,60,120)}
    hp=score(p,y,tie)[1]
    return d,ex,p,b,short,residual,hp

def proposals(rows):
    d,ex,p,b,short,residual,hp=features(rows)
    n=len(p);y=d['y'];tie=d['tie'];truth=np.eye(21)[y]
    comps=[]
    def evaluate(name,q,metadata=None):
        _,hit,logloss=score(q,y,tie)
        rec={'name':name,'all_hits':int(hit.sum()),'logloss':round(float(logloss.mean()),7)}
        for key,(a,c) in FOLDS.items(): rec[key]=int(hit[a:c].sum())
        rec['first504']=int(hit[100:504].sum())
        rec.update(metadata or {})
        comps.append((rec,q))
        return rec
    evaluate('V21_REFERENCE',p,{'family':'baseline'})
    # Candidate A: pair-specific recent residual correction (every entry tied to actual past-pair rates)
    for w,r in residual.items():
        for strength in (.12,.25,.4,.6,1.,1.5,2.):
            for clip in (.01,.035,.08):
                rr=np.clip(r,-clip,clip)
                pred=np.maximum(1e-8,p+strength*rr);pred/=pred.sum(axis=1,keepdims=True)
                evaluate(f'A_residual_w{w}_b{strength}_clip{clip}',pred,{'family':'pair_residual','window':w,'strength':strength,'clip':clip})
    # Candidate B: pair-entry confidence, in history of similar pairs, blend alternative only where supported
    candidate_exp=['recent25','face25','markov','six_bayes','change_bayes_60','V16_best','V15','V7']
    for e in candidate_exp:
        q=ex[e]
        hitq=score(q,y,tie)[1]
        wins=(hitq & ~hp).astype(float); losses=(hp & ~hitq).astype(float)
        for w in (24,60,120):
            gw=past_roll(wins,w).reshape(-1);gl=past_roll(losses,w).reshape(-1)
            for freqw in (25,100):
                utility,n_changed,_,_=entry_utility(p,q,short[freqw],tie);
                # Empirical base-rate support of newly entering pairs, independent of current truth
                for ut in (-.04,0,.04):
                    for margin in (-1,0,1):
                        gate=(n_changed>0)&(utility>=ut)&((gw-gl)>=margin)&((gw+gl)>=2)
                        for mix in (.25,.5,.85):
                            pred=p*(1-mix*gate[:,None])+q*(mix*gate[:,None])
                            evaluate(f'B_{e}_w{w}_fw{freqw}_ut{ut}_m{margin}_a{mix}',pred,{'family':'entry_gate','expert':e,'window':w,'freq_window':freqw,'entry_threshold':ut,'win_margin':margin,'mix':mix,'gated_throws':int(gate.sum())})
    # Candidate C: pair-specific shrinkage of baseline towards past-observed probabilities; controlled ratio
    for w in (15,25,50,100,200):
        empirical=short[w]
        for blend in (.1,.2,.3,.5,.8):
            pred=(1-blend)*p+blend*empirical
            evaluate(f'C_empirical_w{w}_blend{blend}',pred,{'family':'prior_pair_freq','window':w,'mix':blend})
    # Candidate D: use historic under/over-count only for pairs near the top-five boundary
    r0=rank(p,tie); n=len(p)
    is_top=np.zeros_like(p,bool);is_top[np.arange(n)[:,None],r0]=True
    boundary=p[np.arange(n),r0[:,4]]
    for w,r in residual.items():
        for allowance in (.002,.006,.012):
            near=(abs(p-boundary[:,None])<allowance)
            for strength in (.2,.5,1.):
                rr=np.where(near,r,0)
                pred=np.maximum(1e-8,p+strength*rr);pred/=pred.sum(axis=1,keepdims=True)
                evaluate(f'D_boundary_w{w}_a{allowance}_b{strength}',pred,{'family':'near_boundary','window':w,'near':allowance,'strength':strength})
    scores=[c for c,q in comps]
    chosen=max(scores[1:],key=lambda x:(x['first504'],x['validation_351_504'], -x['logloss']))
    historical=max(scores[1:],key=lambda x:(x['all_hits'],x['later_505_672']))
    # conservative: early and validation gains both required, otherwise no promotion
    base=scores[0]
    credible=[c for c in scores[1:] if c['early_101_350']>base['early_101_350'] and c['validation_351_504']>base['validation_351_504']]
    improving=[c for c in scores[1:] if c['early_101_350']>=base['early_101_350'] and c['validation_351_504']>=base['validation_351_504'] and c['later_505_672']>=base['later_505_672'] and c['all_hits']>base['all_hits']]
    return d,p,comps,{'reference':base,'early_selected':chosen,'historical_highest':historical,'strict_both_improving_count':len(credible),'all_3_non_decreasing_count':len(improving),'tested_new_configs':len(scores)-1}

if __name__=='__main__':
    import time
    t0=time.time(); master=R/'throws(2).txt'
    assert hashlib.sha256(master.read_bytes()).hexdigest()==v.V.SHA
    rows=v.V.load(master);assert len(rows)==672
    d,p,comps,res=proposals(rows)
    print('RUNTIME',round(time.time()-t0,2),'RESULTS',json.dumps(res,indent=2),flush=True)
    with (R/'candidate_results.csv').open('w',newline='',encoding='utf-8') as f:
        wr=csv.DictWriter(f,fieldnames=list(dict.fromkeys(k for c,q in comps for k in c)))
        wr.writeheader();wr.writerows(c for c,q in comps)
    (R/'stage1.json').write_text(json.dumps(res,ensure_ascii=False,indent=2))

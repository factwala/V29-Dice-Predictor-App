#!/usr/bin/env python3
"""V15: Regime onset, duration, and causal expert-gating research.
Raw 672 records are never changed. Forecast at row t accesses only rows[:t].
Model selection diagnostics are retrospective: 672 throws have been reused previously.
"""
from pathlib import Path
import sys, csv, hashlib, json, itertools, math
import numpy as np
ROOT=Path(__file__).resolve().parent
PARENT=ROOT
sys.path.insert(0,str(PARENT))
from v14_validate import models as base_models, fast_rank
import v13_core as V
MASTER=ROOT/'throws(2).txt'
EXPECTED=V.SHA
EPS=1.e-12

def rolling_binary(x,w):
    n=len(x);cs=np.r_[0,x.cumsum()];t=np.arange(n);lo=np.maximum(0,t-w)
    return (cs[t]-cs[lo])/np.maximum(t-lo,1)

def rolling_proba(y,w,prior=20):
    n=len(y);cs=np.vstack([np.zeros((1,21)),np.eye(21)[y].cumsum(axis=0)]);t=np.arange(n);lo=np.maximum(0,t-w)
    cnt=cs[t]-cs[lo]
    return (cnt+prior*V.FAIR)/((t-lo+prior)[:,None])

def detect(sig,thr,release=.72,confirm=2,relax=3):
    """Regime switch using prefix derived signal at current timestamp.
    Enter after consecutive threshold exceedance, exit with hysteresis.
    Return currently-active and elapsed regime duration; no current outcome used.
    """
    n=len(sig);active=np.zeros(n,bool);dur=np.zeros(n,int);k=0;lo=0;on=False;last=0
    for t in range(n):
        if sig[t]>=thr:k+=1
        else:k=0
        if sig[t]<thr*release:lo+=1
        else:lo=0
        if not on and k>=confirm:on=True;last=t;k=0;lo=0
        if on and lo>=relax:on=False;lo=0;k=0
        active[t]=on
        if on:dur[t]=t-last+1
    return active,dur

def score(pr,y,tie):
    ranking=np.lexsort((tie,-np.round(pr,12)),axis=1)[:,:5]
    hits=np.any(ranking==y[:,None],axis=1)
    loss=-np.log(np.maximum(pr[np.arange(len(y)),y],EPS))
    return ranking,hits,loss

def metrics(name,pr,y,tie):
    pred,hits,loss=score(pr,y,tie)
    out={'model':name}
    for tag,a,b in [('train_101_350',100,350),('validation_351_504',350,504),('later_505_672',504,672),('all_672',0,672)]:
        out[tag+'_hits']=int(hits[a:b].sum())
        out[tag+'_pct']=round(float(hits[a:b].mean()*100),3)
        out[tag+'_logloss']=round(float(loss[a:b].mean()),6)
    return out

def model_lab(rows):
    m,y,_=base_models(rows)
    n=len(y); t=np.arange(n)
    # deterministic tie breaker matches v7/v13/v14 (using hash of same timestamp and unordered pair)
    hash_order=np.empty((n,21),int)
    for k in range(n):
        vals=[hashlib.sha256(f'TWO-DICE-FAIR-SHA256-V1|{k+1}|{a},{b}'.encode()).hexdigest() for a,b in V.PAIRS]
        for rank,j in enumerate(sorted(range(21),key=lambda j: vals[j])):hash_order[k,j]=rank
    v7=m['V7'];v13=m['V13_233'];v14=m['V14_234']; rec25=rolling_proba(y,25);rec75=rolling_proba(y,75)
    # past-only face probability
    f=np.eye(21)[y];cs=np.vstack([np.zeros((1,21)),f.cumsum(axis=0)]);lo=np.maximum(0,t-25)
    cnt=cs[t]-cs[lo];faces=np.zeros((n,6))
    for j,(a,b) in enumerate(V.PAIRS):faces[:,a-1]+=cnt[:,j];faces[:,b-1]+=cnt[:,j]
    faces=(faces+10/6)/(2*(t-lo)[:,None]+10)
    face25=np.stack([faces[:,a-1]*faces[:,b-1]*(1 if a==b else 2) for a,b in V.PAIRS],axis=1)
    # real past-only Markov transition conditional on prior pair
    markov=np.zeros((n,21))
    last=np.zeros((21,21))
    for i in range(n):
        if i>=2:last[y[i-2],y[i-1]]+=1
        if i==0:markov[i]=V.FAIR
        else:
            h=last[y[i-1]]
            markov[i]=(h+20*V.FAIR)/(h.sum()+20)
    base={**m,'rec25':rec25,'rec75':rec75,'face25':face25,'markov':markov}
    # state signals reflect past records only
    six=np.array([int(6 in z[1:]) for z in rows],float)
    order=np.array([1 if V.orientation(z)=='up' else -1 if V.orientation(z)=='down' else 0 for z in rows],float)
    drift6=np.abs(rolling_binary(six,25)-rolling_binary(six,150))
    driftorient=np.abs(rolling_binary(order,25)-rolling_binary(order,150))
    driftpairs=.5*np.abs(rolling_proba(y,25,1)-rolling_proba(y,150,1)).sum(axis=1)
    six_shift=np.maximum(rolling_binary(six,25)-rolling_binary(six,150),0)
    # prior prediction surprise and recent empirical deviation from baseline
    preqloss=-np.log(np.maximum(v13[np.arange(n),y],EPS))
    surprises=np.r_[0,preqloss[:-1]]
    loss_error=np.maximum(rolling_binary(surprises,20)-rolling_binary(surprises,120),0)
    # signal units differ. threshold grids tailored and fixed without outcome peeking
    signals={
     'six_abs':(drift6,[.12,.20,.28]),
     'six_rise':(six_shift,[.08,.15,.23]),
     'orientation_abs':(driftorient,[.2,.35,.5]),
     'pair_tv':(driftpairs,[.2,.32,.45]),
     'v13_loss_jump':(loss_error,[.08,.15,.25]),
    }
    # score all candidates without caching all probability arrays
    results=[];keep={'V7':v7,'V13_233':v13,'V14_234':v14}
    for name,p in base.items():results.append(metrics(name,p,y,hash_order))
    # drift->duration decay/change point windows drive mixture strength
    for sname,(sig,thresholds) in signals.items():
        for threshold in thresholds:
            for confirm in (1,3):
                active,dur=detect(sig,threshold,confirm=confirm)
                for halflife in (8,25,80):
                    # sustain regime influence in proportion to its longevity,
                    # yet cap so an old regime does not dictate predictions forever
                    persist=(1-np.exp(-dur/max(halflife,1)))*active
                    # onset-focused alternative decays if regime lasts too long
                    onset=np.exp(-np.maximum(dur-1,0)/max(halflife,1))*active
                    for phase,act in [('sustain',persist),('onset',onset)]:
                        for specialist in ('rec25','face25','markov','rec75','V13_233'):
                            for alpha in (.1,.25,.5):
                                pr=(1-alpha*act[:,None])*v14+alpha*act[:,None]*base[specialist]
                                name=f'cp_{sname}_thr{threshold}_cf{confirm}_{phase}_H{halflife}_{specialist}_a{alpha}'
                                results.append(metrics(name,pr,y,hash_order))
    # learned causal expert weights by prior regime and duration bucket.
    # for each expert, only wins recorded strictly before time t.
    top={k:score(p,y,hash_order)[1].astype(float) for k,p in base.items()}
    specialties=('V7','V13_233','rec25','face25','markov')
    for sname,(sig,thresholds) in signals.items():
        threshold=thresholds[1]
        active,dur=detect(sig,threshold,confirm=2)
        for period in (30,100,240):
            for smooth in (10,30):
                scores=np.zeros((n,len(specialties)))
                state=(active.astype(int)*3+np.where(dur<8,0,np.where(dur<30,1,2)))
                for t in range(1,n):
                    a=max(0,t-period);sel=(state[a:t]==state[t]);past=np.where(sel)[0]+a
                    if len(past):
                        z=np.stack([top[k][past] for k in specialties],axis=1)
                        scores[t]=(z.sum(axis=0)+smooth*.28)/(len(past)+smooth)
                    else:scores[t]=.28
                for temp in (3,8):
                    logits=temp*(scores-scores.mean(axis=1,keepdims=True));w=np.exp(logits-logits.max(axis=1,keepdims=True));w/=w.sum(axis=1,keepdims=True)
                    expert=sum(w[:,i,None]*base[k] for i,k in enumerate(specialties))
                    for mix in (.2,.5,.8):
                        pr=(1-mix)*v14+mix*expert
                        name=f'skill_{sname}_W{period}_S{smooth}_T{temp}_a{mix}'
                        results.append(metrics(name,pr,y,hash_order))
    # after assessing, choose ONLY based on first 504 (not final 168)
    # dedicated validation threshold: must show train +4 AND validation +3 vs V7 and not worsen logloss by >0.005
    bv=next(r for r in results if r['model']=='V7')
    passed=[r for r in results if r['model'] not in ('V7','V13_233','V14_234')
      and r['train_101_350_hits']>=bv['train_101_350_hits']+4
      and r['validation_351_504_hits']>=bv['validation_351_504_hits']+3
      and r['train_101_350_logloss']<=bv['train_101_350_logloss']+.005
      and r['validation_351_504_logloss']<=bv['validation_351_504_logloss']+.005]
    retrospective=sorted(results,key=lambda r:(r['all_672_hits'],r['later_505_672_hits']),reverse=True)[0]
    early=sorted(results,key=lambda r:(r['train_101_350_hits']+r['validation_351_504_hits'],r['validation_351_504_hits'],-r['validation_351_504_logloss']),reverse=True)[0]
    chosen=(sorted(passed,key=lambda r:(r['train_101_350_hits']+r['validation_351_504_hits'],-r['validation_351_504_logloss']),reverse=True)[0] if passed else bv)
    return {'results':results,'retrospective':retrospective,'early':early,'chosen':chosen,'passed':passed,'baselines':{k:next(r for r in results if r['model']==k) for k in ('V7','V13_233','V14_234')},'signals':signals,'hash_order':hash_order,'y':y,'forecast_bases':base}

if __name__=='__main__':
    assert hashlib.sha256(MASTER.read_bytes()).hexdigest()==EXPECTED
    rows=V.load(MASTER)
    assert len(rows)==672
    output=model_lab(rows)
    # output candidates
    fields=list(output['results'][0])
    with open(ROOT/'all_candidate_results.csv','w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=fields);w.writeheader();w.writerows(sorted(output['results'],key=lambda r:(-r['all_672_hits'],-r['validation_351_504_hits'])))
    summary={
      'rows':len(rows),'checksum_sha256':EXPECTED,
      'num_candidates':len(output['results']),
      'v7':output['baselines']['V7'],'v13':output['baselines']['V13_233'],'v14':output['baselines']['V14_234'],
      'best_historical_exploratory':output['retrospective'],
      'early_chosen_diagnostic':output['early'],
      'meets_predefined_promotion_gate':len(output['passed']),
      'chosen_by_gate':output['chosen'],
      'warning':'The 672 are reused from multiple V7-V14 rounds, and the 21 known continuation throws have been seen. Not a pristine out-of-sample test.'
    }
    (ROOT/'summary.json').write_text(json.dumps(summary,indent=2,ensure_ascii=False))
    print(json.dumps(summary,ensure_ascii=False,indent=2),flush=True)

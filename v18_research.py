#!/usr/bin/env python3
"""V18 confidence gated online expert switches, strictly prefix-causal.
Each prediction at time t depends exclusively on data before t.
Retrospective parameter search on re-used 672 data is exploratory, not independent evidence.
"""
import csv, json, math, hashlib, itertools
from pathlib import Path
import numpy as np
from scipy.stats import beta as beta_dist
import v13_core as V
from v15_research import score, metrics, detect
from v17_research import arrays
ROOT=Path(__file__).resolve().parent
SECTIONS={'early':(100,350),'val':(350,504),'later':(504,672)}

def lag_roll(v,w):
    """Sum over [max(0,t-w), t), never includes time t's outcome."""
    v=np.asarray(v,dtype=float);n=len(v)
    cum=np.r_[0.,np.cumsum(v)]
    t=np.arange(n);start=np.maximum(0,t-w)
    return cum[t]-cum[start]

def gate_array(p,q,y,tie,mode,w,threshold,min_n,hysteresis,mix,regime=None):
    """Predict with p by default; switch/blend toward q only on past comparative evidence."""
    n=len(y)
    _, hp, lp=score(p,y,tie)
    _, hq, lq=score(q,y,tie)
    # Paired, comparable outcomes: net advantage ONLY from prior disagreement.
    advantage=hq.astype(float)-hp.astype(float)
    favorable=lag_roll((advantage>0).astype(float),w)
    unfavorable=lag_roll((advantage<0).astype(float),w)
    count=favorable+unfavorable
    if mode=='beta':
        # conservative posterior lower quantile on q win-rate, conditional on expert disagreement
        evidence=beta_dist.ppf(.20,favorable+1,unfavorable+1)-.5
        trigger=(count>=min_n)&(evidence>threshold)
        release=(count>=min_n)&(evidence>threshold-hysteresis)
    elif mode=='margin':
        # posterior-smoothed skill contrast per trial
        evidence=(favorable-unfavorable)/(count+6)
        trigger=(count>=min_n)&(evidence>threshold)
        release=(count>=min_n)&(evidence>threshold-hysteresis)
    elif mode=='log':
        # previous log score: reward q for increased probability on actual past outcomes
        edge=lag_roll(lp-lq,w)/(np.minimum(w,np.arange(n))+12)
        evidence=edge
        trigger=(count>=min_n)&(evidence>threshold)
        release=(count>=min_n)&(evidence>threshold-hysteresis)
    else: raise ValueError(mode)
    # duration hysteresis, pure state updates based only on past derived scores
    state=False;active=np.zeros(n,dtype=float)
    for t in range(n):
        if not state and trigger[t]:state=True
        elif state and not release[t]:state=False
        active[t]=float(state)
    if regime is not None: active*=regime
    alpha=mix*active
    return p*(1-alpha[:,None])+q*alpha[:,None], active, evidence, count

def precompute(rows):
    d=arrays(rows);y=d['y'];tie=d['tie'];ex=d['experts'].copy()
    ex['V17_historical']=.9*ex['V15']+.1*ex['six_bayes']
    # Regime flags from original strictly past-only signals
    flags={'always':np.ones(len(y))}
    thresholds={'pair_tv':(.15,.25),'six':(.10,.18),'skill_drop':(.05,.12)}
    for key, thrs in thresholds.items():
        for th in thrs:
            active, duration=detect(d['signals'][key],th,confirm=2,release=.72,relax=3)
            flags[f'{key}_th{th}']=active.astype(float)
            flags[f'{key}_onset{th}']=active.astype(float)*np.exp(-np.maximum(duration-1,0)/35)
    return d,ex,flags

def configurations(d,ex,flags):
    y=d['y']; tie=d['tie'];ref=ex['V15']
    yield 'V7',ex['V7'],{'type':'baseline'}
    yield 'V13',ex['V13'],{'type':'baseline'}
    yield 'V14',ex['V14'],{'type':'baseline'}
    yield 'V15',ex['V15'],{'type':'baseline'}
    yield 'V16_best',ex['V16_best'],{'type':'baseline'}
    yield 'V17_historical',ex['V17_historical'],{'type':'baseline'}
    # Prior-to-current outcome differences determine switches.
    # Restrict parameter grid deliberately, avoid unbounded fishing for individual throw hits.
    challengers=['V16_best','V16_early','V17_historical','change_bayes_20','change_bayes_60','six_bayes','markov','recent25','face25']
    for challenger in challengers:
      q=ex[challenger]
      for mode in ('beta','margin','log'):
       if mode=='beta':params=[(.0,0.05),(-.04,.05)]
       elif mode=='margin':params=[(.06,.04),(.12,.06)]
       else:params=[(.005,.01),(.015,.01)]
       for w in (24,60,120):
        for thr,hyst in params:
         for minn in (2,5):
          for mix in (.25,.60,1.):
           pred,act,evi,num=gate_array(ref,q,y,tie,mode,w,thr,minn,hyst,mix)
           yield f'gate_{challenger}_{mode}_w{w}_thr{thr}_n{minn}_m{mix}',pred,{'type':'switch','expert':challenger,'criterion':mode,'window':w,'threshold':thr,'min_disagreements':minn,'mix':mix,'active_pct':round(float(act.mean()*100),2)}
           # conditional change-point filters, computed solely from past signals
           if challenger in ('V16_best','V17_historical','change_bayes_20','six_bayes') and w in (24,60) and mix in (.60,1.) and minn==2:
            for regime_name in ('pair_tv_th0.15','six_th0.1','skill_drop_th0.05'):
             pred2,act2,_,_=gate_array(ref,q,y,tie,mode,w,thr,minn,hyst,mix,flags[regime_name])
             yield f'regime_{challenger}_{mode}_w{w}_thr{thr}_m{mix}_{regime_name}',pred2,{'type':'regime','expert':challenger,'criterion':mode,'window':w,'threshold':thr,'min_disagreements':minn,'mix':mix,'regime':regime_name,'active_pct':round(float(act2.mean()*100),2)}

def summarise_candidates(rows,save_preds=True):
    d,ex,flags=precompute(rows);y=d['y'];tie=d['tie'];res=[];best={};chosen_pred={}
    for name,p,info in configurations(d,ex,flags):
        stat=metrics(name,p,y,tie)
        stat.update(info)
        stat['first_two_hits']=stat['train_101_350_hits']+stat['validation_351_504_hits']
        stat['min_fold_gain_vs_V15']=min(stat['train_101_350_hits']-86,stat['validation_351_504_hits']-50,stat['later_505_672_hits']-67)
        res.append(stat)
        if save_preds:chosen_pred[name]=p
    return d,res,chosen_pred

def choose(results):
    cand=[r for r in results if r['type']!='baseline']
    # selected by first two periods; not using the later segment. Model selection elsewhere on 672
    # contaminates later as pristine holdout, so interpret as retrospectively diagnostic only.
    early=max(cand,key=lambda r:(r['first_two_hits'],r['validation_351_504_hits'],-r['validation_351_504_logloss']))
    # Require both subsegments strictly improve over V15 before promotion.
    eligible=[r for r in cand if r['train_101_350_hits']>86 and r['validation_351_504_hits']>50]
    promote=max(eligible,key=lambda r:(r['first_two_hits'],r['validation_351_504_hits'],-r['validation_351_504_logloss'])) if eligible else None
    historical=max(cand,key=lambda r:(r['all_672_hits'],r['later_505_672_hits']))
    crossfold=max(cand,key=lambda r:(r['min_fold_gain_vs_V15'],r['first_two_hits'],r['all_672_hits']))
    return early,promote,historical,crossfold,len(eligible)

def write_csv(path,records):
    cols=list(dict.fromkeys(k for r in records for k in r))
    with Path(path).open('w',encoding='utf8',newline='') as f:
        wr=csv.DictWriter(f,fieldnames=cols);wr.writeheader();wr.writerows(records)

def prediction_summary(p,y,tie,rows):
    rank,hit,loss=score(p,y,tie)
    return [{'throw':i+1,'original_triple':','.join(map(str,r)), 'actual_pair':','.join(map(str,V.pair(r))), 'top5':';'.join(','.join(map(str,V.PAIRS[j])) for j in rank[i]),'hit':int(hit[i])} for i,r in enumerate(rows)]

def full_main():
    src=ROOT/'throws(2).txt';assert hashlib.sha256(src.read_bytes()).hexdigest()==V.SHA
    rows=V.load(src);assert len(rows)==672
    d,res,preds=summarise_candidates(rows)
    early,promote,historical,crossfold,npass=choose(res)
    res.sort(key=lambda r:(-r['all_672_hits'],-r['later_505_672_hits']))
    write_csv(ROOT/'candidate_results.csv',res)
    choices={'V7':'V7','V15':'V15','V16_best':'V16_best','V17_historical':'V17_historical','V18_early':early['model'],'V18_historical':historical['model'],'V18_consistent':crossfold['model']}
    if promote:choices['V18_promotable']=promote['model']
    pred={k:prediction_summary(preds[v],d['y'],d['tie'],rows) for k,v in choices.items()}
    out=[]
    for t in range(672):
      line={'throw':t+1,'original_triple':pred['V7'][t]['original_triple'],'actual_pair':pred['V7'][t]['actual_pair']}
      for k in choices:
       line[k+'_top5']=pred[k][t]['top5'];line[k+'_hit']=pred[k][t]['hit']
      out.append(line)
    write_csv(ROOT/'audit_672.csv',out)
    summary={'master_sha256':V.SHA,'n_rows':len(rows),'n_candidates':len(res),'baselines':{k:next(r for r in res if r['model']==k) for k in ('V7','V13','V14','V15','V16_best','V17_historical')},'early_selected':early,'promotable':promote,'promotable_count':npass,'exploratory_max_historical':historical,'best_min_fold_difference':crossfold,'metrics_definition': 'Top5 on unordered pair; each prediction uses only throws before itself; model choice retrospective over previously-reused data.'}
    print('n candidates',len(res),'early',early['model'], 'historic',historical['model'])
    print(json.dumps({k:summary[k] for k in ('early_selected','promotable','exploratory_max_historical','best_min_fold_difference')},ensure_ascii=False,indent=2))
    (ROOT/'precheck_summary.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2))
    return summary

if __name__=='__main__': full_main()

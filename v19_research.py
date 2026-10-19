#!/usr/bin/env python3
"""V19: past-only confidence-gated switch refinement, no outcome memorization.
All forecasts for throw t depend only on rows[:t] (0-indexed).
Candidate search on repeatedly reused data is exploratory, not independent evidence.
"""
from pathlib import Path
import hashlib, json, csv, math
import numpy as np
from scipy.stats import binomtest
import v13_core as V
from v15_research import score, metrics
from v18_research import precompute, gate_array, lag_roll
ROOT=Path(__file__).resolve().parent
BASE='gate_markov_log_w24_thr0.015_n2_m0.25'

def smooth_switch(ref, q, y, tie, *, w=24, threshold=.015, min_dis=2, confirm=1, exit_confirm=1, min_hold=0, cooldown=0, alpha=.25, release=.01, mode='log', age_halflife=0, other_mask=None):
    """Switch to alternate expert on strictly earlier comparison evidence.
    Proposed hold/debounce/cooldowns are online state, never see current y.
    """
    _,h0,l0=score(ref,y,tie); _,h1,l1=score(q,y,tie)
    n=len(y); t=np.arange(n)
    pastdis=lag_roll((h0!=h1).astype(float),w)
    if mode=='log': e=lag_roll(l0-l1,w)/(np.minimum(t,w)+12)
    elif mode=='hit': e=lag_roll(h1.astype(float)-h0.astype(float),w)/(np.minimum(t,w)+8)
    elif mode=='hybrid':
        e= .7*lag_roll(l0-l1,w)/(np.minimum(t,w)+12)+.3*lag_roll(h1.astype(float)-h0.astype(float),w)/(np.minimum(t,w)+8)
    else: raise ValueError(mode)
    active=np.zeros(n,float);st=False; onrun=0;offrun=0; age=0;lockout=0
    for i in range(n):
        good=(pastdis[i]>=min_dis and e[i]>threshold)
        stay=(pastdis[i]>=min_dis and e[i]>threshold-release)
        if other_mask is not None:
            good=good and (other_mask[i]>0)
        if st:
            age+=1
            offrun=0 if stay else offrun+1
            if age>=min_hold and offrun>=exit_confirm:
                st=False;age=0;onrun=0;offrun=0;lockout=cooldown
        else:
            lockout=max(0,lockout-1)
            onrun=onrun+1 if good and lockout==0 else 0
            if onrun>=confirm:
                st=True;age=1;onrun=0;offrun=0
        if st:
            active[i]=np.exp(-(age-1)/age_halflife) if age_halflife>0 else 1.
    result=ref*(1-alpha*active[:,None])+q*(alpha*active[:,None])
    return result,active

def run(rows,full=False):
    d,ex,flags=precompute(rows);y=d['y'];tie=d['tie']; n=len(rows)
    # Recreate exact V18 primary via existing code.
    v18,a18,*_=gate_array(ex['V15'],ex['markov'],y,tie,'log',24,.015,2,.01,.25)
    baseline={'V7':ex['V7'],'V15':ex['V15'],'V16':ex['V16_best'],'V18':v18}
    summaries={k:metrics(k,v,y,tie) for k,v in baseline.items()}
    cand=[];keep={};states={}
    def register(name,p,act,params):
        stat=metrics(name,p,y,tie)
        stat.update(params)
        stat['gain_vs_v18_early']=stat['train_101_350_hits']-summaries['V18']['train_101_350_hits']
        stat['gain_vs_v18_val']=stat['validation_351_504_hits']-summaries['V18']['validation_351_504_hits']
        stat['gain_vs_v18_late']=stat['later_505_672_hits']-summaries['V18']['later_505_672_hits']
        stat['gain_vs_v18_all']=stat['all_672_hits']-summaries['V18']['all_672_hits']
        stat['active_pct']=round(float(np.mean(act))*100,2)
        cand.append(stat)
        # Keep results for winners during search, no need thousands of copies
        if full:keep[name]=(p,act)
    for window in (16,24,36,48,72):
      for threshold in (.005,.015,.025):
       for confirm in (1,2,3):
        for exitconfirm in (1,2,3):
         for alpha in (.15,.25,.35,.50):
          # V18's benefit from Markov varies by onset, confirmation and exit delays
          p,act=smooth_switch(ex['V15'],ex['markov'],y,tie,w=window,threshold=threshold, confirm=confirm,exit_confirm=exitconfirm,alpha=alpha)
          name=f'run_gate_Markov_w{window}_th{threshold}_confirm{confirm}_exit{exitconfirm}_a{alpha}'
          register(name,p,act,{'family':'run_gate','expert':'markov','w':window,'threshold':threshold,'confirm':confirm,'exit_confirm':exitconfirm,'alpha':alpha})
    # Debounce/cooldown and hold are explicit alternative to V18 (smaller grid)
    for expert in ('markov','V16_best','V17_historical','six_bayes','recent25','face25'):
      for window in (16,24,48,72):
       for threshold in (.0,.015,.035):
        for min_hold,cooldown in ((0,0),(3,0),(3,3),(6,6)):
         for alpha in (.15,.25,.4):
          p,act=smooth_switch(ex['V15'],ex[expert],y,tie,w=window,threshold=threshold,confirm=2,exit_confirm=2,min_hold=min_hold,cooldown=cooldown,alpha=alpha)
          name=f'debounce_{expert}_w{window}_th{threshold}_hold{min_hold}_cool{cooldown}_a{alpha}'
          register(name,p,act,{'family':'debounce','expert':expert,'w':window,'threshold':threshold,'min_hold':min_hold,'cooldown':cooldown,'alpha':alpha})
    # Stability of gating: try age-decaying influence after entry. Blend only when
    # evidence in pairwise past-only gate exists, no outcomes of current throw.
    for expert in ('markov','six_bayes','V16_best'):
      for window in (16,24,48,72):
       for threshold in (.0,.015,.035):
        for decay in (8,20,50):
         for alpha in (.25,.5,.8):
          p,act=smooth_switch(ex['V15'],ex[expert],y,tie,w=window,threshold=threshold,confirm=2,exit_confirm=2,age_halflife=decay,alpha=alpha)
          name=f'decay_{expert}_w{window}_th{threshold}_hl{decay}_a{alpha}'
          register(name,p,act,{'family':'decay','expert':expert,'w':window,'threshold':threshold,'halflife':decay,'alpha':alpha})
    # "Repair" V18 only when a second expert has a distinct proven prior edge.
    for expert in ('six_bayes','V16_best','V17_historical','recent25','face25'):
      for window in (16,24,48,72):
       for threshold in (.0,.015,.035):
        for alpha in (.1,.2,.35):
         supplement,act=smooth_switch(v18,ex[expert],y,tie,w=window,threshold=threshold,confirm=2,exit_confirm=2,alpha=alpha)
         name=f'repair_{expert}_w{window}_th{threshold}_a{alpha}'
         register(name,supplement,act,{'family':'repair','expert':expert,'w':window,'threshold':threshold,'alpha':alpha})
    # Weighted mixture guided by forward-only pairwise wins to V15, without binary release
    for expert in ('markov','six_bayes','V16_best'):
      for window in (16,24,48):
       for alpha in (.15,.3,.6):
        for prior in (.02,.05,.1):
         _,_,l0=score(ex['V15'],y,tie); _,_,lq=score(ex[expert],y,tie)
         t=np.arange(n)
         edge=lag_roll(l0-lq,window)/(np.minimum(window,t)+12)
         strength=np.clip(edge/prior,0,1)
         out=ex['V15']*(1-alpha*strength[:,None])+ex[expert]*(alpha*strength[:,None])
         name=f'soft_{expert}_w{window}_alpha{alpha}_prior{prior}'
         register(name,out,strength,{'family':'soft','expert':expert,'w':window,'prior':prior,'alpha':alpha})
    # Selection by only data #101–504, compare 3 windows separately; do not use
    # late period for the selection. Full-historical best is descriptive only.
    def first_key(r):return (min(r['gain_vs_v18_early'],r['gain_vs_v18_val']),r['gain_vs_v18_early']+r['gain_vs_v18_val'],-r['validation_351_504_logloss'])
    selected=max(cand,key=first_key)
    early_top=max(cand,key=lambda r:(r['train_101_350_hits']+r['validation_351_504_hits'],r['validation_351_504_hits'],-r['validation_351_504_logloss']))
    historical=max(cand,key=lambda r:(r['all_672_hits'],r['later_505_672_hits']))
    holds=[r for r in cand if r['gain_vs_v18_early']>=0 and r['gain_vs_v18_val']>=0]
    if full:
        return d,ex,summaries,cand, {'early_stable':selected,'early_peak':early_top,'historical':historical},keep,baseline
    return d,ex,summaries,cand, {'early_stable':selected,'early_peak':early_top,'historical':historical},baseline

def write_csv(path, rows):
    if not rows:return
    keys=list(dict.fromkeys(k for x in rows for k in x))
    with open(path,'w',newline='',encoding='utf8') as f:
        w=csv.DictWriter(f,fieldnames=keys);w.writeheader();w.writerows(rows)

def take_candidate(spec,d,ex,rows):
    y=d['y'];tie=d['tie']; F=spec['family']; expert=spec['expert'];w=int(spec['w']);alpha=float(spec['alpha']);threshold=float(spec.get('threshold',0))
    v18,*_=gate_array(ex['V15'],ex['markov'],y,tie,'log',24,.015,2,.01,.25)
    if F=='run_gate':return smooth_switch(ex['V15'],ex[expert],y,tie,w=w,threshold=threshold,confirm=int(spec['confirm']),exit_confirm=int(spec['exit_confirm']),alpha=alpha)
    if F=='debounce':return smooth_switch(ex['V15'],ex[expert],y,tie,w=w,threshold=threshold,confirm=2,exit_confirm=2,min_hold=int(spec['min_hold']),cooldown=int(spec['cooldown']),alpha=alpha)
    if F=='decay':return smooth_switch(ex['V15'],ex[expert],y,tie,w=w,threshold=threshold,confirm=2,exit_confirm=2,age_halflife=int(spec['halflife']),alpha=alpha)
    if F=='repair':return smooth_switch(v18,ex[expert],y,tie,w=w,threshold=threshold,confirm=2,exit_confirm=2,alpha=alpha)
    if F=='soft':
        _,_,lp=score(ex['V15'],y,tie);_,_,lq=score(ex[expert],y,tie)
        t=np.arange(len(y));s=np.clip((lag_roll(lp-lq,w)/(np.minimum(t,w)+12))/float(spec['prior']),0,1)
        return ex['V15']*(1-alpha*s[:,None])+ex[expert]*(alpha*s[:,None]),s
    raise ValueError(F)

if __name__=='__main__':
    src=ROOT/'throws(2).txt';assert hashlib.sha256(src.read_bytes()).hexdigest()==V.SHA
    rows=V.load(src);assert len(rows)==672
    d,ex,baselines,recs,winners,refs=run(rows)
    recs.sort(key=lambda r:(r['all_672_hits'],r['later_505_672_hits']),reverse=True)
    write_csv(ROOT/'candidate_results.csv',recs)
    print('Candidates',len(recs))
    print('Baselines',[(k,v['all_672_hits']) for k,v in baselines.items()])
    print('Winners',json.dumps(winners,indent=2))
    # Results for chosen contenders
    chosen={'V15':refs['V15'],'V18':refs['V18']}
    acts={}
    for name,spec in winners.items():
        chosen[name],acts[name]=take_candidate(spec,d,ex,rows)
        verify=metrics(name,chosen[name],d['y'],d['tie'])
        for key in ('all_672_hits','train_101_350_hits','validation_351_504_hits','later_505_672_hits'):
            assert verify[key]==spec[key],(name,key,verify[key],spec[key])
    a={}
    for k,p in chosen.items():
        ranks,hits,_=score(p,d['y'],d['tie'])
        a[k]=(ranks,hits)
    lines=[]
    for i,row in enumerate(rows):
        z={'throw':i+1,'original_triple':','.join(map(str,row)),'actual_pair':','.join(map(str,V.pair(row)))}
        for k,(r,h) in a.items():
            z[k+'_top5']=';'.join(','.join(map(str,V.PAIRS[j])) for j in r[i]);z[k+'_hit']=int(h[i])
        lines.append(z)
    write_csv(ROOT/'audit_672.csv',lines)
    (ROOT/'summary.json').write_text(json.dumps({'checksum_sha256':V.SHA,'n_throws':672,'n_candidates':len(recs),'baselines':baselines,'winners':winners},indent=2))

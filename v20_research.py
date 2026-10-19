#!/usr/bin/env python3
"""V20: conservative prefix-only confidence gates, fast loss braking, and time stability.
All forecast arrays at t use raw rows[:t] and past score comparisons indexed <t.
Parameter selection on reused 672 historical throws is exploratory; not blind proof.
"""
import csv, json, hashlib, math
from pathlib import Path
import numpy as np
from scipy.special import betainc
from scipy.stats import binomtest
import v13_core as V
from v18_research import precompute, gate_array, lag_roll
from v19_research import smooth_switch
from v15_research import score, metrics

ROOT=Path(__file__).resolve().parent
PRESPEC={'family':'repair','expert':'recent25','w':72,'threshold':0.0,'alpha':.35}
SEGMENTS={'warmup_1_100':(0,100),'early_101_350':(100,350),'validation_351_504':(350,504),'later_505_672':(504,672)}

def write_csv(file, data):
    if not data:return
    keys=list(dict.fromkeys(k for r in data for k in r))
    with open(file,'w',encoding='utf-8',newline='') as f:
        w=csv.DictWriter(f,fieldnames=keys);w.writeheader();w.writerows(data)

def historical_models(rows):
    d,ex,_=precompute(rows)
    y,tie=d['y'],d['tie']
    v18,*_=gate_array(ex['V15'],ex['markov'],y,tie,'log',24,.015,2,.01,.25)
    v19, active19=smooth_switch(v18,ex['recent25'],y,tie,w=72,threshold=0,confirm=2,exit_confirm=2,alpha=.35)
    return d,ex,{'V7':ex['V7'],'V15':ex['V15'],'V18':v18,'V19':v19},active19

def paired_history(p,q,y,tie,short,long):
    _,hp,lp=score(p,y,tie);_,hq,lq=score(q,y,tie)
    # online evidence at t calculated from previous outcomes ONLY
    win=(hq & ~hp).astype(float);lose=(hp & ~hq).astype(float)
    w=lag_roll(win,short); l=lag_roll(lose,short); wl=lag_roll(win,long);ll=lag_roll(lose,long)
    disagree=w+l; n=np.arange(len(y));
    # P(p_q>0.5 | Jeffreys posterior), conditional on disagreements
    prob=1-betainc(w+.5,l+.5,.5)
    prob_long=1-betainc(wl+.5,ll+.5,.5)
    lossadv=(lag_roll(lp-lq,short))/(np.minimum(n,short)+12)
    lossadv_long=(lag_roll(lp-lq,long))/(np.minimum(n,long)+12)
    # braking: if alt expert lost 2+ recent disagreement rounds, hold/release promptly
    quick_bad=lag_roll(lose,10) -lag_roll(win,10)
    # conditional certainty of difference, n effectively from short window
    return {'short_w':w,'short_l':l,'short_n':disagree,'long_w':wl,'long_l':ll,
            'prob':prob,'prob_long':prob_long,'adv':lossadv,'adv_long':lossadv_long,'quick_bad':quick_bad,
            'hit_q':hq,'hit_p':hp}

def gated(p,q,h, *, conf=.7,min_dis=2, confirm=2, cool=4,alpha=.35, method='posterior', long_check='none', brake=1, rapid_exit=1):
    n=len(p)
    if method=='posterior':
        advantage=(h['prob']>=conf)
        continue_adv=(h['prob'] >= conf-.12)
    elif method=='hybrid':
        advantage=(h['prob']>=conf)&(h['adv']>=0)
        continue_adv=(h['prob'] >= conf-.12)&(h['adv']>=-.015)
    elif method=='log':
        advantage=(h['adv']> (conf-.5)*.08)
        continue_adv=(h['adv']> (conf-.5)*.08-.015)
    else:raise ValueError(method)
    advantage &= h['short_n'] >= min_dis
    continue_adv &= h['short_n'] >= max(1,min_dis-1)
    if long_check=='posterior': advantage &= h['prob_long']>=.5
    if long_check=='log': advantage &= h['adv_long']>=-.005
    trigger=advantage & (h['quick_bad'] < brake)
    hold=continue_adv & (h['quick_bad'] < brake+1)
    on=np.zeros(n,float); state=False;good_run=0;bad_run=0; lock=0
    for i in range(n):
        if state:
            bad_run=0 if hold[i] else bad_run+1
            if bad_run >=rapid_exit:
                state=False;lock=cool;bad_run=0;good_run=0
        else:
            if lock:lock-=1
            if lock==0 and trigger[i]: good_run+=1
            else:good_run=0
            if good_run>=confirm:
                state=True;bad_run=0;good_run=0
        on[i]=int(state)
    return p*(1-alpha*on[:,None])+q*(alpha*on[:,None]),on

def get_candidates(rows,do_full=False):
    d,ex,baseline,act19=historical_models(rows);y,tie=d['y'],d['tie']
    summaries={k:metrics(k,p,y,tie) for k,p in baseline.items()}
    orig=baseline['V19']
    for k in ('V7','V15','V18','V19'):
        print('BASE',k,summaries[k]['all_672_hits'],flush=True)
    family_specs=[('markov',ex['markov']),('recent25',ex['recent25']),('face25',ex['face25']),
                  ('six_bayes',ex['six_bayes']),('V16_best',ex['V16_best']),
                  ('V17_historical',ex['V17_historical']),('V15',ex['V15']),
                  ('change_bayes_60',ex['change_bayes_60'])]
    rec=[];pred_by_name={} if do_full else None
    def register(name,pr,act,params):
        sc=metrics(name,pr,y,tie)
        sc.update(params)
        sc['switched_throws']=int(act.sum())
        sc['delta_early']=sc['train_101_350_hits']-summaries['V19']['train_101_350_hits']
        sc['delta_val']=sc['validation_351_504_hits']-summaries['V19']['validation_351_504_hits']
        sc['delta_later']=sc['later_505_672_hits']-summaries['V19']['later_505_672_hits']
        sc['delta_all']=sc['all_672_hits']-summaries['V19']['all_672_hits']
        rec.append(sc)
        if do_full:pred_by_name[name]=(pr,act)
    for expert,q in family_specs:
      for win in (24,48,96):
       h=paired_history(orig,q,y,tie,win,min(win*3,192))
       for method,conf in [('posterior',.60),('posterior',.75),('posterior',.9),('hybrid',.60),('hybrid',.75),('log',.65)]:
        for min_dis in (2,4):
         for alpha in (.2,.4,.7):
          for long_check in ('none','log'):
           par={'expert':expert,'window':win,'method':method,'confidence':conf,'min_dis':min_dis,'alpha':alpha,'long_check':long_check,'confirm':2,'cool':4,'brake':2,'rapid_exit':1}
           pred,act=gated(orig,q,h,conf=conf,min_dis=min_dis,alpha=alpha,method=method,long_check=long_check,brake=2)
           name=f'gated_{expert}_w{win}_{method}_{conf}_d{min_dis}_a{alpha}_L{long_check}'
           register(name,pred,act,par)
       # conservative short-term 'bad run' emergency exit sensitivity around plausible gate
       for conf in (.6,.75):
        for alpha in (.3,.6):
         for brake in (0,1,3):
          for rapid_exit in (1,2):
           name=f'brake_{expert}_w{win}_c{conf}_a{alpha}_b{brake}_ex{rapid_exit}'
           pred,act=gated(orig,q,h,conf=conf,min_dis=2,alpha=alpha,method='hybrid',long_check='log',brake=brake,rapid_exit=rapid_exit)
           register(name,pred,act,{'expert':expert,'window':win,'method':'brake','confidence':conf,'min_dis':2,'alpha':alpha,'long_check':'log','confirm':2,'cool':4,'brake':brake,'rapid_exit':rapid_exit})
    # choose candidate using only first two segments for settings; final segment merely diagnostic
    search=max(rec,key=lambda z:(min(z['delta_early'],z['delta_val']),z['delta_early']+z['delta_val'],-z['validation_351_504_logloss']))
    stable=max(rec,key=lambda z:(min(z['delta_early'],z['delta_val'],z['delta_later']),sum((z['delta_early'],z['delta_val'],z['delta_later'])),z['all_672_hits']))
    historical=max(rec,key=lambda z:(z['all_672_hits'],z['later_505_672_hits']))
    winning=[z for z in rec if z['delta_early']>0 and z['delta_val']>0 and z['delta_later']>0]
    return d,ex,baseline,summaries,rec,{'early_selected':search,'retrospective_stable':stable,'historical_max':historical,'strict_improvement_3period_count':len(winning)},pred_by_name

def build_specific(rows,spec):
    d,ex,base,_=historical_models(rows);y,tie=d['y'],d['tie']
    p=base['V19'];q=ex[spec['expert']]; w=int(spec['window']);h=paired_history(p,q,y,tie,w,min(w*3,192))
    pr,act=gated(p,q,h, conf=float(spec['confidence']),min_dis=int(spec['min_dis']),alpha=float(spec['alpha']),method=('hybrid' if spec['method']=='brake' else spec['method']),long_check=spec['long_check'],confirm=int(spec['confirm']),cool=int(spec['cool']),brake=int(spec['brake']),rapid_exit=int(spec['rapid_exit']))
    return d,base,pr,act

if __name__=='__main__':
    import time
    st=__import__('time').time()
    src=ROOT/'throws(2).txt';assert hashlib.sha256(src.read_bytes()).hexdigest()==V.SHA
    rows=V.load(src);assert len(rows)==672
    d,ex,base,bs,results,winners,cache=get_candidates(rows,True)
    print('Total candidates',len(results),'runtime',round(time.time()-st,2),flush=True)
    for k,v in winners.items():print(k,v['model'] if isinstance(v,dict) else v, 'hits',v['all_672_hits'] if isinstance(v,dict) else '',flush=True)
    write_csv(ROOT/'candidate_results.csv',sorted(results,key=lambda r:(r['all_672_hits'],r['later_505_672_hits']),reverse=True))
    chosen={'V7':base['V7'],'V15':base['V15'],'V18':base['V18'],'V19':base['V19']}
    acts={}
    for label in ('early_selected','retrospective_stable','historical_max'):
        spec=winners[label];pr,act=cache[spec['model']];chosen[label]=pr;acts[label]=act
    audit=[]
    for i,z in enumerate(rows):
        entry={'throw':i+1,'original_triple':','.join(map(str,z)),'actual_pair':','.join(map(str,V.pair(z)))}
        for name,p in chosen.items():
            ranking,hit,_=score(p[i:i+1],d['y'][i:i+1],d['tie'][i:i+1]);
            entry[name+'_top5']=';'.join(f'{V.PAIRS[j][0]},{V.PAIRS[j][1]}' for j in ranking[0]);entry[name+'_hit']=int(hit[0])
            if name in acts:entry[name+'_gate_on']=int(acts[name][i])
        audit.append(entry)
    write_csv(ROOT/'audit_672.csv',audit)
    # Known 21 actual throws used for retrospective diagnostics, not new blind proof
    with (ROOT/'LIVE_673_693.csv').open() as f:
        live=[tuple(map(int,r[1:])) for r in csv.reader(f)]
    assert len(live)==21
    ld,lbase,lp,lact=build_specific(rows+live,winners['early_selected']);ly,lt=ld['y'],ld['tie']
    _,livehits,_=score(lp,ly,lt)
    _,basehits,_=score(lbase['V19'],ly,lt)
    known={'V20_selected':int(livehits[672:].sum()),'V19':int(basehits[672:].sum()),'n':len(live),'not_blind':True}
    chosen_pred=chosen['early_selected'];_,hnew,_=score(chosen_pred,d['y'],d['tie']);_,hold,_=score(base['V19'],d['y'],d['tie'])
    gains=int((hnew & ~hold).sum());losses=int((~hnew & hold).sum())
    result={'original_sha256':V.SHA,'n_rows':672,'n_configurations':len(results),
            'baseline':bs,'winners':winners,'known_673_693':known,
            'early_selected_vs_v19':{'new_hits':gains,'lost_hits':losses,'paired_sign_p':float(binomtest(gains,gains+losses,.5).pvalue) if gains+losses else 1.},
            'selection_note':'V20 early selection based on #101-504 only; but reused 672 across prior V1-V19, not pristine validation',
            'time_elapsed_s':round(time.time()-st,2)}
    (ROOT/'summary.json').write_text(json.dumps(result,ensure_ascii=False,indent=2))
    print('LIVE',known,'DIFF',gains,losses, 'Total runtime',round(time.time()-st,2),flush=True)

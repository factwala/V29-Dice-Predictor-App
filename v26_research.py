#!/usr/bin/env python3
"""V26 offline calibration-drift research. Predictions at row t are functions of rows[:t].
Do not confuse retrospectively selected max score with prospective evidence.
"""
from pathlib import Path
import sys,json,csv,hashlib,time
import numpy as np
from scipy.stats import binomtest
R=Path(__file__).resolve().parent
sys.path.insert(0,str(R))
import v20_research as v20, v22_research as v22,v25_validate as v25
from v15_research import score
FOLDS={'warmup_1_100':(0,100),'early_101_350':(100,350),'validation_351_504':(350,504),'later_505_672':(504,672)}
EXPERTS=['markov','face25','recent25','six_bayes','change_bayes_60','V16_best','V15','V7']

def precalc(rows):
    d, _, baseline, *_=v25.forecast(rows,v25.SPECS['V25_selected_254'])
    d2, experts, _,_=v22.get_base(rows)
    assert np.array_equal(d['y'],d2['y'])
    n=len(rows); y=d['y'];tie=d['tie']; eye=np.eye(21)[y]
    # full-sample truth is used only in lagged past_roll (excludes row t)
    idx=np.arange(n)
    pastfreq={}
    for fw in (25,50,100,200):
        count=v22.past_roll(eye,fw)
        pastfreq[fw]=(count+25*v20.V.FAIR)/(np.minimum(idx,fw)[:,None]+25)
    baserank,bh,_=score(baseline,y,tie)
    expert_features={}
    for e in EXPERTS:
        q=experts[e];rank,h,_=score(q,y,tie)
        isbase=np.zeros((n,21),bool);isq=np.zeros((n,21),bool)
        isbase[idx[:,None],baserank]=1;isq[idx[:,None],rank]=1
        ent=isq&~isbase; out=isbase&~isq
        changed=ent.any(axis=1)
        # actual net utility of making full swap, revealed only after throw; past_roll excludes t
        actual_reward=((ent.astype(float)-out.astype(float))*eye).sum(axis=1)
        assert np.all(np.isin(actual_reward,[-1.,0.,1.]))
        # rolling historical realized advantage, residual to past-probability forecast, calibration drift
        prior_edge={fw:np.sum((ent.astype(float)-out.astype(float))*pastfreq[fw],axis=1) for fw in pastfreq}
        obs={}
        for w in (12,24,48,96,192):
            r=v22.past_roll(actual_reward,w).ravel()
            opportunities=v22.past_roll(changed.astype(float),w).ravel()
            wins=v22.past_roll((actual_reward>0).astype(float),w).ravel()
            losses=v22.past_roll((actual_reward<0).astype(float),w).ravel()
            # average regret per all previous throws, equivalent to E[swap utility]
            obs[w]={'reward':r,'opp':opportunities,'wins':wins,'losses':losses,
              'posterior':(wins+2)/(wins+losses+4),
              'adv':(wins-losses)/(wins+losses+6),
              'success_fraction':wins/(np.maximum(1.,wins+losses))}
        # Prequential estimate for the *last* w different top5 events, using historical predicted edge and observed result
        cal={}
        for w in (24,48,96):
            # realized reward minus expected utility from pair marginal observed before that throw
            lag_resid=actual_reward-prior_edge[100]
            lag_sum=v22.past_roll(lag_resid,w).ravel()
            denom=v22.past_roll(changed.astype(float),w).ravel()
            cal[w]=lag_sum/(denom+8)
        expert_features[e]={'q':q,'rank':rank,'ent':ent,'out':out,'changed':changed,'reward':actual_reward,
                             'edge':prior_edge,'obs':obs,'cal':cal}
    return d,baseline,experts,expert_features

def stat(name,pred,d,basehit,extra=None):
    r,h,loss=score(pred,d['y'],d['tie'])
    res={'model':name,'hits':int(h.sum()),'top5_pct':round(float(h.mean()*100),4),
          'first504_train':int(h[100:504].sum()),'log_loss':float(loss.mean())}
    for k,(a,b) in FOLDS.items():res[k]=int(h[a:b].sum())
    new=h&~basehit;lost=basehit&~h
    res['new_hits']=int(new.sum());res['lost_hits']=int(lost.sum())
    if extra:res.update(extra)
    return res

def candidate_for_params(base,f,settings):
    # causal gate. All f at t are computed from earlier observed results and past model forecasts.
    (fw,wc,wp,risk_z,cal_weight,drift_weight,threshold,alpha,min_decisions)=settings
    edge=f['edge'][fw];C=f['cal'][wc];obs=f['obs'][wp];slow=f['obs'][192]
    # shrink raw calibration residual, reward drift if short avg exceeds long avg.
    fast_reward=obs['reward']/(obs['opp']+6.)
    slow_reward=slow['reward']/(slow['opp']+12.)
    drift=fast_reward-slow_reward
    freq_n=np.minimum(np.arange(len(base)),fw)
    p=f['edge'][fw]
    # Risk proxy shrinks with number of observations
    unc=np.sqrt(np.maximum(1e-9,0.12)/(freq_n+25))
    utility=edge+cal_weight*C+drift_weight*drift-risk_z*unc
    gate=(f['changed'] & (obs['wins']+obs['losses']>=min_decisions) &
          (obs['wins']-obs['losses']>=-1) & (utility>=threshold))
    forecast=base + (gate*alpha)[:,None]*(f['q']-base)
    return forecast,gate,utility

def research(rows,verbose=False):
    d,baseline,experts,features=precalc(rows)
    base_rank,bhit,_=score(baseline,d['y'],d['tie'])
    records=[]
    scores={}
    def eval_candidate(name,pred,meta={}):
        rec=stat(name,pred,d,bhit,meta);records.append(rec);return rec
    eval_candidate('V25_SELECTED_254',baseline,{'family':'reference'})
    # Fixed V25 historical comparator
    _,_,hist,*_=v25.forecast(rows,v25.SPECS['V25_historical_255'])
    eval_candidate('V25_HISTORICAL_255',hist,{'family':'reference'})
    for e in EXPERTS:
        f=features[e]
        for fw in (25,100,200):
          for wc in (24,96):
           for wp in (24,48,96):
            for risk_z in (0.,.5):
             for cal_weight in (0.,.5,1.0):
              for drift_weight in (0.,.5,1.0):
               for threshold in (-.04,0,.03):
                for alpha in (.25,.5):
                 settings=(fw,wc,wp,risk_z,cal_weight,drift_weight,threshold,alpha,2)
                 pred,gate,utility=candidate_for_params(baseline,f,settings)
                 name=f'{e}|fw{fw}|cw{wc}|pw{wp}|rz{risk_z}|cb{cal_weight}|dr{drift_weight}|th{threshold}|a{alpha}'
                 rec=eval_candidate(name,pred,{'family':'V26_calibration_drift','expert':e,'gates':int(gate.sum()),
                     'parameters':json.dumps(settings)})
        if verbose:print('expert completed',e, 'count',len(records),flush=True)
    return d,baseline,features,records

def selected(records):
    base=records[0]
    cand=records[2:]
    historical=max(cand,key=lambda z:(z['hits'],z['later_505_672'],-z['log_loss']))
    train=max(cand,key=lambda z:(z['first504_train'],z['validation_351_504'],-z['log_loss']))
    early=max(cand,key=lambda z:(z['early_101_350'],z['warmup_1_100'],-z['log_loss']))
    all_no_loss=[z for z in cand if all(z[k]>=base[k] for k in FOLDS) and z['hits']>base['hits']]
    return {'V25_selected_reference':base, 'V25_hist_reference':records[1], 'V26_best_history':historical,
            'V26_selected_first504':train,'V26_selected_first350':early,'V26_nonloss_count':len(all_no_loss),
            'V26_configs':len(cand)}

if __name__=='__main__':
    master=R/'throws(2).txt';assert hashlib.sha256(master.read_bytes()).hexdigest()==v20.V.SHA
    rows=v20.V.load(master);assert len(rows)==672
    start=time.time();d,baseline,features,recs=research(rows,True)
    summary=selected(recs)
    fields=list(dict.fromkeys(k for r in recs for k in r))
    with (R/'candidate_results.csv').open('w',newline='') as out:
        writer=csv.DictWriter(out,fieldnames=fields);writer.writeheader();writer.writerows(recs)
    (R/'selection.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2))
    print('SECONDS',round(time.time()-start,2),'SUMMARY',json.dumps(summary,indent=2),flush=True)

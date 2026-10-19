#!/usr/bin/env python3
"""V27 precommitted small-policy Online Confidence Calibration.
At row t probabilities, calibration, gates and weights use only rows before t.
No optimizing candidate params against row t outcomes. Retrospective policy selection is described as such.
"""
from pathlib import Path
import json,hashlib,csv,sys,time
import numpy as np
from scipy.stats import binomtest
import v20_research as v20
import v22_research as v22
import v25_validate as v25
import v26_research as v26
from v15_research import score
ROOT=Path(__file__).resolve().parent
FOLDS={'warmup_1_100':(0,100),'early_101_350':(100,350),'validation_351_504':(350,504),'later_505_672':(504,672)}

# FIXED expert pool: selected before score inspection in this V27 investigation.
EXPERT_NAMES=('V26_255','V26_256_historical','V25_255_historical','V7','V16')
POLICIES=(
 ('posterior_fast','posterior',32,0.35,2,3),
 ('posterior_slow','posterior',96,0.35,2,3),
 ('posterior_fast_tolerant','posterior',32,0.50,1,2),
 ('posterior_slow_tolerant','posterior',96,0.50,1,2),
 ('loss_fast','loss',32,0.35,1,2),
 ('loss_slow','loss',96,0.35,1,2),
 ('hybrid_fast','hybrid',32,0.50,1,3),
 ('hybrid_slow','hybrid',96,0.50,1,3),
 ('hedge_fast','hedge',32,0.25,0,0),
 ('hedge_slow','hedge',96,0.25,0,0),
 ('hedge_fast_high','hedge',32,0.50,0,0),
 ('hedge_slow_high','hedge',96,0.50,0,0),
 ('regret_fast','regret',32,0.35,1,3),
 ('regret_slow','regret',96,0.35,1,3),
)

def past_roll(a,w):return v22.past_roll(a,w).reshape(-1) if np.asarray(a).ndim==1 else v22.past_roll(a,w)

def build_experts(rows):
    selection=json.loads((ROOT/'selection.json').read_text())
    # fixed specs from previous research, not retuned based on V27 outcomes
    d,base,all_ex,features=v26.precalc(rows)
    s1=selection['V26_selected_first504'];s2=selection['V26_best_history']
    v26s,_,_=v26.candidate_for_params(base,features[s1['expert']],json.loads(s1['parameters']))
    v26h,_,_=v26.candidate_for_params(base,features[s2['expert']],json.loads(s2['parameters']))
    _,_,v25h,*_=v25.forecast(rows,v25.SPECS['V25_historical_255'])
    models={'V26_255':v26s,'V26_256_historical':v26h,
            'V25_255_historical':v25h,'V7':all_ex['V7'],'V16':all_ex['V16_best']}
    return d,models

def predict_policies(d,models):
    y=d['y'];tie=d['tie'];n=len(y);i=np.arange(n)
    ps=np.stack([models[k] for k in EXPERT_NAMES],axis=1)
    assert ps.shape==(n,5,21) and np.all(np.isfinite(ps))
    assert np.max(np.abs(ps.sum(axis=-1)-1))<1e-5
    base=ps[:,0]
    hits=np.stack([score(ps[:,j],y,tie)[1] for j in range(5)],axis=1).astype(float)
    losses=-np.log(np.maximum(ps[i,:,y],1e-12)) # shape n,5, but indexing advanced beware
    assert losses.shape==(n,5)
    out={};stats={}
    for name,kind,w,alpha,margin,mindis in POLICIES:
        h=past_roll(hits,w);loss=past_roll(losses,w)
        # all history statistics are prefix-only. Negative values mean challenger underperforms.
        diff=h[:,1:]-h[:,[0]]
        decisive=past_roll(((hits[:,1:]!=hits[:,[0]]).astype(float)),w)
        wl=(diff+0.)/(decisive+6.)
        nwin=np.minimum(i,w)
        logadv=(loss[:,[0]]-loss[:,1:])/(nwin[:,None]+12.)
        utility=wl+.06*logadv
        active=np.zeros(n,bool);choice=np.zeros(n,int)
        pred=base.copy()
        for t in range(n):
            if kind=='posterior':
                qual=(diff[t]>=margin)&(decisive[t]>=mindis)
                vals=wl[t]
            elif kind=='loss':
                qual=(logadv[t]>.01)&(diff[t]>=0)&(decisive[t]>=mindis)
                vals=logadv[t]
            elif kind=='hybrid':
                qual=(diff[t]>=margin)&(decisive[t]>=mindis)&(logadv[t]>=-.025)
                vals=utility[t]
            elif kind=='regret':
                qual=(diff[t]>=margin)&(decisive[t]>=mindis)&(logadv[t]>=0)
                vals=utility[t]
            elif kind=='hedge':
                # bounded softmax expert weights from previous logloss and hits.
                scorev=(h[t]-h[t,0])/(nwin[t]+16) + .15*(loss[t,0]-loss[t])/(nwin[t]+16)
                scorev=np.clip(scorev*25,-8,8)
                weights=np.exp(scorev-scorev.max());weights/=weights.sum()
                pred[t]=(1-alpha)*base[t]+alpha*np.sum(weights[:,None]*ps[t],axis=0)
                active[t]=True;choice[t]=int(np.argmax(weights))
                continue
            else:raise ValueError(kind)
            if np.any(qual):
                valid=np.where(qual,vals,-1e9);j=1+int(np.argmax(valid))
                choice[t]=j;active[t]=True
                pred[t]=(1-alpha)*base[t]+alpha*ps[t,j]
        pred/=pred.sum(axis=1,keepdims=True)
        out[name]=pred;stats[name]={'activated':int(active.sum()),'choices':{EXPERT_NAMES[j]:int((choice==j).sum()) for j in range(1,5)}}
    return out,stats

def summary(name,p,d,ref):
    r,h,loss=score(p,d['y'],d['tie']);refhit=score(ref,d['y'],d['tie'])[1]
    row={'model':name,'hits':int(h.sum()),'top5_pct':round(100*h.mean(),3),
         'logloss':round(float(loss.mean()),6),'new_hits':int((h&~refhit).sum()),'lost_hits':int((~h&refhit).sum())}
    for k,(a,b) in FOLDS.items():row[k]=int(h[a:b].sum())
    row['first504']=int(h[:504].sum());row['first350']=int(h[:350].sum())
    return row

def evaluate(rows,write=False):
    d,models=build_experts(rows)
    policies,policy_stats=predict_policies(d,models)
    ref=models['V26_255']
    all_preds={'V26_SELECTED_255':ref,'V26_HISTORICAL_256':models['V26_256_historical'],
               'V25_HISTORICAL_255':models['V25_255_historical'],**policies}
    scores=[summary(k,v,d,ref) for k,v in all_preds.items()]
    # predeclared selection uses only pre-505 outcomes, not all_672 maximizing.
    fitted=[v for v in scores if v['model'] in policies]
    early=max(fitted,key=lambda z:(z['first350'],z['warmup_1_100']+z['early_101_350'],-z['logloss']))
    selected=max(fitted,key=lambda z:(z['first504'],z['validation_351_504'],-z['logloss']))
    highest=max(fitted,key=lambda z:(z['hits'],z['later_505_672'],-z['logloss']))
    result={'historical':highest,'selected_first504':selected,'selected_first350':early,
            'reference':scores[0],'V26_retrospective':scores[1],'number_policies':len(fitted),
            'expert_pool':list(EXPERT_NAMES),'policy_switch_stats':policy_stats}
    if write:
        with (ROOT/'candidate_results.csv').open('w',newline='') as f:
            wr=csv.DictWriter(f,fieldnames=list(scores[0]));wr.writeheader();wr.writerows(scores)
        (ROOT/'selection_v27.json').write_text(json.dumps(result,indent=2,ensure_ascii=False))
        y=d['y'];tie=d['tie'];n=len(y)
        compare={'V26_SELECTED_255':ref,'V26_HISTORICAL_256':models['V26_256_historical'],
                 selected['model']:policies[selected['model']],highest['model']:policies[highest['model']]}
        ranks={k:score(v,y,tie)[0] for k,v in compare.items()}
        hits={k:score(v,y,tie)[1] for k,v in compare.items()}
        with (ROOT/'audit_672.csv').open('w',newline='') as f:
            cols=['throw','original_triple','actual_unordered_pair']
            for k in compare:cols+=[k+'_Top5',k+'_hit']
            wr=csv.DictWriter(f,fieldnames=cols);wr.writeheader()
            for t,z in enumerate(rows):
                rr={'throw':t+1,'original_triple':','.join(map(str,z)),
                    'actual_unordered_pair':','.join(map(str,v20.V.pair(z)))}
                for k in compare:
                    rr[k+'_Top5']=';'.join(','.join(map(str,v20.V.PAIRS[int(j)])) for j in ranks[k][t]);rr[k+'_hit']=int(hits[k][t])
                wr.writerow(rr)
    return result,scores,all_preds

if __name__=='__main__':
    master=ROOT/'throws(2).txt'
    assert hashlib.sha256(master.read_bytes()).hexdigest()==v20.V.SHA
    rows=v20.V.load(master);assert len(rows)==672
    t0=time.perf_counter();r,s,_=evaluate(rows,True)
    print('SECS',round(time.perf_counter()-t0,2));print(json.dumps({k:r[k] for k in ('historical','selected_first504','selected_first350','reference','V26_retrospective','number_policies')},indent=2),flush=True)

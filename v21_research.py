#!/usr/bin/env python3
"""V21 online switching utility: strictly prefix-only predictors; retrospective model search.
The model score at t uses only forecasts based on x[:t] and scoring of prior t' < t.
"""
import hashlib, json, csv
from pathlib import Path
from itertools import product
import numpy as np
import v20_research as v
from v15_research import score
ROOT=Path(__file__).resolve().parent

def mean_past(x,w):
    x=np.asarray(x,float);cs=np.r_[0,np.cumsum(x)];t=np.arange(len(x));a=np.maximum(t-w,0)
    return (cs[t]-cs[a])/np.maximum(1,t-a)

def audit_rank(pr,y,tie):
    r,h,_=score(pr,y,tie);return r,h

def build_base(rows):
    d,ex,baselines,_=v.historical_models(rows);p=baselines['V19'];q=ex['V16_best'];
    h=v.paired_history(p,q,d['y'],d['tie'],48,144)
    v20,gate=v.gated(p,q,h,conf=.60,min_dis=2,alpha=.70,method='posterior',long_check='none',confirm=2,cool=4,brake=2,rapid_exit=1)
    return d,ex,p,q,v20,gate,h

def get_features(rows,d,p,q,v20,on):
    n=len(rows);y=d['y'];tie=d['tie']
    rp,hp= audit_rank(p,y,tie);rq,hq=audit_rank(q,y,tie);r20,h20=audit_rank(v20,y,tie)
    # Future label only in lagged computations. Pair-membership sets available before current throw.
    masks={}
    for name,r in [('p',rp),('q',rq),('v20',r20)]:
        m=np.zeros((n,21),bool);m[np.arange(n)[:,None],r]=True;masks[name]=m
    add=masks['v20'] & ~masks['p']; remove=masks['p'] & ~masks['v20']
    add2=masks['q'] & ~masks['p']; remove2=masks['p'] & ~masks['q']
    d20=add.sum(axis=1);dq=add2.sum(axis=1)
    actual=np.eye(21)[y]
    prior_pair_freq={w:np.maximum(0,np.cumsum(actual,axis=0)-np.vstack([np.zeros((w,21)), np.cumsum(actual,axis=0)[:-w]])[:n]) if False else None for w in []}
    # historic pair prevalence: strictly previous outcome counts
    for w in (15,25,40,75,120):
        cs=np.vstack([np.zeros((1,21)), actual.cumsum(axis=0)])
        t=np.arange(n);a=np.maximum(0,t-w)
        prior_pair_freq[w]=(cs[t]-cs[a]+10*v.V.FAIR)/(np.maximum(0,t-a)[:,None]+10)
    features={}
    for w,prev in prior_pair_freq.items():
        features[f'novel_utility{w}']=((add*prev).sum(axis=1)-(remove*prev).sum(axis=1))
        features[f'expert_utility{w}']=((add2*prev).sum(axis=1)-(remove2*prev).sum(axis=1))
    features['set_changed']=d20>0
    features['q_changed']=dq>0
    features['gate_on']=on>0
    features['top5_p']=masks['p'];features['top5_v20']=masks['v20']
    features['marginal_pred_gain']=((add2*q).sum(axis=1)-(remove2*p).sum(axis=1))
    # previous V20 vs V19 hit disagreement for each context; compare on past
    gain=(h20&~hp).astype(float);loss=(hp&~h20).astype(float)
    # special pairs with six, available in the forecast sets
    six=np.array([int(6 in pair) for pair in v.V.PAIRS]);features['six_gain']= (add*six).sum(axis=1)-(remove*six).sum(axis=1)
    # effective on-set utility: previous examples of the same *kind* of replacement.
    for w in (40,100,240):
        for cat,sel in [('six_plus',features['six_gain']>0),('six_minus',features['six_gain']<0),('same_six',features['six_gain']==0),
                        ('multi',d20>1),('single',d20==1)]:
            weights=(sel&features['set_changed']).astype(float)
            den=v.lag_roll(weights,w)
            gr=v.lag_roll(gain*weights,w);lo=v.lag_roll(loss*weights,w)
            features[f'context_{w}_{cat}']=(gr-lo)/(den+8)
    return features, masks, hp,hq,h20

def calc(rows,verbose=True):
    d,ex,p,q,v20,on,h=build_base(rows)
    y,tie=d['y'],d['tie'];f,m,hp,hq,h20=get_features(rows,d,p,q,v20,on)
    n=len(rows)
    candidates=[]
    def eval_candidate(name,pr,extra=None):
        r,hits,ll=score(pr,y,tie)
        rec={'model':name,'all_hits':int(hits.sum()),'warm_1_100':int(hits[:100].sum()),
             'early_101_350':int(hits[100:350].sum()),'val_351_504':int(hits[350:504].sum()),
             'later_505_672':int(hits[504:672].sum()),'train_101_504':int(hits[100:504].sum()),
             'known_last21':int(hits[672:693].sum()) if n>=693 else '',
             'logloss_101_504':float(np.mean(ll[100:504])),
             'different_sets':int(np.sum(np.any(r!=score(v20,y,tie)[0],axis=1))) if False else '',
             **(extra or {})}
        candidates.append(rec)
        return rec
    base=eval_candidate('V20',v20)
    # Hypothesis A: retain V20 only if the changed *pair* has positive prior support.
    for fname in ['novel_utility15','novel_utility25','novel_utility40','novel_utility75','novel_utility120',
                  'expert_utility25','expert_utility75','expert_utility120','marginal_pred_gain']:
        for thr in (-.06,-.025,0,.015,.04):
            # If V20 set change appears poorly supported, use V19.
            allow=(f[fname]>=thr).astype(float)
            mix=np.maximum(0,on)*allow
            pr=p*(1-.7*mix[:,None])+q*(.7*mix[:,None])
            eval_candidate(f'A_{fname}_thr{thr:g}',pr,{'family':'utility_filter','feature':fname,'threshold':thr,'gate_on':int(mix.sum())})
    # Hypothesis B: early-exit with prefix-past conditional switch skill, only in true set-changes
    for ctxw in (40,100,240):
        for typ in ('six_plus','six_minus','same_six','multi','single'):
            k=f'context_{ctxw}_{typ}'
            for th in (-.05,-.02,0,.02,.05):
                allow=f[k]>=th
                mix=on*allow
                pr=p*(1-.7*mix[:,None])+q*(.7*mix[:,None])
                eval_candidate(f'B_{k}_thr{th:g}',pr,{'family':'context_filter','feature':k,'threshold':th,'gate_on':int(mix.sum())})
    # Hypothesis C: previously better other expert with gated evidence, supplement V20.
    # Three-way arbitration: V20 and chosen expert with past hit differences, only when recent utility wins.
    for expert in ('recent25','face25','markov','six_bayes','V15','V7','V16_best','change_bayes_60'):
        alt=ex[expert];rh, alt_hit,_=score(alt,y,tie);gain=(alt_hit&~h20).astype(float);loss=(h20&~alt_hit).astype(float)
        for w in (24,48,96,160):
            g=v.lag_roll(gain,w);l=v.lag_roll(loss,w)
            for margin in (0,1,2,3):
                for min_n in (2,4):
                    active=((g-l)>=margin)&((g+l)>=min_n)
                    for alpha in (.25,.5,.8):
                        pr=v20*(1-alpha*active[:,None])+alt*(alpha*active[:,None])
                        eval_candidate(f'C_{expert}_{w}_m{margin}_d{min_n}_a{alpha:g}',pr,{'family':'expert_override','expert':expert,'w':w,'margin':margin,'min_n':min_n,'alpha':alpha,'override_on':int(active.sum())})
    # Hypothesis D: shrink V20 when its switch is fragile based on previous *gain ratio* and pair margin
    for w in (24,48,96):
        g=v.lag_roll((h20&~hp).astype(float),w);l=v.lag_roll((hp&~h20).astype(float),w)
        for shrink in (.3,.5,.7):
            for balance in (-1,0,1,2):
                alpha=np.where((g-l)>=balance,.7,shrink)
                pr=p*(1-alpha[:,None]*on[:,None])+q*(alpha[:,None]*on[:,None]);
                eval_candidate(f'D_shrink_w{w}_balance{balance}_to{shrink}',pr,{'family':'shrink','w':w,'balance':balance,'shrink':shrink})
    # Fixed selection by first 504 data; prioritizes held-out #351-504 too, though all have been reused in past work.
    ranked=sorted(candidates[1:],key=lambda x:(x['train_101_504'],x['val_351_504'],-x['logloss_101_504']),reverse=True)
    hist=sorted(candidates[1:],key=lambda x:(x['all_hits'],x['later_505_672']),reverse=True)
    stable=sorted(candidates[1:],key=lambda x:(min(x['early_101_350']-89,x['val_351_504']-55,x['later_505_672']-68),x['all_hits']),reverse=True)
    return d,ex,p,q,v20,on,h,f,candidates,{'base':base,'preselected':ranked[0],'historical':hist[0],'stable':stable[0],'num_candidates':len(candidates)-1}

if __name__=='__main__':
    rows=v.V.load(ROOT/'throws(2).txt')
    assert len(rows)==672 and hashlib.sha256((ROOT/'throws(2).txt').read_bytes()).hexdigest()==v.V.SHA
    d,ex,p,q,v20,on,h,f,candidates,picks=calc(rows)
    print(json.dumps(picks,indent=2),flush=True)
    with open(ROOT/'candidate_results.csv','w',newline='') as file:
        writer=csv.DictWriter(file,fieldnames=list(candidates[0].keys())+sorted(set.union(*(set(c) for c in candidates))-set(candidates[0])))
        writer.writeheader();writer.writerows(candidates)
    (ROOT/'stage1.json').write_text(json.dumps(picks,indent=2))

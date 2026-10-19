#!/usr/bin/env python3
"""V13: Past-only context-sensitive expert gating and online calibration.
The original master file is immutable and SHA256 protected.
Returns audited walk-forward predictions; model choice based on earlier windows only.
"""
import csv, hashlib, json, math, shutil
from pathlib import Path
from collections import Counter, defaultdict, deque
import numpy as np

ROOT=Path(__file__).resolve().parent
MASTER=ROOT/'throws(2).txt'
SHA='578a7efc3d1c4f69adbcbc17976a4a79a2bcf2bd889eec232ce776b0fcdaf418'
PAIRS=[(a,b) for a in range(1,7) for b in range(a,7)]
IX={p:i for i,p in enumerate(PAIRS)}
FAIR=np.array([1 if a==b else 2 for a,b in PAIRS],float)/36
EXP=('recent50','ordersum','balance5','ewma40')

def load(path=MASTER):
    raw=Path(path).read_bytes()
    if path==MASTER: assert hashlib.sha256(raw).hexdigest()==SHA
    rows=[tuple(map(int,x.split(','))) for x in raw.decode('utf-8-sig').splitlines() if x.strip()]
    assert len(rows)>=672 and all(len(z)==3 and z[0] in (1,2) and 1<=z[1]<=6 and 1<=z[2]<=6 for z in rows)
    return rows

def pair(z):return tuple(sorted(z[1:]))
def orientation(z):
    a,b=z[1:]
    return 'eq' if a==b else ('6last' if b==6 else ('up' if a<b else 'down'))

def order(probs,n):
    def key(k):
        a,b=PAIRS[k]
        h=hashlib.sha256(f'TWO-DICE-FAIR-SHA256-V1|{n}|{a},{b}'.encode()).hexdigest()
        return (-round(float(probs[k]),12),h)
    return sorted(range(21),key=key)[:5]

def components(rows):
    """V7 probabilities and 4 expert probabilities. Every output t based exclusively on rows[:t]."""
    nmax=len(rows)
    ex_probs=np.zeros((nmax,4,21),float); base=np.zeros((nmax,21),float); weights=np.zeros((nmax,4),float)
    recent=deque(maxlen=50);scores=[deque(maxlen=40) for _ in range(4)];cond_sum=defaultdict(lambda:np.zeros(21)); cond_bal=defaultdict(lambda:np.zeros(21)); ewma=np.zeros(21)
    hist=[]; decay=.5**(1/40)
    for t,z in enumerate(rows):
        rsum=np.zeros(21)
        for p in recent:rsum[IX[pair(p)]]+=1
        r50=(rsum+30*FAIR)/(len(recent)+30)
        if hist:
            prev=hist[-1];a,b=prev[1:]; cs=(orientation(prev),(a+b)//4)
            cb=sum(1 if orientation(p)=='up' else -1 if orientation(p)=='down' else 0 for p in hist[-5:])
        else:cs=('start',-1);cb=0
        s=cond_sum[cs];b=cond_bal[cb]
        pp=np.stack([r50,(s+100*FAIR)/(s.sum()+100),(b+100*r50)/(b.sum()+100),(ewma+20*FAIR)/(ewma.sum()+20)])
        ex_probs[t]=pp
        hlen=len(scores[0])
        if hlen:
            successes=np.array([sum(ss) for ss in scores]); logits=1.5*(successes-hlen*10/36)/math.sqrt(hlen)
            raw=np.exp(logits-logits.max()); wt=.9*raw/raw.sum()+.1/4
        else:wt=np.ones(4)/4
        weights[t]=wt; base[t]=wt@pp
        tr=IX[pair(z)]
        for i in range(4):scores[i].append(int(tr in order(pp[i],t+1)))
        cond_sum[cs][tr]+=1;cond_bal[cb][tr]+=1
        ewma*=decay;ewma[tr]+=1;recent.append(z);hist.append(z)
    return ex_probs,base,weights

def contexts(rows):
    out=defaultdict(list); h=[]
    for z in rows:
        if h:
            last=h[-1]; a,b=last[1:]; ori=orientation(last)
            last5=h[-5:]; balance=sum(1 if orientation(r)=='up' else -1 if orientation(r)=='down' else 0 for r in last5)
            sign='up' if balance>0 else 'down' if balance<0 else 'balanced'
            present6=any(6 in x[1:] for x in last5)
            keys={'ori':ori,'prev_six':int(6 in last[1:]),'prev_special':last[0],
                  'sum_bin':(a+b)//3,'gap_bin':abs(a-b)//2,'balance':sign,
                  'ori_balance':(ori,sign),'ori_six':(ori,int(present6)),
                  'sum_special':((a+b)//4,last[0]),
                  'prev_repeat': int(len(h)>1 and pair(last)==pair(h[-2]))}
        else:keys={k:'start' for k in ('ori','prev_six','prev_special','sum_bin','gap_bin','balance','ori_balance','ori_six','sum_special','prev_repeat')}
        for k,v in keys.items():out[k].append(v)
        h.append(z)
    return out

def hits_from_proba(p,truth):
    n=len(truth); predicted=np.array([order(p[t],t+1) for t in range(n)])
    return (predicted==truth[:,None]).any(axis=1).astype(int),predicted

def evaluate(probs,truth):
    hits,pred=hits_from_proba(probs,truth)
    loss=-np.log(np.clip(probs[np.arange(len(truth)),truth],1e-10,1)).astype(float)
    return hits,pred,loss

def contextual_skill(ctx, ex_probs, truth, horizon, metric, prior):
    nmax=len(truth)
    skill=np.zeros((nmax,4))
    hit=np.zeros((nmax,4))
    for t in range(nmax):
        for e in range(4):hit[t,e]=float(truth[t] in order(ex_probs[t,e],t+1))
    if metric=='hit': perf=hit
    elif metric=='logprob':perf=np.log(np.clip(ex_probs[np.arange(nmax),:,truth],1e-10,1))
    else:raise ValueError(metric)
    for t in range(nmax):
        start=max(0,t-horizon)
        mask=np.array([ctx[i]==ctx[t] for i in range(start,t)],bool)
        if mask.any():
            local=perf[start:t][mask]
            center=local.mean(axis=0)
            centered=center-center.mean()
            strength=float(local.shape[0])/(local.shape[0]+prior)
            skill[t]=centered*strength
    return skill

def construct_candidates(rows):
    ex_probs,v7,w7=components(rows)
    truth=np.array([IX[pair(z)] for z in rows])
    ctxs=contexts(rows)
    result={'V7':v7}
    # Gating candidate: experts are not memorized; only their previous errors in the same context are used.
    for key in ('ori','prev_six','prev_special','sum_bin','gap_bin','balance','ori_balance','ori_six','sum_special','prev_repeat'):
        for horizon in (80,240):
            for metric in ('hit','logprob'):
                skill=contextual_skill(ctxs[key],ex_probs,truth,horizon,metric,prior=15)
                for temp in (2.0,6.0):
                    logits=np.log(np.clip(w7,1e-12,1))+temp*skill
                    s=np.exp(logits-logits.max(axis=1,keepdims=True));s/=s.sum(axis=1,keepdims=True)
                    gated=np.einsum('ne,nej->nj',s,ex_probs)
                    for mix in (.4,.8):
                        name=f'gate_{key}_W{horizon}_{metric}_T{temp:g}_M{mix:g}'
                        result[name]=(1-mix)*v7+mix*gated
    # Online bias correction of V7 forecasts, based only on previous forecast errors.
    actual=np.eye(21)[truth]
    for horizon in (30,80,160):
        residual=actual-v7
        for shrink in (40,100):
            s=np.zeros_like(v7)
            for t in range(1,len(rows)):
                start=max(0,t-horizon)
                s[t]=residual[start:t].sum(axis=0)/(t-start+shrink)
            for amount in (.2,.5):
                p=np.maximum(1e-8,v7+amount*s);p/=p.sum(axis=1,keepdims=True)
                result[f'calibrate_W{horizon}_S{shrink}_M{amount}']=p
    return result,truth,ex_probs,w7,ctxs

def assess_all(configs,truth):
    detail={};pred_cache={}
    spans={'train_101_350':(100,350),'validation_351_504':(350,504),'later_505_672':(504,672),'all_672':(0,672)}
    for name,p in configs.items():
        hits,pred,loss=evaluate(p,truth)
        row={'model':name}
        for label,(a,b) in spans.items():
            row[f'{label}_hits']=int(hits[a:b].sum())
            row[f'{label}_pct']=round(100*float(hits[a:b].mean()),3)
            row[f'{label}_logloss']=round(float(loss[a:b].mean()),6)
        detail[name]=row;pred_cache[name]=(hits,pred,loss)
    return detail,pred_cache

def choose(scores):
    base=scores['V7']; others=[v for k,v in scores.items() if k!='V7']
    # Predeclared conservative gate matches previous V12, counts and logloss both.
    winners=[r for r in others if r['train_101_350_hits']>=base['train_101_350_hits']+4 and r['validation_351_504_hits']>=base['validation_351_504_hits']+3 and r['train_101_350_logloss']<=base['train_101_350_logloss']-.005 and r['validation_351_504_logloss']<=base['validation_351_504_logloss']-.005]
    selected=(sorted(winners,key=lambda r:(-r['train_101_350_hits']-r['validation_351_504_hits'],r['validation_351_504_logloss']))[0]['model'] if winners else 'V7')
    best_early=sorted(others,key=lambda r:(-r['train_101_350_hits']-r['validation_351_504_hits'],r['validation_351_504_logloss']))[0]
    best_all=sorted(others,key=lambda r:-r['all_672_hits'])[0]
    return selected,best_early,best_all,len(winners)

def main():
    rows=load(); configs,truth,exp,w7,ctx=construct_candidates(rows)
    print('candidate_count',len(configs)-1,flush=True)
    scores,pred=assess_all(configs,truth)
    reference=list(csv.DictReader(open('/mnt/data/Adaptive_Dice_V12_Analog_States/V7_672_reference_audit.csv')))
    assert all(';'.join(','.join(map(str,PAIRS[j])) for j in pred['V7'][1][i]) == reference[i]['top5'] for i in range(672)), 'V7 reimplementation failed'
    picked,best_early,best_all,passed=choose(scores)
    out=sorted(scores.values(),key=lambda x:(-x['train_101_350_hits']-x['validation_351_504_hits'],x['validation_351_504_logloss']))
    with open(ROOT/'candidate_results.csv','w',encoding='utf8',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(out[0]));w.writeheader();w.writerows(out)
    with open(ROOT/'audit_672.csv','w',encoding='utf8',newline='') as f:
        fieldnames=['throw','actual_triple','actual_pair','v7_top5','v7_hit','v13_chosen_model','v13_chosen_top5','v13_hit','top1_hit','chosen_vs_v7_different']
        writer=csv.DictWriter(f,fieldnames=fieldnames);writer.writeheader()
        for i,z in enumerate(rows):
            h7,p7,_=pred['V7'];h,p,_=pred[picked]
            fmt=lambda y:';'.join(','.join(map(str,PAIRS[k])) for k in y)
            writer.writerow({'throw':i+1,'actual_triple':','.join(map(str,z)),'actual_pair':','.join(map(str,pair(z))), 'v7_top5':fmt(p7[i]),'v7_hit':int(h7[i]),'v13_chosen_model':picked,'v13_chosen_top5':fmt(p[i]),'v13_hit':int(h[i]),'top1_hit':int(p[i][0]==truth[i]),'chosen_vs_v7_different':int(not np.array_equal(p[i],p7[i]))})
    # Full model prefix test; alter actual at row #451 and verify predictions for all <=451. 
    modified=rows.copy();modified[450]=(2 if rows[450][0]==1 else 1, 6,6)
    modconfigs,_,_,_,_=construct_candidates(modified)
    unchanged=all(np.allclose(configs[name][:451],modconfigs[name][:451],atol=1e-13) for name in configs)
    diverged=bool(not np.allclose(configs[picked][451:],modconfigs[picked][451:],atol=1e-13))
    assert unchanged and diverged,{'unchanged':unchanged,'later_changed':diverged}
    summary={'master_sha256':SHA,'master_rows':672,'candidate_count_excluding_v7':len(configs)-1,'selected':picked,'promotion_passed':bool(passed),'passed_candidates':passed,'v7':scores['V7'],'chosen':scores[picked], 'best_early_diagnostic':best_early,'best_overall_diagnostic':best_all,'prefix_leakage_check':{'all_candidate_forecasts_through_451_unchanged_after_mutating_451':unchanged,'later_chosen_forecasts_changed':diverged},'limitations':'Original 672 throws reused in prior research, not a pristine independent test; 80% not supported by these historical results.'}
    (ROOT/'summary.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2),encoding='utf8')
    shutil.copy2(MASTER,ROOT/'throws(2).txt')
    print(json.dumps(summary,ensure_ascii=False,indent=2))
if __name__=='__main__':main()

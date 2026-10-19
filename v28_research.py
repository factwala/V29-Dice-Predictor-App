#!/usr/bin/env python3
"""V28: small preregistered set of strictly prefix-only pairwise calibration tests.
All forecasts for throw t are computed using outcomes rows[:t].
Candidate selection on first 504 reported separately. No claim of blind holdout.
"""
from pathlib import Path
import sys,csv,json,hashlib
import numpy as np
import v27_research as v27
import v20_research as v20
from v15_research import score
P=Path(__file__).resolve().parent
PERIODS={'warmup_1_100':(0,100),'train_101_350':(100,350),'validation_351_504':(350,504),'later_505_672':(504,672)}
# Limited preregistered configurations, not large hyperparameter search.
CANDIDATES=(
    ('pair_residual_w40_low','pair',40,0.50,40),
    ('pair_residual_w40_mid','pair',40,1.00,40),
    ('pair_residual_w100_low','pair',100,0.50,40),
    ('pair_residual_w100_mid','pair',100,1.00,40),
    ('pair_residual_w200_low','pair',200,0.50,40),
    ('pair_residual_w200_mid','pair',200,1.00,40),
    ('face_residual_w40_low','face',40,0.40,40),
    ('face_residual_w40_mid','face',40,0.80,40),
    ('face_residual_w100_low','face',100,0.40,40),
    ('face_residual_w100_mid','face',100,0.80,40),
    ('face_residual_w200_low','face',200,0.40,40),
    ('face_residual_w200_mid','face',200,0.80,40),
    ('context6_w100_low','context6',100,0.75,35),
    ('context6_w100_mid','context6',100,1.5,35),
    ('context6_w200_low','context6',200,0.75,35),
    ('context6_w200_mid','context6',200,1.5,35),
    ('contextsum_w100_low','contextsum',100,0.75,35),
    ('contextsum_w100_mid','contextsum',100,1.5,35),
    ('contextsum_w200_low','contextsum',200,0.75,35),
    ('contextsum_w200_mid','contextsum',200,1.5,35),
)

def face_map():
    pairs=v20.V.PAIRS
    f=np.zeros((21,6))
    for k,(a,b) in enumerate(pairs):
        f[k,a-1]+=1.0
        f[k,b-1]+=1.0
    return f

def prior_roll(obs,w):
    n=len(obs);zeros=np.zeros((1,)+obs.shape[1:]);cs=np.concatenate([zeros,np.cumsum(obs,axis=0)],axis=0)
    t=np.arange(n);return cs[t]-cs[np.maximum(0,t-w)]

def estimate(d,base,spec):
    nm,typ,w,alpha,shrink=spec
    y=d['y'];n=len(y);idx=np.arange(n)
    eye=np.eye(21)[y]
    err=eye-base
    if typ=='pair':
        count=prior_roll(err,w)
        nseen=np.minimum(idx,w)
        adj=count/(nseen[:,None]+shrink)
        # shrink residuals to unitless log-odds scale
        logits=np.log(np.maximum(base,1e-10))+alpha*(adj/0.052)
    elif typ=='face':
        fmap=face_map();resid=err@fmap
        summed=prior_roll(resid,w)
        seen=np.minimum(idx,w)
        adj=(summed/(seen[:,None]+shrink))@fmap.T
        logits=np.log(np.maximum(base,1e-10))+alpha*adj/0.34
    else:
        # Context on PREVIOUS recorded throw, no current throw labels.
        rows=d['rows']
        key=np.zeros(n,int)
        for t in range(1,n):
            a,b=sorted((rows[t-1][1],rows[t-1][2]))
            key[t]=int(b==6) if typ=='context6' else int((a+b)>=7)
        # Same-context observations only, as known before predicting t.
        delta=np.zeros_like(base)
        for t in range(n):
            lo=max(0,t-w);mask=(key[lo:t]==key[t]);nseen=int(mask.sum())
            if nseen:
                delta[t]=err[lo:t][mask].sum(axis=0)/(nseen+shrink)
        logits=np.log(np.maximum(base,1e-10))+alpha*delta/0.052
    logits=np.clip(logits,-40,40)
    logits-=np.max(logits,axis=1,keepdims=True)
    result=np.exp(logits);result/=result.sum(axis=1,keepdims=True)
    return result

def summarize(name,p,base,d):
    r,h,loss=score(p,d['y'],d['tie']);r0,h0,l0=score(base,d['y'],d['tie'])
    z={'model':name,'hits':int(h.sum()),'accuracy_pct':round(float(100*h.mean()),3),
      'new_hits':int((h&~h0).sum()),'lost_hits':int((h0&~h).sum()),
      'avg_logloss':round(float(loss.mean()),7),'top5_sets_changed':int(sum(set(r[t])!=set(r0[t]) for t in range(len(r))))}
    for k,(a,b) in PERIODS.items():z[k]=int(h[a:b].sum())
    z['first504']=int(h[:504].sum())
    return z

def main():
    path=P/'throws(2).txt';assert hashlib.sha256(path.read_bytes()).hexdigest()==v20.V.SHA
    rows=v20.V.load(path);assert len(rows)==672
    d,models=v27.build_experts(rows);d['rows']=rows
    base=models['V26_255'];ranks={};summary=[]
    summary.append(summarize('V26_reference',base,base,d));ranks['V26_reference']=base
    v27_policy=v27.predict_policies(d,models)[0]['hybrid_slow']
    summary.append(summarize('V27_hybrid_slow',v27_policy,base,d));ranks['V27_hybrid_slow']=v27_policy
    for spec in CANDIDATES:
        name=spec[0];pred=estimate(d,base,spec)
        summary.append(summarize(name,pred,base,d));ranks[name]=pred
    # policy chosen on first504 ONLY, label as historical holdout due past exposure to data
    selected=max(summary[2:],key=lambda z:(z['first504'],z['validation_351_504'],-z['avg_logloss']))
    hist=max(summary[2:],key=lambda z:(z['hits'],z['later_505_672'],-z['avg_logloss']))
    with (P/'candidate_results.csv').open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(summary[0]));w.writeheader();w.writerows(summary)
    cmp={'V26_reference':base,'V27_hybrid_slow':v27_policy,'V28_selected_first504':ranks[selected['model']],'V28_historical_best':ranks[hist['model']]}
    rr={k:score(v,d['y'],d['tie']) for k,v in cmp.items()}
    with (P/'audit_672.csv').open('w',newline='') as f:
        fields=['throw','actual_triple','actual_pair']+[v+'Top5' for v in cmp]+[v+'Hit' for v in cmp]
        w=csv.DictWriter(f,fieldnames=fields);w.writeheader()
        for t,row in enumerate(rows):
            out={'throw':t+1,'actual_triple':','.join(map(str,row)),'actual_pair':','.join(map(str,v20.V.pair(row)))}
            for k in cmp:
                out[k+'Top5']=';'.join(','.join(map(str,v20.V.PAIRS[int(j)])) for j in rr[k][0][t])
                out[k+'Hit']=int(rr[k][1][t])
            w.writerow(out)
    result={'baseline':summary[0],'v27':summary[1],'selected_first504':selected,'historical_best':hist,'number_candidates':len(CANDIDATES)}
    (P/'selection_v28.json').write_text(json.dumps(result,indent=2,ensure_ascii=False))
    print(json.dumps(result,indent=2,ensure_ascii=False))
    return result
if __name__=='__main__':main()

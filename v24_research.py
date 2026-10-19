#!/usr/bin/env python3
"""V24 exploratory Pair Replacement Risk Model.
Only historical prefix data at index t is used to predict result t.
Data reused after V1-V23: backtests are NOT prospective validation.
"""
from pathlib import Path
import csv, json, hashlib, time
import numpy as np
import v23_validate as v23
import v22_research as w22
import v20_research as w20
from v15_research import score
ROOT=Path(__file__).resolve().parent
FOLDS={'warmup_1_100':(0,100), 'early_101_350':(100,350),
       'validation_351_504':(350,504),'later_505_672':(504,672)}
EXPERTS=('face25','recent25','markov','six_bayes','change_bayes_60','V16_best','V15','V7')


def precompute(rows):
    d,p22,base,*_=v23.predict(rows)
    d2,ex,*_=w22.get_base(rows)
    assert np.array_equal(d['y'],d2['y'])
    y,tie=d['y'],d['tie'];n=len(rows)
    assert n>0
    rank0,h0,_=score(base,y,tie)
    actual=np.eye(21)[y]
    cumulative=np.vstack([np.zeros((1,21)),actual.cumsum(axis=0)])
    count_by_window={}
    for w in (15,25,50,100,150):
        idx=np.arange(n);count_by_window[w]=cumulative[idx]-cumulative[np.maximum(0,idx-w)]
    profile={}
    for name in EXPERTS:
        q=ex[name]
        rq,hq,_=score(q,y,tie)
        incoming=np.zeros((n,21),bool);outgoing=np.zeros((n,21),bool)
        incoming[np.arange(n)[:,None],rq]=True
        outgoing[np.arange(n)[:,None],rank0]=True
        entrant=incoming &~outgoing; dropout=outgoing&~incoming
        changed=entrant.any(axis=1)
        wins=(hq &~h0).astype(float);losses=(h0&~hq).astype(float)
        evidence={w:(w22.past_roll(wins,w).ravel(),w22.past_roll(losses,w).ravel()) for w in (24,48,96)}
        profile[name]={'q':q,'r':rq,'h':hq,'entered':entrant,'dropped':dropout,'changed':changed,'evidence':evidence}
    return d,base,profile,count_by_window,rank0,h0


def candidate_stats(pred,y,tie,name,meta=None):
    rank,hits,loss=score(pred,y,tie)
    rec={'model':name,'all_hits':int(hits.sum()),'mean_log_loss':round(float(loss.mean()),8)}
    for k,(lo,hi) in FOLDS.items():rec[k]=int(hits[lo:hi].sum())
    rec['train_101_504']=int(hits[100:504].sum())
    if meta:rec.update(meta)
    return rec


def utility_arrays(ent,drop,cnt,lookback,prior=20):
    n=len(cnt)
    N=np.minimum(np.arange(n),lookback)[:,None]+prior
    sm=(cnt+prior*w20.V.FAIR)/N
    # multinomial beta marginals; conservative (independence approximation over changed pairs)
    sdev=np.sqrt(np.maximum(sm*(1-sm),1e-12)/(N+1))
    expected=((ent.astype(float)-drop.astype(float))*sm).sum(axis=1)
    uncertainty=np.sqrt(((ent|drop)*sdev**2).sum(axis=1))
    return sm, expected,uncertainty


def run(rows, print_progress=False):
    d,base,experts,counts,rank0,h0=precompute(rows)
    y,tie=d['y'],d['tie'];n=len(rows); records=[]
    rec0=candidate_stats(base,y,tie,'V23_BASELINE',{'family':'baseline'})
    records.append(rec0)
    cache={}
    def test(pred, name, meta, *, store=False):
        result=candidate_stats(pred,y,tie,name,meta)
        records.append(result)
        if store:cache[name]=pred
        return result
    # FAMILY A: risk-adjusted Bayesian gain from top-5 entries relative to exits,
    # with past disagreements between the two predictors as a separate confidence term.
    for exp in EXPERTS:
        meta=experts[exp];q=meta['q'];ent=meta['entered'];drop=meta['dropped'];changed=meta['changed']
        for w in (15,25,50,100):
            probs, edge, uncertainty=utility_arrays(ent,drop,counts[w],w)
            for ew in (24,48):
                wins,losses=meta['evidence'][ew]
                denom=wins+losses+4
                past_adv=(wins-losses)/denom
                sufficient=(wins+losses>=2)
                for risk_z in (0.0, .5,1.0,1.5):
                    for bonus in (0.0,0.1,0.25):
                        net=edge-risk_z*uncertainty+bonus*past_adv
                        for thr in (-.02,0.,.02):
                            allowed=changed & sufficient & (past_adv>=-0.1) & (net>=thr)
                            if not allowed.any():continue
                            for alpha in (.25,.5,.75):
                                pred=base + (alpha*allowed)[:,None]*(q-base)
                                name=f'A_{exp}_w{w}_ew{ew}_z{risk_z}_b{bonus}_t{thr}_a{alpha}'
                                test(pred,name,{'family':'bayesian_expected_replacement','expert':exp,'lookback':w,'evidence_window':ew,'risk_z':risk_z,'confidence_bonus':bonus,'threshold':thr,'alpha':alpha,'gates':int(allowed.sum())})
        if print_progress:print('Finished A:',exp,len(records)-1,flush=True)
    # FAMILY B: direct single-pair replacement. Controlled swap of the lowest-supported
    # exit for highest-supported entry; evidence is from prefix only.
    for exp in ('face25','recent25','change_bayes_60','V16_best','markov'):
        m=experts[exp]; ent=m['entered'];drop=m['dropped'];change=m['changed']
        for w in (15,25,50,100):
            prob,edge,uncertainty=utility_arrays(ent,drop,counts[w],w)
            idx_entry=np.argmax(np.where(ent,prob,-1.),axis=1)
            idx_exit=np.argmin(np.where(drop,prob,2.),axis=1)
            pick_gain=prob[np.arange(n),idx_entry]-prob[np.arange(n),idx_exit]
            u_pair=np.sqrt(prob[np.arange(n),idx_entry]*(1-prob[np.arange(n),idx_entry])/(np.minimum(np.arange(n),w)+21)+prob[np.arange(n),idx_exit]*(1-prob[np.arange(n),idx_exit])/(np.minimum(np.arange(n),w)+21))
            for evidence_window in (24,48):
                wins,losses=m['evidence'][evidence_window]
                evidence_ok=(wins+losses>=2)&(wins>=losses)
                for z in (0.,.5,1.,1.5):
                    for threshold in (0.,.02,.04):
                        allow=change&evidence_ok&(pick_gain-z*u_pair>=threshold)
                        if not allow.any():continue
                        # Force exactly one top5 pair replacement; no answer access.
                        pred=base.copy(); tt=np.flatnonzero(allow)
                        # Lower bound is worst ranked incumbent score. Increase entrant just past that boundary.
                        boundary=pred[tt,rank0[tt,4]]
                        pred[tt,idx_entry[tt]]=np.maximum(pred[tt,idx_entry[tt]],boundary+1e-7)
                        pred=pred/np.sum(pred,axis=1)[:,None]
                        name=f'B_{exp}_w{w}_ev{evidence_window}_z{z}_t{threshold}'
                        test(pred,name,{'family':'single_replacement_risk','expert':exp,'lookback':w,'evidence_window':evidence_window,'risk_z':z,'threshold':threshold,'alpha':'swap','gates':int(allow.sum())})
    if print_progress:print('Finished B:',len(records)-1,flush=True)
    return d,base,records,cache


def winners(records):
    ref=records[0]; others=records[1:]
    # early-only selection is based on 101-350, while later folds are merely reported
    early_selected=max(others,key=lambda r:(r['early_101_350'],r['warmup_1_100'],-r['mean_log_loss']))
    # first504 selection separate from later505-672
    first504_selected=max(others,key=lambda r:(r['train_101_504'],r['validation_351_504'],-r['mean_log_loss']))
    hist=max(others,key=lambda r:(r['all_hits'],r['later_505_672']))
    both=[r for r in others if r['early_101_350']>ref['early_101_350'] and r['validation_351_504']>ref['validation_351_504']]
    holdout_nonloss=[r for r in others if all(r[f]>=ref[f] for f in FOLDS) and r['all_hits']>ref['all_hits']]
    return {'reference':ref,'highest_historical':hist,'selected_on_101_350':early_selected,
           'selected_on_101_504':first504_selected,'both_early_and_mid_strictly_better':len(both),
           'nondecreasing_four_periods':len(holdout_nonloss),'tested':len(others)}

if __name__=='__main__':
    p=ROOT/'throws(2).txt'
    assert hashlib.sha256(p.read_bytes()).hexdigest()==w20.V.SHA
    rows=w20.V.load(p);assert len(rows)==672
    t=time.monotonic();d,base,records,_=run(rows,True)
    summary=winners(records)
    print('SECONDS',round(time.monotonic()-t,2),'SUMMARY',json.dumps(summary,indent=2),flush=True)
    with (ROOT/'candidate_results.csv').open('w',newline='') as f:
        keys=list(dict.fromkeys(k for rec in records for k in rec))
        w=csv.DictWriter(f,fieldnames=keys);w.writeheader();w.writerows(records)
    (ROOT/'selection.json').write_text(json.dumps(summary,indent=2,ensure_ascii=False))

#!/usr/bin/env python3
"""V25: Online Replacement Risk Calibration.
At time t forecast is computed only from outcomes in rows[:t].
IMPORTANT retrospective hyperparameter search on previously inspected 672 data does NOT prove future lift.
"""
from pathlib import Path
import sys,csv,json,hashlib,time
import numpy as np
R=Path(__file__).resolve().parent
sys.path.insert(0,str(R))
import v24_validate as v24
import v24_research as research24
import v22_research as v22
import v20_research as v20
from v15_research import score
FOLDS={'warmup_1_100':(0,100),'early_101_350':(100,350),'validation_351_504':(350,504),'later_505_672':(504,672)}
EXPERTS=('markov','face25','recent25','six_bayes','change_bayes_60','V16_best','V15','V7')

def precompute(rows):
    d,base23,base24,gate24,*_=v24.predict(rows,v24.SPECS['V24_253_retrospective'])
    d2,ex,_,_=v22.get_base(rows)
    assert np.array_equal(d['y'],d2['y'])
    y=d['y'];tie=d['tie'];n=len(y)
    ranks,h0,_=score(base24,y,tie)
    truth=np.eye(21)[y]
    cum=np.vstack((np.zeros((1,21)),truth.cumsum(axis=0)))
    idx=np.arange(n)
    freq={}
    for w in (15,25,50,100,200):
        count=cum[idx]-cum[np.maximum(0,idx-w)]
        N=np.minimum(idx,w)[:,None]
        # shrink frequency estimates toward fair unconditional pair rates
        freq[w]=(count+25*v20.V.FAIR)/(N+25)
    delta={}
    for exp in EXPERTS:
        q=ex[exp]
        rankq,hq,_=score(q,y,tie)
        inc=np.zeros((n,21),bool);out=np.zeros((n,21),bool)
        inc[idx[:,None],rankq]=True
        out[idx[:,None],ranks]=True
        entry=inc&~out;exit_=out&~inc;changed=entry.any(axis=1)
        win=(hq&~h0).astype(float);loss=(h0&~hq).astype(float)
        # observed swap utility: +1 if outcome in entrant set, -1 if in dropout set
        realized=((entry.astype(float)-exit_.astype(float))*truth).sum(axis=1)
        assert np.array_equal((realized>0).astype(float),win) and np.array_equal((realized<0).astype(float),loss)
        perf={}
        for w in (24,48,96,192):
            W=v22.past_roll(win,w).ravel();L=v22.past_roll(loss,w).ravel()
            C=v22.past_roll(changed.astype(float),w).ravel()
            # conservative posterior mean net reward shrink to zero; no current outcome
            perf[w]={'W':W,'L':L,'C':C,'adv':(W-L)/(W+L+6),
                     'prob':(W+3)/(W+L+6),
                     'se':np.sqrt((W+3)*(L+3)/((W+L+6)**2*(W+L+7)))}
        delta[exp]={'q':q,'entry':entry,'exit':exit_,'changed':changed,'perf':perf}
    return d,base23,base24,gate24,ex,freq,delta

def candidate_stats(pred,y,tie,name,extra=None):
    rank,h,loss=score(pred,y,tie)
    z={'model':name,'hits':int(h.sum()),'log_loss':float(loss.mean()),'train_101_504':int(h[100:504].sum())}
    for key,(a,b) in FOLDS.items():z[key]=int(h[a:b].sum())
    if extra:z.update(extra)
    return z

def generate(rows,progress=False):
    d,base23,base,gate24,ex,freq,delta=precompute(rows)
    y=d['y'];tie=d['tie'];n=len(rows)
    scores=[];kept={};
    def assess(pred,name,meta={},keep=False):
        z=candidate_stats(pred,y,tie,name,meta);scores.append(z)
        if keep:kept[name]=pred
        return z
    assess(base,'V24_253_BASE',{'family':'base'},True)
    assess(base23,'V23_250_REFERENCE',{'family':'base'},True)
    # A: Online-calibrated replacement risk from entry/exit probability difference
    for e in EXPERTS:
        dlt=delta[e];q=dlt['q'];entry=dlt['entry'];exit_=dlt['exit'];changed=dlt['changed'];
        for fw in (25,50,100,200):
            f=freq[fw]; diff=((entry.astype(float)-exit_.astype(float))*f).sum(axis=1)
            count=np.minimum(np.arange(n),fw)
            # finite-sample uncertainty with multinomial candidate counts
            un=np.sqrt(((entry|exit_)*f*(1-f)).sum(axis=1)/(count+26))
            for ew in (24,48,96):
                p=dlt['perf'][ew];W,L,C,adv=p['W'],p['L'],p['C'],p['adv']
                for risk in (0.,0.5,1.0):
                    for history_bonus in (0.,.05,.15,.3):
                        utility=diff-risk*un+history_bonus*adv
                        for threshold in (-.03,0,.03):
                            for margin in (-1,0,1):
                                approved=changed&(W+L>=2)&((W-L)>=margin)&(utility>=threshold)
                                if not approved.any():continue
                                for alpha in (.25,.5,.75):
                                    pred=base+(alpha*approved)[:,None]*(q-base)
                                    name=f'A_{e}_fw{fw}_ew{ew}_r{risk}_bonus{history_bonus}_t{threshold}_m{margin}_a{alpha}'
                                    assess(pred,name,{'family':'calibrated_expected_gain','expert':e,'gates':int(approved.sum()),'set_changes':int(np.sum(approved&changed))})
        if progress:print('A complete',e,'candidates',len(scores)-2,flush=True)
    # B: adaptive posterior Bayesian utility learned only on past disagreement outcomes
    for e in EXPERTS:
        dlt=delta[e];q=dlt['q'];changed=dlt['changed'];
        for ew in (24,48,96,192):
            p=dlt['perf'][ew];W,L,C,adv=p['W'],p['L'],p['C'],p['adv']
            for z in (0.,.5,1.,1.5):
                # confidence lower bound in a 2-way split of decisive previous cases
                lower=p['prob']-z*p['se']
                for th in (.4,.45,.5,.55):
                    for min_decisions in (1,2,4,6):
                        approved=changed&(W+L>=min_decisions)&(lower>=th)
                        if not approved.any():continue
                        for alpha in (.25,.5):
                            pred=base+(alpha*approved)[:,None]*(q-base)
                            name=f'B_{e}_ew{ew}_z{z}_th{th}_n{min_decisions}_a{alpha}'
                            assess(pred,name,{'family':'posterior_reliability','expert':e,'gates':int(approved.sum())})
        if progress:print('B complete',e,'candidates',len(scores)-2,flush=True)
    # C: Continuous shrink toward pairwise empirical rates with calibrated expected lift
    for fw in (25,50,100,200):
        residual=freq[fw]-base
        for b in (.025,.05,.10,.20,.35):
            pred=np.maximum(1e-9,base+b*residual)
            pred/=pred.sum(axis=1,keepdims=True)
            assess(pred,f'C_pair_calibration_fw{fw}_a{b}',{'family':'online_pair_probability_calibration','expert':'none','gates':n})
    return d,base23,base,scores,delta,freq,kept

def selection(scores):
    ref=scores[0];alt=scores[2:]
    hist=max(alt,key=lambda z:(z['hits'],z['later_505_672'],-z['log_loss']))
    train=max(alt,key=lambda z:(z['train_101_504'],z['validation_351_504'],-z['log_loss']))
    early=max(alt,key=lambda z:(z['early_101_350'],z['warmup_1_100'],-z['log_loss']))
    nonloss=[z for z in alt if all(z[k]>=ref[k] for k in FOLDS) and z['hits']>ref['hits']]
    return {'reference':ref,'highest_historical':hist,'selected_first504':train,'selected_first350':early,'nonloss_candidates':len(nonloss),'tested':len(alt)}

if __name__=='__main__':
    f=R/'throws(2).txt';assert hashlib.sha256(f.read_bytes()).hexdigest()==v20.V.SHA
    rows=v20.V.load(f);assert len(rows)==672
    t=time.perf_counter();d,p23,p24,scores,*_=generate(rows,True)
    summary=selection(scores)
    print('elapsed',round(time.perf_counter()-t,1),'TOTAL',len(scores),'SUMMARY',json.dumps(summary,indent=2),flush=True)
    keys=list(dict.fromkeys(k for s in scores for k in s))
    with (R/'candidate_results.csv').open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=keys);w.writeheader();w.writerows(scores)
    (R/'selection.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2))

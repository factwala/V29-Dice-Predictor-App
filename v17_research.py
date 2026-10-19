#!/usr/bin/env python3
"""V17: Bayesian online change point & causally adaptive experts.
- All forecasts at t use only throws[:t].
- Retrospective search acknowledged; selection windows are diagnostic, not clean holdout.
- No historical result lookup, memorization or post-t outcome features.
"""
from pathlib import Path
import csv, json, math, hashlib
import numpy as np
import v13_core as V
from v16_research import build as v16_build, run_candidates as v16_candidates
from v15_research import detect, score, metrics, rolling_binary
ROOT=Path(__file__).resolve().parent


def bocpd(y, fair, hazard=0.03, maxrun=120, concentration=18):
    """Online Bayesian hazard model, Dirichlet-multinomial on unordered 21 pairs.
    At t predictions from posterior last updated on t-1. Duration likelihoods use
    only prior outcomes. Truncated tail approximates a long-running regime.
    """
    n=len(y);K=len(fair);out=np.zeros((n,K));ch=np.zeros(n); age=np.zeros(n)
    prior=concentration*fair
    w=np.zeros(maxrun+1);w[0]=1.
    cnt=np.zeros((maxrun+1,K),float)
    for t,j in enumerate(y):
        forecast=(cnt+prior[None,:])/(cnt.sum(axis=1)[:,None]+concentration)
        out[t]=(w[:,None]*forecast).sum(axis=0)
        ch[t]=w[:min(5,len(w))].sum()
        age[t]=w@np.arange(len(w))
        lik=forecast[:,j]
        neww=np.zeros_like(w);newcnt=np.zeros_like(cnt)
        # after seeing y[t], either reset run or continue; action affects only t+1
        neww[0]=(w*lik).sum()*hazard
        neww[1:]=w[:-1]*lik[:-1]*(1-hazard)
        # mass at capped run length retained, combine count histories
        tail=w[-1]*lik[-1]*(1-hazard)
        tot=neww[-1]+tail
        if tot>0:
            accum=neww[-1]*(cnt[-2].copy())+tail*cnt[-1]
            newcnt[-1]=accum/tot
        for r in range(1,maxrun):newcnt[r]=cnt[r-1]
        # each carried regime gets the observed y
        newcnt[1:,j]+=1
        neww[-1]+=tail
        norm=neww.sum()
        if norm<=0: raise ArithmeticError('Invalid posterior weights')
        w=neww/norm;cnt=newcnt
    assert np.allclose(out.sum(axis=1),1)
    return out,ch,age


def binary_bocpd_six(rows, hazard=.03,maxrun=120,concentration=10):
    n=len(rows);posterior=np.zeros(n);sums=np.zeros(maxrun+1);w=np.zeros(maxrun+1);w[0]=1
    num=np.zeros(maxrun+1)
    for t,z in enumerate(rows):
        p=(num+concentration*11/36)/(sums+concentration)
        posterior[t]=w@p
        value=int(6 in z[1:]);lik=p if value else 1-p
        neww=np.zeros(maxrun+1);neww[0]=(w*lik).sum()*hazard
        neww[1:]=w[:-1]*lik[:-1]*(1-hazard)
        tail=w[-1]*lik[-1]*(1-hazard)
        nw=neww[-1]+tail
        newnum=np.zeros_like(num);newsum=np.zeros_like(sums)
        newnum[1:maxrun]=num[:maxrun-1]+value
        newsum[1:maxrun]=sums[:maxrun-1]+1
        if nw>0:
            newnum[-1]=(neww[-1]*num[-2]+tail*num[-1])/nw+value
            newsum[-1]=(neww[-1]*sums[-2]+tail*sums[-1])/nw+1
        neww[-1]+=tail;w=neww/neww.sum();num=newnum;sums=newsum
    return posterior


def arrays(rows):
    d=v16_build(rows);y=d['y'];tie=d['tie'];n=len(y);t=np.arange(n)
    experts=d['experts'].copy()
    old_v16={'change_pair_0.32_cf3_onset_H25_markov_a0.5', 'change_v13_surprise_0.08_cf1_sustain_H8_markov_a0.5'}
    _, v16_cached=v16_candidates(d,old_v16)
    experts['V16_best']=v16_cached['change_pair_0.32_cf3_onset_H25_markov_a0.5']
    experts['V16_early']=v16_cached['change_v13_surprise_0.08_cf1_sustain_H8_markov_a0.5']
    for H in (20,60,150):
        p,ch,age=bocpd(y,V.FAIR,1/H,maxrun=120,concentration=20)
        experts[f'change_bayes_{H}']=p
        if H==60:change_recent=ch;change_age=age
    six_p=binary_bocpd_six(rows,hazard=.04)
    mask=np.array([int(6 in pair) for pair in V.PAIRS],bool)
    baseline=experts['V14']
    p6=np.maximum(baseline[:,mask].sum(axis=1),1e-9)
    conditioned=np.zeros_like(baseline)
    conditioned[:,mask]=baseline[:,mask]*(six_p/p6)[:,None]
    conditioned[:,~mask]=baseline[:,~mask]*((1-six_p)/(1-p6))[:,None]
    experts['six_bayes']=conditioned
    # binary top-5 and pointwise log probability feedback, assessed after prediction
    hit={k:score(p,y,tie)[1].astype(float) for k,p in experts.items()}
    logp={k:np.log(np.maximum(p[np.arange(n),y],1e-15)) for k,p in experts.items()}
    group=['V7','V13','V14','V15','V16_best','V16_early','recent25','face25','markov','change_bayes_20','change_bayes_60','change_bayes_150','six_bayes']
    # all rolling feedback obtained from prefix [0,t) only
    def rolling_feedback(src,w,prior=.28,smooth=15):
        vals=np.stack([src[k] for k in group],axis=1)
        cs=np.vstack([np.zeros((1,len(group))),vals.cumsum(axis=0)])
        lo=np.maximum(0,t-w);raw=cs[t]-cs[lo];return (raw+smooth*prior)/((t-lo+smooth)[:,None])
    h12=rolling_feedback(hit,12);h30=rolling_feedback(hit,30);h100=rolling_feedback(hit,100)
    log20=rolling_feedback(logp,20,prior=-3.2,smooth=25)
    log75=rolling_feedback(logp,75,prior=-3.2,smooth=25)
    # change signals from past-only rows; prior forecaster underperformance flag
    drop=np.maximum(h100[:,group.index('V14')]-h12[:,group.index('V14')],0)
    drift6=d['signals']['six']; driftpair=d['signals']['pair']
    signals={'pair_tv':driftpair,'six':drift6,'skill_drop':drop,
             'bayes_short_run': np.clip(change_recent-.08,0,1)}
    return {'experts':experts,'group':group,'hits':hit,'y':y,'tie':tie,'h12':h12,'h30':h30,'h100':h100,'log20':log20,'log75':log75,'signals':signals,'bayes_age':change_age}


def candidates(d):
    n=len(d['y']);group=d['group'];E=np.stack([d['experts'][k] for k in group],axis=1)
    ref=d['experts']['V14'];v15=d['experts']['V15']
    yield 'V7',d['experts']['V7']
    yield 'V13',d['experts']['V13']
    yield 'V14',ref
    yield 'V15',v15
    yield 'V16_best',d['experts']['V16_best']
    yield 'V16_early',d['experts']['V16_early']
    # V16's fixed, previously discovered forecast is a baseline for fair comparisons.
    for spec in ('V15','V16_best','V16_early'):
        for contrast in ('change_bayes_20','change_bayes_60','six_bayes'):
            for alpha in (.05,.1,.2,.4):
                yield f'refine_{spec}_{contrast}_a{alpha}',(1-alpha)*d['experts'][spec]+alpha*d['experts'][contrast]
    # Bayesian change point forecasts, mixing with reference to limit variance
    for spec in ('change_bayes_20','change_bayes_60','change_bayes_150','six_bayes'):
        for alpha in (.1,.25,.5,1.):
            yield f'bayes_{spec}_a{alpha}',(1-alpha)*ref+alpha*d['experts'][spec]
    # Feedback gating: experts penalized when recently inaccurate and erratic.
    for signal,s in d['signals'].items():
        thresholds={'pair_tv':(.15,.25), 'six':(.10,.18),'skill_drop':(.05,.12),'bayes_short_run':(.08,.2)}[signal]
        for threshold in thresholds:
            active,duration=detect(s,threshold,confirm=2,release=.7,relax=3)
            for hl in (10,30,70):
                severity=active*np.exp(-np.maximum(duration-1,0)/hl)
                for criterion in ('hits','log'):
                    for speed in (.4,.8):
                        if criterion=='hits':
                            skill=(1-speed)*d['h100']+speed*d['h30']
                            fast=d['h12']; scale=7
                        else:
                            skill=(1-speed)*d['log75']+speed*d['log20']
                            fast=d['log20'];scale=2
                        # During onset increase short feedback; punish unreliable experts
                        sk=(1-.4*severity[:,None])*skill + .4*severity[:,None]*fast
                        for reliability in (0,.5):
                            # penalize unstable experts: skill must persist across windows
                            penalty=np.abs(d['h30']-d['h100'])
                            logits=scale*(sk - reliability*penalty)
                            logits=logits-logits.max(axis=1,keepdims=True)
                            w=np.exp(logits);w/=w.sum(axis=1,keepdims=True)
                            ensemble=np.einsum('ne,nek->nk',w,E)
                            for mix in (.25,.5,.8):
                                pr=(1-mix)*ref+mix*ensemble
                                yield f'gate_{signal}_th{threshold}_hl{hl}_{criterion}_speed{speed}_rel{reliability}_m{mix}',pr
    # Combined state detection: error drops AND face drift needed, focus on single best expert
    for signal,s in d['signals'].items():
        thresholds={'pair_tv':(.15,.25), 'six':(.10,.18),'skill_drop':(.05,.12),'bayes_short_run':(.08,.2)}[signal]
        for threshold in thresholds:
            active,duration=detect(s,threshold,confirm=3,release=.7,relax=4)
            onset=active*np.exp(-(np.maximum(duration-1,0))/20)
            for expert in ('change_bayes_20','change_bayes_60','six_bayes','V15'):
                for mix in (.1,.25,.5):
                    yield f'onset_{signal}_th{threshold}_{expert}_m{mix}', (1-mix*onset[:,None])*ref+(mix*onset[:,None])*d['experts'][expert]


def evaluate(rows,d):
    outcome=[];extra={};n=len(rows)
    for name,p in candidates(d):
        sc=metrics(name,p,d['y'],d['tie']);outcome.append(sc)
        if name in ('V7','V13','V14','V15'):extra[name]=p
    return outcome,extra

def choose(recs):
    base=next(r for r in recs if r['model']=='V7')
    v15=next(r for r in recs if r['model']=='V15')
    possible=[r for r in recs if r['model'].startswith(('gate_','onset_','bayes_','refine_'))]
    # Only pick by first #504; no one gets to use #505+ results in selection.
    # More stringent: sustained uplift in both #101-350, #351-504 over V7.
    good=[r for r in possible if r['train_101_350_hits']>=base['train_101_350_hits']+3
          and r['validation_351_504_hits']>=base['validation_351_504_hits']+3
          and r['train_101_350_logloss']<=base['train_101_350_logloss']+.005
          and r['validation_351_504_logloss']<=base['validation_351_504_logloss']+.005]
    early=max(possible,key=lambda r:(r['train_101_350_hits']+r['validation_351_504_hits'],r['validation_351_504_hits'],-r['validation_351_504_logloss']))
    pick=max(good,key=lambda r:(r['train_101_350_hits']+r['validation_351_504_hits'],r['validation_351_504_hits'],-r['validation_351_504_logloss'])) if good else base
    hist=max(recs,key=lambda r:(r['all_672_hits'],r['later_505_672_hits']))
    return pick,early,hist,len(good)

def main():
    src=ROOT/'throws(2).txt';data=src.read_bytes();assert hashlib.sha256(data).hexdigest()==V.SHA
    rows=V.load(src);assert len(rows)==672
    d=arrays(rows);rec,ex=evaluate(rows,d);pick,early,hist,npass=choose(rec)
    sortedrec=sorted(rec,key=lambda r:(-r['all_672_hits'],-r['later_505_672_hits']))
    with (ROOT/'candidate_results.csv').open('w',newline='',encoding='utf8') as f:
        w=csv.DictWriter(f,fieldnames=list(sortedrec[0]));w.writeheader();w.writerows(sortedrec)
    names={a for a in ['V7','V13','V14','V15','V16_best','V16_early',pick['model'],early['model'],hist['model']]}
    selected={k:p for k,p in candidates(d) if k in names}
    predictions={k:score(p,d['y'],d['tie']) for k,p in selected.items()}
    fields=['throw','original_triple','actual_pair']+[f'{tag}_{field}' for tag in ('V7','V15','V16_best','V17_selected','V17_historical') for field in ('top5','hit')]
    aliases={'V7':'V7','V15':'V15','V16_best':'V16_best','V17_selected':pick['model'],'V17_historical':hist['model']}
    with (ROOT/'audit_672.csv').open('w',newline='',encoding='utf8') as f:
        writer=csv.DictWriter(f,fields);writer.writeheader()
        for i,r in enumerate(rows):
            line={'throw':i+1,'original_triple':','.join(map(str,r)),'actual_pair':','.join(map(str,V.pair(r)))}
            for tag,key in aliases.items():
                top,h,_=predictions[key];line[f'{tag}_top5']=';'.join(','.join(map(str,V.PAIRS[j])) for j in top[i]);line[f'{tag}_hit']=int(h[i])
            writer.writerow(line)
    # Previously observed continuation, NOT independent.
    with (ROOT/'LIVE_673_693.csv').open() as f: live=[tuple(map(int,r[1:])) for r in csv.reader(f)]
    dc=arrays(rows+live); names2={k for k in names if k in selected}
    post={k:p for k,p in candidates(dc) if k in names2}
    known={k:int(score(p,dc['y'],dc['tie'])[1][672:].sum()) for k,p in post.items()}
    # Six prefix invariance probes using full recomputation; changed results in RAM only.
    probes=[]
    for loc in (49,100,350,450,503,671):
        r=rows[loc];m=rows.copy();m[loc]=(1 if r[0]==2 else 2,6 if r[1]!=6 else 1, 6 if r[2]!=6 else 2)
        da=arrays(m); modified={k:p for k,p in candidates(da) if k in names}
        invariant=all(np.allclose(selected[k][:loc+1],modified[k][:loc+1],rtol=0,atol=1e-12) for k in names)
        assert invariant,loc
        probes.append({'throw':loc+1,'prefix_unchanged':True})
    assert hashlib.sha256(src.read_bytes()).hexdigest()==V.SHA
    report={'master_checksum':V.SHA,'rows':672,'candidates':len(rec),'reference':{k:next(r for r in rec if r['model']==k) for k in ('V7','V13','V14','V15','V16_best','V16_early')},'early_chosen':early,'selected_by_gate':pick,'gate_pass_count':npass,'highest_historical_exploratory':hist,'known_continuation_hits_21':known,'prefix_checks':probes, 'warning':'Past 672 and known 21 continuation used in previous studies; retrospective selection is not independent validation.'}
    (ROOT/'summary.json').write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf8')
    print(json.dumps(report,ensure_ascii=False,indent=2))

if __name__=='__main__':main()

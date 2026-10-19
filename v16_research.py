#!/usr/bin/env python3
"""V16: causal regime onset/exit/duration & rolling expert-arbitration.

All row-t forecasts use rows[:t]. Master data immutable/checksum checked.
Selection only uses #101-504; #505-672 later window is not used in selection.
Caution: earlier research repeatedly viewed the entire 672 sequence.
"""
from pathlib import Path
import csv, json, hashlib, math, sys
import numpy as np
import v13_core as V
from v14_validate import models
from v15_research import rolling_binary, rolling_proba, detect, score, metrics
ROOT=Path(__file__).resolve().parent


def build(rows, controls=None):
    refs, y, _=models(rows); n=len(rows); t=np.arange(n)
    tie=np.zeros((n,21),dtype=np.int16)
    for i in range(n):
        hashes=[hashlib.sha256(f'TWO-DICE-FAIR-SHA256-V1|{i+1}|{a},{b}'.encode()).hexdigest() for a,b in V.PAIRS]
        for j,k in enumerate(sorted(range(21),key=lambda j:hashes[j])):tie[i,k]=j
    v7=refs['V7'];v13=refs['V13_233'];v14=refs['V14_234']
    rec25=rolling_proba(y,25);rec75=rolling_proba(y,75)
    yhot=np.eye(21)[y]
    cum=np.vstack([np.zeros((1,21)),yhot.cumsum(axis=0)]);lo=np.maximum(0,t-25)
    paircnt=cum[t]-cum[lo]
    faces=np.zeros((n,6))
    for j,(a,b) in enumerate(V.PAIRS):faces[:,a-1]+=paircnt[:,j];faces[:,b-1]+=paircnt[:,j]
    faces=(faces+10/6)/(2*(t-lo)[:,None]+10)
    face25=np.stack([faces[:,a-1]*faces[:,b-1]*(1 if a==b else 2) for a,b in V.PAIRS],axis=1)
    markov=np.zeros((n,21)); counts=np.zeros((21,21))
    for i in range(n):
        if i>=2:counts[y[i-2],y[i-1]]+=1
        if i:markov[i]=(counts[y[i-1]]+20*V.FAIR)/(counts[y[i-1]].sum()+20)
        else:markov[i]=V.FAIR
    # Last known V15 configuration, fixed from previous research (not re-chosen here)
    six=np.array([int(6 in z[1:]) for z in rows],float)
    orient=np.array([1 if V.orientation(z)=='up' else -1 if V.orientation(z)=='down' else 0 for z in rows],float)
    loss13=-np.log(np.maximum(v13[np.arange(n),y],1e-12))
    surprise=np.r_[0,loss13[:-1]]
    jump=np.maximum(rolling_binary(surprise,20)-rolling_binary(surprise,120),0)
    active,duration=detect(jump,.08,confirm=3)
    onset=active*np.exp(-np.maximum(duration-1,0)/25)
    v15=(1-.25*onset[:,None])*v14+(.25*onset[:,None])*face25
    experts={'V7':v7,'V13':v13,'V14':v14,'V15':v15,'recent25':rec25,'recent75':rec75,'face25':face25,'markov':markov}
    # causal binary skill streams per expert, evaluated at end of preceding throw
    hits={k:score(p,y,tie)[1].astype(float) for k,p in experts.items()}
    def recent_skill(k,w):
        z=hits[k]; c=np.r_[0,z.cumsum()];lo=np.maximum(0,t-w)
        return (c[t]-c[lo]+10*5/21)/(t-lo+10)
    short={k:recent_skill(k,25) for k in experts}
    long={k:recent_skill(k,120) for k in experts}
    # strictly-past-only state features
    drift6=np.abs(rolling_binary(six,25)-rolling_binary(six,150))
    driftori=np.abs(rolling_binary(orient,25)-rolling_binary(orient,150))
    drifttv=.5*np.abs(rolling_proba(y,25,1)-rolling_proba(y,150,1)).sum(axis=1)
    v14hits=hits['V14'];cs=np.r_[0,v14hits.cumsum()];
    def hrate(w):
        lo=np.maximum(0,t-w);return (cs[t]-cs[lo]+3)/(t-lo+10)
    lossdrift=np.maximum(hrate(120)-hrate(25),0)
    signals={'six':drift6,'pair':drifttv,'direction':driftori,'v13_surprise':jump,'v14_hit_drop':lossdrift}
    conf={'six':(.12,.2),'pair':(.20,.32),'direction':(.2,.35),'v13_surprise':(.08,.15),'v14_hit_drop':(.06,.12)}
    return {'experts':experts,'hits':hits,'short':short,'long':long,'signals':signals,'thresholds':conf,'y':y,'tie':tie}


def run_candidates(data, target_names=None):
    experts=data['experts']; y=data['y']; tie=data['tie'];n=len(y)
    records={};chosen={}
    def add(name, p, keep=False):
        if target_names is not None and name not in target_names:return
        records[name]=metrics(name,p,y,tie)
        if keep or target_names is not None:chosen[name]=p
    for name in ('V7','V13','V14','V15'):
        add(name,experts[name],True)
    # separate changes in baseline and alternative expert. Post-onset decay / persistent regime
    for signal,arr in data['signals'].items():
        for threshold in data['thresholds'][signal]:
            for cf in (1,3):
                active,duration=detect(arr,threshold,confirm=cf,release=.7,relax=3)
                for form in ('onset','sustain'):
                    for half in (8,25,80):
                        influence=(np.exp(-np.maximum(duration-1,0)/half) if form=='onset'
                                   else (1-np.exp(-duration/half)))*active
                        for specialist in ('recent25','face25','markov','V13','V15'):
                            for alpha in (.10,.25,.50):
                                name=f'change_{signal}_{threshold}_cf{cf}_{form}_H{half}_{specialist}_a{alpha}'
                                if target_names is not None and name not in target_names:continue
                                p=(1-alpha*influence[:,None])*experts['V14']+alpha*influence[:,None]*experts[specialist]
                                add(name,p)
    # Regime-dependent online arbitration based solely on previous expert hits
    for signal,arr in data['signals'].items():
        for threshold in data['thresholds'][signal]:
            active,age=detect(arr,threshold,confirm=2,release=.7,relax=3)
            for temp in (3,9):
                for speed in (.25,.75):
                    # balance short- and long-run skills; give new regime more short-run weight
                    onset=np.exp(-np.maximum(age-1,0)/25)*active
                    wshort=np.clip(speed+.35*onset,0,1)
                    specialists=('V7','V13','V15','recent25','face25','markov')
                    skills=np.stack([wshort*data['short'][k]+(1-wshort)*data['long'][k] for k in specialists],axis=1)
                    logits=temp*(skills-skills.mean(axis=1,keepdims=True));w=np.exp(logits-logits.max(axis=1,keepdims=True));w/=w.sum(axis=1,keepdims=True)
                    expert=sum(w[:,i,None]*experts[k] for i,k in enumerate(specialists))
                    for alpha in (.25,.50,.80):
                        name=f'arbitrate_{signal}_{threshold}_T{temp}_speed{speed}_a{alpha}'
                        if target_names is not None and name not in target_names:continue
                        add(name,(1-alpha)*experts['V14']+alpha*expert)
    return records,chosen


def select(records):
    ref=records['V7'];b14=records['V14']
    eligible=[r for r in records.values() if r['model'] not in ('V7','V13','V14','V15')]
    # The only selection-relevant windows are #101-350 and #351-504, not #505-672.
    # Verify no regression on validation as a floor, and require both periods positive v7.
    gated=[r for r in eligible if
           r['train_101_350_hits']>=ref['train_101_350_hits']+2 and
           r['validation_351_504_hits']>=ref['validation_351_504_hits']+2 and
           r['train_101_350_logloss']<=ref['train_101_350_logloss']+.010 and
           r['validation_351_504_logloss']<=ref['validation_351_504_logloss']+.010]
    # chosen pre holdout, no 505-672 information in sorting
    by_early=lambda r:(r['train_101_350_hits']+r['validation_351_504_hits'],r['validation_351_504_hits'],-r['validation_351_504_logloss'])
    early=max(eligible,key=by_early)
    pick=max(gated,key=by_early) if gated else ref
    historical=max(records.values(),key=lambda r:(r['all_672_hits'],r['later_505_672_hits']))
    return pick,early,historical,len(gated)


def main():
    source=ROOT/'throws(2).txt'; original=source.read_bytes();assert hashlib.sha256(original).hexdigest()==V.SHA
    rows=V.load(source); assert len(rows)==672
    data=build(rows);sc, _=run_candidates(data)
    pick,early,historical,passing=select(sc)
    ordered=sorted(sc.values(),key=lambda r:(-r['all_672_hits'],-r['validation_351_504_hits']))
    with open(ROOT/'candidate_results.csv','w',newline='',encoding='utf8') as f:
        w=csv.DictWriter(f,list(ordered[0]));w.writeheader();w.writerows(ordered)
    names=set(['V7','V13','V14','V15',pick['model'],historical['model'],early['model']]);_,forecasts=run_candidates(data,names)
    scored={name:score(p,data['y'],data['tie']) for name,p in forecasts.items()}
    with open(ROOT/'audit_672.csv','w',newline='',encoding='utf8') as f:
        col=['throw','original_triple','real_unordered_pair']+[f'{k}_{suffix}' for k in ('V7','V14','V15','V16_selected','V16_historical') for suffix in ('top5','hit')]
        w=csv.DictWriter(f,col);w.writeheader()
        aliases={'V7':'V7','V14':'V14','V15':'V15','V16_selected':pick['model'],'V16_historical':historical['model']}
        for i,z in enumerate(rows):
            r={'throw':i+1,'original_triple':','.join(map(str,z)),'real_unordered_pair':','.join(map(str,V.pair(z)))}
            for label,key in aliases.items():
                pred,h,_=scored[key];r[f'{label}_top5']=';'.join(','.join(map(str,V.PAIRS[j])) for j in pred[i]);r[f'{label}_hit']=int(h[i])
            w.writerow(r)
    # evaluate already-known continuation, NOT a fresh blind test.
    live=[]
    for r in csv.reader(open(ROOT/'LIVE_673_693.csv')):live.append(tuple(map(int,r[1:])))
    combined=rows+live; d2=build(combined);_,post=run_candidates(d2,names)
    continuation={k: {'hits':int(score(p,d2['y'],d2['tie'])[1][672:693].sum()),'out_of':21} for k,p in post.items()}
    # full recomputation prefix invariance: V7/V14/V15/selected/historical are sensitive to altered past but not actual at same timestep
    tests=[]
    for index in (49,350,450,600,671):
        z=rows[index]; swapped=rows.copy(); swapped[index]=(2 if z[0]==1 else 1, 6 if z[1]!=6 else 1, 6 if z[2]!=6 else 2)
        altdata=build(swapped);_,alt=run_candidates(altdata,names)
        invariant=all(np.allclose(forecasts[k][:index+1],alt[k][:index+1],rtol=0,atol=1e-12) for k in names)
        tests.append({'throw_modified_in_memory':index+1,'predictions_up_to_and_including_this_throw_unchanged':bool(invariant)})
        assert invariant,(index,names)
    assert source.read_bytes()==original
    stats={'original_sha256':V.SHA,'rows':672,'configurations_evaluated':len(sc),'selection_uses_first_504_only':True,'main_model':'V7','reference':{k:sc[k] for k in ('V7','V13','V14','V15')},'best_early_candidate':early,'gate_pass_count':passing,'model_chosen_by_gate':pick,'highest_retrospective_score':historical,'known_continuation':continuation,'prefix_tests':tests,'note':'672 outcomes reused across V7-V15 research; later_505_672 and known_673_693 are NOT clean independent holdouts; even prefix safety cannot prevent search overfitting.'}
    (ROOT/'summary.json').write_text(json.dumps(stats,ensure_ascii=False,indent=2))
    print(json.dumps(stats,ensure_ascii=False,indent=2),flush=True)

if __name__=='__main__':main()

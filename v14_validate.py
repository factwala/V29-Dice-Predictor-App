#!/usr/bin/env python3
"""Independent reproduction and causal prefix leakage tests of specific V14 champion.
The master is strictly read only; mutations for controls are RAM-only.
"""
import sys,json,hashlib,csv
from pathlib import Path
import numpy as np
import v13_core as V
OUT=Path(__file__).resolve().parent
SHA=V.SHA
PAIRS=V.PAIRS

def fast_rank(prob):
    n=len(prob)
    h=np.zeros((n,21),int)
    for t in range(n):
        hashes=[hashlib.sha256(f'TWO-DICE-FAIR-SHA256-V1|{t+1}|{a},{b}'.encode()).hexdigest() for a,b in PAIRS]
        for i,k in enumerate(sorted(range(21),key=lambda k:hashes[k])):h[t,k]=i
    return np.lexsort((h,-np.round(prob,12)),axis=1)[:,:5]

def models(rows):
    n=len(rows)
    ex,v7,w7=V.components(rows)
    truth=np.array([V.IX[V.pair(z)] for z in rows])
    sk=V.contextual_skill(V.contexts(rows)['prev_special'],ex,truth,240,'hit',15)
    l=np.log(np.maximum(w7,1e-12))+6*sk
    s=np.exp(l-l.max(axis=1,keepdims=True));s/=s.sum(axis=1,keepdims=True)
    v13=.2*v7+.8*np.einsum('ne,nej->nj',s,ex)
    x=np.array([int(6 in z[1:]) for z in rows]); c=np.r_[0,x.cumsum()];t=np.arange(n)
    def avg(window):
        lo=np.maximum(0,t-window)
        return (c[t]-c[lo])/np.maximum(t-lo,1)
    drift=np.abs(avg(25)-avg(150))
    intensity=1/(1+np.exp(-np.clip((drift-.3)*12,-20,20)))
    # Causal 25-throw face estimates with pseudocount 10 (Dirichlet on 6 faces)
    y=np.array([V.IX[V.pair(z)] for z in rows]);onehot=np.eye(21)[y];cum=np.vstack([np.zeros((1,21)),onehot.cumsum(axis=0)])
    lo=np.maximum(0,t-25);f=cum[t]-cum[lo];num=t-lo
    ff=np.zeros((n,6))
    for j,(a,b) in enumerate(PAIRS):ff[:,a-1]+=f[:,j];ff[:,b-1]+=f[:,j]
    face=(ff+10/6)/(2*num[:,None]+10)
    p25=np.stack([face[:,a-1]*face[:,b-1]*(1 if a==b else 2) for a,b in PAIRS],axis=1)
    p25/=p25.sum(axis=1,keepdims=True)
    v14=(1-.1*intensity[:,None])*v13+(.1*intensity[:,None])*p25
    return {'V7':v7,'V13_233':v13,'V14_234':v14},truth,{'drift':drift,'intensity':intensity}

def stats(p,t):
    pred=fast_rank(p)
    hit=(pred==t[:,None]).any(axis=1)
    loss=-np.log(np.maximum(1e-15,p[np.arange(len(t)),t]))
    ranges={'1-672':(0,672),'101-350':(100,350),'351-504':(350,504),'505-672':(504,672)}
    if len(t)>672:ranges['673-'+str(len(t))]=(672,len(t))
    return pred,hit,{k:{'hits':int(hit[a:b].sum()),'denom':b-a,'percent':round(100*float(hit[a:b].mean()),3),'mean_log_loss':round(float(loss[a:b].mean()),6)} for k,(a,b) in ranges.items()}

def main():
    raw=(Path(__file__).resolve().parent/'throws(2).txt').read_bytes()
    assert hashlib.sha256(raw).hexdigest()==SHA
    master=V.load()
    m,t,d=models(master)
    computed={}
    for name,p in m.items():
        pred,hit,sc=stats(p,t);computed[name]=dict(pred=pred,hit=hit,sc=sc)
    assert computed['V7']['sc']['1-672']['hits']==229
    assert computed['V13_233']['sc']['1-672']['hits']==233
    assert computed['V14_234']['sc']['1-672']['hits']==234
    # multiple exact prefix causal checks for both predictions and model selection forecasts
    tests=[]
    for pos in (49,100,350,450,503,600,671):
        altered=master.copy();a,b,c=altered[pos]
        altered[pos]=(1 if a==2 else 2, 6 if b!=6 else 1,6 if c!=6 else 2)
        alt,_,_=models(altered)
        changed_after=False
        for name,p in m.items():
            assert np.allclose(p[:pos+1],alt[name][:pos+1],atol=1e-13), (name,pos)
            changed_after |= not np.allclose(p[pos+1:],alt[name][pos+1:],atol=1e-13)
        tests.append({'changed_throw':pos+1,'all_predictions_through_changed_throw_unchanged':True,'any_later_forecast_changed':bool(changed_after)})
    assert hashlib.sha256((Path(__file__).resolve().parent/'throws(2).txt').read_bytes()).hexdigest()==SHA
    with open(OUT/'audit_672.csv','w',newline='',encoding='utf8') as f:
        names=['throw','actual_triple','actual_unordered_pair','v7_top5','v7_hit','v13_top5','v13_hit','v14_top5','v14_hit','v14_top1_hit','v14_six_rate_drift','v14_regime_intensity']
        w=csv.DictWriter(f,fieldnames=names);w.writeheader()
        for i,z in enumerate(master):
            fmt=lambda pred:';'.join(','.join(map(str,PAIRS[k])) for k in pred)
            w.writerow({'throw':i+1,'actual_triple':','.join(map(str,z)),'actual_unordered_pair':','.join(map(str,V.pair(z))),**{name:fmt(computed[k]['pred'][i]) for k,name in [('V7','v7_top5'),('V13_233','v13_top5'),('V14_234','v14_top5')]},'v7_hit':int(computed['V7']['hit'][i]),'v13_hit':int(computed['V13_233']['hit'][i]),'v14_hit':int(computed['V14_234']['hit'][i]),'v14_top1_hit':int(computed['V14_234']['pred'][i,0]==t[i]),'v14_six_rate_drift':round(float(d['drift'][i]),6),'v14_regime_intensity':round(float(d['intensity'][i]),6)})
    live=[]
    with open(OUT/'LIVE_673_693.csv') as f:
        for r in csv.reader(f):live.append(tuple(map(int,r[1:])))
    combined=master+live
    ml,tl,_=models(combined)
    live_stats={name:stats(p,tl)[2]['673-693'] for name,p in ml.items()}
    with open(OUT/'known_673_693_audit.csv','w',newline='',encoding='utf8') as f:
        w=csv.DictWriter(f,fieldnames=['throw','actual','v7_hit','v13_hit','v14_hit','v14_top5']);w.writeheader()
        scores={k:stats(p,tl) for k,p in ml.items()}
        for i in range(672,693):
            w.writerow({'throw':i+1,'actual':','.join(map(str,combined[i])),'v7_hit':int(scores['V7'][1][i]),'v13_hit':int(scores['V13_233'][1][i]),'v14_hit':int(scores['V14_234'][1][i]),'v14_top5':';'.join(','.join(map(str,PAIRS[k])) for k in scores['V14_234'][0][i])})
    report={'master_sha256':SHA,'historical_scores':{name:z['sc'] for name,z in computed.items()},'known_673_693':live_stats,'leakage_tests':tests,'v14_vs_v13_flips':{'v14_extra_hits':int(((computed['V14_234']['hit']==1)&(computed['V13_233']['hit']==0)).sum()),'v14_lost_hits':int(((computed['V14_234']['hit']==0)&(computed['V13_233']['hit']==1)).sum())}}
    (OUT/'validation_summary.json').write_text(json.dumps(report,ensure_ascii=False,indent=2))
    print(json.dumps(report,ensure_ascii=False,indent=2),flush=True)
if __name__=='__main__':main()

"""Recovered original V28 generator + historical V29 scoring for NEXT throw.

For a live app, supply verified chronological history to next_top5().
This does NOT read current/next result, nor reproduce original holdout claims
as a promise of future performance.
"""
from collections import Counter
import v13_core as core
from v15_research import score
from v28_validate import run_model

PAIRS = [f'{a},{b}' for a in range(1, 7) for b in range(a, 7)]

def parse_throw(s):
    vals = tuple(int(x.strip()) for x in str(s).split(','))
    if len(vals)!=3 or vals[0] not in (1,2) or any(x<1 or x>6 for x in vals):
        raise ValueError('Expected special,die2,die3 e.g. 1,3,2')
    return vals

def next_v28(history):
    rows = [parse_throw(s) if isinstance(s,str) else tuple(s) for s in history]
    if not rows: raise ValueError('No prior observations')
    # The generator emits one prediction per row using strictly prior results.
    # Supply a dummy row to obtain the one-step-ahead forecast, and prove that
    # different dummy outcomes give identical forecasts.
    out=[]
    for dummy in ((1,1,1),(2,6,6)):
        data, _, predictions, _ = run_model(rows+[dummy])
        ranks, _, _ = score(predictions, data['y'], data['tie'])
        out.append([','.join(map(str,core.PAIRS[int(j)])) for j in ranks[-1]])
    if out[0]!=out[1]:
        raise RuntimeError('Future outcome affected V28 prediction; stop')
    return out[0]

def next_top5(history):
    rows = [parse_throw(s) if isinstance(s,str) else tuple(s) for s in history]
    prior=next_v28(rows)
    observed=[','.join(map(str,sorted(r[1:]))) for r in rows[-400:]]
    counts=Counter(observed)
    rank={p:j for j,p in enumerate(prior)}
    scores={p:0.25*max(0,5-rank.get(p,5))/15 + 0.75*(counts[p]+1)/(len(observed)+21) for p in PAIRS}
    chosen=sorted(PAIRS,key=lambda p:(-scores[p],p))[:5]
    return {'throw':len(rows)+1,'v28_prior':prior,'v29_top5':chosen,'history_used':len(observed)}

if __name__=='__main__':
    import v20_research as v20
    historical=v20.V.load('throws(2).txt')
    live=[(1,6,6),(2,5,2),(1,5,1),(1,3,2)]
    print(next_top5(historical+live))

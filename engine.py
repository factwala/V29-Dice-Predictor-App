"""Frozen V29 scoring, with explicit provisional fixed-V28-prior continuation."""

import hashlib
import json
from collections import Counter



PAIRS = [f'{a},{b}' for a in range(1,7) for b in range(a,7)]
WINDOW = 400
MIX = 0.75


def validate_throw(s):
    vals = [x.strip() for x in str(s).split(',')]
    if len(vals) != 3 or any(x not in ('1','2','3','4','5','6') for x in vals) or vals[0] not in ('1','2'):
        raise ValueError('तीन अंक लिखें: पहला 1 या 2, बाकी दोनों 1 से 6 तक (जैसे 1,5,6)।')
    return ','.join(vals)


def unordered_pair(s):
    x = validate_throw(s).split(',')
    return ','.join(sorted(x[1:], key=int))





def score_v29(history, prior):
    if len(prior)!=5 or len(set(prior))!=5 or not set(prior)<=set(PAIRS):
        raise ValueError('Invalid baseline V28 ranking')
    # Important: current throw is NOT part of history.
    observed=[unordered_pair(x) for x in history[-WINDOW:]]
    counts=Counter(observed)
    rank={p:j for j,p in enumerate(prior)}
    freq={p:(counts[p]+1)/(len(observed)+len(PAIRS)) for p in PAIRS}
    scores={p:(1-MIX)*max(0,5-rank.get(p,5))/15+MIX*freq[p] for p in PAIRS}
    return sorted(PAIRS,key=lambda p:(-scores[p],p))[:5]




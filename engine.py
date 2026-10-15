"""Frozen V29 scoring, with explicit provisional fixed-V28-prior continuation."""
import csv
import hashlib
import json
from collections import Counter
from pathlib import Path

DATA_FILE = Path(__file__).parent / 'data' / 'historical_v28_audit.csv'
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


def seed_rows():
    with DATA_FILE.open(newline='', encoding='utf-8') as fh:
        rows = list(csv.DictReader(fh))
    if len(rows) != 672 or [int(x['throw']) for x in rows] != list(range(1,673)):
        raise RuntimeError('Historical audit must have precisely 672 ordered rows.')
    return rows


def seed_payload():
    rows = seed_rows()
    data = [[str(r['throw']), validate_throw(r['original_result']), r['V28_top5']] for r in rows]
    digest = hashlib.sha256(json.dumps(data,separators=(',', ':'),ensure_ascii=False).encode()).hexdigest()
    return data,digest


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


def reproduce_history():
    rows=seed_rows(); history=[]
    for row in rows:
        got=score_v29(history,row['V28_top5'].split(';'))
        # V29 prior historical predictions not stored here; compare against local audit separately in tests.
        history.append(validate_throw(row['original_result']))
    return history

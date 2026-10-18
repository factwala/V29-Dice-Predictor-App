"""V29 legacy scorer and experimental alternatives; no future data is used."""
from collections import Counter

PAIRS = [f"{a},{b}" for a in range(1, 7) for b in range(a, 7)]
WINDOW = 400
MIX = 0.75


def validate_throw(s):
    vals = [x.strip() for x in str(s).split(",")]
    if len(vals) != 3 or any(x not in ("1", "2", "3", "4", "5", "6") for x in vals) or vals[0] not in ("1", "2"):
        raise ValueError("तीन अंक लिखें: पहला 1 या 2, बाकी दोनों 1 से 6 तक (जैसे 1,5,6)।")
    return ",".join(vals)


def unordered_pair(s):
    x = validate_throw(s).split(",")
    return ",".join(sorted(x[1:], key=int))


def _ranking(history, prior, window, mix):
    if not 0 <= mix <= 1:
        raise ValueError("mix must be between 0 and 1")
    if prior is not None and (len(prior) != 5 or len(set(prior)) != 5 or not set(prior) <= set(PAIRS)):
        raise ValueError("Invalid baseline V28 ranking")
    if window is not None and (not isinstance(window, int) or window <= 0):
        raise ValueError("window must be a positive integer or None")
    observed = [unordered_pair(x) for x in (history[-window:] if window is not None else history)]
    counts = Counter(observed)
    ranks = {p: j for j, p in enumerate(prior or [])}
    frequency = {p: (counts[p] + 1) / (len(observed) + len(PAIRS)) for p in PAIRS}
    scores = {
        p: (1 - mix) * max(0, 5 - ranks.get(p, 5)) / 15 + mix * frequency[p]
        for p in PAIRS
    }
    ordered = sorted(PAIRS, key=lambda p: (-scores[p], p))
    return ordered, scores, counts, len(observed)


def score_v29(history, prior):
    """EXACT legacy V29: last 400 observations and fixed V28 prior."""
    return _ranking(history, prior, WINDOW, MIX)[0][:5]


def score_details(history, prior):
    ordered, scores, counts, _ = _ranking(history, prior, WINDOW, MIX)
    return [
        {"Rank": i + 1, "Pair": p, "V29 Score": round(scores[p], 6), "Last 400 Frequency": counts[p]}
        for i, p in enumerate(ordered)
    ]


def compare_models(history, prior):
    """Diagnostic comparison only; does NOT change live predictions."""
    configurations = [
        ("V29 legacy (400 + fixed V28)", WINDOW, MIX, prior),
        ("Full history + fixed V28", None, MIX, prior),
        ("Rolling frequency only (400)", WINDOW, 1.0, None),
        ("Full history frequency only", None, 1.0, None),
    ]
    output = []
    for label, window, mix, baseline in configurations:
        ordered, scores, counts, used = _ranking(history, baseline, window, mix)
        output.append({
            "Model": label,
            "History Used": used,
            "Top-5": ";".join(ordered[:5]),
            "Rank 5/6 Gap": round(scores[ordered[4]] - scores[ordered[5]], 6),
        })
    return output

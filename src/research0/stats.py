"""Agreement and accuracy statistics used by the audit (Phase 4) and the quality gates (Phase 7). Pure functions, no database."""
import math
from collections import Counter


def wilson(k, n, z=1.96):
    """Wilson score interval for a proportion. Returns (estimate, low, high); all None when n is 0."""
    if n <= 0:
        return None, None, None
    p = k / n
    denominator = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / denominator
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denominator
    return p, max(0.0, centre - half), min(1.0, centre + half)


def cohen_kappa(pairs):
    """Cohen's kappa for two raters and any set of categories. pairs: [(a, b)]. kappa is None when chance agreement is 1."""
    n = len(pairs)
    if not n:
        raise ValueError("At least one paired rating is required")
    observed = sum(a == b for a, b in pairs) / n
    ca, cb = Counter(a for a, _ in pairs), Counter(b for _, b in pairs)
    expected = sum(ca[c] * cb[c] for c in set(ca) | set(cb)) / (n * n)
    return {"n": n, "agreement": observed, "kappa": (observed - expected) / (1 - expected) if expected < 1 else None}


def krippendorff_alpha_nominal(units):
    """Krippendorff's alpha for nominal data. units: list of lists with the values given to each unit by whoever rated it
    (any number of raters, missing ratings simply absent). Units with fewer than two ratings are ignored."""
    units = [list(u) for u in units if len(u) >= 2]
    total = sum(len(u) for u in units)
    if not units or total <= 1:
        raise ValueError("At least one unit with two ratings is required")
    coincidences = Counter()
    for u in units:
        counts = Counter(u)
        m = len(u)
        for c, nc in counts.items():
            for k, nk in counts.items():
                coincidences[(c, k)] += (nc * (nc - 1) if c == k else nc * nk) / (m - 1)
    marginals = Counter()
    for (c, _), v in coincidences.items():
        marginals[c] += v
    observed = sum(v for (c, k), v in coincidences.items() if c != k)
    expected = sum(marginals[c] * marginals[k] for c in marginals for k in marginals if c != k) / (total - 1)
    return {"units": len(units), "alpha": 1 - observed / expected if expected else None}


def prf(tp, fp, fn):
    precision = tp / (tp + fp) if tp + fp else None
    recall = tp / (tp + fn) if tp + fn else None
    f1 = 2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else None
    return {"precision": precision, "recall": recall, "f1": f1, "tp": tp, "fp": fp, "fn": fn}


def macro_f1(truth, predicted, labels):
    """Macro-averaged F1 over `labels`. A label with no truth and no prediction is left out of the average."""
    scores = []
    for label in labels:
        tp = sum(t == label and p == label for t, p in zip(truth, predicted))
        fp = sum(t != label and p == label for t, p in zip(truth, predicted))
        fn = sum(t == label and p != label for t, p in zip(truth, predicted))
        f1 = prf(tp, fp, fn)["f1"]
        if f1 is not None:
            scores.append(f1)
    return sum(scores) / len(scores) if scores else None


def icc_agreement(matrix):
    """ICC(2,1): two-way random effects, absolute agreement, single rater (Shrout and Fleiss). matrix: rows = items, columns = raters,
    complete (no missing). Returns None when the variance is degenerate."""
    n, k = len(matrix), len(matrix[0]) if matrix else 0
    if n < 2 or k < 2 or any(len(r) != k for r in matrix):
        raise ValueError("A complete items x raters matrix with at least 2 items and 2 raters is required")
    grand = sum(map(sum, matrix)) / (n * k)
    row_means = [sum(r) / k for r in matrix]
    col_means = [sum(matrix[i][j] for i in range(n)) / n for j in range(k)]
    ss_rows = k * sum((m - grand) ** 2 for m in row_means)
    ss_cols = n * sum((m - grand) ** 2 for m in col_means)
    ss_total = sum((x - grand) ** 2 for r in matrix for x in r)
    ms_r, ms_c = ss_rows / (n - 1), ss_cols / (k - 1)
    ms_e = (ss_total - ss_rows - ss_cols) / ((n - 1) * (k - 1))
    denominator = ms_r + (k - 1) * ms_e + k * (ms_c - ms_e) / n
    return {"items": n, "raters": k, "icc": (ms_r - ms_e) / denominator if denominator else None}


def spearman(x, y):
    from scipy.stats import spearmanr
    if len(x) != len(y) or len(x) < 3:
        raise ValueError("At least 3 paired values are required")
    rho = spearmanr(x, y).statistic
    return {"n": len(x), "rho": None if math.isnan(rho) else float(rho)}

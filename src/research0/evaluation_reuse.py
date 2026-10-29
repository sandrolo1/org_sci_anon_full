"""TEMPORARY (21 Oct 2026, the user's request): a stratified sample that starts from the papers already judged, so the ABS 1-2 papers added to the 2026 set can be judged without
paying again for the others. To be deleted once that benchmark is built; the standard stratified sample (`paper_evaluation.sample(stratify=True)`, from zero or extending one run) stays.

How it works: in the menu "Extend an earlier run" of the page, the value "judged" asks `paper_evaluation.with_keep` for `keep_judged` instead of the papers of one run. In each ABS grade
it keeps the papers already judged by any run (test runs excluded) - first those judged on the three criteria, then those judged on novelty only, each in the usual hash order - up to the
grade's share of n (`paper_evaluation.allocate`); the standard sample then fills each grade with new papers. What the judges already answered comes from the register of the calls,
so a kept paper costs nothing as long as its prompt is the same (a paper judged on novelty only still needs its methodology and impact votes, and a card made since changes them).

To remove it: delete this file and its test (tests/test_evaluation_reuse.py), and the lines marked TEMPORARY in paper_evaluation.py and dashboard_template.html."""
from . import paper_evaluation as pe


def keep_judged(con, where, *, n=pe.DEFAULT_N, seed=pe.DEFAULT_SEED, min_words=pe.DEFAULT_MIN_WORDS, require_grade=True, require_fulltext=True, stratify=False, **_):
    """The ids to keep: the papers of the pool already judged, at most each grade's share of n when `stratify` (else at most n). -> list of paper ids."""
    pool, _, display, _, _ = pe.pool_of(where, min_words=min_words, require_grade=require_grade, require_fulltext=require_fulltext)
    judged = pe._judged(con)
    good = ("ok", "partial")
    rank = lambda p: 0 if (judged.get(p) or {}).get("full") in good else 1 if (judged.get(p) or {}).get("novelty") in good else None
    done = [p for p in pool if rank(p) is not None]
    ordered = lambda ids: sorted(pe.draw(ids, len(ids), int(seed)), key=rank)      # three criteria first, then the hash: sorted() is stable
    if not stratify:
        return ordered(done)[:int(n)]
    capacity = {}
    for p in pool:
        g = pe.stratum(display, p)
        capacity[g] = capacity.get(g, 0) + 1
    share = pe.allocate(int(n), capacity)
    out = []
    for g in pe.strata_order(capacity):
        out += ordered([p for p in done if pe.stratum(display, p) == g])[:share[g]]
    return out

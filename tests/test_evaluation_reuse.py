"""TEMPORARY, with src/research0/evaluation_reuse.py (delete both together): the stratified sample that keeps the papers already judged, so they are not paid for again."""
import re
import uuid

from research0 import evaluation_reuse, paper_evaluation as ev

from test_forum_scoring import EMBEDDER, Judge
from test_forum import needs_db
from test_paper_evaluation import db, where          # noqa: F401 (fixtures)


@needs_db
def test_the_papers_already_judged_are_kept_up_to_each_grade_and_asked_nothing_again(db, where):
    with db.connect() as c:
        made = ev.create_run(c, n=5, seed=3, min_words=20, require_fulltext=False, where=where, synthetic=True, max_tokens=500_000)
        first = uuid.UUID(made["run_id"])
        judge = Judge(vote=6)
        assert ev.execute(c, first, where=where, synthetic_provider=judge, embedder=EMBEDDER)["state"] == "completed"
        c.execute("UPDATE evaluation_run SET synthetic=false WHERE run_id=%s", (first,))           # test runs are never reused: this one stands for a real run
        c.commit()
        judged = c.execute("SELECT parameters->'paper_ids' FROM evaluation_run WHERE run_id=%s", (first,)).fetchone()[0]
        display = ev.read_display(where["directory"])
        options = {"n": 6, "seed": 3, "min_words": 20, "require_fulltext": False, "stratify": True}
        keep = evaluation_reuse.keep_judged(c, where, **options)
        per_grade = lambda ids: {g: sum(1 for p in ids if ev.stratum(display, p) == g) for g in {ev.stratum(display, p) for p in ids}}
        assert set(keep) <= set(judged) and all(n <= 2 for n in per_grade(keep).values())                  # the share of each grade is 6 / 3 = 2
        assert sum(per_grade(keep).values()) == sum(min(2, n) for n in per_grade(judged).values())
        assert evaluation_reuse.keep_judged(c, where, **{**options, "stratify": False}) == ev.draw(judged, len(judged), 3)     # unstratified: all of them, in the hash order
        shown = ev.pipeline(c, where=where, extend_run="judged", **options)
        assert shown["sample"]["kept"] == len(keep) and {g["grade"]: g["papers"] for g in shown["sample"]["strata"]} == {"4*": 2, "3": 2, "2": 2}
        asked_before = len(judge.requests)
        second = ev.create_run(c, where=where, synthetic=True, max_tokens=500_000, extend_run="judged", **options)
        params = c.execute("SELECT parameters FROM evaluation_run WHERE run_id=%s", (second["run_id"],)).fetchone()[0]
        assert params["extends_run"] == "judged" and params["kept"] == len(keep) and params["stratify"] and params["paper_ids"][:len(keep)] == keep
        assert ev.execute(c, uuid.UUID(second["run_id"]), where=where, synthetic_provider=judge, embedder=EMBEDDER)["state"] == "completed"
        ids = lambda reqs: {re.search(r"INPUT_ID: (\S+)", r.prompt).group(1) for r in reqs}
        assert len(ids(judge.requests[asked_before:])) == 6 - len(keep)                                       # only the new papers were asked

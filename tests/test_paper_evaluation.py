"""The evaluation of published papers by the novelty judges (paper_evaluation.py): what the judge may see, the sample, the cutoff rule, a whole run offline, and the statistics the page shows.
The judge is the scripted stand-in of test_forum_scoring and the embeddings are the offline lexical ones, so nothing here calls a model."""
import csv
import datetime as dt
import io
import json
import re
import textwrap
import threading
import time
import uuid

import pytest

from research0 import mock_world, paper_cards, paper_evaluation as ev, scoring_io, scoring_judges
from research0.forum_providers import Cancelled, TurnResult

from test_forum_scoring import EMBEDDER, TOPICS, Judge, build_corpora
from test_forum import needs_db

# the same papers the corpus has, in other words, so the neighbours are real neighbours
JOURNALS = [("Journal of Imaginary Studies", "4*"), ("Organization Tests", "3"), ("Review of Mock Methods", "2"), ("Annals of Scripted Papers", "")]


def abstract(topic, i):
    return (f"This paper examines {topic} in firms of type {i}. We theorise a mechanism about {topic} and test it on a panel of firms, and we discuss what the result changes for the literature "
            f"on {topic} and for managers who deal with {topic} every day.")


def make_folder(root, count=12, extra=None):
    """A local copy of the layout of inputs/evaluation_2026: the file the judge reads, the file only the page reads, no raw folder."""
    (root / "evaluator_input").mkdir(parents=True)
    (root / "display_metadata").mkdir()
    inputs, display = [], []
    for i in range(count):
        pid = f"e26_{i:012d}"
        topic = TOPICS[i % len(TOPICS)]
        journal, grade = JOURNALS[i % len(JOURNALS)]
        text = abstract(topic, i) if i != 5 else "Too short to be an abstract."
        inputs.append({"paper_id": pid, "title": f"A study {i} of {topic}", "abstract": text, "abstract_sha256": scoring_io.sha256_text(text)})
        display.append({"paper_id": pid, "doi": f"10.9999/pub.{i}", "journal": journal, "ajg2024": grade, "ajg_matched_by": "issn" if grade else "", "ajg_file": "", "online_date": f"2026-0{1 + i % 9}-15",
                        "source_sets": "publishability_dataset", "abstract_source": "crossref", "abstract_words": str(len(text.split()))})
    if extra:
        inputs[0] = {**inputs[0], **extra}
    with open(root / "evaluator_input" / "abstracts.jsonl", "w", encoding="utf-8", newline="\n") as f:
        for r in inputs:
            f.write(json.dumps(r) + "\n")
    with open(root / "display_metadata" / "papers.csv", "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(display[0]))
        w.writeheader()
        w.writerows(display)
    return inputs, display


@pytest.fixture
def where(tmp_path):
    folder = tmp_path / "evaluation_2026"
    make_folder(folder)
    corpora = build_corpora(tmp_path / "corpora")
    cfg = scoring_judges.load_config(ev.ROOT / "config" / "evaluation_pre2026.json")
    cfg = {**cfg, "run_live": False, "embedding": "hash-rp-v1", "k": 3, "lexical_k": 2, "concurrency": 1, "budget": {"max_live_calls": 0, "max_tokens": 0, "max_seconds": 0}}
    cfg["judges"] = [{**j, "provider": "synthetic"} for j in cfg["judges"]]
    (tmp_path / "cfg.json").write_text(json.dumps(scoring_judges.validate_config(cfg)), encoding="utf-8")
    return {"directory": folder, "config": tmp_path / "cfg.json", "workspace": tmp_path / "ws", "novelty": corpora["novelty"], "methodology": corpora["methodology"],
            "impact": corpora["impact"], "cards_config": ev.ROOT / "config" / "evaluation_cards.json"}


@pytest.fixture
def db(world):
    with world.connect() as c:
        for table in ("evaluation_score", "evaluation_run", "evaluation_card", "evaluation_fulltext", "evaluation_paper"):
            c.execute(f"DELETE FROM {table}")
        c.commit()
    yield world
    with world.connect() as c:
        for table in ("evaluation_score", "evaluation_run", "evaluation_card", "evaluation_fulltext", "evaluation_paper"):
            c.execute(f"DELETE FROM {table}")
        c.commit()


# --- what the judge may see ------------------------------------------------------------------------------------------------------------------

def test_the_file_the_judge_reads_may_hold_nothing_but_the_title_and_the_abstract(tmp_path):
    make_folder(tmp_path / "ok")
    assert len(ev.read_inputs(tmp_path / "ok")) == 12
    make_folder(tmp_path / "leak", extra={"journal": "Journal of Imaginary Studies"})
    with pytest.raises(ValueError, match="must not see"):
        ev.read_inputs(tmp_path / "leak")
    make_folder(tmp_path / "leak2", extra={"ajg2024": "4*"})
    with pytest.raises(ValueError, match="must not see"):
        ev.read_inputs(tmp_path / "leak2")


def test_the_real_folder_has_no_journal_no_grade_and_no_doi_in_what_the_judge_reads():
    directory = ev.settings()["directory"]
    if not (directory / "evaluator_input" / "abstracts.jsonl").is_file():
        pytest.skip("inputs/evaluation_2026 is local and not in this checkout")
    inputs = ev.read_inputs(directory)
    assert inputs and all(set(r) <= {"title", "abstract", "abstract_sha256"} for r in inputs.values())
    assert not any(k.startswith("10.") for k in inputs)


# --- the sample ------------------------------------------------------------------------------------------------------------------------------------

def test_the_sample_is_drawn_by_a_hash_of_the_seed_so_the_order_of_the_file_does_not_matter(tmp_path):
    inputs, display = make_folder(tmp_path / "f")
    read, shown = ev.read_inputs(tmp_path / "f"), ev.read_display(tmp_path / "f")
    pool = ev.eligible(read, shown)
    assert "e26_000000000005" not in pool                                       # an abstract of six words is no abstract
    assert all(shown[p]["ajg2024"] for p in pool)                               # a grade is required by default
    assert len(ev.eligible(read, shown, require_grade=False)) > len(pool)
    first, again = ev.draw(pool, 4, 7), ev.draw(list(reversed(pool)), 4, 7)
    assert first == again and len(set(first)) == 4
    assert ev.draw(pool, 4, 8) != first                                         # another seed, another sample


def test_an_extended_sample_keeps_every_paper_of_the_earlier_one_even_when_the_pool_changes(where):
    first = ev.sample(where, n=4, seed=3, min_words=20, require_fulltext=False)["ids"]
    again = ev.sample(where, n=7, seed=3, min_words=20, require_fulltext=False, keep=first)
    assert again["ids"][:4] == first and len(again["ids"]) == 7 and len(set(again["ids"])) == 7
    assert again["pool"] >= 7
    # another seed and no grade filter: the pool is not the one of the first draw, and the kept papers stay all the same
    other = ev.sample(where, n=7, seed=99, min_words=20, require_grade=False, require_fulltext=False, keep=first)
    assert set(first) <= set(other["ids"]) and len(other["ids"]) == 7
    # the same arguments give the same sample
    assert ev.sample(where, n=7, seed=3, min_words=20, require_fulltext=False, keep=first)["ids"] == again["ids"]


def test_a_job_sample_ignores_an_empty_run_to_extend(where):
    # the page sends extend_run "" for "no, draw a new sample": the Batch of the cards failed with a TypeError (21 Oct 2026)
    options = {"n": 4, "seed": 3, "min_words": 20, "require_fulltext": False, "stratify": True}
    assert ev._sample_for("unused://", where, {**options, "extend_run": ""})["ids"] == ev.sample(where, **options)["ids"]


def test_an_extended_sample_cannot_be_smaller_than_what_is_kept_nor_keep_a_paper_that_is_gone(where):
    first = ev.sample(where, n=4, seed=3, min_words=20, require_fulltext=False)["ids"]
    with pytest.raises(ValueError, match="n must be between 4"):
        ev.sample(where, n=3, seed=3, min_words=20, require_fulltext=False, keep=first)
    with pytest.raises(ValueError, match="no longer among the abstracts"):
        ev.sample(where, n=5, seed=3, min_words=20, require_fulltext=False, keep=first + ["e26_999999999999"])


def test_the_allocation_is_as_equal_as_the_strata_allow_and_respects_what_is_kept():
    assert ev.allocate(10, {"4*": 10, "2": 10, "1": 2}) == {"4*": 4, "2": 4, "1": 2}                  # a small grade gives all it has, the others share the rest
    assert ev.allocate(7, {"1": 5, "4*": 5, "3": 5}) == {"4*": 3, "3": 2, "1": 2}                      # a tie goes to the higher grade
    assert ev.allocate(9, {"4*": 2, "3": 2}) == {"4*": 2, "3": 2}                                       # never more than the capacity
    assert ev.allocate(6, {"4*": 9, "3": 9, ev.UNGRADED: 9}, {"4*": 5}) == {"4*": 5, "3": 1, ev.UNGRADED: 0}   # the kept papers stay, the rest go to the smallest
    assert ev.strata_order(["2", ev.UNGRADED, "4*", "1"]) == ["4*", "2", "1", ev.UNGRADED]


def test_a_stratified_sample_takes_the_same_number_of_papers_from_each_grade(where):
    # the fixture's pool: 3 papers graded 4*, 2 graded 3 (one abstract is too short), 3 graded 2, 3 without a grade
    s = ev.sample(where, n=6, seed=3, min_words=20, require_fulltext=False, stratify=True)
    assert {g["grade"]: g["papers"] for g in s["strata"]} == {"4*": 2, "3": 2, "2": 2} and len(set(s["ids"])) == 6
    assert [g["grade"] for g in s["strata"]] == ["4*", "3", "2"] and {g["grade"]: g["pool"] for g in s["strata"]} == {"4*": 3, "3": 2, "2": 3}
    assert ev.sample(where, n=6, seed=3, min_words=20, require_fulltext=False, stratify=True)["ids"] == s["ids"]        # the same arguments, the same papers
    bigger = ev.sample(where, n=7, seed=3, min_words=20, require_fulltext=False, stratify=True)
    assert {g["grade"]: g["papers"] for g in bigger["strata"]} == {"4*": 3, "3": 2, "2": 2}                      # the grade 3 has no third paper
    loose = ev.sample(where, n=8, seed=3, min_words=20, require_grade=False, require_fulltext=False, stratify=True)
    assert {g["grade"]: g["papers"] for g in loose["strata"]} == {"4*": 2, "3": 2, "2": 2, ev.UNGRADED: 2}       # without the grade filter, the ungraded are a stratum
    four = [p for p in s["ids"] if s["display"][p]["ajg2024"] == "4*"]
    kept = ev.sample(where, n=6, seed=3, min_words=20, require_fulltext=False, stratify=True, keep=four + [p for p in s["ids"] if p not in four][:1])
    assert kept["ids"][:3] == four + [p for p in s["ids"] if p not in four][:1] and sum(g["papers"] for g in kept["strata"]) == 6
    assert {g["grade"]: g["kept"] for g in kept["strata"] if g["kept"]} == {"4*": 2, s["display"][kept["ids"][2]]["ajg2024"]: 1}
    plain = ev.sample(where, n=6, seed=3, min_words=20, require_fulltext=False)
    assert plain["ids"] == ev.draw(ev.pool_of(where, min_words=20, require_fulltext=False)[0], 6, 3)               # without stratify, the draw of before


def test_a_paper_already_in_the_corpus_is_left_out_whether_the_doi_or_the_title_matches(tmp_path):
    inputs, display = make_folder(tmp_path / "f")
    read, shown = ev.read_inputs(tmp_path / "f"), ev.read_display(tmp_path / "f")
    shown["e26_000000000001"]["doi"] = "10.1/3"                                  # the DOI of corpus entry P3
    read["e26_000000000002"]["title"] = "Paper 4 on venture capital and startup scaling"          # the title of corpus entry P4
    entries = [{"id": "P3", "doi": "10.1/3", "title": "x"}, {"id": "P4", "doi": "", "title": "Paper 4 on venture capital and startup scaling"}]
    assert ev.already_in_corpus(read, shown, entries) == {"e26_000000000001", "e26_000000000002"}


# --- the cutoff rule ---------------------------------------------------------------------------------------------------------------------------

def test_the_shipped_configuration_declares_a_cutoff_that_precedes_the_papers_and_no_tool_is_offered():
    cfg = scoring_judges.load_config(ev.ROOT / "config" / "evaluation_pre2026.json")
    assert {j["model"]["id"] for j in cfg["judges"]} == {"gpt-5.2-2025-12-11"}
    assert ev.check_cutoffs(cfg, dt.date(2026, 1, 1)) == dt.date(2025, 8, 31)


def test_a_run_is_refused_when_the_model_could_have_seen_the_papers_or_declares_no_cutoff():
    cfg = scoring_judges.load_config(ev.ROOT / "config" / "evaluation_pre2026.json")
    with pytest.raises(ValueError, match="does not precede"):
        ev.check_cutoffs(cfg, dt.date(2025, 8, 31))
    undeclared = json.loads(json.dumps(cfg))
    del undeclared["judges"][1]["model"]["knowledge_cutoff"]
    with pytest.raises(ValueError, match="not declared"):
        ev.check_cutoffs(undeclared, dt.date(2026, 1, 1))
    forum_cfg = scoring_judges.load_config(ev.ROOT / "config" / "forum_scoring.json")             # gpt-6-luna: no declared cutoff, and the provider says May 2026
    with pytest.raises(ValueError, match="not declared"):
        ev.check_cutoffs(forum_cfg, dt.date(2026, 1, 1))


# --- a whole run, offline -----------------------------------------------------------------------------------------------------------------------

@needs_db
def test_a_run_gives_every_paper_the_two_novelty_votes_and_the_judge_never_sees_the_journal(db, where):
    with db.connect() as c:
        made = ev.create_run(c, n=6, seed=3, min_words=20, require_fulltext=False, where=where, synthetic=True, max_tokens=500_000)
        run_id = __import__("uuid").UUID(made["run_id"])
        assert made["papers"] == 6 and made["model"] == "gpt-5.2-2025-12-11" and made["knowledge_cutoff"] == "2025-08-31"
        judge = Judge(vote=6, hybrid_vote=7)
        out = ev.execute(c, run_id, where=where, synthetic_provider=judge, embedder=EMBEDDER)
        assert out["state"] == "completed" and out["scored"] == 6 and out["failed"] == 0
        rows = c.execute("SELECT status,novelty,novelty_hybrid FROM evaluation_score WHERE run_id=%s", (run_id,)).fetchall()
        assert len(rows) == 6 and {r[0] for r in rows} == {"ok"} and {float(r[1]) for r in rows} == {6.0} and {float(r[2]) for r in rows} == {7.0}
        assert c.execute("SELECT status FROM evaluation_run WHERE run_id=%s", (run_id,)).fetchone()[0] == "completed"
    assert judge.requests
    for request in judge.requests:
        assert request.web_search is False and request.tools == []                          # no internet, no tool
        shown = request.prompt.split("<<<PROPOSAL\n", 1)[1].split("\n>>>END_PROPOSAL", 1)[0]
        assert shown.startswith("TITLE: A study ") and "ABSTRACT:\n" in shown             # the very text a proposal is judged on
        assert not any(name in shown for name, _ in JOURNALS) and "10.9999" not in shown and "ABS 2024" not in shown and "ajg" not in shown.lower() and "2026" not in shown
        assert request.model == "gpt-5.2-2025-12-11"


@needs_db
def test_the_journal_and_the_grade_are_kept_for_the_page_and_the_abstract_is_not_stored(db, where):
    with db.connect() as c:
        made = ev.create_run(c, n=6, seed=3, min_words=20, require_fulltext=False, where=where, synthetic=True, max_tokens=500_000)
        run_id = __import__("uuid").UUID(made["run_id"])
        ev.execute(c, run_id, where=where, synthetic_provider=Judge(vote=5), embedder=EMBEDDER)
        detail = ev.run_detail(c, run_id)
        assert len(detail["rows"]) == 6 and all(r["journal"] and r["ajg2024"] for r in detail["rows"])
        document = c.execute("SELECT document FROM evaluation_score WHERE run_id=%s LIMIT 1", (run_id,)).fetchone()[0]
        assert all("proposal_text" not in document["criteria"][c] for c in ("novelty", "methodology", "impact")) and "examines" not in json.dumps(document["criteria"]["novelty"]["similarity"])
        assert document["models"]["judge_1"]["knowledge_cutoff"] == "2025-08-31" and document["criteria"]["novelty"]["hybrid"]["vote"] == 5


@needs_db
def test_the_statistics_read_the_votes_by_abs_grade_and_compare_the_two_judges(db, where):
    with db.connect() as c:
        made = ev.create_run(c, n=10, seed=1, min_words=20, require_fulltext=False, require_grade=False, where=where, synthetic=True, max_tokens=500_000)
        run_id = __import__("uuid").UUID(made["run_id"])
        ev.execute(c, run_id, where=where, synthetic_provider=Judge(vote=6, hybrid_vote=8), embedder=EMBEDDER)
        detail = ev.run_detail(c, run_id)
        stats = detail["stats"]
        assert stats["first"]["n"] == 10 and stats["first"]["mean"] == 6.0 and stats["hybrid"]["mean"] == 8.0
        assert stats["judges"]["n"] == 10 and stats["judges"]["mean_difference"] == 2.0 and stats["judges"]["within_one_share"] == 0.0
        grades = {g["grade"]: g for g in stats["by_grade"]}
        assert "not graded" in grades and grades["not graded"]["first"]["n"] >= 1 and sum(g["first"]["n"] for g in grades.values()) == 10
        assert stats["spearman_with_grade"]["first"] is None                       # every vote is 6: no rank to correlate
        listed = ev.list_runs(c)["runs"][0]
        assert listed["scored"] == 10 and listed["mean_first"] == 6.0 and listed["mean_hybrid"] == 8.0 and listed["status"] == "completed"
        one = ev.paper_detail(c, run_id, detail["rows"][0]["paper_id"], where)
        assert one["shown_to_the_judges"]["title"].startswith("A study") and one["score"]["criteria"]["novelty"]["judges"]["judge_1"]["vote"] == 6


def test_a_cell_that_starts_like_a_formula_is_defused_and_an_unknown_format_is_refused():
    assert ev._cell("=HYPERLINK(\"x\")") == "'=HYPERLINK(\"x\")" and ev._cell("@cmd") == "'@cmd" and ev._cell("+1") == "'+1"
    assert ev._cell("A study") == "A study" and ev._cell(None) == "" and ev._cell(6.5) == "6.5" and ev._cell(True) == "yes"
    assert ev._cell({"impact": "no card"}) == "impact: no card"
    with pytest.raises(ValueError, match="Unknown export format"):
        ev.export_run(None, uuid.uuid4(), "xlsx")


@needs_db
def test_the_export_gives_the_page_s_rows_as_csv_and_the_rows_with_the_statistics_as_json(db, where):
    with db.connect() as c:
        made = ev.create_run(c, n=6, seed=3, min_words=20, require_fulltext=False, where=where, synthetic=True, max_tokens=500_000, label="Pilot, 2026")
        run_id = uuid.UUID(made["run_id"])
        ev.execute(c, run_id, where=where, synthetic_provider=Judge(vote=6, hybrid_vote=7), embedder=EMBEDDER)
        detail = ev.run_detail(c, run_id)
        name, kind, data = ev.export_run(c, run_id, "csv")
        assert name == f"evaluation_Pilot-2026_{made['run_id'][:8]}.csv" and kind.startswith("text/csv") and data.startswith(b"\xef\xbb\xbf")
        rows = list(csv.DictReader(io.StringIO(data.decode("utf-8-sig"))))
        assert [r["paper_id"] for r in rows] == [r["paper_id"] for r in detail["rows"]] and len(rows) == 6
        assert {r["novelty_first"] for r in rows} == {"6.0"} and {r["novelty_hybrid"] for r in rows} == {"7.0"} and {r["test_data"] for r in rows} == {"yes"}
        assert all(r["journal"] and r["abs_grade_2024"] and r["run_id"] == made["run_id"] and r["model"] == "gpt-5.2-2025-12-11" for r in rows)
        jname, jkind, jdata = ev.export_run(c, run_id, "json")
        body = json.loads(jdata)
        assert jname.endswith(".json") and jkind.startswith("application/json") and body["run"]["run_id"] == made["run_id"] and len(body["rows"]) == 6
        assert body["stats"]["first"]["mean"] == 6.0 and body["counts"]["papers"] == 6 and "by_grade" in body["stats"]
        with pytest.raises(ValueError, match="Unknown evaluation run"):
            ev.export_run(c, uuid.uuid4(), "csv")


@needs_db
def test_a_stopped_run_keeps_what_it_scored_and_a_resumed_one_does_only_the_rest(db, where):
    with db.connect() as c:
        made = ev.create_run(c, n=6, seed=3, min_words=20, require_fulltext=False, where=where, synthetic=True, max_tokens=500_000)
        run_id = __import__("uuid").UUID(made["run_id"])
        seen = []

        def gate():
            seen.append(1)
            if len(seen) > 12:
                raise Cancelled("stopped by a person")
        out = ev.execute(c, run_id, where=where, synthetic_provider=Judge(vote=6), embedder=EMBEDDER, gate=gate)
        assert out["state"] == "stopped" and 0 < out["scored"] < 6
        assert ev.list_runs(c)["runs"][0]["status"] == "stopped"
        before = c.execute("SELECT count(*) FROM evaluation_score WHERE run_id=%s", (run_id,)).fetchone()[0]
        again = ev.execute(c, run_id, where=where, synthetic_provider=Judge(vote=6), embedder=EMBEDDER)
        assert again["state"] == "completed" and before + again["scored"] == 6
        assert c.execute("SELECT count(DISTINCT paper_id) FROM evaluation_score WHERE run_id=%s AND status='ok'", (run_id,)).fetchone()[0] == 6


@needs_db
def test_a_run_that_extends_another_holds_all_its_papers_and_asks_nothing_again_for_them(db, where):
    with db.connect() as c:
        made = ev.create_run(c, n=3, seed=3, min_words=20, require_fulltext=False, where=where, synthetic=True, max_tokens=500_000)
        first = uuid.UUID(made["run_id"])
        judge = Judge(vote=6)
        assert ev.execute(c, first, where=where, synthetic_provider=judge, embedder=EMBEDDER)["state"] == "completed"
        asked_first = len(judge.requests)
        bigger = ev.create_run(c, n=6, seed=3, min_words=20, require_fulltext=False, where=where, synthetic=True, max_tokens=500_000, extend_run=str(first))
        params = c.execute("SELECT parameters FROM evaluation_run WHERE run_id=%s", (bigger["run_id"],)).fetchone()[0]
        old = c.execute("SELECT parameters FROM evaluation_run WHERE run_id=%s", (first,)).fetchone()[0]["paper_ids"]
        assert params["paper_ids"][:3] == old and len(params["paper_ids"]) == 6 and params["extends_run"] == str(first) and params["kept"] == 3
        shown = ev.pipeline(c, where=where, n=6, seed=3, min_words=20, require_fulltext=False, extend_run=str(first))
        assert shown["sample"]["n"] == 6 and shown["sample"]["kept"] == 3
        assert ev.execute(c, uuid.UUID(bigger["run_id"]), where=where, synthetic_provider=judge, embedder=EMBEDDER)["state"] == "completed"
        ids = lambda reqs: {re.search(r"INPUT_ID: (\S+)", r.prompt).group(1) for r in reqs}
        assert len(ids(judge.requests[asked_first:])) == 3 and not ids(judge.requests[:asked_first]) & ids(judge.requests[asked_first:])     # only the three new papers were asked
        with pytest.raises(ValueError, match="unknown"):
            ev.create_run(c, n=6, seed=3, min_words=20, require_fulltext=False, where=where, synthetic=True, extend_run=str(uuid.uuid4()))


@needs_db
def test_a_run_nobody_is_running_reads_as_interrupted(db, where):
    with db.connect() as c:
        made = ev.create_run(c, n=3, seed=3, min_words=20, require_fulltext=False, where=where, synthetic=True, max_tokens=500_000)
        assert ev.list_runs(c, active=None)["runs"][0]["status"] == "interrupted"
        assert ev.list_runs(c, active=made["run_id"])["runs"][0]["status"] == "running"


@needs_db
def test_a_sample_larger_than_the_pool_and_a_corpus_duplicate_are_refused_or_left_out(db, where):
    with db.connect() as c:
        with pytest.raises(ValueError, match="only"):
            ev.create_run(c, n=12, seed=3, min_words=20, require_fulltext=False, where=where, synthetic=True)
        shown = ev.read_display(where["directory"])
        rows = list(csv.DictReader(open(where["directory"] / "display_metadata" / "papers.csv", encoding="utf-8")))
        rows[0]["doi"] = "10.1/3"                                                  # a corpus DOI
        with open(where["directory"] / "display_metadata" / "papers.csv", "w", encoding="utf-8", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0]))
            w.writeheader()
            w.writerows(rows)
        made = ev.create_run(c, n=8, seed=3, min_words=20, require_fulltext=False, require_grade=False, where=where, synthetic=True)
        ids = c.execute("SELECT parameters FROM evaluation_run WHERE run_id=%s", (made["run_id"],)).fetchone()[0]
        assert shown and rows[0]["paper_id"] not in ids["paper_ids"] and ids["left_out_in_corpus"] == 1


@needs_db
def test_the_job_runs_in_a_thread_one_at_a_time_and_can_be_stopped(db, where, monkeypatch):
    url = db.url
    job = ev.EvaluationJob()
    monkeypatch.setattr(ev, "settings", lambda: where)
    monkeypatch.setattr(ev, "execute", lambda con, run_id, **kw: (time.sleep(0.3), kw["gate"](), {"run_id": str(run_id), "state": "completed", "note": "", "scored": 0, "failed": 0, "usage": {}})[2])
    snap = job.start(url, provider_factory=lambda: object(), n=3, seed=3, min_words=20, require_fulltext=False, max_tokens=500_000)
    assert snap["state"] == "running"
    with pytest.raises(ValueError, match="already going"):
        job.start(url, provider_factory=lambda: object(), n=3, seed=3, min_words=20, require_fulltext=False, max_tokens=500_000)
    deadline = time.time() + 5
    while job.snapshot()["state"] != "idle" and time.time() < deadline:
        time.sleep(0.05)
    assert job.snapshot()["state"] == "idle" and job.snapshot()["last"]["state"] == "completed"
    assert threading.active_count() >= 1


def test_the_rank_correlation_is_none_without_variation_and_signed_with_it():
    assert ev._spearman([1, 2, 3, 4], [6, 6, 6, 6]) is None
    assert ev._spearman([1, 2, 3, 4], [2, 4, 6, 8]) == 1.0
    assert ev._spearman([1, 2, 3, 4], [8, 6, 4, 2]) == -1.0
    assert ev._spearman([1, 2], [1, 2]) is None


# --- the steps that make the cards, and a run on the three criteria ------------------------------------------------------------------------------

SECTIONS = ("Introduction", "Methods", "Data and sample", "Results", "Discussion", "Conclusions")


def paper_pdf(i, topic):
    """A small full text with the sections the card selection looks for. NOT a real paper."""
    lines = [f"A study {i} of {topic}", ""]
    for name in SECTIONS:
        sentences = [f"Sentence {j} of the {name.lower()} section explains how {topic} works in the firms we observed, case {i}." for j in range(8)]
        lines += [name] + textwrap.wrap(" ".join(sentences), 92) + [""]
    lines += ["References", "Author, A. (2020). An earlier paper. Some Journal."]
    return mock_world.make_pdf([lines[k:k + 52] for k in range(0, len(lines), 52)])


class CardMaker:
    """Fills a card with passages copied from the text it is shown. NOT a model. `journal` = a journal name it writes into the first field, to test the guard."""
    name, synthetic, cache_identity = "synthetic", True, "scripted-card-maker"

    def __init__(self, journal=None):
        self.requests, self.journal = [], journal

    def run(self, request):
        self.requests.append(request)
        kind = "method" if "_method_" in request.task else "impact"
        source = request.prompt.split("<<<TESTO\n", 1)[1].split("\n>>>FINE_TESTO", 1)[0]
        sentences = [x.strip() for x in re.split(r"(?<=[.])\s+", source) if 40 <= len(x.strip()) <= 180]
        fields = paper_cards.PROFILES[kind].fields
        card = {f: {"text": f"The {f.replace('_', ' ')} the paper states.", "status": "extracted", "passages": [sentences[n % len(sentences)][:120]]} for n, f in enumerate(fields)}
        if self.journal:
            card[fields[0]]["text"] = f"The question, as framed in {self.journal}."
        card["info_gaps"] = []
        text = json.dumps(card)
        return TurnResult(text=text, model=request.model, resolved_model="scripted", llm_calls=1, input_tokens=len(request.prompt) // 4, output_tokens=len(text) // 4, latency_ms=1)


def put_archive(where, display, pids, pdfs, version="submittedVersion"):
    """The layout of fulltext_archive/: the PDFs the project already had, named after the DOI, and their index."""
    folder = where["directory"] / "fulltext_archive"
    folder.mkdir(exist_ok=True)
    rows = []
    for pid in pids:
        name = ev.doi_slug(display[pid]["doi"]) + ".pdf"
        (folder / name).write_bytes(pdfs[pid])
        rows.append({"file": name, "doi": display[pid]["doi"], "openalex_id": "W1", "title": "x", "journal": display[pid]["journal"], "publication_date": "2026-03-01",
                     "version": version, "oa_status": "green", "license": "cc-by", "download_url": "https://repository.example.org/x.pdf", "pages": "20", "sha256": "", "source": "git 87c4aaf:x.pdf"})
    with open(folder / "archive_index.csv", "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)


def test_a_card_field_that_names_the_journal_is_removed_and_the_card_stays_if_enough_is_left():
    profile = paper_cards.PROFILES["method"]
    card = {f: {"text": f"{f} stated", "status": "extracted", "passages": [f"a passage about {f} that is long enough"]} for f in profile.fields}
    card["question"]["text"] = "As the Journal of Imaginary Studies puts it."
    card["design"]["passages"] = ["published in journal of imaginary studies, vol 3"]
    out, off = ev.guard_card(json.loads(json.dumps({**card, "info_gaps": []})), "Journal of Imaginary Studies", profile)
    assert set(off) == {"question", "design"} and out["question"]["status"] == "missing" and out["measures"]["status"] == "extracted"
    assert any("publication venue" in g for g in out["info_gaps"]) and not any("imaginary" in g.lower() for g in out["info_gaps"])
    assert ev.guard_card(json.loads(json.dumps({**card, "info_gaps": []})), "Short", profile)[1] == []     # a name too short to be told from ordinary words is not looked for
    plain = {f: {"text": f"How the innovation is evaluated ({f})", "status": "extracted", "passages": [f"hyped innovations polarise moral evaluations of {f}"]} for f in profile.fields}
    out, off = ev.guard_card(json.loads(json.dumps({**plain, "info_gaps": ["the innovation studied is not named"]})), "Innovation", profile)
    assert off == [] and out["info_gaps"] == ["the innovation studied is not named"]                  # a one-word journal name used as a plain word stays (20 Oct 2026)
    plain["design"]["text"] = "As shown in an earlier paper published in Innovation, the design is a survey."
    plain["measures"]["passages"] = ["see the special issue of Organization and the Organization journal"]
    assert ev.guard_card(json.loads(json.dumps({**plain, "info_gaps": []})), "Innovation", profile)[1] == ["design"]       # named as the venue: removed
    assert ev.guard_card(json.loads(json.dumps({**plain, "info_gaps": []})), "Organization", profile)[1] == ["measures"]
    plain["question"]["passages"] = ["This special issue of the Journal of Innovation Economics & Management explores the topic."]
    assert ev.guard_card(json.loads(json.dumps({**plain, "info_gaps": []})), "Journal of Innovation Economics &amp; Management", profile)[1] == ["question"]


@needs_db
def test_the_full_texts_are_those_the_project_already_has_and_nothing_is_downloaded(db, where, monkeypatch):
    from research0 import fulltext
    monkeypatch.setattr(fulltext, "acquire", lambda *a, **k: pytest.fail("the evaluation must not look for PDFs on the web"))
    with db.connect() as c:
        s = ev.sample(where, n=6, seed=3, min_words=20, require_fulltext=False)
        ids, display = s["ids"], s["display"]
        ev.sync_papers(c, ids, s["inputs"], display)
        c.commit()
        pdfs = {pid: paper_pdf(n, "platform governance") for n, pid in enumerate(ids)}
        put_archive(where, display, ids[2:4], pdfs)                                                  # in the project's archive, by DOI
        manual = where["directory"] / "fulltext_manual"
        manual.mkdir()
        (manual / f"{ids[0]}.pdf").write_bytes(pdfs[ids[0]])                                         # put by hand, named after the paper id
        (manual / f"{ev.doi_slug(display[ids[1]]['doi'])}.pdf").write_bytes(pdfs[ids[1]])            # named after the DOI
        (manual / f"{ids[2]}.pdf").write_bytes(paper_pdf(99, "a later version"))                      # a file put by hand wins over the archive
        out = ev.import_fulltexts(c, ids, where)
        assert out["project_archive"] == 1 and out["manual_upload"] == 3 and out["not_in_the_project"] == 2 and out["problems"] == []
        again = ev.import_fulltexts(c, ids, where)
        assert again["unchanged"] == 4 and again["project_archive"] == again["manual_upload"] == 0      # the same files: nothing new
        assert all(ev.parsed_text(where, p) for p in ids[:4]) and not ev.parsed_text(where, ids[4])
        assert "a later version" in ev.parsed_text(where, ids[2])
        pipe = ev.pipeline(c, where=where, n=6, seed=3, min_words=20, require_fulltext=False)
        assert pipe["counts"]["with_fulltext"] == 4 and pipe["counts"]["parsed"] == 4 and pipe["counts"]["cards"]["method"]["to_make"] == 4
        rows = {r["paper_id"]: r for r in pipe["rows"]}
        assert rows[ids[3]]["fulltext"]["origin"] == "project_archive" and rows[ids[3]]["fulltext"]["version"] == "submitted"
        assert rows[ids[2]]["fulltext"]["origin"] == "manual_upload"
        only = ev.sample(where, n=4, seed=3, min_words=20)                                             # by default the sample is drawn among the papers that have a full text
        assert sorted(only["ids"]) == sorted(ids[:4])
        with pytest.raises(ValueError, match="only 4 papers"):
            ev.sample(where, n=5, seed=3, min_words=20)


@needs_db
def test_the_cards_are_made_as_the_corpus_cards_and_a_run_on_three_criteria_judges_them_like_a_proposals(db, where):
    with db.connect() as c:
        s = ev.sample(where, n=6, seed=3, min_words=20, require_fulltext=False)
        ids, display = s["ids"], s["display"]
        ev.sync_papers(c, ids, s["inputs"], display)
        c.commit()
        pdfs = {pid: paper_pdf(n, "platform governance") for n, pid in enumerate(ids)}
        put_archive(where, display, ids[:3], pdfs)
        ev.import_fulltexts(c, ids, where)
        journal = display[ids[0]]["journal"]
        maker = CardMaker(journal=journal)
        made = ev.make_cards(c, ids, where, synthetic_provider=maker)
        assert made["state"] == "completed" and made["valid"] == 6 and made["rejected"] == 0 and made["model"] == "gpt-6-luna"
        for request in maker.requests:                                                                 # the request of the corpus cards, field for field
            kind = "method" if "_method_" in request.task else "impact"
            assert request.model == "gpt-6-luna" and request.reasoning_effort is None and request.max_output_tokens == 8000 and request.web_search is False
            assert request.system == paper_cards.PROFILES[kind].card_system and request.prompt.startswith("INPUT_ID: ") and "<<<TESTO\n" in request.prompt
            assert not any(display[p]["doi"] in request.prompt for p in ids)
        cards = ev.latest_cards(c, ids)
        if display[ids[1]]["journal"] != journal:
            assert cards[(ids[1], "method")]["card"]["question"]["status"] == "extracted"
        assert cards[(ids[0], "method")]["card"]["question"]["status"] == "missing"                    # it named the paper's own journal
        assert ev.make_cards(c, ids, where, synthetic_provider=CardMaker())["asked"] == 0                 # nothing left to make
        run = ev.create_run(c, n=6, seed=3, min_words=20, require_fulltext=False, where=where, synthetic=True, max_tokens=500_000, criteria="full")
        assert run["criteria"] == "full"
        judge = Judge(vote=6)
        out = ev.execute(c, uuid.UUID(run["run_id"]), where=where, synthetic_provider=judge, embedder=EMBEDDER)
        assert out["state"] == "completed"
        rows = {p: (st, m, i) for p, st, m, i in c.execute("SELECT paper_id,status,methodology,impact FROM evaluation_score WHERE run_id=%s", (uuid.UUID(run["run_id"]),)).fetchall()}
        assert all(rows[p][0] == "ok" and float(rows[p][1]) == 6 and float(rows[p][2]) == 6 for p in ids[:3])     # the three votes of a proposal
        assert all(rows[p][0] == "partial" and rows[p][1] is None for p in ids[3:])                            # no full text: novelty only
        method_prompts = [r.prompt for r in judge.requests if r.task in ("forum_score_methodology", "forum_score_impact")]
        assert len(method_prompts) >= 6 and all("CARD" in p for p in method_prompts)
        owner = {scoring_io.opaque_id(ev.SALT, p): p for p in ids}
        for p in method_prompts:                                                                       # the judged paper's own journal never reaches its judge
            pid = owner[re.search(r"INPUT_ID: (\S+)", p).group(1)]
            assert display[pid]["journal"] not in p.split(">>>END_PROPOSAL", 1)[0] and "10.9999" not in p
        detail = ev.run_detail(c, uuid.UUID(run["run_id"]))
        assert detail["partial"] == 3 and detail["stats"]["methodology"]["n"] == 3 and detail["stats"]["mean"]["mean"] == 6.0
        assert {r["paper_id"]: r["mean"] for r in detail["rows"]}[ids[0]] == 6.0
        one = ev.paper_detail(c, uuid.UUID(run["run_id"]), ids[1], where)
        assert one["cards"]["method"]["status"] == "valid" and one["score"]["criteria"]["impact"]["judges"]["judge_1"]["vote"] == 6
        listed = ev.list_runs(c)["runs"][0]
        assert listed["kind"] == "full" and listed["complete"] == 3 and listed["partial"] == 3 and listed["mean_methodology"] == 6.0


class FakeBatchClient:
    """The two calls of the OpenAI client the collection makes. NOT OpenAI: it answers with what it is given."""

    def __init__(self, status, output):
        from types import SimpleNamespace as NS
        self.retrieved = []
        self.batches = NS(retrieve=lambda bid: self.retrieved.append(bid) or NS(status=status, output_file_id="file-out" if output else None, error_file_id=None,
                                                                                    request_counts=NS(total=4, completed=4, failed=0)))
        self.files = NS(content=lambda fid: NS(read=lambda: output.encode("utf-8")))


def batch_answer(line, text, status_code=200, status="completed"):
    return json.dumps({"custom_id": line["custom_id"], "error": None, "response": {"status_code": status_code, "body": {
        "status": status, "model": line["body"]["model"], "output": [{"type": "message", "content": [{"type": "output_text", "text": text}]}]}}})


@needs_db
def test_the_cards_by_batch_send_the_synchronous_request_and_store_each_answer_once(db, where):
    from types import SimpleNamespace as NS
    with db.connect() as c:
        s = ev.sample(where, n=6, seed=3, min_words=20, require_fulltext=False)
        ids, display = s["ids"], s["display"]
        ev.sync_papers(c, ids, s["inputs"], display)
        c.commit()
        put_archive(where, display, ids[:3], {pid: paper_pdf(n, "platform governance") for n, pid in enumerate(ids)})
        ev.import_fulltexts(c, ids, where)
        assert ev.make_cards(c, ids[:1], where, synthetic_provider=CardMaker())["valid"] == 2        # the cards made synchronously stay and are not sent again
        submitted = []

        def submit(directory):
            submitted.append(directory)
            state = {"input_file_id": "file-in", "batch_id": f"batch_{len(submitted)}", "status": "validating"}
            (directory / "batch.json").write_text(json.dumps(state), encoding="utf-8")
            return state
        with pytest.raises(ValueError, match="above the ceiling"):
            ev.submit_card_batch(c, ids, where, max_tokens=1000, submit=submit)
        assert submitted == [] and ev.card_batch_dirs(where) == []                                   # nothing written, nothing sent
        out = ev.submit_card_batch(c, ids, where, max_tokens=2_000_000, submit=submit)
        assert out["cards"] == 4 and out["batch_id"] == "batch_1"
        lines = [json.loads(x) for x in (submitted[0] / "requests.jsonl").read_text(encoding="utf-8").splitlines()]
        maker = CardMaker()
        for line in lines:                                                                           # the body of the synchronous call, field for field
            body, kind = line["body"], line["custom_id"].split("-", 1)[0]
            assert line["url"] == "/v1/responses" and body["model"] == "gpt-6-luna" and body["max_output_tokens"] == 8000 and body["store"] is False
            assert body["instructions"] == paper_cards.PROFILES[kind].card_system and "reasoning" not in body and "temperature" not in body and "tools" not in body
            assert body["text"]["format"]["name"] == paper_cards.PROFILES[kind].card_task and body["input"][0]["content"].startswith("INPUT_ID: ")
            assert not any(p in json.dumps(line) for p in ids) and not any(display[p]["doi"] in json.dumps(line) for p in ids)
        with pytest.raises(ValueError, match="in a Batch not yet collected"):
            ev.submit_card_batch(c, ids, where, max_tokens=2_000_000, submit=submit)
        assert ev.make_cards(c, ids, where, synthetic_provider=CardMaker())["asked"] == 0              # nor by the synchronous path
        pipe = ev.pipeline(c, where=where, n=6, seed=3, min_words=20, require_fulltext=False)
        assert pipe["counts"]["cards"]["method"]["in_batch"] == 2 and pipe["batches"][0]["status"] == "validating" and pipe["estimate"]["cards"] == 0

        waiting = ev.refresh_card_batches(c, where, FakeBatchClient("in_progress", ""))
        assert waiting[0]["status"] == "in_progress" and waiting[0]["collected"] is None and len(ev.cards_in_batch(where)) == 4
        good = [maker.run(NS(task=f"evaluation_{x['custom_id'].split('-', 1)[0]}_card", prompt=x["body"]["input"][0]["content"], model="gpt-6-luna")).text for x in lines]
        output = "\n".join([batch_answer(lines[0], good[0]), batch_answer(lines[1], "{not json"), batch_answer(lines[2], good[2], status_code=500), batch_answer(lines[3], good[3], status="incomplete")])
        done = ev.refresh_card_batches(c, where, FakeBatchClient("completed", output))
        assert done[0]["collected"] == {**done[0]["collected"], "status": "completed", "valid": 1, "rejected": 2, "failed_requests": 1, "missing_requests": 0}
        rows = c.execute("SELECT status,rejection,source_info FROM evaluation_card WHERE source_info ? 'batch' ORDER BY status").fetchall()
        assert [r[0] for r in rows] == ["rejected", "rejected", "valid"] and rows[2][2]["batch"]["batch_id"] == "batch_1"
        assert any("cut off" in r[1] for r in rows[:2]) and any("not valid JSON" in r[1] for r in rows[:2])
        client = FakeBatchClient("completed", output)
        ev.refresh_card_batches(c, where, client)
        assert client.retrieved == [] and c.execute("SELECT count(*) FROM evaluation_card WHERE source_info ? 'batch'").fetchone()[0] == 3   # collected once
        assert ev.store_card_answers(c, submitted[0], where, output)["valid"] == 1 and c.execute("SELECT count(*) FROM evaluation_card").fetchone()[0] == 5   # same ids: no double row
        again = ev.submit_card_batch(c, ids, where, max_tokens=2_000_000, submit=submit)
        assert again["cards"] == 3                                                                   # the two refused and the failed request; not the valid one

        pid, kind = ids[1], "method"
        source, _ = ev.card_source(ev.parsed_text(where, pid), kind)
        for _ in range(ev.MAX_CARD_REFUSALS):
            c.execute("INSERT INTO evaluation_card(card_id,paper_id,kind,profile_version,model,source_sha256,source_info,status,rejection) VALUES (%s,%s,%s,%s,'m',%s,'{}','rejected','x')",
                      (uuid.uuid4(), pid, kind, paper_cards.PROFILES[kind].version, scoring_io.sha256_text(source)))
        (submitted[1] / "collected.json").write_text("{}", encoding="utf-8")
        send, held, stopped = ev.cards_for_batch(c, ids, where)
        assert (pid, kind) in stopped and (pid, kind) not in {(t[0], t[1]) for t in send}


@needs_db
def test_a_full_run_needs_the_card_corpora_and_a_known_set_of_criteria(db, where):
    with db.connect() as c:
        broken = {**where, "methodology": where["directory"] / "nowhere"}
        with pytest.raises(ValueError, match="card corpus"):
            ev.create_run(c, n=3, seed=3, min_words=20, require_fulltext=False, where=broken, synthetic=True, criteria="full")
        with pytest.raises(ValueError, match="criteria"):
            ev.create_run(c, n=3, seed=3, min_words=20, require_fulltext=False, where=where, synthetic=True, criteria="everything")


# --- the download of the missing full texts, the reconciliation and the list to fetch by hand (21 Oct 2026) -----------------------------------------

def fake_acquire(pdfs, outcomes):
    """A stand-in for `fulltext.acquire`: a PDF for the DOIs of `pdfs`, the outcome of `outcomes` for the others. No network."""
    asked = []

    def acquire(doi, email, get=None, openalex_key=None, semantic_scholar_key=None):
        asked.append(doi)
        base = {"source": "oa-pages-v3", "found_by": "openalex", "doi": doi, "oa_status": "green", "license": "cc-by", "version": "accepted", "pdf_url": f"https://repo.example.org/{doi}.pdf",
                "landing_url": f"https://repo.example.org/{doi}", "http_status": None, "detail": None, "data": None}
        if doi in pdfs:
            return {**base, "outcome": "downloaded", "data": pdfs[doi]}
        return {**base, "outcome": outcomes.get(doi, "not_open_access"), "version": None, "pdf_url": None}
    return acquire, asked


@needs_db
def test_the_download_tries_only_the_usable_papers_without_a_full_text_and_records_every_attempt(db, where, monkeypatch):
    from research0 import fulltext
    from research0.migrate import apply_migrations
    with db.connect() as c:
        display = ev.read_display(where["directory"])
        pool = ev.eligible(ev.read_inputs(where["directory"]), display, min_words=20, require_grade=True)
        order = ev.draw(pool, len(pool), 3)
        put_archive(where, display, order[:1], {order[0]: paper_pdf(0, "platform governance")})                 # one paper already has its full text
        todo, inputs, _ = ev.download_todo(c, where, min_words=20, require_grade=True, seed=3)
        assert todo == order[1:]                                                                               # in the order of the draw; papers without a grade are not tried
        doi = lambda p: display[p]["doi"]
        acquire, asked = fake_acquire({doi(order[1]): paper_pdf(1, "open innovation")}, {doi(order[2]): "lookup_failed", doi(order[3]): "blocked_by_site"})
        monkeypatch.setattr(fulltext, "acquire", acquire)
        ev.sync_papers(c, todo, inputs, display)
        c.commit()
        out = ev.download_fulltexts(c, todo[:4], where, "someone@example.org", workers=1)
        assert out["outcomes"] == {"imported": 1, "lookup_failed": 1, "blocked_by_site": 1, "not_open_access": 1} and not out["stopped"]
        assert asked == [doi(p) for p in todo[:4]]
        assert ev.parsed_text(where, order[1]) and ev.version_of(ev.download_index(where)[doi(order[1])]) == "accepted"
        have = ev.available_files(where, {p: doi(p) for p in order})
        assert have[order[1]][1] == "open_access_download" and have[order[0]][1] == "project_archive"
        again, _, _ = ev.download_todo(c, where, min_words=20, require_grade=True, seed=3)
        assert again[0] == order[2] and order[3] not in again and order[4] not in again                       # a lookup that failed is tried again; a final answer is not
        assert ev.import_fulltexts(c, [order[1]], where)["unchanged"] == 1                                    # the import finds the same file: nothing new
        assert order[1] in ev.sample(where, n=2, seed=3, min_words=20)["ids"]                                  # the sample sees the downloaded paper
        apply_migrations(c)                                                                                    # every init runs every migration again: the rows of 033 stay valid
        c.commit()
        with pytest.raises(ValueError, match="UNPAYWALL_EMAIL"):
            ev.download_fulltexts(c, todo, where, "")


@needs_db
def test_the_reconciliation_says_where_each_paper_stops_and_the_excel_lists_those_to_fetch_by_hand(db, where, monkeypatch):
    from openpyxl import load_workbook
    from research0 import fulltext
    with db.connect() as c:
        display = ev.read_display(where["directory"])
        path = where["directory"] / "evaluator_input" / "abstracts.jsonl"
        rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if json.loads(line)["paper_id"] != "e26_000000000001"]   # one paper has no abstract
        rows[0]["title"] = '=HYPERLINK("http://evil.example")'                                               # a title that looks like a formula
        path.write_text("".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8")
        inputs = ev.read_inputs(where["directory"])
        ids = ev.sample(where, n=2, seed=3, min_words=20, require_fulltext=False)["ids"]
        put_archive(where, display, ids, {p: paper_pdf(n, "platform governance") for n, p in enumerate(ids)})
        (where["directory"] / "fulltext_manual").mkdir()
        (where["directory"] / "fulltext_manual" / "a paper I found.pdf").write_bytes(b"%PDF-1.4")             # a name the import cannot recognise
        ev.sync_papers(c, ids, inputs, display)
        c.commit()
        ev.import_fulltexts(c, ids[:1], where)
        rec = ev.reconcile(c, where, min_words=20, require_grade=True, seed=3)
        stops = {r["paper_id"]: r["stops_at"] for r in rec["rows"]}
        assert rec["total"] == 12 and stops["e26_000000000001"] == "abstract" and stops["e26_000000000005"] == "words"
        assert stops["e26_000000000003"] == "grade" and stops[ids[0]] == "cards" and stops[ids[1]] == "parsed"
        funnel = {f["stage"]: f for f in rec["funnel"]}
        assert funnel["abstract"]["papers"] == 11 and funnel["fulltext"]["papers"] == 2 and funnel["judged"]["papers"] == 0
        assert sum(f["left_out"] for f in rec["funnel"]) == 12                                                 # every paper stops somewhere (none is judged)
        assert rec["folders"]["archive"]["by_stage"] == {"usable": 2} and rec["folders"]["manual"]["not_recognised"] == ["a paper I found.pdf"]
        assert rec["todo"]["to_fetch"] == funnel["fulltext"]["left_out"] == rec["todo"]["to_fetch_never_tried"] and rec["todo"]["to_import"] == 1
        total = lambda key, of=rec: sum(g[key] for g in of["by_grade"])                                       # the funnel by ABS grade adds up to the funnel
        assert total("papers") == 12 and total("usable") == funnel["grade"]["papers"] and total("usable_with_fulltext") == funnel["fulltext"]["papers"] == total("with_fulltext")
        assert total("parsed") == funnel["parsed"]["papers"] and total("cards") == funnel["cards"]["papers"] and total("judged") == 0
        assert rec["by_grade"][-1]["grade"] == "not graded" and rec["by_grade"][-1]["usable"] == 0                # without a grade, never usable when the sample asks for one
        shown = [g["grade"] for g in rec["by_grade"][:-1]]
        assert shown == sorted(shown, key=lambda g: -ev.GRADES.index(g))                                       # highest grade first
        loose = ev.reconcile(c, where, min_words=20, require_grade=False, seed=3)
        assert {f["stage"]: f for f in loose["funnel"]}["grade"]["skipped"] and not [r for r in loose["rows"] if r["stops_at"] == "grade"]
        assert loose["by_grade"][-1]["usable"] > 0
        acquire, _ = fake_acquire({}, {})
        monkeypatch.setattr(fulltext, "acquire", acquire)
        todo, _, _ = ev.download_todo(c, where, min_words=20, require_grade=True, seed=3)
        ev.sync_papers(c, todo, inputs, display)
        c.commit()
        ev.download_fulltexts(c, todo[-2:], where, "someone@example.org", workers=1)
        name, content_type, data = ev.export_to_fetch(c, where, min_words=20, require_grade=True, seed=3)
        assert name.endswith(".xlsx") and "spreadsheetml" in content_type
        book = load_workbook(io.BytesIO(data))
        assert book.sheetnames == ["Read me", "To fetch by hand", "All papers"]
        readme = [[cell.value for cell in row] for row in book["Read me"].iter_rows()]
        at = next(i for i, row in enumerate(readme) if row[0] == "By ABS 2024 grade")
        assert readme[at + 1][0] == "ABS 2024 grade" and readme[at + 1 + len(rec["by_grade"])][0] == "not graded"
        sheet = book["To fetch by hand"]
        listed = [sheet.cell(row=r, column=2).value for r in range(2, sheet.max_row + 1)]
        assert sorted(listed) == sorted(r["paper_id"] for r in rec["rows"] if r["stops_at"] == "fulltext") and listed[:2] == todo[-2:]   # those tried in vain come first
        assert sheet.cell(row=2, column=3).value == f"{todo[-2]}.pdf" and sheet.cell(row=2, column=9).value == "Not open access"
        assert sheet.cell(row=2, column=4).hyperlink.target.startswith("https://doi.org/")
        every = book["All papers"]
        titles = [every.cell(row=r, column=4) for r in range(2, every.max_row + 1)]
        assert every.max_row == 13 and any(str(t.value).startswith("=") for t in titles) and all(t.data_type == "s" for t in titles if t.value)   # a title is never a formula


@needs_db
def test_the_download_job_runs_in_the_task_slot_and_refuses_without_an_address(db, where, monkeypatch):
    from research0 import fulltext
    acquire, _ = fake_acquire({}, {})
    monkeypatch.setattr(fulltext, "acquire", acquire)
    monkeypatch.setattr(ev, "settings", lambda: where)
    job = ev.EvaluationJob()
    with pytest.raises(ValueError, match="UNPAYWALL_EMAIL"):
        job.start_download(db.url, email="", min_words=20)
    snap = job.start_download(db.url, email="someone@example.org", limit=2, min_words=20, seed=3)
    assert snap["total"] == 2
    deadline = time.time() + 10
    while job.snapshot()["state"] != "idle" and time.time() < deadline:
        time.sleep(0.05)
    assert job.snapshot()["last"]["outcomes"] == {"not_open_access": 2} and "fetch by hand" in job.snapshot()["message"]

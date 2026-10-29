import json
import uuid
import pytest

from research0 import community_dashboard as dashboard


def test_assignment_stats_keep_unassigned_and_excluded_separate(tmp_path, monkeypatch):
    monkeypatch.setattr(dashboard, "RUN_ROOT", tmp_path)
    run = tmp_path / "example"
    run.mkdir()
    (run / "manifest.json").write_text(json.dumps({"all_unique_records": 7, "selected_records": 5, "request_count": 5}), encoding="utf-8")
    (run / "batch.json").write_text(json.dumps({"status": "completed", "request_counts": {"total": 5, "completed": 3, "failed": 2}}), encoding="utf-8")
    (run / "assignments.csv").write_text("record_id,cto,tim\na,1,0\nb,0,1\nc,1,1\n", encoding="utf-8")
    status = dashboard.assignment_status("example")
    assert (status["assigned"], status["unassigned"], status["excluded"]) == (3, 2, 2)
    assert (status["cto_only"], status["tim_only"], status["both"], status["neither"]) == (1, 1, 1, 0)
    assert status["request_counts"] == {"total": 5, "completed": 3, "failed": 2}


class PaperRows:
    """Answers the one query `stats` makes: (canonical_id, title, authors, year, journal) of the catalog run."""

    def __init__(self, rows):
        self.rows = rows

    def execute(self, query, params=None):
        assert query.startswith("SELECT canonical_id,title,authors,publication_year,source_title FROM canonical_paper")
        return iter(self.rows)


def test_stats_count_each_community_separately_and_pick_the_same_examples_for_a_seed(tmp_path, monkeypatch):
    monkeypatch.setattr(dashboard, "RUN_ROOT", tmp_path)
    run = tmp_path / "example"
    run.mkdir()
    ids = [uuid.uuid4() for _ in range(7)]
    (run / "manifest.json").write_text(json.dumps({"all_unique_records": 9, "selected_records": 8, "request_count": 1}), encoding="utf-8")
    (run / "batch.json").write_text(json.dumps({"status": "completed"}), encoding="utf-8")
    # ids[6] has a label but is not in the catalog run: it is not counted
    (run / "assignments.csv").write_text("record_id,cto,tim\n" + "".join(f"{cid},{c},{t}\n" for cid, (c, t) in zip(
        ids, [(1, 0), (1, 0), (0, 1), (1, 1), (0, 0), (0, 0), (1, 0)])), encoding="utf-8")
    papers = [(cid, f"Paper {n}", "Ana", 2020 + n % 2, "Journal A" if n < 3 else None) for n, cid in enumerate(ids[:6])]
    result = dashboard.stats(PaperRows(papers), "example", seed=3, per_group=1)
    assert result["available"] and result["labelled"] == 6 and result["validated"] is False
    assert {g["key"]: g["papers"] for g in result["groups"]} == {"both": 1, "cto_only": 2, "tim_only": 1, "neither": 2}
    assert result["communities"]["CTO"]["papers"] == 3 and result["communities"]["TIM"]["papers"] == 2    # both counts for each community
    assert [r["year"] for r in result["by_year"]] == [2020, 2021] and sum(r["total"] for r in result["by_year"]) == 6
    assert {r["source"] for r in result["by_source"]} == {"Journal A", "(no journal)"}
    assert all(len(rows) <= 1 for rows in result["examples"].values()) and len(result["examples"]["cto_only"]) == 1
    assert dashboard.stats(PaperRows(papers), "example", seed=3, per_group=1)["examples"] == result["examples"]


def test_stats_without_labels_say_so(tmp_path, monkeypatch):
    monkeypatch.setattr(dashboard, "RUN_ROOT", tmp_path)
    result = dashboard.stats(PaperRows([]), "never-started")
    assert result["available"] is False and result["labelled"] == 0


def test_binary_kappa_accounts_for_chance_agreement():
    assert dashboard.kappa([(0, 0), (0, 0), (1, 1), (1, 1)])["kappa"] == 1
    result = dashboard.kappa([(0, 0), (0, 1), (1, 0), (1, 1)])
    assert result["agreement"] == 0.5
    assert result["kappa"] == 0


class SampleConnection:
    """Records the sample inserts; `eligible` are the retained, unflagged canonical papers."""

    def __init__(self, eligible):
        self.eligible, self.inserted, self.rows = eligible, [], []

    def execute(self, query, params=None):
        if query.startswith("SELECT 1 FROM blind_sample"):
            self.rows = []
        elif query.startswith("SELECT canonical_id FROM canonical_paper") and "random()" not in query:
            self.rows = [(cid,) for cid in self.eligible]
        elif "random()" in query:
            self.rows = [(cid,) for cid in params[1][:params[2]]]
        elif query.startswith("INSERT INTO blind_sample_paper"):
            self.inserted.append(params[1])
        return self

    def fetchone(self):
        return self.rows[0] if self.rows else None

    def __iter__(self):
        return iter(self.rows)

    def transaction(self):
        import contextlib
        return contextlib.nullcontext()


def test_sample_can_be_drawn_before_the_batch_has_returned_any_label(tmp_path, monkeypatch):
    monkeypatch.setattr(dashboard, "RUN_ROOT", tmp_path)
    run = tmp_path / "example"
    run.mkdir()
    ids = [uuid.uuid4() for _ in range(150)]
    (run / "manifest.json").write_text(json.dumps({"all_unique_records": 150, "selected_records": 150, "request_count": 1,
                                                   "requests": {"community-000000": [str(i) for i in ids]}}), encoding="utf-8")
    monkeypatch.setattr(dashboard, "sample_status", lambda con, run_id, validator=None: {"created": True})
    con = SampleConnection(ids)
    dashboard.create_sample(con, "example")
    assert len(con.inserted) == 100 and set(con.inserted) <= set(ids)


def test_sample_ignores_papers_not_in_the_submitted_batch(tmp_path, monkeypatch):
    monkeypatch.setattr(dashboard, "RUN_ROOT", tmp_path)
    run = tmp_path / "example"
    run.mkdir()
    ids = [uuid.uuid4() for _ in range(120)]
    (run / "manifest.json").write_text(json.dumps({"requests": {"community-000000": [str(i) for i in ids[:50]]}}), encoding="utf-8")
    with pytest.raises(ValueError, match="Only 50 eligible"):
        dashboard.create_sample(SampleConnection(ids), "example")


def test_agreement_reports_human_kappa_while_batch_labels_are_missing(tmp_path, monkeypatch):
    ids = [uuid.uuid4() for _ in range(4)]
    run = tmp_path / "example"
    run.mkdir()
    (run / "assignments.csv").write_text(f"record_id,cto,tim\n{ids[0]},1,0\n", encoding="utf-8")
    monkeypatch.setattr(dashboard, "RUN_ROOT", tmp_path)
    ratings = {"A": dict(zip(ids, [(0, 1), (1, 0), (0, 1), (1, 0)])), "B": dict(zip(ids, [(0, 1), (1, 0), (0, 1), (1, 0)]))}
    monkeypatch.setattr(dashboard, "completed_ratings", lambda con, run_id, minimum=2: ({"threshold": 0.8}, ["A", "B"], ratings))

    class Connection:
        def execute(self, query, params):
            return []

    result = dashboard.agreement(Connection(), "example")
    assert result["batch_missing"] == 3
    assert result["CTO"]["annotators"]["kappa"] == 1
    assert "annotator_a_batch" not in result["CTO"] and "final_batch" not in result["TIM"]


def test_agreement_with_one_annotator_compares_only_with_batch(tmp_path, monkeypatch):
    ids = [uuid.uuid4() for _ in range(4)]
    run = tmp_path / "example"
    run.mkdir()
    (run / "assignments.csv").write_text(
        "record_id,cto,tim\n" + "".join(
            f"{cid},{cto},{tim}\n" for cid, cto, tim in zip(ids, [0, 1, 0, 1], [1, 0, 1, 0])), encoding="utf-8")
    monkeypatch.setattr(dashboard, "RUN_ROOT", tmp_path)
    ratings = {"A": dict(zip(ids, [(0, 1), (1, 0), (0, 1), (0, 1)]))}
    monkeypatch.setattr(dashboard, "completed_ratings", lambda con, run_id, minimum=2: ({"threshold": 0.8}, ["A"], ratings))
    result = dashboard.agreement(object(), "example")
    assert result["mode"] == "single_annotator" and result["annotators"] == ["A"]
    assert result["CTO"]["annotator_a_batch"]["agreement"] == 0.75
    assert result["TIM"]["annotator_a_batch"]["agreement"] == 0.75
    assert "annotators" not in result["CTO"] and "disagreements" not in result
    (run / "assignments.csv").write_text(f"record_id,cto,tim\n{ids[0]},0,1\n", encoding="utf-8")
    partial = dashboard.agreement(object(), "example")
    assert partial["batch_missing"] == 3 and partial["CTO"] == {}


def test_agreement_compares_annotators_with_batch_and_adjudicated_labels(tmp_path, monkeypatch):
    ids = [uuid.uuid4() for _ in range(4)]
    run = tmp_path / "example"
    run.mkdir()
    (run / "assignments.csv").write_text(
        "record_id,cto,tim\n" + "".join(
            f"{cid},{cto},{tim}\n" for cid, cto, tim in zip(ids, [0, 1, 0, 1], [1, 0, 1, 0])
        ), encoding="utf-8")
    monkeypatch.setattr(dashboard, "RUN_ROOT", tmp_path)
    ratings = {
        "A": dict(zip(ids, [(0, 1), (1, 0), (0, 1), (1, 0)])),
        "B": dict(zip(ids, [(0, 1), (1, 0), (1, 1), (1, 0)])),
    }
    monkeypatch.setattr(dashboard, "completed_ratings", lambda con, run_id, minimum=2:
                        ({"threshold": 0.8}, ["A", "B"], ratings))

    class Connection:
        def __init__(self, resolved):
            self.resolved = resolved

        def execute(self, query, params):
            assert "community_adjudication" in query
            return [(ids[2], 0, 1)] if self.resolved else []

    pending = dashboard.agreement(Connection(False), "example")
    assert pending["CTO"]["annotators"]["agreement"] == 0.75
    assert pending["CTO"]["annotator_a_batch"]["kappa"] == 1
    assert pending["CTO"]["annotator_b_batch"]["agreement"] == 0.75
    assert pending["pending_adjudication"] == 1
    assert "final_batch" not in pending["CTO"]
    final = dashboard.agreement(Connection(True), "example")
    assert final["pending_adjudication"] == 0
    assert final["CTO"]["final_batch"]["kappa"] == 1
    assert final["TIM"]["final_batch"]["kappa"] == 1


def test_partial_batch_comparison_is_opt_in_and_reports_what_it_used(tmp_path, monkeypatch):
    ids = [uuid.uuid4() for _ in range(4)]
    run = tmp_path / "example"
    run.mkdir()
    (run / "assignments.csv").write_text(
        "record_id,cto,tim\n" + "".join(f"{cid},{cto},{tim}\n" for cid, cto, tim in zip(ids[:3], [0, 1, 0], [1, 0, 1])), encoding="utf-8")
    monkeypatch.setattr(dashboard, "RUN_ROOT", tmp_path)
    ratings = {"A": dict(zip(ids, [(0, 1), (1, 0), (1, 1), (0, 0)]))}
    monkeypatch.setattr(dashboard, "completed_ratings", lambda con, run_id, minimum=2: ({"threshold": 0.8}, ["A"], ratings))
    strict = dashboard.agreement(object(), "example")
    assert strict["batch_compared"] == 0 and strict["CTO"] == {} and not strict["batch_partial"]
    partial = dashboard.agreement(object(), "example", allow_partial=True)
    assert partial["batch_partial"] and partial["batch_compared"] == 3 and partial["batch_missing_ids"] == [str(ids[3])]
    assert partial["CTO"]["annotator_a_batch"]["n"] == 3
    assert partial["CTO"]["annotator_a_batch"]["agreement"] == 2 / 3

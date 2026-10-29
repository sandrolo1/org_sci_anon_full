import json

import pytest

from research0 import community_v2 as v2
from research0.mock_world import scripted_preassignment_output

SCOPE = "config/community_scope_v2.json"


def test_abstract_rules():
    assert v2.abstract_state("[No abstract available]")[0] == "absent" and v2.abstract_state(None)[0] == "absent"
    assert v2.abstract_state("word " * 10)[0] == "short" and v2.abstract_state("word " * 50)[0] == "abstract"


def test_yes_needs_a_quote_that_really_occurs():
    fields = {"title": "Platform work", "abstract": "We study  algorithmic management of gig workers.", "keywords": "gig economy"}
    ok = v2.validate_side(dict(label="yes", confidence=0.9, evidence_field="abstract", evidence_quote="Algorithmic management of GIG workers", rationale="r"), fields, "CTO")
    assert ok["evidence_field"] == "abstract"
    moved = v2.validate_side(dict(label="yes", confidence=0.9, evidence_field="title", evidence_quote="gig economy", rationale="r"), fields, "CTO")
    assert moved["evidence_field"] == "keywords"  # the quote is real, the field was wrong: store where it was found
    for bad in (dict(label="yes", confidence=0.9, evidence_field="abstract", evidence_quote="", rationale="r"),
                dict(label="yes", confidence=0.9, evidence_field="abstract", evidence_quote="invented text", rationale="r"),
                dict(label="maybe", confidence=0.9, rationale="r"), dict(label="no", confidence=1.5, rationale="r"),
                dict(label="no", confidence=0.5, rationale=" "), None):
        with pytest.raises(ValueError):
            v2.validate_side(bad, fields, "CTO")
    no = v2.validate_side(dict(label="no", confidence=0.4, evidence_field="abstract", evidence_quote="whatever", rationale="r"), fields, "TIM")
    assert no["evidence_quote"] is None


def test_prepare_collect_store_end_to_end(world, con, tmp_path):
    manifest = v2.prepare(con, world.run_id, tmp_path / "run", SCOPE)
    eligible = world.eligible()
    no_text = [p for p in eligible if p.abstract == "[No abstract available]"]
    assert manifest["eligible_papers"] == len(eligible) and len(manifest["insufficient"]) >= len(no_text) > 0
    assert sum(len(v) for v in manifest["requests"].values()) == len(eligible) - len(manifest["insufficient"])
    request_lines = (tmp_path / "run" / "requests.jsonl").read_text().splitlines()
    request = json.loads(request_lines[0])
    assert request["body"]["text"]["format"]["strict"] and "Missing or insufficient information" in request["body"]["instructions"]
    sent = {p["record_id"] for line in request_lines for p in json.loads(json.loads(line)["body"]["input"])}
    assert not sent & set(manifest["insufficient"])  # papers without a usable abstract are never sent to the model

    victim = next(iter(manifest["model_input"]))
    dropped = list(manifest["requests"].items())[1][1][0]
    lines = scripted_preassignment_output(world, request_lines, fabricate_for=victim, drop_paper=dropped)
    (tmp_path / "run" / "batch_output.jsonl").write_text("\n".join(lines) + "\n", encoding="utf-8")
    result = v2.collect_file(con, tmp_path / "run", SCOPE)
    assert result["rejected"] >= 2 and result["missing_requests"] == 0   # the fabricated quote and the request with a missing paper
    run_id = result["preassignment_run_id"]
    labels = {(str(c), k): l for c, k, l in con.execute("SELECT canonical_id,community,label FROM community_preassignment WHERE preassignment_run_id=%s", (run_id,))}
    assert (victim, "CTO") not in labels and (dropped, "CTO") not in labels
    for p in no_text:
        assert labels[(str(p.canonical_id), "CTO")] == labels[(str(p.canonical_id), "TIM")] == "uncertain"
    some = next(p for p in eligible if p.truth["CTO"] and not p.truth["TIM"] and str(p.canonical_id) in manifest["model_input"]
                and str(p.canonical_id) not in (victim, dropped) and str(p.canonical_id) not in manifest["requests"]["preassign-000001"])
    assert (labels[(str(some.canonical_id), "CTO")], labels[(str(some.canonical_id), "TIM")]) == ("yes", "no")
    # Collecting again changes nothing (append-only, idempotent)
    again = v2.collect_file(con, tmp_path / "run", SCOPE)
    assert again["preassignment_run_id"] == run_id
    assert con.execute("SELECT count(*) FROM community_preassignment WHERE preassignment_run_id=%s", (run_id,)).fetchone()[0] == len(labels)
    assert v2.summary(con, run_id)["CTO"]["uncertain"] >= len(no_text)


def test_codebook_change_blocks_collection(world, con, tmp_path):
    v2.prepare(con, world.run_id, tmp_path / "run", SCOPE)
    other = tmp_path / "scope.json"
    other.write_text(open(SCOPE, encoding="utf-8").read().replace("aom-scope-v2", "changed"), encoding="utf-8")
    with pytest.raises(ValueError, match="codebook changed"):
        v2.collect_file(con, tmp_path / "run", other)


def test_v1_suggestions_become_labels_and_untrusted_ones_become_uncertain(world, con, tmp_path):
    eligible = world.eligible()
    with_text = next(p for p in eligible if p.abstract != "[No abstract available]")
    without = next(p for p in eligible if p.abstract == "[No abstract available]")
    (tmp_path / "assignments.csv").write_text(
        "record_id,cto,tim,cto_rationale,tim_rationale,custom_id,model,scope_version,prompt_version\n"
        f"{with_text.canonical_id},1,0,because,not,c,m,aom-scope-v1,community-binary-v1\n{without.canonical_id},0,0,none,none,c,m,aom-scope-v1,community-binary-v1\n", encoding="utf-8")
    run_id, n = v2.import_v1_suggestions(con, world.run_id, tmp_path, "config/community_scope_v1.json")
    assert n == 4
    rows = {(str(c), k): (l, b) for c, k, l, b in con.execute("SELECT canonical_id,community,label,text_basis FROM community_preassignment WHERE preassignment_run_id=%s", (run_id,))}
    assert rows[(str(with_text.canonical_id), "CTO")] == ("yes", "abstract") and rows[(str(with_text.canonical_id), "TIM")] == ("no", "abstract")
    assert rows[(str(without.canonical_id), "CTO")] == ("uncertain", "insufficient")
    assert con.execute("SELECT source FROM preassignment_run WHERE preassignment_run_id=%s", (run_id,)).fetchone()[0] == "v1_suggestion"

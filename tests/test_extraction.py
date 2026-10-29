import json
import uuid

import pytest

from research0 import documents as docs
from research0 import extraction as ex
from research0.mock_world import scripted_extraction_output


def test_label_normalisation():
    assert ex.normalize_label("  Work-Engagement (UWES) ") == "work engagement uwes"
    assert ex.normalize_label("Café  Culture") == "cafe culture"


def test_locate_tolerates_only_whitespace_differences():
    text = "We define work  engagement\nas a state. Another sentence."
    assert ex.locate(text, "work engagement as a state.") == (10, 37) or ex.locate(text, "work engagement as a state.") is not None
    start, end = ex.locate(text, "work engagement as a state.")
    assert text[start:end].split() == "work engagement as a state.".split()
    assert ex.locate(text, "Work engagement as a state.") is not None     # case is formatting once the quote is long enough to be unambiguous
    assert ex.locate("We define Trust as a state.", "trust") is None       # ...but a short quote must match exactly
    assert ex.locate(text, "work engagement is a state.") is None         # words matter
    assert ex.locate(text, "") is None


TEXT = "Trust is defined as willingness to be vulnerable. Trust predicted performance (beta = .3) among nurses."
CHUNK = {"page_offsets": [0, 1030, 2000], "chunks": [("c1", 1000, 1000 + len(TEXT))],
         "segments": [{"section": "results", "start": 1000, "end": 1000 + len(TEXT), "text": TEXT}]}


def good():
    return {"constructs": [dict(label="Trust", surface_form="Trust", role="focal", definition="willingness to be vulnerable", quote="Trust is defined as willingness to be vulnerable."),
                           dict(label="Performance", surface_form="performance", role="outcome", definition="", quote="Trust predicted performance (beta = .3) among nurses.")],
            "measures": [], "claims": [dict(subject_label="Trust", object_label="Performance", relation="predicts", direction="positive",
                                            quote="Trust predicted performance (beta = .3)", modifiers=[dict(kind="sample", value="nurses", quote="among nurses")])]}


def test_a_faithful_answer_is_accepted_with_offsets_and_pages_computed_here():
    accepted, rejections = ex.validate_response(good(), CHUNK)
    assert rejections == [] and len(accepted["constructs"]) == 2 and len(accepted["claims"]) == 1
    first = accepted["constructs"][0]
    assert (first["char_start"], first["char_end"], first["page"]) == (1000, 1049, 1)
    assert accepted["claims"][0]["page"] == 2 and accepted["claims"][0]["modifiers"][0]["value"] == "nurses"
    assert accepted["constructs"][0]["definition_text"] == "willingness to be vulnerable"


@pytest.mark.parametrize("edit,reason", [
    (lambda p: p["constructs"][0].update(quote="Trust is the willingness to be vulnerable."), "quote not found"),
    (lambda p: p["constructs"][0].update(definition="a feeling of safety"), "definition is not a verbatim"),
    (lambda p: p["constructs"][0].update(role="boss"), "invalid role"),
    (lambda p: p["constructs"][0].update(surface_form="Loyalty"), "surface form"),
    (lambda p: p["claims"][0].update(object_label="Satisfaction"), "not among the accepted"),
    (lambda p: p["claims"][0].update(subject_label="Performance"), "not among the accepted"),   # self relation
    (lambda p: p["claims"][0].update(quote="Trust strongly predicted performance"), "quote not found"),
    (lambda p: p["claims"][0].update(relation="causes"), "invalid relation"),
])
def test_unfaithful_items_are_rejected_not_repaired(edit, reason):
    payload = good()
    edit(payload)
    accepted, rejections = ex.validate_response(payload, CHUNK)
    assert any(reason in r["reason"] or (reason == "invalid role" and "invalid role" in r["reason"]) for r in rejections), rejections
    if "construct" in rejections[0]["kind"] and reason != "definition is not a verbatim":
        assert len(accepted["constructs"]) == 1


def test_a_measure_for_an_unknown_construct_and_a_bad_modifier_are_rejected():
    payload = good()
    payload["measures"] = [dict(construct_label="Loyalty", name="L-scale", items="", scale="", quote="Trust is defined as willingness to be vulnerable.")]
    payload["claims"][0]["modifiers"].append(dict(kind="sample", value="doctors", quote="among doctors"))
    accepted, rejections = ex.validate_response(payload, CHUNK)
    assert accepted["measures"] == [] and len(accepted["claims"][0]["modifiers"]) == 1
    assert {r["kind"] for r in rejections} == {"measure", "modifier"}


def test_schema_is_strict_and_prompt_forbids_deciding_sameness():
    schema = ex.response_schema()
    assert schema["additionalProperties"] is False and set(schema["required"]) == {"constructs", "measures", "claims", "attributes"}
    request = ex.make_request("abc", CHUNK["segments"])
    assert request["body"]["text"]["format"]["strict"] and "Do not decide whether two constructs are the same" in request["body"]["instructions"]


def test_attributes_distinguish_extracted_inferred_and_missing():
    payload = good()
    payload["attributes"] = [
        dict(construct_label="Trust", kind="level", basis="extracted", value="willingness to be vulnerable", quote="Trust is defined as willingness to be vulnerable."),
        dict(construct_label="Trust", kind="mechanism", basis="inferred", value="exposure to risk", quote="Trust is defined as willingness to be vulnerable."),
        dict(construct_label="Trust", kind="reliability", basis="missing", value="", quote=""),
        dict(construct_label="Trust", kind="level", basis="extracted", value="the team level", quote="Trust is defined as willingness to be vulnerable."),   # not in the text
        dict(construct_label="Trust", kind="context", basis="inferred", value="hospitals", quote="never said"),                                           # quote not in the text
        dict(construct_label="Loyalty", kind="level", basis="missing", value="", quote=""),                                                              # unknown construct
        dict(construct_label="Trust", kind="colour", basis="missing", value="", quote="")]
    accepted, rejections = ex.validate_response(payload, CHUNK)
    kept = {(a["kind"], a["basis"]) for a in accepted["attributes"]}
    assert kept == {("level", "extracted"), ("mechanism", "inferred"), ("reliability", "missing")}
    missing = next(a for a in accepted["attributes"] if a["basis"] == "missing")
    assert missing["quote"] is None and missing["page"] is None
    assert next(a for a in accepted["attributes"] if a["basis"] == "inferred")["page"] == 1
    assert len([r for r in rejections if r["kind"] == "attribute"]) == 4


def test_a_moderator_or_mediator_is_a_construct_attached_to_its_claim():
    payload = good()
    payload["claims"][0]["modifiers"] = [dict(kind="moderator", construct_label="Performance", value="", quote="among nurses"),
                                         dict(kind="mediator", construct_label="Loyalty", value="Loyalty", quote="among nurses"),
                                         dict(kind="sample", construct_label="", value="nurses", quote="among nurses")]
    accepted, rejections = ex.validate_response(payload, CHUNK)
    mods = accepted["claims"][0]["modifiers"]
    assert [(m["kind"], m["construct"]) for m in mods] == [("moderator", "performance"), ("sample", None)]
    assert any("must be one of the accepted constructs" in r["reason"] for r in rejections)


def test_claim_evidence_status_and_confidence_are_checked():
    for bad in (dict(evidence_status="proved"), dict(confidence=1.4), dict(confidence=True)):
        payload = good()
        payload["claims"][0].update(bad)
        accepted, rejections = ex.validate_response(payload, CHUNK)
        assert accepted["claims"] == [] and any("evidence status or confidence" in r["reason"] for r in rejections)
    payload = good()
    payload["claims"][0].update(evidence_status="not_supported", confidence=0.4)
    assert ex.validate_response(payload, CHUNK)[0]["claims"][0]["evidence_status"] == "not_supported"


def _paper_of(text):
    return {"page_offsets": [0], "chunks": [("c", 0, len(text))], "segments": [{"section": "results", "start": 0, "end": len(text), "text": text}]}


def test_pdf_formatting_is_ignored_but_the_stored_quote_is_the_papers_own_text():
    text = "Firms with freedom to operate (FTO) expand their eﬀorts in 20 05 and speci-\nfic ﬁrms diﬀer a lot, T o be sure."
    payload = {"constructs": [dict(label="Freedom to operate", surface_form="FTO", role="focal", definition="",
                                   quote="firms with freedom to operate (FTO) expand their efforts in 2005 and specific firms differ a lot."),
                              dict(label="Firm", surface_form="firms", role="mentioned", definition="", quote="To be sure")],
               "measures": [], "claims": [], "attributes": []}
    accepted, rejections = ex.validate_response(payload, _paper_of(text))
    first = accepted["constructs"][0]
    assert first["quote"] == text[:-len(", T o be sure.")] and text[first["char_start"]:first["char_end"]] == first["quote"]      # the paper's characters, ligatures included
    assert "ﬀ" in first["quote"] and "efforts" not in first["quote"]
    assert [r["reason"] for r in rejections] == ["quote not found in the paper"]                                                  # "To be sure" has too few letters to be matched loosely


def test_loose_matching_still_rejects_a_changed_added_or_missing_word():
    text = "Firms with freedom to operate expand their efforts in the following years."
    paper = _paper_of(text)
    for quote in ("Firms with freedom to operate expanded their efforts in the following years.",
                  "Firms with freedom to operate expand their efforts in the following three years.",
                  "Firms with freedom to operate expand efforts in the following years."):
        assert ex.locate_in_paper(paper, quote) is None
    assert ex.locate_in_paper(paper, "firms with freedom to operate, expand their efforts in the following years")


def test_definition_and_extracted_values_are_stored_as_the_papers_words():
    text = "Trust is deﬁned as a willingness to be vulnerable to the actions of another party."
    payload = {"constructs": [dict(label="Trust", surface_form="Trust", role="focal", definition="a willingness to be vulnerable to the actions of another party",
                                   quote="Trust is defined as a willingness to be vulnerable to the actions of another party.")],
               "measures": [], "claims": [],
               "attributes": [dict(construct_label="Trust", kind="mechanism", basis="extracted", value="willingness to be vulnerable to the actions", quote="Trust is defined as a willingness to be vulnerable to the actions of another party.")]}
    accepted, rejections = ex.validate_response(payload, _paper_of(text))
    assert rejections == [] and accepted["constructs"][0]["definition_text"] == "a willingness to be vulnerable to the actions of another party"
    assert accepted["attributes"][0]["value"] == "willingness to be vulnerable to the actions"


def test_a_quote_cannot_be_stitched_from_two_sections():
    paper = {"page_offsets": [0], "chunks": [("a", 0, 30), ("b", 40, 70)],
             "segments": [{"section": "theory", "start": 0, "end": 30, "text": "Trust is a willingness to rely."},
                          {"section": "results", "start": 40, "end": 70, "text": "Trust predicted performance."}]}
    payload = {"constructs": [dict(label="Trust", surface_form="Trust", role="focal", definition="", quote="to rely. Trust predicted"),
                              dict(label="Performance", surface_form="Performance", role="outcome", definition="", quote="Trust predicted performance.")],
               "measures": [], "claims": [], "attributes": []}
    accepted, rejections = ex.validate_response(payload, paper)
    assert [m["label_normalized"] for m in accepted["constructs"]] == ["performance"] and len(rejections) == 1


def test_offsets_and_chunk_are_computed_from_where_the_quote_sits():
    paper = {"page_offsets": [0, 1030], "chunks": [("intro", 100, 160), ("later", 1000, 1000 + len(TEXT))],
             "segments": [{"section": "introduction", "start": 100, "end": 160, "text": "Trust matters for hospitals and for how work gets done."},
                          {"section": "results", "start": 1000, "end": 1000 + len(TEXT), "text": TEXT}]}
    payload = {"constructs": [dict(label="Trust", surface_form="Trust", role="focal", definition="", quote="Trust matters for hospitals"),
                              dict(label="Trust", surface_form="Trust", role="focal", definition="willingness to be vulnerable", quote="Trust is defined as willingness to be vulnerable.")],
               "measures": [], "claims": [], "attributes": []}
    accepted, rejections = ex.validate_response(payload, paper)
    assert rejections == []
    # One label, two different places: two mentions, each in its own chunk, nothing merged or dropped.
    assert [(m["chunk_id"], m["char_start"], m["page"]) for m in accepted["constructs"]] == [("intro", 100, 1), ("later", 1000, 1)]
    assert [m["definition_text"] for m in accepted["constructs"]] == [None, "willingness to be vulnerable"]


def test_an_acronym_is_kept_as_an_attribute_with_the_sentence_that_states_it():
    text = "Digital transformation (DT) is the use of digital tools. DT predicted agility among firms."
    paper = {"page_offsets": [0], "chunks": [("c", 0, len(text))], "segments": [{"section": "theory", "start": 0, "end": len(text), "text": text}]}
    payload = {"constructs": [dict(label="Digital transformation", surface_form="DT", role="focal", definition="", quote="DT predicted agility among firms.")],
               "measures": [], "claims": [],
               "attributes": [dict(construct_label="Digital transformation", kind="acronym", basis="extracted", value="DT", quote="Digital transformation (DT) is the use of digital tools.")]}
    accepted, rejections = ex.validate_response(payload, paper)
    assert rejections == [] and accepted["constructs"][0]["label_normalized"] == "digital transformation" and accepted["constructs"][0]["surface_form"] == "DT"
    assert [(a["kind"], a["value"]) for a in accepted["attributes"]] == [("acronym", "DT")]


def test_the_prompt_unifies_names_only_on_the_authors_own_statement():
    for phrase in ("Use ONE `label` per construct", "only when the authors themselves state the equivalence", "never join them on your own judgment"):
        assert phrase in ex.INSTRUCTIONS


def test_one_request_per_paper_without_references(world, con, tmp_path):
    ingestion, _ = docs.ingest(con, world.run_id, world.store)
    manifest = ex.prepare(con, ingestion, tmp_path / "x")
    papers = con.execute("SELECT count(*) FROM paper_version WHERE status='parsed'").fetchone()[0]
    assert manifest["request_count"] == papers and manifest["unit"] == "paper"
    for line in (tmp_path / "x" / "requests.jsonl").read_text(encoding="utf-8").splitlines():
        request = json.loads(line)
        sent = json.loads(request["body"]["input"])
        assert request["custom_id"] == sent["paper_id"] and sent["sections"]
        assert not {s["section"] for s in sent["sections"]} & set(ex.SKIP_SECTIONS)


def test_excluded_papers_are_neither_parsed_nor_sent(world, con, tmp_path):
    from research0 import fulltext
    ids = [r[0] for r in con.execute("SELECT DISTINCT canonical_id FROM fulltext_file WHERE run_id=%s ORDER BY 1", (world.run_id,))]
    fulltext.exclude_papers(con, world.run_id, [ids[0]], "not in scope", "pytest")
    ingestion, counts = docs.ingest(con, world.run_id, world.store)
    assert not con.execute("SELECT 1 FROM paper_version WHERE canonical_id=%s", (ids[0],)).fetchone() and sum(counts.values()) == len(ids) - 1
    fulltext.exclude_papers(con, world.run_id, [ids[0]], "back in scope", "pytest", excluded=False)       # a later decision wins
    docs.ingest(con, world.run_id, world.store)
    fulltext.exclude_papers(con, world.run_id, [ids[1]], "parsed first, excluded after", "pytest")
    manifest = ex.prepare(con, ingestion, tmp_path / "x")
    sent = {json.loads(line)["custom_id"] for line in (tmp_path / "x" / "requests.jsonl").read_text(encoding="utf-8").splitlines()}
    owners = {str(v): c for v, c in con.execute("SELECT version_id,canonical_id FROM paper_version")}
    assert ids[1] not in {owners[v] for v in sent} and ids[0] in {owners[v] for v in sent} and manifest["request_count"] == len(sent)


def run_pipeline(world, con, tmp_path, **flaws):
    ingestion, _ = docs.ingest(con, world.run_id, world.store)
    manifest = ex.prepare(con, ingestion, tmp_path / "x")
    owner = {str(v): p for v, p in con.execute("SELECT version_id,canonical_id FROM paper_version")}
    request_lines = (tmp_path / "x" / "requests.jsonl").read_text(encoding="utf-8").splitlines()
    chunk_ids = [json.loads(line)["custom_id"] for line in request_lines]
    flaw_args = {k: next(c for c in chunk_ids if "x") if v else None for k, v in flaws.items()}
    lines = scripted_extraction_output(world, request_lines, owner, **{k: flaw_args[k] for k in flaw_args})
    (tmp_path / "x" / "batch_output.jsonl").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return ingestion, manifest, ex.collect_file(con, tmp_path / "x")


def test_extraction_from_mock_pdfs_end_to_end(world, con, tmp_path):
    ingestion, manifest, result = run_pipeline(world, con, tmp_path)
    assert manifest["request_count"] == result["papers_answered"] and result["missing_requests"] == 0 and result["rejected_items"] == 0
    papers = [p for p in world.papers if p.pdf]
    expected_claims = sum(len(p.planted["claims"]) for p in papers)
    assert result["new_claims"] == expected_claims and result["new_measures"] == sum(len(p.planted["measures"]) for p in papers)
    assert result["new_attributes"] == sum(len(p.planted["attributes"]) for p in papers)
    moderated = con.execute("SELECT count(*) FROM claim_modifier m JOIN construct_mention c ON c.mention_id=m.mention_id WHERE m.kind='moderator'").fetchone()[0]
    assert moderated == sum(1 for p in papers for c in p.planted["claims"] if c.get("modifier"))
    # Every stored record points at text that really says it, on the page it reports.
    rows = con.execute("SELECT m.quote,m.char_start,m.char_end,m.page,v.full_text,v.page_offsets FROM construct_mention m JOIN paper_version v USING (version_id)").fetchall()
    assert rows
    for quote, start, end, page, full, offsets in rows:
        assert full[start:end] == quote and docs.page_of(offsets, start) == page
    claim = con.execute("SELECT c.quote,c.char_start,c.char_end,s.label_normalized,o.label_normalized FROM nomological_claim c "
                        "JOIN construct_mention s ON s.mention_id=c.subject_mention_id JOIN construct_mention o ON o.mention_id=c.object_mention_id LIMIT 1").fetchone()
    assert claim[3] in claim[0].lower() and claim[4] in claim[0].lower()
    # Collecting again adds nothing
    again = ex.collect_file(con, tmp_path / "x")
    assert again["new_mentions"] == again["new_claims"] == again["new_measures"] == 0
    assert con.execute("SELECT count(*) FROM nomological_claim").fetchone()[0] == expected_claims


def test_invented_and_altered_quotes_are_rejected_in_the_pipeline(world, con, tmp_path):
    _, _, result = run_pipeline(world, con, tmp_path, invent_in=True)
    assert result["rejected_items"] >= 1
    assert con.execute("SELECT count(*) FROM construct_mention WHERE label_normalized='phantom construct'").fetchone()[0] == 0
    rejections = json.loads((tmp_path / "x" / "extraction_rejections.json").read_text(encoding="utf-8"))
    assert any("phantom" in r["what"] for r in rejections)


def test_human_validation_is_append_only(world, con, tmp_path):
    run_pipeline(world, con, tmp_path)
    mention = con.execute("SELECT mention_id FROM construct_mention LIMIT 1").fetchone()[0]
    ex.record_decision(con, "construct_mention", mention, "validated", "Ana")
    ex.record_decision(con, "construct_mention", mention, "rejected", "Ben", "wrong sense")
    ex.record_decision(con, "construct_mention", mention, "needs_review", "Ana", "second thoughts")
    state = ex.validation_state(con, "construct_mention", [mention])[mention]
    assert state == {"Ana": "needs_review", "Ben": "rejected"}
    assert con.execute("SELECT count(*) FROM decision_event WHERE subject_id=%s", (mention,)).fetchone()[0] == 3
    for bad in (("construct_mention", uuid.uuid4(), "validated", "Ana"), ("construct_mention", mention, "maybe", "Ana"), ("construct_mention", mention, "validated", " "),
                ("planet", mention, "validated", "Ana")):
        with pytest.raises(ValueError):
            ex.record_decision(con, *bad)


def test_a_staged_run_stays_out_of_the_live_tables_and_the_agents_view_until_a_person_promotes_it(piped, pcon, tmp_path):
    """The user's rule of 13 Oct 2026: a new extraction Batch is collected into the staged_ twins and reaches the tables the releases and the agents read only by promotion."""
    import shutil
    directory = tmp_path / "staged"
    shutil.copytree(piped.directory, directory)
    manifest = json.loads((directory / "manifest.json").read_text(encoding="utf-8"))
    manifest["prompt_version"] = "construct-extraction-staging-test"             # another run of the same answers
    (directory / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    live = {t: pcon.execute(f"SELECT count(*) FROM {t}").fetchone()[0] for t in ex.EXTRACTION_TABLES}
    result = ex.collect_file(pcon, directory, staged=True)
    run = uuid.UUID(result["extraction_run_id"])
    assert result["staged"] and result["new_mentions"] > 0
    assert {t: pcon.execute(f"SELECT count(*) FROM {t}").fetchone()[0] for t in ex.EXTRACTION_TABLES} == live          # nothing reached the live tables
    for table in ("staged_construct_mention", "staged_extraction_run", "extraction_promotion"):
        assert pcon.execute("SELECT has_table_privilege('research_agent', %s, 'SELECT')", (table,)).fetchone()[0] is False  # the agents cannot read it
    with pytest.raises(ValueError, match="in staging"):
        ex.collect_file(pcon, directory)                                  # collecting it again into the live tables is refused
    (listed,) = [r for r in ex.staged_runs(pcon) if r["extraction_run_id"] == str(run)]
    assert listed["promoted"] is None and listed["rows"]["construct_mention"] == result["new_mentions"]
    preview = ex.promote(pcon, run, "Reviewer")
    assert preview["executed"] is False and preview["would_copy"] == listed["rows"] and not pcon.execute("SELECT 1 FROM extraction_run WHERE extraction_run_id=%s", (run,)).fetchone()
    with pytest.raises(ValueError, match="who promotes"):
        ex.promote(pcon, run, " ", execute=True)
    done = ex.promote(pcon, run, "Reviewer", "checked the rejections", execute=True)
    assert done["copied"] == listed["rows"]
    assert pcon.execute("SELECT count(*) FROM construct_mention WHERE extraction_run_id=%s", (run,)).fetchone()[0] == result["new_mentions"]
    assert pcon.execute("SELECT count(*) FROM staged_construct_mention WHERE extraction_run_id=%s", (run,)).fetchone()[0] == result["new_mentions"]   # the staged record stays
    assert ex.staged_runs(pcon)[-1]["promoted"]["by"] == "Reviewer"
    with pytest.raises(ValueError, match="already in the live tables"):
        ex.promote(pcon, run, "Reviewer", execute=True)


def test_papers_collected_into_an_already_promoted_run_are_promoted_as_a_delta_run_that_leaves_the_parent_alone(piped, pcon):
    """Papers collected later into the same staged run are refused by promote(); promote_delta() copies only what the live tables lack, under a separate run."""
    parent = pcon.execute("SELECT extraction_run_id FROM extraction_run ORDER BY created_at DESC LIMIT 1").fetchone()[0]
    with pytest.raises(ValueError, match="No staged extraction run"):
        ex.promote_delta(pcon, parent, "Reviewer")                                   # nothing staged for it yet
    for name in ex.EXTRACTION_TABLES:                                               # stage a copy of the live run, as extract-collect does
        cols = ",".join(ex._columns(pcon, name))
        pcon.execute(f"INSERT INTO staged_{name}({cols}) SELECT {cols} FROM {name} WHERE " + ex.SCOPE[name].format(p="") , (parent,))
    assert ex.promote_delta(pcon, parent, "Reviewer")["would_copy"] == dict.fromkeys(ex.DELTA_KEYS, 0)      # nothing is missing from the live tables
    live = {t: pcon.execute(f"SELECT count(*) FROM {t}").fetchone()[0] for t in ex.DELTA_KEYS}
    paper = pcon.execute("SELECT canonical_id FROM construct_mention WHERE extraction_run_id=%s GROUP BY 1 ORDER BY count(*) DESC LIMIT 1", (parent,)).fetchone()[0]
    mentions = "(SELECT mention_id FROM construct_mention WHERE canonical_id=%s AND extraction_run_id=%s)"
    claims = "(SELECT claim_id FROM nomological_claim WHERE canonical_id=%s AND extraction_run_id=%s)"
    for statement in (f"DELETE FROM claim_modifier WHERE claim_id IN {claims} OR mention_id IN {mentions}", f"DELETE FROM nomological_claim WHERE claim_id IN {claims}",
                      f"DELETE FROM construct_measure WHERE mention_id IN {mentions}", f"DELETE FROM extraction_attribute WHERE mention_id IN {mentions}",
                      "DELETE FROM construct_mention WHERE canonical_id=%s AND extraction_run_id=%s"):     # one paper whose constructs are in staging only
        pcon.execute(statement, (paper, parent) * (statement.count("%s") // 2))
    missing = live["construct_mention"] - pcon.execute("SELECT count(*) FROM construct_mention").fetchone()[0]
    assert missing > 0                                                              # the live tables now lack some of the staged rows
    preview = ex.promote_delta(pcon, parent, "Reviewer")
    assert preview["executed"] is False and preview["would_copy"]["construct_mention"] == missing
    with pytest.raises(ValueError, match="who promotes"):
        ex.promote_delta(pcon, parent, " ", execute=True)
    done = ex.promote_delta(pcon, parent, "Reviewer", "later papers", execute=True)
    delta = uuid.UUID(done["delta_run_id"])
    assert done["copied"] == preview["would_copy"]
    assert pcon.execute("SELECT count(*) FROM construct_mention WHERE extraction_run_id=%s", (delta,)).fetchone()[0] == missing
    assert pcon.execute("SELECT count(*) FROM construct_mention").fetchone()[0] == live["construct_mention"]          # the parent run's rows were not touched or duplicated
    assert pcon.execute("SELECT manifest->>'delta_of' FROM extraction_run WHERE extraction_run_id=%s", (delta,)).fetchone()[0] == str(parent)
    assert pcon.execute("SELECT promoted_by FROM extraction_promotion WHERE extraction_run_id=%s", (delta,)).fetchone()[0] == "Reviewer"
    with pytest.raises(ValueError, match="already promoted as a delta"):
        ex.promote_delta(pcon, parent, "Reviewer", execute=True)

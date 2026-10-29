"""Forum, second step: agents that run Python for their analyses (a local subprocess with limits) and the research proposal an episode writes from its question.
The sandbox and the PDF are tested on their own; whole episodes run on the mock world with the scripted provider, so no model is called."""
import json
import os
import sys
import threading
import urllib.request
import uuid

import pytest

from research0 import forum, forum_cards as fcards, forum_proposal as fprop, forum_protocol as proto, forum_sandbox as sandbox, paper_cards, scoring_io
from forum_fake import ScriptedProvider

# the mock-world fixtures and helpers of the main forum tests
from test_forum import config, fw, forum_world, make_episode, messages, needs_db, run_episode, server, http, wait_for   # noqa: F401

MODELS = forum.load_models()
FLAG = sandbox.ENV_FLAG


# --- The sandbox -------------------------------------------------------------------------------------------------------------------------

def run(code, files=None, **kw):
    return sandbox.run_python(code, files or {}, **{"timeout": 20, "memory_mb": 512, **kw})


def test_code_runs_in_a_fresh_process_reads_its_data_and_keeps_only_allowed_outputs():
    data, described = sandbox.csv_file(["a", "b"], [[1, 2], [3, None]])
    assert described["rows"] == 2 and described["columns"] == ["a", "b"] and len(described["sha256"]) == 64
    code = "\n".join(["import csv", "rows = list(csv.DictReader(open('data/t.csv', encoding='utf-8')))", "print(len(rows), rows[1]['b'] == '')",
                      "open('out/result.csv', 'w', newline='').write('x\\n1\\n')", "open('out/skip.exe', 'w').write('no')", "open('notes.txt', 'w').write('inside the folder is fine')"])
    result = run(code, {"t.csv": data})
    assert result.ok and result.stdout.strip() == "2 True"
    assert [f["name"] for f in result.files] == ["result.csv"] and result.files[0]["content"] == b"x\n1\n" and len(result.files[0]["sha256"]) == 64
    assert any("skip.exe" in d for d in result.dropped)
    again = run("print(open('notes.txt').read())")                                 # nothing survives between runs
    assert not again.ok and "No such file" in again.stderr


def test_the_environment_is_scrubbed_so_secrets_of_the_server_are_not_visible(monkeypatch):
    monkeypatch.setenv("DATABASE_URL", "postgresql://secret")
    monkeypatch.setenv("OPENAI_API_KEY", "sk-secret")
    result = run("import os; print(sorted(k for k in os.environ if 'SECRET' in k.upper() or k in ('DATABASE_URL', 'OPENAI_API_KEY', 'AWS_SECRET_ACCESS_KEY')))")
    assert result.ok and result.stdout.strip() == "[]"


@pytest.mark.parametrize("code, message", [
    ("print(open(r'" + os.path.abspath(__file__).replace("\\", "/") + "').read()[:20])", "only lets code read data/"),
    ("open('../escape.txt', 'w').write('x')", "only lets code read data/"),
    ("import socket; socket.socket().connect(('127.0.0.1', 9))", "no network"),
    ("import subprocess; subprocess.run(['python', '-c', '1'])", "no network and starts no processes"),
    ("import os; os.system('echo hi')", "no network and starts no processes"),
])
def test_the_prelude_makes_the_obvious_escapes_fail_loudly(code, message):
    result = run(code)
    assert not result.ok and message in result.stderr


def test_a_run_that_takes_too_long_prints_too_much_or_asks_for_too_much_memory_is_stopped():
    slow = run("import time; time.sleep(30)", timeout=2)
    assert slow.timed_out and not slow.ok and "time limit" in slow.killed_for and slow.duration_ms < 10_000
    noisy = run("while True: print('x' * 1000)")
    assert not noisy.ok and "output limit" in noisy.killed_for and len(noisy.stdout) <= sandbox.MAX_STDOUT_CHARS
    hungry = run("x = bytearray(900 * 1024 * 1024); print(len(x))", memory_mb=200)
    assert not hungry.ok and ("MemoryError" in hungry.stderr or hungry.exit_code not in (0, None)) and "900" not in hungry.stdout
    assert hungry.limits_applied                                                   # the platform really applied a memory limit
    assert run("x" * (sandbox.MAX_CODE_CHARS + 1)).exit_code == 2


def test_a_person_pressing_stop_ends_a_run():
    stop = threading.Event()
    threading.Timer(1.0, stop.set).start()
    result = run("import time; time.sleep(30)", cancel=stop)
    assert result.cancelled and not result.ok and result.duration_ms < 10_000


def test_the_libraries_listed_for_the_agent_are_the_ones_that_import():
    libraries = sandbox.available_libraries()
    assert "numpy" in libraries
    assert run("import " + ", ".join(libraries) + "; print('ok')", timeout=120).stdout.strip() == "ok"
    assert sandbox.environment()["python"] == sys.version.split()[0]


# --- Configuration and rules --------------------------------------------------------------------------------------------------------------

def test_proposal_is_on_by_default_code_is_off_and_the_code_limits_exist_only_when_it_is_on():
    cfg = proto.normalize_config(config("C"), MODELS)
    assert cfg["proposal"] == {"enabled": True, "page_limit": 4, "max_tokens": 9000} and cfg["analysis_tools"] == "none" and cfg["code"] is None
    with_code = proto.normalize_config(config("C", analysis_tools="python_local", code={"max_runs": 3}), MODELS)
    assert with_code["code"] == {"max_runs": 3, "timeout_seconds": 60, "memory_mb": 512} and with_code["prompt_version"] == proto.PROMPT_VERSION_V4_CODE
    assert cfg["prompt_version"] == proto.PROMPT_VERSION_V4
    assert proto.budgets_checksum(with_code) != proto.budgets_checksum(cfg)
    assert proto.budgets_checksum(proto.normalize_config(config("C", analysis_tools="python_local", code={"max_runs": 4}), MODELS)) != proto.budgets_checksum(with_code)
    assert proto.budgets_checksum(proto.normalize_config(config("C", proposal={"enabled": False}), MODELS)) != proto.budgets_checksum(cfg)


@pytest.mark.parametrize("bad, message", [
    ({"proposal": {"enabled": "yes"}}, "proposal.enabled must be true or false"),
    ({"proposal": {"page_limit": 5}}, "proposal_pages must be between 1 and 4"),
    ({"analysis_tools": "python_local", "code": {"max_runs": 0}}, "max_code_runs must be between"),
    ({"analysis_tools": "python_local", "code": {"memory_mb": 8}}, "code_memory_mb must be between"),
    ({"analysis_tools": "docker"}, "analysis_tools must be one of"),
    ({"budgets": {"max_llm_calls": 4}}, "with their proposal"),
])
def test_configuration_errors_for_code_and_proposal(bad, message):
    with pytest.raises(ValueError, match=message):
        proto.normalize_config(config("C", **bad), MODELS)


def test_the_proposal_is_paid_from_the_part_of_the_budget_held_back_and_the_rules_follow_the_tools():
    on, off = proto.normalize_config(config("C"), MODELS), proto.normalize_config(config("C", proposal={"enabled": False}), MODELS)
    spec, order = proto.CONDITIONS["C"], proto.layout("C")
    reserve = lambda c: proto.synthesis_reserve(c["agents"], c["protocol"], spec, order, c["proposal"])
    assert reserve(on) - reserve(off) == 3 * (9000 + 12_000) + proto.CARD_CALLS * (proto.CARD_MAX_TOKENS + 12_000)      # draft, critique, revision; the methodology and the impact card
    solo = proto.normalize_config(config("A"), MODELS)
    assert proto.proposal_calls(proto.CONDITIONS["A"]) == 1 and proto.proposal_calls(spec) == 3 and solo["proposal"]["enabled"]
    agent = proto.layout("C")[0]
    assert "No analysis tools exist" in proto.framing("C", agent, trigger="anomaly") and "run_python" not in proto.framing("C", agent, trigger="anomaly")
    coded = proto.framing("C", agent, "python_local", trigger="anomaly")
    assert "run_python" in coded and "No analysis tools exist" not in coded and proto.PROMPT_VERSION_V4_CODE in coded


def test_a_question_may_call_an_analysis_obtained_only_if_a_run_that_finished_is_cited():
    base = {"rq": "Why?", "sources": ["E1", "A1"], "predictions": [{"statement": "s", "would_support": "a", "would_contradict": "b"}], "argument_steps": []}
    obtained = lambda run_id: {**base, "analyses": [{"status": "obtained", "description": "d", "data": "x", "limitations": "l", "run_id": run_id}]}
    known = {"E1", "A1", "A2"}
    assert not [w for w in proto.validate_question(obtained("A1"), known, {"A1": True})[1] if "obtained" in w]
    assert any("cites no run" in w for w in proto.validate_question(obtained("A9"), known, {"A1": True})[1])
    assert any("cites no run" in w for w in proto.validate_question(obtained(""), known, {"A1": True})[1])
    assert any("did not finish" in w for w in proto.validate_question(obtained("A2"), known, {"A1": True, "A2": False})[1])
    assert any("no analysis tool" in w for w in proto.validate_question(obtained("A1"), known, None)[1])     # without the tool nothing can be obtained
    assert proto.cited_ids("see [A2] and [E10] and [P1] but not [X3]") == ["A2", "E10", "P1"]
    assert "run_id" in proto.QUESTION_SCHEMA["properties"]["analyses"]["items"]["required"]


# --- The proposal text and its PDF --------------------------------------------------------------------------------------------------------

def paragraph(words, cite="[E1]"):
    return ("Platforms reorganise how firms coordinate innovation and the label hides two constructs " + cite + ". ") * max(1, words // 13)


def plain_abstract(words=220):
    return " ".join(("Platforms reorganise how firms coordinate innovation and the label may hide two constructs so we expect a design that separates them to predict two patterns " * 40).split()[:words])


def proposal(words=130, **overrides):
    return {"title": "A title", "abstract": plain_abstract(220), **{k: paragraph(words) for k, _, _ in fprop.SECTIONS}, **overrides}


def test_a_proposal_is_an_abstract_an_introduction_and_a_theoretical_background_and_nothing_else():
    assert [s[0] for s in fprop.SECTIONS] == ["introduction", "theoretical_background"]
    content, warnings = fprop.validate_proposal({**proposal(), "methods": paragraph(130), "expected_outcomes": paragraph(130)}, {"E1"})
    assert content["_validation"]["cited"] == ["E1"] and set(content["_validation"]["words"]) == {k for k, _, _ in fprop.SECTIONS} and warnings == []
    assert set(content) == {"title", "abstract", "introduction", "theoretical_background", "_validation"}      # whatever else the model sent is not kept
    _, warnings = fprop.validate_proposal(proposal(abstract=plain_abstract(130)), {"E1"})
    assert any("abstract has only 130 words" in w for w in warnings)                                          # a short abstract is flagged, kept


def test_a_proposal_needs_two_real_sections_and_flags_ids_nobody_retrieved():
    _, warnings = fprop.validate_proposal(proposal(theoretical_background=paragraph(130, "[E7] [A3]")), {"E1"})
    assert any("E7" in w and "A3" in w for w in warnings)
    assert any("cites no evidence" in w for w in fprop.validate_proposal(proposal(**{k: "Plain words " * 40 for k, _, _ in fprop.SECTIONS}), {"E1"})[1])
    with pytest.raises(ValueError, match="Theoretical background"):
        fprop.validate_proposal(proposal(theoretical_background="too short"), {"E1"})
    with pytest.raises(ValueError, match="no title"):
        fprop.validate_proposal(proposal(title=" "), {"E1"})


def test_a_section_far_under_its_word_budget_is_kept_but_flagged_as_shallow():
    total, per = fprop.word_budget(4)
    deep, warnings = fprop.validate_proposal(proposal(abstract=plain_abstract(220), **{k: paragraph(per[k]) for k, _, _ in fprop.SECTIONS}), {"E1"}, pages=4)
    assert not any("shallower" in w or "abstract has only" in w for w in warnings)
    thin, warnings = fprop.validate_proposal(proposal(words=100, abstract=plain_abstract(100)), {"E1"}, pages=4)
    assert sum("shallower" in w for w in warnings) == 2 and any("abstract has only 100 words" in w for w in warnings)      # both sections, and the abstract
    assert thin["introduction"]                                                                                           # flagged, never dropped
    assert not any("shallower" in w for w in fprop.validate_proposal(proposal(words=100), {"E1"})[1])                      # without the page limit there is no budget to compare with
    assert per["theoretical_background"] > per["introduction"] > 600                                                        # the budget is deep: the old sections had ~300 and ~600


def test_references_are_built_from_what_exists_never_from_what_the_model_wrote():
    content, _ = fprop.validate_proposal(proposal(theoretical_background=paragraph(130, "[E1] [P1] [A1] [E99]")), {"E1", "P1", "A1"})
    sources = {"E1": {"kind": "passage", "title": "Platform ecosystems", "year": 2021, "pages": [3, 5], "doi": "10.1/x"},
               "P1": {"kind": "diagnosis", "category": "Jingle", "basis": "decided by human reviewers", "labels": ["engagement", "engagement"]},
               "A1": {"kind": "analysis", "purpose": "count rows", "code_sha256": "ab" * 32, "ok": False}}
    lines = fprop.reference_lines(content, sources)
    assert len(lines) == 3 and not any("E99" in line for line in lines)           # an id that does not exist gets no reference
    assert "(2021). Platform ecosystems. Cited at pp. 3-5." in lines[0] and "doi:10.1/x" in lines[0] and "FAILED" in lines[2]


def test_references_carry_authors_journal_and_year_when_the_catalog_has_them():
    content, _ = fprop.validate_proposal(proposal(theoretical_background=paragraph(130, "[E1] [E2] [E3]")), {"E1", "E2", "E3"})
    sources = {"E1": {"kind": "passage", "title": "Platform ecosystems.", "year": 2021, "pages": [3, 5], "doi": "10.1/x", "authors": "Gao F.; Chen J.", "journal": "Management Science"},
               "E2": {"kind": "construct", "title": "Open source", "year": 2017, "pages": [7, 7], "doi": None, "journal": "Information Systems Research",
                      "authors": "Germonprez M.; Kendall J.E.; Kendall K.E.; Mathiassen L."},
               "E3": {"kind": "passage", "title": "No metadata", "year": None, "pages": [0, 0], "authors": None, "journal": None}}
    lines = fprop.reference_lines(content, sources)
    assert lines[0] == "[E1] Gao F. & Chen J. (2021). Platform ecosystems. Management Science. Cited at pp. 3-5. doi:10.1/x"
    assert lines[1] == "[E2] Germonprez M. et al. (2017). Open source. Information Systems Research. Cited at p. 7. Construct record."
    assert lines[2] == "[E3] (n.d.). No metadata."
    assert fprop.format_authors("Satterstrom P.; Kerrissey M.; DiBenigno J.") == "Satterstrom P., Kerrissey M. & DiBenigno J."
    assert fprop.format_authors("[No author name available]") == ""


def test_the_pdf_page_count_is_measured_and_unicode_outside_the_standard_fonts_does_not_break_it():
    meta = {"condition": "D (Jingle reasoning)", "authors": ["CTO specialist with TIM specialist"], "episode": "episode abcd1234", "date": "2026-10-11", "notice": "EXPLORATORY: x."}
    short, _ = fprop.validate_proposal(proposal(words=90, title="Ligatures ﬁ and signs ≥ → ≈ — “quotes”"), {"E1"})
    pdf, pages = fprop.render_pdf(short, meta, ["[E1] A reference (2021)."])
    assert pdf.startswith(b"%PDF") and pages == 1
    assert fprop.pdf_text("a ≥ b → c ﬁ") == "a >= b -> c fi"
    long, _ = fprop.validate_proposal(proposal(words=2600), {"E1"})
    assert fprop.render_pdf(long, meta, [])[1] > 4                                  # 2 x 2600 words do not fit in four pages
    total, per = fprop.word_budget(4)
    assert total == 4 * fprop.WORDS_PER_PAGE and sum(per.values()) <= total - fprop.ABSTRACT_RESERVE + 4          # the sections leave room for the abstract
    fits, _ = fprop.validate_proposal(proposal(abstract=plain_abstract(260), **{k: paragraph(per[k]) for k, _, _ in fprop.SECTIONS}), {"E1"})
    assert fprop.render_pdf(fits, meta, ["[E1] ref"] * 25)[1] <= 4                  # the word budget the agents are given really fits, with the longest abstract asked and 25 references


# --- Whole episodes: the proposal -------------------------------------------------------------------------------------------------------

def proposals(fw, episode):
    with fw.world.connect() as c:
        return forum.episode_detail(c, episode)["proposals"]


class LengthProvider(ScriptedProvider):
    """The scripted agents, but the proposal text has the length given for each stage (words per section)."""

    def __init__(self, **words):
        super().__init__(delay=0)
        self.words, self.stages, self.requests = words, [], []

    def run(self, request):
        self.requests.append(request)
        result = super().run(request)
        if request.task in self.words:
            self.stages.append(request.task)
            obj = json.loads(result.text)
            for key, _, _ in fprop.SECTIONS:
                obj[key] = paragraph(self.words[request.task])
            result.text = json.dumps(obj)
        elif request.task.startswith("proposal_"):
            self.stages.append(request.task)
        return result


@needs_db
def test_a_discussion_ends_in_a_proposal_drafted_by_the_lead_criticised_by_the_other_and_revised(fw):
    episode = make_episode(fw, "C", protocol={"rounds": 2, "lead": "cto"})
    provider = LengthProvider()
    assert run_episode(fw, episode, provider) == "completed"
    assert provider.stages == ["proposal_draft", "proposal_critique", "proposal_revise"]
    (found,) = proposals(fw, episode)
    assert found["author"] == "cto" and found["critic"] == "tim" and found["status"] == "ok" and 1 <= found["pages"] <= found["page_limit"] == 4
    assert all(found["content"][k] for k, _, _ in fprop.SECTIONS) and "methods" not in found["content"] and "expected_outcomes" not in found["content"]
    assert found["content"]["title"].startswith("[SIMULATION]")
    assert found["refs"] and all(line.startswith("[") for line in found["refs"])                       # every reference is an id the episode really has
    critique = [m for m in messages(fw, episode, kind="message") if m["data"].get("step") == "Proposal critique"]
    assert len(critique) == 1 and critique[0]["agent"] == "tim" and critique[0]["audience"] == "all"
    stages = [m["data"]["task"] for m in messages(fw, episode, kind="note") if m["data"].get("role") == "proposal_stage"]
    assert stages == ["proposal_draft", "proposal_revise"]
    asked = {r.task: r for r in provider.requests}
    critic, draft = asked["proposal_critique"], asked["proposal_draft"]
    assert "THE WORK SO FAR" in draft.prompt and "THE WORK SO FAR" not in critic.prompt and "YOUR PRIVATE NOTES" not in critic.prompt   # the critic reads the draft, not the discussion again
    assert "THE EVIDENCE THE DRAFT CITES" in critic.prompt and "Valid source ids: " in critic.prompt and len(critic.prompt) < len(draft.prompt)
    for kind in fcards.KINDS:                                       # the cards are asked as the papers' cards are: the instructions as the system, the text alone as the message
        card = asked[fcards.TASKS[kind]]
        assert card.system == fcards.INSTRUCTIONS[kind] and fcards.INSTRUCTIONS[kind] not in card.prompt and card.prompt.startswith("INPUT_ID: ")
    for kind in fcards.KINDS:                                       # the effort each card asks for (the test model does not reason, so None stays None: see the unit test)
        assert asked[fcards.TASKS[kind]].reasoning_effort == fcards.effort(kind, draft.reasoning_effort) == found["content"]["cards"][kind]["reasoning_effort"]
    reflect = next(r for r in provider.requests if r.task == "reflect")
    assert reflect.prompt.startswith("THE FORUM SO FAR:") and reflect.prompt.index("Budget left") > reflect.prompt.index("YOUR PRIVATE NOTES SO FAR")    # the header after the transcript
    with fw.world.connect() as c:
        question = c.execute("SELECT question_id FROM question WHERE episode_id=%s", (episode,)).fetchone()[0]
        assert c.execute("SELECT question_id FROM forum_proposal WHERE episode_id=%s", (episode,)).fetchone()[0] == question
        assert forum.list_episodes(c, [fw.release_id])[0]["proposals"] == 1


@needs_db
def test_one_agent_writes_its_proposal_alone_and_isolated_specialists_each_write_their_own(fw):
    solo = make_episode(fw, "A", protocol={"rounds": 1})
    provider = LengthProvider()
    assert run_episode(fw, solo, provider) == "completed" and provider.stages == ["proposal_draft"]
    (found,) = proposals(fw, solo)
    assert found["author"] == "generalist" and found["critic"] is None
    isolated = make_episode(fw, "B", protocol={"rounds": 1, "reasoner": "tim"})
    provider = LengthProvider()
    assert run_episode(fw, isolated, provider) == "completed" and provider.stages == ["proposal_draft"]
    found = proposals(fw, isolated)
    assert [p["author"] for p in found] == ["tim"] and found[0]["critic"] is None                                      # a specialist alone: nobody to critique
    assert not [m for m in messages(fw, isolated, kind="message") if m["data"].get("step") == "Proposal critique"]


@needs_db
def test_a_proposal_over_the_page_limit_is_shortened_by_its_author_and_if_still_too_long_kept_and_marked(fw):
    episode = make_episode(fw, "A", protocol={"rounds": 1})
    provider = LengthProvider(proposal_draft=2600, proposal_shorten=140)
    assert run_episode(fw, episode, provider) == "completed" and provider.stages == ["proposal_draft", "proposal_shorten"]
    (found,) = proposals(fw, episode)
    assert found["status"] == "ok" and found["pages"] <= 4 and not any("limit" in w for w in found["warnings"])
    stubborn = make_episode(fw, "A", protocol={"rounds": 1})
    provider = LengthProvider(proposal_draft=2600, proposal_shorten=2600)
    assert run_episode(fw, stubborn, provider) == "completed" and provider.stages.count("proposal_shorten") == 2      # two attempts, then it stops
    over = next(p for p in proposals(fw, stubborn))
    assert over["status"] == "over_limit" and over["pages"] > 4 and "was not cut by hand" in over["warnings"][0] and over["content"]["theoretical_background"].startswith("Platforms")


@needs_db
def test_the_proposal_says_when_it_rests_on_an_unvalidated_release_and_the_pdf_is_served(fw, server):
    from research0 import release
    blocked = None
    try:
        with fw.world.connect() as c:
            blocked, *_ = release.build(c, fw.world.run_id, "unsigned-for-proposal", fw.pre, None, fw.ingestion_id)
            c.commit()
            episode = forum.create_episode(c, blocked, config("0", protocol={"rounds": 1}), "Tester")
        assert run_episode(fw, episode) == "completed"
        (found,) = proposals(fw, episode)
        assert found["meta"]["notice"].startswith("EXPLORATORY, NOT A FINDING") and "not validated" in found["meta"]["notice"] and found["warnings"][0] == found["meta"]["notice"]
        port, run_id = server()
        with urllib.request.urlopen(f"http://127.0.0.1:{port}/api/forum/proposal.pdf?id={found['proposal_id']}") as response:
            body = response.read()
            assert response.status == 200 and response.headers["Content-Type"] == "application/pdf" and "inline" in response.headers["Content-Disposition"]
        assert body.startswith(b"%PDF")
        assert http(port, "GET", f"/api/forum/proposal.pdf?id={uuid.uuid4()}")[0] == 400
    finally:
        with fw.world.connect() as c:
            for table in ("forum_proposal_score", "forum_proposal_evaluation", "forum_proposal", "question", "forum_analysis_file", "forum_analysis"):
                c.execute(f"DELETE FROM {table}")
            c.execute("DELETE FROM forum_message"); c.execute("DELETE FROM forum_agent"); c.execute("DELETE FROM forum_episode")
            c.execute("DELETE FROM retrieval_hit"); c.execute("DELETE FROM retrieval_run"); c.execute("DELETE FROM agent_profile WHERE agent_id LIKE 'forum:%'")
            blocked and c.execute("DELETE FROM release WHERE release_id=%s", (blocked,))
            c.commit()


@needs_db
def test_without_budget_for_it_the_question_is_kept_and_the_missing_proposal_is_reported_and_it_can_be_switched_off(fw):
    off = make_episode(fw, "A", protocol={"rounds": 1}, proposal={"enabled": False})
    assert run_episode(fw, off) == "completed" and proposals(fw, off) == [] and "proposal_draft" not in {m["data"].get("task") for m in messages(fw, off)}
    tight = make_episode(fw, "A", protocol={"rounds": 2}, budgets={"max_llm_calls": 6})        # the smallest the reserve allows: the final question, the draft and the two cards, plus 2
    run_episode(fw, tight)
    with fw.world.connect() as c:
        assert c.execute("SELECT count(*) FROM question WHERE episode_id=%s", (tight,)).fetchone()[0] == 1
        status, detail = c.execute("SELECT status,status_detail FROM forum_episode WHERE episode_id=%s", (tight,)).fetchone()
    assert status in ("completed", "budget_exhausted")
    assert proposals(fw, tight) or "proposal" in (detail or "")                        # either it fit in the reserve or the episode says it did not


# --- Whole episodes: code -----------------------------------------------------------------------------------------------------------------

class CodeProvider(ScriptedProvider):
    """The scripted agents, but the first reflection of each agent runs these programs through the run_python tool."""

    def __init__(self, *programs):
        super().__init__(delay=0)
        self.programs, self.replies, self.done = list(programs), {}, set()

    def run(self, request):
        names = {t.name for t in request.tools}
        if request.task == "reflect" and "run_python" in names and request.hints["agent_label"] not in self.done:
            self.done.add(request.hints["agent_label"])
            for code in self.programs:
                self.replies.setdefault(request.hints["agent_label"], []).append(json.loads(request.run_tool("run_python", {"purpose": "test run", "code": code})))
        return super().run(request)


@pytest.fixture
def code_on(monkeypatch):
    monkeypatch.setenv(FLAG, "1")


@needs_db
def test_code_is_refused_unless_the_server_allows_it(fw, monkeypatch):
    monkeypatch.delenv(FLAG, raising=False)
    with fw.world.connect() as c:
        with pytest.raises(ValueError, match="Code execution is off on this server"):
            forum.create_episode(c, fw.release_id, config("A", analysis_tools="python_local"), "Tester")
    monkeypatch.setenv(FLAG, "1")
    episode = make_episode(fw, "A", analysis_tools="python_local", protocol={"rounds": 1})
    monkeypatch.delenv(FLAG)
    with pytest.raises(ValueError, match="Code execution is off"):
        forum.Runner(episode, ScriptedProvider(), app_url=fw.world.url)               # the flag is checked again when the episode starts


@needs_db
def test_an_agent_runs_code_on_its_tables_and_the_run_is_stored_cited_and_verified(fw, code_on):
    episode = make_episode(fw, "A", analysis_tools="python_local", code={"max_runs": 3}, protocol={"rounds": 2})
    assert run_episode(fw, episode, ScriptedProvider(delay=0)) == "completed"
    with fw.world.connect() as c:
        detail = forum.episode_detail(c, episode)
    (run_a,) = detail["analyses"]
    assert run_a["run_id"] == "A1" and run_a["agent"] == "generalist" and run_a["exit_code"] == 0 and "rows in constructs.csv" in run_a["stdout"]
    assert run_a["code_sha256"] and set(run_a["inputs"]) >= {"papers.csv", "constructs.csv", "measures.csv", "claims.csv", "modifiers.csv", "passages.csv"}
    assert "diagnoses.csv" not in run_a["inputs"]                                                    # A starts from the constructs: no diagnosis, not even as a table
    assert all(len(v["sha256"]) == 64 for v in run_a["inputs"].values()) and run_a["environment"]["python"]
    assert detail["config"]["analysis_tools"] == "python_local" and detail["prompt_version"] == proto.PROMPT_VERSION_V4_CODE
    result = next(m for m in detail["messages"] if m["kind"] == "tool_result" and m["data"].get("tool") == "run_python")
    assert result["content"].startswith("A1 finished")
    (question,) = detail["questions"]
    assert question["content"]["analyses"][0]["status"] == "obtained" and question["content"]["analyses"][0]["run_id"] == "A1"
    assert not [w for w in question["content"]["_validation"]["warnings"] if "obtained" in w]       # a finished, cited run backs "obtained"
    assert any(e["id"] == "A1" and e["kind"] == "analysis" and e["verified"] for e in question["evidence"])
    (proposal_row,) = detail["proposals"]
    assert "A1" in proposal_row["content"]["_validation"]["cited"] and any(line.startswith("[A1] Python analysis run in this forum") for line in proposal_row["refs"])


@needs_db
def test_the_tables_an_agent_gets_follow_its_community_and_condition_0_has_no_database(fw, code_on):
    program = "import csv; print(len(list(csv.DictReader(open('data/papers.csv', encoding='utf-8')))))"
    with fw.world.connect() as c:
        from research0 import rag
        manifest = rag.release_scope(c, fw.release_id, True)[1]
        sizes = {k: len(rag.community_papers(c, manifest, k, True)) for k in ("CTO", "TIM")}
        union = len({*rag.community_papers(c, manifest, "CTO", True), *rag.community_papers(c, manifest, "TIM", True)})
    counts = {}
    for name in ("cto", "tim"):
        isolated = make_episode(fw, "B", analysis_tools="python_local", protocol={"rounds": 1, "reasoner": name})
        provider = CodeProvider(program)
        assert run_episode(fw, isolated, provider) == "completed"
        counts.update({label: int(r[0]["stdout"]) for label, r in provider.replies.items()})
    assert counts == {"CTO specialist": sizes["CTO"], "TIM specialist": sizes["TIM"]}
    baseline = make_episode(fw, "0", analysis_tools="python_local", protocol={"rounds": 1})
    provider = CodeProvider("import os; print(sorted(os.listdir('data')))")
    assert run_episode(fw, baseline, provider) == "completed"
    assert provider.replies["Baseline agent"][0]["stdout"].strip() == "['passages.csv']"            # the baseline reads raw passages only, by design
    assert union >= max(sizes.values())


@needs_db
def test_failed_runs_do_not_back_an_obtained_analysis_the_budget_is_enforced_and_files_can_be_downloaded(fw, server, code_on):
    episode = make_episode(fw, "A", analysis_tools="python_local", code={"max_runs": 2, "timeout_seconds": 30}, protocol={"rounds": 1}, proposal={"enabled": False})
    provider = CodeProvider("1/0", "open('out/summary.csv', 'w', newline='').write('a,b\\n1,2\\n'); print('wrote')", "print('third')")
    assert run_episode(fw, episode, provider) == "completed"
    first, second, third = provider.replies["Generalist agent"]
    assert first["exit_code"] != 0 and "ZeroDivisionError" in first["stderr"]
    assert second["exit_code"] == 0 and second["files"] == [{"name": "summary.csv", "size": 8}]
    assert "budget" in third["error"]                                                                    # two runs allowed, the third is refused
    with fw.world.connect() as c:
        assert c.execute("SELECT count(*) FROM forum_analysis WHERE episode_id=%s", (episode,)).fetchone()[0] == 2
        (question,) = c.execute("SELECT content FROM question WHERE episode_id=%s", (episode,)).fetchone()
    port, _ = server()
    with urllib.request.urlopen(f"http://127.0.0.1:{port}/api/forum/analysis/file?episode={episode}&run=A2&name=summary.csv") as response:
        assert response.read() == b"a,b\n1,2\n" and "attachment" in response.headers["Content-Disposition"]
    assert http(port, "GET", f"/api/forum/analysis/file?episode={episode}&run=A2&name=missing.csv")[0] == 400
    assert http(port, "GET", f"/api/forum/analysis/file?episode={uuid.uuid4()}&run=A2&name=summary.csv")[0] == 400


@needs_db
def test_the_settings_page_reports_whether_code_is_allowed_and_an_episode_with_code_needs_the_flag(fw, server, monkeypatch):
    monkeypatch.delenv(FLAG, raising=False)
    port, run_id = server()
    cfg = http(port, "GET", "/api/forum/config")[1]
    assert cfg["code"]["available"] is False and cfg["code"]["flag"] == FLAG and cfg["code"]["defaults"]["max_runs"] == 8 and cfg["proposal"]["defaults"]["page_limit"] == 4
    release_id = http(port, "GET", "/api/forum/releases")[1]["releases"][0]["release_id"]
    body = {"run_id": run_id, "release_id": release_id, "created_by": "T", "config": config("A", analysis_tools="python_local")}
    status, err = http(port, "POST", "/api/forum/episode/create", body)
    assert status == 400 and "Code execution is off" in err["error"]


def test_no_file_under_deploy_turns_local_code_on():
    from pathlib import Path
    for path in (Path(__file__).resolve().parents[1] / "deploy").rglob("*"):
        if path.is_file() and path.suffix in (".yaml", ".yml", ".caddy", ".sh", ".py", ".env", ""):
            assert FLAG not in path.read_text(encoding="utf-8", errors="ignore"), f"{path.name} mentions {FLAG}: the production stack must never enable it"


def test_the_proposal_needs_an_abstract_that_reads_without_the_references():
    content, warnings = fprop.validate_proposal(proposal(), {"E1"})
    assert content["abstract"] == plain_abstract(220) and content["_validation"]["abstract_words"] == 220 and warnings == []
    assert "abstract" not in content["_validation"]["words"]                      # the abstract is outside the two sections' word budget (it has its own reserve)
    for bad in ("", "too short", " ".join(["word"] * 59)):
        with pytest.raises(ValueError, match="abstract"):
            fprop.validate_proposal(proposal(abstract=bad), {"E1"})
    with pytest.raises(ValueError, match="abstract"):
        fprop.validate_proposal({k: v for k, v in proposal().items() if k != "abstract"}, {"E1"})
    _, warnings = fprop.validate_proposal(proposal(abstract=plain_abstract(220) + " [E1]"), {"E1"})
    assert any("abstract cites evidence ids" in w for w in warnings)
    _, warnings = fprop.validate_proposal(proposal(abstract=plain_abstract(320)), {"E1"})
    assert any("320 words" in w for w in warnings)


def test_the_request_schema_and_the_instructions_ask_for_the_abstract():
    assert "abstract" in fprop.PROPOSAL_SCHEMA["properties"] and "abstract" in fprop.PROPOSAL_SCHEMA["required"]
    for task in (fprop.draft_task(3), fprop.revise_task(3), fprop.shorten_task(4, 3)):
        assert "abstract" in task and "no evidence ids" in task
    assert "a title, an abstract, an introduction and a theoretical background" in fprop.draft_task(3)
    for gone in ("methods:", "expected_outcomes:", "four sections"):
        assert gone not in fprop.draft_task(3) and gone not in fprop.critique_task()
    assert "NO methods section and NO expected-outcomes section" in fprop.draft_task(3)       # the agents are told where the design went


def test_the_abstract_states_expected_results_and_never_results_as_obtained():
    _, warnings = fprop.validate_proposal(proposal(abstract=" ".join(["Platforms reorganise how firms coordinate innovation and the label hides two constructs"] * 12)), {"E1"})
    assert any("does not state any expected result" in w for w in warnings)
    _, warnings = fprop.validate_proposal(proposal(abstract=plain_abstract(130) + " We found that the effect is large and our results show it."), {"E1"})
    assert any("worded as obtained" in w or "words a result as obtained" in w for w in warnings)
    for task in (fprop.draft_task(3), fprop.revise_task(3)):
        assert "EXPECTED results" in task and "never as results obtained" in task and "abstract of a published paper" in task


# --- The methodology card and the impact card of a proposal ---------------------------------------------------------------------------------

CARD_TEXT = proposal(abstract="Platforms reorganise how firms coordinate innovation across unrelated partners, so we expect two patterns. " * 8,
                     introduction="Scholars of digital platforms disagree about which processes link openness and innovation outcomes [E1]. " * 6,
                     theoretical_background="A governance mechanism could raise complementor entry only when the platform controls access to its boundary resources [E1]. " * 6)


def good_card(kind, source, **overrides):
    passage = "Scholars of digital platforms disagree about which processes link openness and innovation outcomes."
    assert scoring_io.squash_alnum(passage) in scoring_io.squash_alnum(source)
    card = {f: {"text": f"A short sentence about {f}.", "status": "inferred", "passages": [passage]} for f in fcards.PROFILES[kind].fields}
    return {**card, "info_gaps": ["What data exist is still to be found out."], **overrides}


def test_the_cards_of_a_proposal_have_exactly_the_structure_of_the_cards_of_the_papers():
    assert fcards.KINDS == ("method", "impact") and fcards.PROFILES == {"method": paper_cards.METHOD, "impact": paper_cards.IMPACT}
    for kind, profile in fcards.PROFILES.items():
        assert fcards.schema(kind) == paper_cards.card_schema(profile.fields)                        # the very schema the Batch of the papers asks for
        assert set(fcards.provider_schema(kind)["required"]) == set(profile.fields) | {"info_gaps"}
    assert fcards.INSTRUCTIONS["impact"] == paper_cards.IMPACT_SYSTEM                                   # the impact card is made with the instructions of the papers
    assert "YOURS to make" in fcards.INSTRUCTIONS["method"] and fcards.INSTRUCTIONS["method"] != paper_cards.METHOD_SYSTEM    # the method card is designed, not described
    assert fcards.CALLS == proto.CARD_CALLS == len(fcards.KINDS) and set(fcards.TASKS.values()) <= set(forum.FINAL_TASKS)


def test_a_card_is_made_from_the_final_text_without_the_markers_of_the_internal_evidence():
    source = fcards.source_text(CARD_TEXT)
    assert "[E1]" not in source and "disagree about which processes" in source and source.startswith("Title\nA title") and "Abstract" in source and "Theoretical background" in source
    assert "[E1]" not in fcards.prompt("impact", "proposal-ab12", source) and "INPUT_ID: proposal-ab12" in fcards.prompt("impact", "proposal-ab12", source)
    again = fcards.prompt("method", "proposal-ab12", source, ["design: a passage is not an exact piece"])
    assert "REFUSED BY THE CHECK" in again and "design: a passage is not an exact piece" in again and "REFUSED" not in fcards.prompt("method", "proposal-ab12", source)


def test_a_card_is_asked_with_its_own_instructions_and_the_impact_card_with_the_effort_of_the_papers_cards():
    for kind in fcards.KINDS:
        assert fcards.system(kind) == fcards.INSTRUCTIONS[kind] and fcards.INSTRUCTIONS[kind] not in fcards.prompt(kind, "proposal-ab12", "Title. A text.")
    assert fcards.effort("impact", "high") == "low" and fcards.effort("method", "high") == "high"
    assert fcards.effort("impact", None) is None and fcards.effort("method", None) is None           # a model that does not reason is never asked to


def test_every_passage_of_a_card_must_be_an_exact_piece_of_the_proposal_and_what_cannot_be_proved_becomes_missing():
    source = fcards.source_text(CARD_TEXT)
    for kind in fcards.KINDS:
        card, demoted, problems = fcards.check(kind, good_card(kind, source), source)
        assert card and demoted == [] and problems == []
    bad = good_card("method", source)
    bad["measures"] = {"text": "Survey scales.", "status": "inferred", "passages": ["A sentence that the proposal never wrote anywhere at all."]}
    card, demoted, problems = fcards.check("method", bad, source)
    assert demoted == ["measures"] and card["measures"] == {"text": None, "status": "missing", "passages": []} and any("measures" in g for g in card["info_gaps"])
    mostly_bad = {f: {"text": "x.", "status": "inferred", "passages": ["Nothing of this is in the text, not a word of it."]} for f in paper_cards.METHOD.fields}
    card, demoted, problems = fcards.check("method", {**mostly_bad, "info_gaps": []}, source)
    assert card is None and problems and "exact quotation" in problems[0]                           # fewer than half survive: refused, never patched
    assert fcards.check("impact", {"not": "a card"}, source)[0] is None


def test_what_is_stored_for_a_card_says_whether_it_is_valid_and_what_is_known_about_it():
    source = fcards.source_text(CARD_TEXT)
    card = good_card("impact", source)
    stored = fcards.record("impact", "valid", source=source, model="m", card=card)
    assert stored["status"] == "valid" and stored["profile_version"] == paper_cards.IMPACT.version and stored["completeness"]["share"] == 1.0 and len(stored["source_sha256"]) == 64
    refused = fcards.record("method", "rejected", source=source, model="m", problems=["only 1 of 8 fields have an exact quotation"])
    missing = fcards.record("method", "not_made", source=source, reason="the token limit was reached")
    assert refused["card"] is None and missing["card"] is None and "completeness" not in refused
    assert fcards.valid_cards({"cards": {"impact": stored, "method": refused}}) == {"impact": card}
    assert fcards.valid_cards({}) == {} and fcards.valid_cards(None) == {}                             # a proposal written before 18 Oct 2026 has no cards


class CardProvider(ScriptedProvider):
    """The scripted agents; the cards named in `once` come back unusable the first time they are asked, those in `always` every time, and asking for those in `fail` raises."""

    def __init__(self, once=(), always=(), fail=()):
        super().__init__(delay=0)
        self.once, self.always, self.fail, self.cards = set(once), set(always), set(fail), []

    def run(self, request):
        if request.task in self.fail:
            self.cards.append(request.task)
            raise ConnectionError("the provider dropped the connection")
        result = super().run(request)
        if request.task in fcards.TASKS.values():
            self.cards.append(request.task)
            if request.task in self.always or (request.task in self.once and self.cards.count(request.task) == 1):
                result.text = json.dumps({"not": "a card"})
        return result


@needs_db
def test_the_author_writes_the_two_cards_after_the_final_text_and_they_are_stored_with_the_proposal(fw):
    episode = make_episode(fw, "C", protocol={"rounds": 2, "lead": "cto"})
    provider = CardProvider()
    assert run_episode(fw, episode, provider) == "completed"
    assert provider.cards == ["card_method", "card_impact"]                                              # one call each, after the proposal
    (found,) = proposals(fw, episode)
    cards = found["content"]["cards"]
    assert set(cards) == {"method", "impact"} and all(c["status"] == "valid" for c in cards.values())
    source = fcards.source_text(found["content"])
    for kind, profile in fcards.PROFILES.items():
        card = cards[kind]["card"]
        assert set(card) == set(profile.fields) | {"info_gaps"} and cards[kind]["profile_version"] == profile.version
        assert all(f["status"] in ("extracted", "inferred") and f["text"] for f in (card[x] for x in profile.fields))
        assert all(scoring_io.squash_alnum(p) in scoring_io.squash_alnum(source) for x in profile.fields for p in card[x]["passages"])     # every passage is a piece of the text
    assert not any("card" in w for w in found["warnings"]) and found["pages"] <= found["page_limit"]       # the cards are not part of the 4 pages
    stages = [m["data"] for m in messages(fw, episode, kind="note") if m["data"].get("role") == "card_stage"]
    assert [(s["kind"], s["attempt"]) for s in stages] == [("method", 1), ("impact", 1)]


@needs_db
def test_a_refused_card_is_asked_for_once_more_and_a_card_that_stays_refused_never_fails_the_episode(fw):
    episode = make_episode(fw, "A", protocol={"rounds": 1})
    provider = CardProvider(once={"card_method"}, always={"card_impact"})
    assert run_episode(fw, episode, provider) == "completed"
    assert provider.cards == ["card_method", "card_method", "card_impact", "card_impact"]                # a second attempt each, then it stops
    (found,) = proposals(fw, episode)
    method, impact = found["content"]["cards"]["method"], found["content"]["cards"]["impact"]
    assert method["status"] == "valid" and impact["status"] == "rejected" and impact["card"] is None and impact["problems"]
    assert fcards.valid_cards(found["content"]).keys() == {"method"}
    assert any("impact card could not be made" in w for w in found["warnings"]) and found["status"] == "ok"
    assert all(found["content"][k] for k, _, _ in fprop.SECTIONS)                                       # the proposal itself is untouched


@needs_db
def test_an_error_of_the_provider_in_a_card_never_loses_the_proposal_that_was_already_written(fw):
    episode = make_episode(fw, "A", protocol={"rounds": 1})
    provider = CardProvider(fail={"card_method"})
    assert run_episode(fw, episode, provider) == "completed"
    assert provider.cards == ["card_method", "card_impact"]                                              # no retry after an error; the other card is still tried
    (found,) = proposals(fw, episode)
    method, impact = found["content"]["cards"]["method"], found["content"]["cards"]["impact"]
    assert method["status"] == "not_made" and "ConnectionError" in method["reason"] and method["card"] is None and impact["status"] == "valid"
    assert any("method card could not be made" in w for w in found["warnings"]) and all(found["content"][k] for k, _, _ in fprop.SECTIONS)

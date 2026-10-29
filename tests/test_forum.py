"""The discovery forum: protocol and configuration (no database), the OpenAI provider against a stand-in client, and whole episodes on the mock world
with the tests' offline stand-in (forum_fake.py; the application has no simulation). No model is called and nothing is written outside the throw-away mock schema."""
import json
import os
import threading
import uuid
from pathlib import Path
from types import SimpleNamespace

import psycopg
import pytest

from forum_fake import ScriptedProvider
from research0 import forum, forum_providers, forum_protocol as proto
from research0.forum_providers import BudgetExceeded, Cancelled, OpenAIProvider, ToolSpec, TurnRequest

MODELS = forum.load_models()
UNCLEAR = {"same_referent": "unclear", "compatible_properties": "unclear", "substitution_changes": "unclear"}


def config(condition="C", /, **overrides):
    return {"provider": "scripted", "condition": condition, **overrides}


# --- Protocol and configuration (no database) -------------------------------------------------------------------------------------------

def test_the_design_has_four_conditions_and_d_e_f_are_kept_only_to_read_old_episodes():
    assert proto.DESIGN == ("0", "A", "B", "C") and list(proto.CONDITIONS) == ["0", "A", "B", "C", "D", "E", "F"]
    assert all(proto.CONDITIONS[c].get("legacy") for c in "DEF") and not any(proto.CONDITIONS[c].get("legacy") for c in proto.DESIGN)
    assert [a["key"] for a in proto.layout("0")] == ["baseline"] and [a["key"] for a in proto.layout("A")] == ["generalist"]
    for c in "BCDEF":
        assert [(a["key"], a["community"]) for a in proto.layout(c)] == [("cto", "CTO"), ("tim", "TIM")]
    assert not proto.CONDITIONS["0"]["organized"] and all(proto.CONDITIONS[c]["organized"] for c in "ABC")
    assert proto.CONDITIONS["C"]["name"] == "Open specialist encounter" and proto.CONDITIONS["C"]["encounter"]
    for c in "DEF":
        with pytest.raises(ValueError, match="removed from the design"):
            proto.normalize_config(config(c), MODELS)
    assert proto.prompt_version("D") == proto.PROMPT_VERSION                 # what the old episodes ran with stays nameable


def test_defaults_fill_in_and_are_frozen_with_prompt_version():
    cfg = proto.normalize_config(config("B"), MODELS)
    assert cfg["agents"]["cto"]["discussion_model"] == "scripted-v1" and cfg["agents"]["cto"]["reflection_model"] == "scripted-v1" and set(cfg["agents"]) == {"cto"}
    assert cfg["budgets"] == proto.DEFAULT_BUDGETS and cfg["external_access"] == "off" and cfg["analysis_tools"] == "none" and cfg["prompt_version"] == proto.PROMPT_VERSION_V4
    assert cfg["protocol"]["lead"] == "cto" and cfg["protocol"]["reflection"] is True
    assert {proto.normalize_config(config(c), MODELS)["prompt_version"] for c in proto.DESIGN} == {proto.PROMPT_VERSION_V4}
    assert proto.normalize_config(config("A", analysis_tools="python_local"), MODELS)["prompt_version"] == proto.PROMPT_VERSION_V4_CODE


def test_every_episode_gets_one_of_the_nine_triggers_chosen_or_drawn_from_the_seed_alone():
    assert list(proto.TRIGGERS) == ["phenomenon", "anomaly", "challenged_assumption", "paradox", "missing_mechanism", "boundary_reversal", "process", "translation", "construct_problem"]
    drawn = {proto.normalize_config(config("A", seed=s), MODELS)["trigger"]["key"] for s in range(120)}
    assert drawn == set(proto.TRIGGERS)                                                                  # random by default, and every trigger can come up
    for seed in (3, 77, 2024):                                                                           # matched across conditions: the same seed, the same trigger
        assert len({proto.normalize_config(config(c, seed=seed), MODELS)["trigger"]["key"] for c in proto.DESIGN}) == 1
    pinned = proto.normalize_config(config("0", trigger="anomaly"), MODELS)["trigger"]
    assert pinned == {"choice": "anomaly", "key": "anomaly", "name": "Anomaly", "cross_community": False}
    assert proto.normalize_config(config("B", trigger={"choice": "translation"}), MODELS)["trigger"]["cross_community"] is True      # B cannot do it: analysed apart
    assert proto.normalize_config(config("C", trigger="translation"), MODELS)["trigger"]["cross_community"] is False
    with pytest.raises(ValueError, match="trigger must be random or one of"):
        proto.normalize_config(config("A", trigger="horoscope"), MODELS)
    # the trigger is protocol, not budget: it never makes two episodes incomparable
    assert proto.budgets_checksum(proto.normalize_config(config("A", trigger="anomaly"), MODELS)) == proto.budgets_checksum(proto.normalize_config(config("A", trigger="paradox"), MODELS))


def test_the_trigger_is_told_as_a_way_to_search_never_as_a_topic_with_what_it_demands():
    agent = proto.layout("A")[0]
    for key, (name, instruction, move) in proto.TRIGGERS.items():
        text = proto.framing("A", agent, trigger=key, dossiers=key == "construct_problem")
        assert name in text and instruction in text and move in text and "not what to study" in text
    assert "abstain from claiming one" in proto.framing("0", proto.layout("0")[0], trigger="phenomenon")
    assert "genuine theoretical tension" in proto.framing("C", proto.layout("C")[0], trigger="paradox")
    assert proto.ISOLATED_TRANSLATION in proto.framing("B", proto.layout("B")[0], trigger="translation") and proto.ISOLATED_TRANSLATION not in proto.framing("C", proto.layout("C")[0], trigger="translation")
    organized = proto.framing("A", agent, trigger="construct_problem", dossiers=True)
    assert proto.DOSSIER_NOTE in organized and proto.RULE_3_DOSSIERS in organized and "[P2] for a construct-problem dossier" in organized and proto.NO_DIAGNOSIS not in organized
    raw = proto.framing("0", proto.layout("0")[0], trigger="construct_problem")
    assert proto.RAW_NOTE in raw and proto.DOSSIER_NOTE not in raw and "[P2]" not in raw and proto.RULE_3_FULL_TEXT in raw
    other = proto.framing("A", agent, trigger="anomaly")
    assert proto.NO_DIAGNOSIS in other and "[P2]" not in other and proto.RULE_3_CONSTRUCTS in other


def test_separate_models_for_reflection_discussion_and_synthesis_are_kept_per_agent():
    cfg = proto.normalize_config({"condition": "C", "provider": "openai", "agents": {"cto": {"reflection_model": "gpt-6-luna", "discussion_model": "my-custom-model", "synthesis_model": "gpt-6-luna", "reasoning_effort": "high"},
                                                                                   "tim": {"discussion_model": "gpt-6-luna", "reasoning_effort": "none"}}}, MODELS)
    cto, tim = cfg["agents"]["cto"], cfg["agents"]["tim"]
    assert (cto["reflection_model"], cto["discussion_model"], cto["synthesis_model"], cto["reasoning_effort"]) == ("gpt-6-luna", "my-custom-model", "gpt-6-luna", "high")
    assert tim["reflection_model"] == "gpt-6-luna" and tim["reasoning_effort"] is None


@pytest.mark.parametrize("bad, message", [
    ({"condition": "Z"}, "condition must be"),
    ({"provider": "carrier-pigeon"}, "provider must be"),
    ({"budgets": {"max_total_tokens": 10**9}}, "max_total_tokens must be between"),
    ({"budgets": {"max_llm_calls": "many"}}, "whole number"),
    ({"protocol": {"rounds": 0}}, "rounds must be between"),
    ({"agents": {"cto": {"temperature": 3}}}, "temperature must be between"),
    ({"agents": {"cto": {"reasoning_effort": "extreme"}}}, "reasoning_effort"),
    ({"agents": {"cto": {"discussion_model": "gpt-6-luna"}}}, "belongs to the openai provider"),
    ({"external_access": "open"}, "tool of the openai provider"),
    ({"analysis_tools": "python"}, "analysis_tools must be one of"),
    ({"protocol": {"lead": "ghost"}}, "lead must be"),
    ({"budgets": {"max_total_tokens": 5000}}, "leave nothing for the rounds"),
    ({"trigger": "construct_problem", "diagnoses": {"limit": 9}}, "count against max_pairs"),
    ({"trigger": "construct_problem", "diagnoses": {"limit": 0}}, "at least one construct relationship"),
    ({"title": "x" * 200}, "limited to"),
])
def test_configuration_errors_say_what_is_wrong(bad, message):
    with pytest.raises(ValueError, match=message):
        proto.normalize_config(config("C", **bad), MODELS)


def test_the_tests_stand_in_only_runs_its_own_model():
    with pytest.raises(ValueError, match="only runs the scripted-v1"):
        proto.normalize_config(config("A", agents={"generalist": {"discussion_model": "something-else"}}), MODELS)


def test_the_simulation_is_gone_from_the_application(monkeypatch):
    """Removed on 13 Oct 2026 at the user's request: without the tests' stand-in (registered by conftest.py) nothing can run, rate or offer a simulated episode."""
    from research0 import forum_providers, pipeline_cli
    monkeypatch.setattr(proto, "PROVIDERS", {"openai": None})
    monkeypatch.setattr(forum_providers, "FACTORIES", {"openai": OpenAIProvider})
    with pytest.raises(ValueError, match="simulation was removed"):
        proto.normalize_config(config("A"), MODELS)
    with pytest.raises(ValueError, match="simulation was removed"):
        forum_providers.get_provider("scripted")
    cfg = proto.normalize_config({"condition": "A"}, MODELS)
    assert cfg["provider"] == "openai" and cfg["agents"]["generalist"]["discussion_model"] == "gpt-6-luna"
    assert {m["provider"] for m in MODELS["models"]} == {"openai"}
    assert "forum-demo" not in pipeline_cli.COMMANDS and not (Path(forum.__file__).parent / "forum_demo.py").exists()
    page = (Path(forum.__file__).parent / "forum_template.html").read_text(encoding="utf-8")
    assert "simulat" not in page.lower() and "scripted" not in page                 # no trace of it in the frontend (the user's request)


def test_budgets_checksum_is_the_comparability_key_across_conditions():
    same = {c: proto.budgets_checksum(proto.normalize_config(config(c), MODELS)) for c in proto.DESIGN}
    assert len(set(same.values())) == 1                                            # defaults are identical across conditions
    changed = proto.budgets_checksum(proto.normalize_config(config("C", budgets={"max_passages": 10}), MODELS))
    assert changed != same["C"]
    models_only = proto.budgets_checksum(proto.normalize_config(config("C", protocol={"rounds": 2}, seed=5), MODELS))
    assert models_only == same["C"]                                                # rounds and seed are the protocol, not the budget


def test_the_lead_is_chosen_or_picked_by_the_seed_and_the_choice_is_frozen_with_the_name():
    picks = {proto.normalize_config(config("C", seed=s), MODELS)["protocol"]["lead"] for s in range(40)}
    assert picks == {"cto", "tim"}                                                                      # C defaults to random: the seed picks, and both can open
    first = proto.normalize_config(config("C", seed=77), MODELS)["protocol"]
    assert first["lead_choice"] == "random" and first["lead"] == proto.normalize_config(config("C", seed=77), MODELS)["protocol"]["lead"]   # the same seed always picks the same agent
    for name in ("cto", "tim"):
        pinned = proto.normalize_config(config("C", seed=77, protocol={"lead": name}), MODELS)["protocol"]
        assert (pinned["lead"], pinned["lead_choice"]) == (name, name)
    assert proto.normalize_config(config("B"), MODELS)["protocol"]["lead_choice"] == "cto"             # elsewhere the first agent, unless asked
    # the opener and the trigger come from the same seed but from independent draws: knowing one does not fix the other
    pairs = {(proto.normalize_config(config("C", seed=s), MODELS)["protocol"]["lead"], proto.normalize_config(config("C", seed=s), MODELS)["trigger"]["key"]) for s in range(200)}
    assert len(pairs) > 12


def test_condition_c_opens_on_a_theme_the_opener_proposes_and_keeps_the_earlier_openings_as_auxiliary_controls():
    cfg = proto.normalize_config(config("C"), MODELS)
    assert cfg["encounter"] == {"theme_mode": "opener_theme", "theme_size": 5, "items_per_round": 0} and cfg["start"] == "constructs"
    assert proto.AUXILIARY_THEME_MODES == ("constructs", "rag_cluster", "random_passages")
    assert proto.normalize_config(config("C", encounter={"theme_mode": "rag_cluster"}), MODELS)["start"] == "theme"
    assert proto.normalize_config(config("C", encounter={"theme_mode": "constructs"}), MODELS)["start"] == "constructs"
    cto, tim = proto.layout("C")
    opener, other = (proto.framing("C", a, encounter={"theme_mode": "opener_theme"}, trigger="anomaly", lead="cto") for a in (cto, tim))
    assert "You open: following the trigger, propose a theme supported by evidence" in opener and "You test or reframe it" in other
    assert proto.steps_for("C", "opener_theme") is proto.STEPS_V4_ENCOUNTER and proto.chain_step("C", "opener_theme", 1, 4)[0] == "Opening theme"
    assert proto.synth_task("C", "opener_theme").endswith(proto.SYNTH_ENCOUNTER + proto.SYNTH_TIMELINESS)
    assert proto.synth_task("C", "rag_cluster") == proto.SYNTH_V4 + proto.SYNTH_BETWEEN + proto.SYNTH_TIMELINESS
    enc = proto.normalize_config(config("C", encounter={"theme_mode": "rag_cluster"}), MODELS)["encounter"]
    assert enc == {"theme_mode": "rag_cluster", "theme_size": 5, "items_per_round": 0}
    assert proto.normalize_config(config("C", encounter={"theme_mode": "random_passages"}), MODELS)["encounter"]["items_per_round"] == 3        # the control keeps its random items
    with pytest.raises(ValueError, match="theme_mode must be"):
        proto.normalize_config(config("C", encounter={"theme_mode": "dice"}), MODELS)
    with pytest.raises(ValueError, match="theme_size must be between"):
        proto.normalize_config(config("C", encounter={"theme_size": 1}), MODELS)
    assert proto.steps_for("C", "rag_cluster") is proto.STEPS_THEME and proto.steps_for("C", "random_passages") is proto.STEPS_RANDOM and proto.steps_for("C") is proto.STEPS_RANDOM
    assert proto.steps_for("C", "constructs") is proto.STEPS_CONSTRUCTS and proto.steps_for("D") is proto.STEPS
    assert "neither corpus could answer alone" in proto.framing("C", cto, "none", {"theme_mode": "rag_cluster"})
    assert "neither corpus could answer alone" not in proto.framing("C", cto, "none", {"theme_mode": "random_passages"})


def test_the_dossier_options_belong_to_the_construct_problem_trigger_only():
    for c in proto.DESIGN:
        cfg = proto.normalize_config(config(c, trigger="anomaly", diagnoses={"provisional": True, "pipeline_hints": True}), MODELS)
        assert cfg["diagnoses"]["provisional"] is False and cfg["diagnoses"]["pipeline_hints"] is False          # no misleading tag where nothing is handed over
        cfg = proto.normalize_config(config(c, trigger="construct_problem", diagnoses={"provisional": True, "pipeline_hints": True}), MODELS)
        assert cfg["diagnoses"]["provisional"] is True and cfg["diagnoses"]["pipeline_hints"] is True
    with pytest.raises(ValueError, match="pipeline_hints"):
        proto.normalize_config(config("C", trigger="construct_problem", diagnoses={"pipeline_hints": "yes"}), MODELS)
    proto.normalize_config(config("0", trigger="construct_problem", diagnoses={"limit": 9}), MODELS)      # 0 reads raw passages, not dossiers: max_pairs does not bind it


def test_every_condition_walks_the_reasoning_chain_and_never_skips_the_second_step():
    for chain in (proto.STEPS_V4_SOLO, proto.STEPS_V4_ENCOUNTER):
        n = len(chain)
        assert [proto.phase_for(r, 4, n) for r in range(1, 5)] == [0, 1, 2, 3]
        assert [proto.phase_for(r, 3, n) for r in range(1, 4)] == [0, 1, 3]
    assert [proto.chain_step(c, "opener_theme", r, 4)[0] for c in "0AB" for r in (1, 4)] == ["Search by the trigger", "Mechanism, question and predictions"] * 3
    titles = [proto.chain_step("A", None, r, 4)[0] for r in range(1, 5)]
    assert titles == ["Search by the trigger", "Puzzles and rival interpretations", "Select the puzzle", "Mechanism, question and predictions"]
    assert [proto.chain_step("C", "opener_theme", r, 4)[0] for r in range(1, 5)] == ["Opening theme", "Rival puzzles", "Compare and select", "Mechanism, question and predictions"]
    assert "ONE short sentence" in proto.chain_step("A", None, 4, 4)[1] and "do not commit to one yet" in proto.chain_step("0", None, 1, 4)[1]
    assert proto.chain_step("C", "constructs", 3, 4)[0] == "Compare the questions"                     # an auxiliary opening keeps its chain
    for condition in proto.DESIGN:
        text = proto.framing(condition, proto.layout(condition)[0], trigger="anomaly")
        assert proto.RULE_9 in text and proto.RULE_8_V4 in text and text.endswith(f"Prompt version {proto.PROMPT_VERSION_V4}.")


def test_every_agent_is_told_what_cto_and_tim_are_the_same_way_in_every_condition():
    for condition in proto.CONDITIONS:
        for agent in proto.layout(condition):
            text = proto.framing(condition, agent, trigger="anomaly")
            assert text.startswith(proto.CORPUS_CONTEXT) and "not job titles" in text
            assert all(name in text and scope in text for name, scope in proto.COMMUNITIES.values())
            assert text.count(proto.COMMUNITIES["CTO"][1]) == 1        # a specialist is not told its own domain twice


def test_condition_b_runs_one_specialist_alone_with_the_whole_budget_and_both_at_once_is_refused():
    default = proto.normalize_config(config("B"), MODELS)
    assert default["protocol"]["reasoner_choice"] == "cto" and default["protocol"]["reasoners"] == ["cto"] and set(default["agents"]) == {"cto"}    # when nobody says
    for name, other in (("cto", "tim"), ("tim", "cto")):
        one = proto.normalize_config(config("B", protocol={"reasoner": name}), MODELS)
        assert one["protocol"]["reasoner_choice"] == name and one["protocol"]["reasoners"] == [name] and set(one["agents"]) == {name}
        assert one["protocol"]["budget_split"] == "shared" and one["protocol"]["lead"] == name         # nothing to halve: it has the whole budget
        assert [a["key"] for a in proto.active_agents("B", one["protocol"])] == [name] and other not in one["agents"]
        assert proto.budgets_checksum(one) == proto.budgets_checksum(proto.normalize_config(config("A"), MODELS)) == proto.budgets_checksum(proto.normalize_config(config("C"), MODELS))
    proto.normalize_config(config("B", budgets=dict(proto.DEFAULT_BUDGETS, max_passages=61, max_llm_calls=61), protocol={"reasoner": "tim"}), MODELS)   # nothing is split: odd is fine
    proto.normalize_config(config("B", budgets={"max_total_tokens": 100_000}), MODELS)                       # the whole budget covers its final question
    with pytest.raises(ValueError, match="runs one specialist alone since 14 Oct 2026"):
        proto.normalize_config(config("B", protocol={"reasoner": "both"}), MODELS)
    with pytest.raises(ValueError, match="reasoner must be one of"):
        proto.normalize_config(config("B", protocol={"reasoner": "ghost"}), MODELS)
    for c in "0AC":
        with pytest.raises(ValueError, match="only be chosen in the isolated condition"):
            proto.normalize_config(config(c, protocol={"reasoner": "cto"}), MODELS)
        assert "reasoners" not in proto.normalize_config(config(c), MODELS)["protocol"]                  # the other conditions keep the shape they always had
    assert [a["key"] for a in proto.active_agents("B", {"rounds": 2})] == ["cto", "tim"]                 # an episode made before the choice existed ran with both


def test_rounds_of_the_legacy_diagnosis_chain_still_spread_and_end_on_refinement():
    assert [proto.step_for(r, 6) for r in range(1, 7)] == [0, 1, 2, 3, 4, 5]
    assert proto.step_for(1, 3) == 0 and proto.step_for(3, 3) == len(proto.STEPS) - 1 and proto.step_for(1, 1) == len(proto.STEPS) - 1
    assert proto.STEPS[0][0].startswith("Diagnosis") and proto.STEPS[-1][0] == "Refinement"


def test_categories_follow_the_two_decision_fields():
    assert proto.category_of("distinct", "jingle") == "jingle" and proto.category_of("equivalent", "jangle") == "jangle"
    assert proto.category_of("insufficient_evidence", "unresolved") == "unresolved"
    assert proto.category_of("related_but_distinct", "none") == "related_but_distinct" and proto.category_of("distinct", "none") == "no_fallacy"


def test_a_construct_problem_dossier_begins_with_the_reasons_and_not_with_a_score():
    side = {"label": "x", "paper": {"title": "T", "year": 2020}, "mentions": [], "measures": [], "attributes": [], "claims": []}
    dossier = {"a": side, "b": side, "signals": {"s_sem_op": 0.91}, "candidate_relation": {"same_label": True, "contrary_evidence": [{"quote": "q", "page": 2}]}}
    assert "taken by people" in proto.compact_dossier(dossier, 5000)[0]
    assert "NO PERSON HAS REVIEWED" in proto.compact_dossier(dossier, 5000, reviewed=False)[0] and "taken by people" not in proto.compact_dossier(dossier, 5000, reviewed=False)[0]
    text, _ = proto.compact_dossier(dossier, 5000, rationale=["The two definitions name different referents."], signals=False)
    body = json.loads(text)
    assert list(body)[0] == "reviewers_rationale" and "signals" not in body and "0.91" not in text and body["contrary_evidence"] == [{"quote": "q", "page": 2}]
    assert body["notice"] == proto.RATIONALE_NOTICE


TIMELY = {"why_now": "Adoption doubled since 2023 [E1].", "recent_evidence": "The latest study [E1] and a 2026 preprint [W1].",
          "already_answered": "Searched the corpus and the web; [W1] studies the main effect, not the boundary condition.", "check_scope": "Corpus to 2025 and the web."}
V4 = {"rq": "When does prior knowledge enable or constrain organizational novelty?", "trigger_evidence": "Firms with large patent stocks innovate less [E1].",
      "puzzle": "Knowledge is assumed to enable novelty.", "rival_interpretations": [{"interpretation": "Rigidity", "evidence": "[E1]"}, {"interpretation": "Selection", "evidence": "[E2]"}],
      "alternative_puzzles": [{"puzzle": "Size effects", "why_discarded": "Answered"}], "mechanism": "m", "predictions": [{"statement": "s", "would_support": "a", "would_contradict": "b"}],
      "theoretical_contribution": "c", "conceptual_signature": {k: "x" for k in proto.SIGNATURE_FIELDS}, "empirical_design": {k: "y" for k in proto.DESIGN_FIELDS},
      "boundary_conditions": [], "uncertainty": "u", "timeliness": TIMELY, "argument_steps": [], "verification_plan": {"starting_point": "p", "new_relation": "n"},
      "analyses": [], "sources": ["E1", "E2"]}


def test_a_question_keeps_what_the_agent_wrote_and_flags_what_cannot_be_trusted():
    good = {"rq": "Why?", "sources": ["E1", "P2"], "predictions": [{"statement": "s"}], "argument_steps": [{"step": "x", "claim": "c", "sources": ["E1"]}], "analyses": [{"status": "proposed"}],
            "timeliness": TIMELY}
    content, warnings = proto.validate_question(good, {"E1", "P2", "W1"}, years={"E1": 2025}, web=True)
    assert warnings == [] and content["rq"] == "Why?" and content["_validation"]["unknown_sources"] == []
    bad = {"rq": "Why?", "sources": ["E9"], "analyses": [{"status": "obtained"}], "predictions": []}
    content, warnings = proto.validate_question(bad, {"E1"})
    assert content["_validation"]["unknown_sources"] == ["E9"] and any("never retrieved" in w for w in warnings) and any("obtained" in w for w in warnings)
    assert any("No distinguishable prediction" in w for w in warnings)
    with pytest.raises(ValueError, match="no research question"):
        proto.validate_question({"rq": "  "}, set())


def test_a_v4_question_is_checked_for_rivals_alternatives_a_short_question_its_signature_and_a_shown_observation():
    known = {"E1", "E2", "W1"}
    assert proto.validate_question(V4, known, years={"E1": 2025}, web=True, trigger="anomaly")[1] == []
    weak = {**V4, "rq": " ".join(["word"] * 50), "rival_interpretations": V4["rival_interpretations"][:1], "alternative_puzzles": [],
            "conceptual_signature": {**V4["conceptual_signature"], "level": "", "mechanism": " "}, "trigger_evidence": "Firms innovate less."}
    warnings = proto.validate_question(weak, known, years={"E1": 2025}, web=True, trigger="phenomenon")[1]
    assert any("1 rival interpretation(s)" in w for w in warnings) and any("No alternative puzzle" in w for w in warnings)
    assert any("runs 50 words" in w for w in warnings) and any("incomplete (mechanism, level)" in w for w in warnings)
    assert any("observation is not shown" in w for w in warnings)
    assert not any("observation" in w for w in proto.validate_question(weak, known, years={"E1": 2025}, web=True, trigger="paradox")[1])     # only the observation triggers ask for it
    invented = proto.validate_question({**V4, "trigger_evidence": "As [E44] shows."}, known, years={"E1": 2025}, web=True)[0]
    assert "E44" in invented["_validation"]["unknown_sources"]                                    # an id invented where the question starts is caught too


def test_every_agent_must_argue_that_its_question_is_timely_and_not_already_answered():
    for condition in proto.DESIGN:
        agent = proto.layout(condition)[0]
        offline, online = proto.framing(condition, agent, today="2026-10-13", trigger="anomaly"), proto.framing(condition, agent, web=True, trigger="anomaly")
        assert proto.RULE_8 in offline and "Today is 2026-10-13." in offline and "covers only the corpus, whose papers end in 2025" in offline and "[W1]" not in offline
        assert proto.RULE_8_WEB in online and "[W1] for a web source" in online and "Today is" not in online
        assert proto.SYNTH_TIMELINESS in proto.synth_task(condition, "opener_theme") and proto.TIMELINESS_STEP in proto.chain_step(condition, "opener_theme", 4, 4)[1]
    assert "most recent studies on the same relation" in proto.chain_step("A", None, 2, 4)[1]
    assert set(proto.QUESTION_SCHEMA["properties"]["timeliness"]["required"]) == set(proto.TIMELINESS_FIELDS)
    known = {"E1", "E2", "W1"}
    base = {"rq": "Why?", "sources": ["E1"], "predictions": [{"statement": "s"}]}
    assert proto.validate_question(base, known)[1] == ["The question does not argue why it matters now or whether it has already been answered."]
    empty = proto.validate_question({**base, "timeliness": {k: "" for k in proto.TIMELINESS_FIELDS}}, known)[1]
    assert empty == ["The question does not argue why it matters now or whether it has already been answered."]
    uncited = proto.validate_question({**base, "timeliness": {**TIMELY, "why_now": "x", "recent_evidence": "y", "already_answered": "z"}}, known, web=True)[1]
    assert any("cites no evidence id" in w for w in uncited) and any("no web source supports" in w for w in uncited)
    old = {**TIMELY, "recent_evidence": "The latest study [E2].", "why_now": "x", "already_answered": "z"}
    warnings = proto.validate_question({**base, "timeliness": old}, known, years={"E2": 2019})[1]
    assert any("most recent study cited for timeliness is from 2019" in w for w in warnings) and any("against the corpus only" in w for w in warnings)
    assert not any("from 2019" in w for w in proto.validate_question({**base, "timeliness": {**old, "why_now": "[W1]"}}, known, years={"E2": 2019}, web=True)[1])      # a later web source covers it
    content, warnings = proto.validate_question({**base, "timeliness": {**TIMELY, "why_now": "[E7]"}}, known, years={"E1": 2025}, web=True)
    assert "E7" in content["_validation"]["unknown_sources"]              # an id invented in the timeliness block is caught like any other


def test_question_schema_is_strict_enough_for_structured_output():
    def check(node):
        if node.get("type") == "object":
            assert node["additionalProperties"] is False and set(node["required"]) == set(node["properties"])
            [check(v) for v in node["properties"].values()]
        if node.get("type") == "array":
            check(node["items"])
    check(proto.QUESTION_SCHEMA)
    props = proto.QUESTION_SCHEMA["properties"]
    assert {"trigger_evidence", "puzzle", "rival_interpretations", "alternative_puzzles", "theoretical_contribution", "conceptual_signature", "empirical_design"} <= set(props)
    assert "diagnosis_premise" not in props and "unit_of_analysis" not in props                     # the empirical choices live in empirical_design, not beside the question
    assert set(props["conceptual_signature"]["required"]) == set(proto.SIGNATURE_FIELDS) and set(props["empirical_design"]["required"]) == set(proto.DESIGN_FIELDS)
    for condition in proto.DESIGN:
        assert proto.synth_task(condition, "opener_theme").startswith(proto.SYNTH_V4)
    assert proto.SYNTH_OBSERVATION in proto.synth_task("0", trigger="anomaly") and proto.SYNTH_OBSERVATION not in proto.synth_task("0", trigger="paradox")
    assert proto.SYNTH_DOSSIER in proto.synth_task("A", trigger="construct_problem", dossiers=True) and proto.SYNTH_CROSS in proto.synth_task("B", trigger="translation")


def test_dossier_for_an_agent_is_shortened_without_losing_its_shape():
    side = lambda label: {"label": label, "paper": {"title": "T", "year": 2020}, "mentions": [{"definition": "d" * 400, "quote": "q" * 400, "page": 3}] * 3,
                          "measures": [{"name": "m", "items": "i" * 300, "scale": "s", "page": 4}] * 5, "attributes": [{"kind": "level", "basis": "extracted", "value": "v", "page": 1}] * 12,
                          "claims": [{"as": "subject", "other_label": "o", "relation": "r", "sign": "+", "evidence_status": "tested", "page": 2}] * 12}
    dossier = {"a": side("A"), "b": side("B"), "signals": {"s_sem_op": None, "unavailable": ["s_sem_op"]}, "candidate_relation": {"same_label": False}}
    text, cut = proto.compact_dossier(dossier, 6000)
    assert not cut and json.loads(text)["a"]["label"] == "A" and json.loads(text)["signals"]["s_sem_op"] is None
    text, cut = proto.compact_dossier(dossier, 1800)
    assert len(text) <= 1800 + 20 and (not cut or text.endswith("[truncated]"))


# --- The OpenAI provider against a stand-in client ---------------------------------------------------------------------------------------

class FakeResponses:
    def __init__(self, scripted):
        self.scripted, self.calls = list(scripted), []

    def create(self, **kwargs):
        self.calls.append(kwargs)
        return self.scripted.pop(0)


def fake_response(output, text="", status="completed", usage=(100, 20, 5)):
    return SimpleNamespace(output=output, output_text=text, status=status, usage=SimpleNamespace(input_tokens=usage[0], output_tokens=usage[1], output_tokens_details=SimpleNamespace(reasoning_tokens=usage[2])))


def function_call(name, arguments, call_id="c1"):
    item = SimpleNamespace(type="function_call", name=name, arguments=json.dumps(arguments), call_id=call_id)
    item.model_dump = lambda exclude_none=True: {"type": "function_call", "name": name, "arguments": json.dumps(arguments), "call_id": call_id}
    return item


def reasoning_item(text):
    item = SimpleNamespace(type="reasoning", summary=[SimpleNamespace(text=text)])
    item.model_dump = lambda exclude_none=True: {"type": "reasoning", "summary": [{"text": text}], "encrypted_content": "x"}
    return item


def make_request(**kwargs):
    base = dict(model="gpt-6-luna", system="sys", prompt="go", task="reflect", tools=[ToolSpec("search_corpus", "d", {"type": "object", "properties": {}})],
                run_tool=lambda name, args: json.dumps({"hits": []}), max_output_tokens=500, max_tool_rounds=2)
    base.update(kwargs)
    return TurnRequest(**base)


def test_openai_provider_runs_the_tool_loop_and_accounts_for_every_call():
    client = SimpleNamespace(responses=FakeResponses([
        fake_response([reasoning_item("thinking about it"), function_call("search_corpus", {"query": "q"})]),
        fake_response([], text="final notes")]))
    seen, usages = [], []
    result = OpenAIProvider(client).run(make_request(run_tool=lambda n, a: seen.append((n, a)) or "{}", on_usage=usages.append, reasoning_effort="high", temperature=0.3))
    assert result.text == "final notes" and result.llm_calls == 2 and seen == [("search_corpus", {"query": "q"})]
    assert (result.input_tokens, result.output_tokens, result.reasoning_tokens) == (200, 40, 10) and len(usages) == 2
    assert "thinking about it" in result.reasoning_summary
    first, second = client.responses.calls
    assert first["store"] is False and first["reasoning"] == {"effort": "high", "summary": "auto"} and first["include"] == ["reasoning.encrypted_content"] and first["temperature"] == 0.3
    assert first["tools"][0]["name"] == "search_corpus" and first["max_output_tokens"] == 500
    assert [i.get("type") for i in second["input"][1:]] == ["reasoning", "function_call", "function_call_output"]    # the reasoning travels with the call it led to


def test_openai_provider_sends_no_reasoning_options_when_none_is_asked_and_adds_web_search_on_request():
    client = SimpleNamespace(responses=FakeResponses([fake_response([], text="ok")]))
    OpenAIProvider(client).run(make_request(web_search=True))
    call = client.responses.calls[0]
    assert "reasoning" not in call and "include" not in call and "temperature" not in call and {"type": "web_search"} in call["tools"]


def test_openai_provider_forces_text_on_the_last_round_and_passes_a_strict_schema():
    client = SimpleNamespace(responses=FakeResponses([fake_response([function_call("search_corpus", {"query": "a"})]), fake_response([function_call("search_corpus", {"query": "b"}, "c2")]),
                                                      fake_response([], text="{}")]))
    schema = {"name": "q", "schema": {"type": "object"}}
    OpenAIProvider(client).run(make_request(max_tool_rounds=2, json_schema=schema))
    calls = client.responses.calls
    assert len(calls) == 3 and "tools" in calls[0] and "tools" in calls[1] and "tools" not in calls[2]
    assert calls[2]["text"]["format"] == {"type": "json_schema", "name": "q", "strict": True, "schema": {"type": "object"}}


def test_openai_provider_stops_before_a_call_when_the_gate_says_so_and_reports_cut_off_answers():
    client = SimpleNamespace(responses=FakeResponses([fake_response([], text="never")]))
    def closed():
        raise BudgetExceeded("tokens")
    with pytest.raises(BudgetExceeded):
        OpenAIProvider(client).run(make_request(gate=closed))
    assert client.responses.calls == []
    cut = SimpleNamespace(responses=FakeResponses([fake_response([], text="half", status="incomplete")]))
    assert OpenAIProvider(cut).run(make_request()).finish == "incomplete"


def test_openai_provider_survives_a_tool_call_with_unreadable_arguments():
    broken = SimpleNamespace(type="function_call", name="search_corpus", arguments="{not json", call_id="c1")
    broken.model_dump = lambda exclude_none=True: {"type": "function_call", "name": "search_corpus", "arguments": "{not json", "call_id": "c1"}
    client = SimpleNamespace(responses=FakeResponses([fake_response([broken]), fake_response([], text="done")]))
    result = OpenAIProvider(client).run(make_request())
    assert result.text == "done" and "bad tool call" in client.responses.calls[1]["input"][-1]["output"]


class RateLimitError(Exception):        # same name as the SDK's: the provider recognises it by name
    def __init__(self, message="Error code: 429 - rate limit reached", retry_after=None, code=None):
        super().__init__(message)
        self.code = code
        self.response = SimpleNamespace(headers={"retry-after": retry_after} if retry_after is not None else {})


class FlakyResponses(FakeResponses):
    """Raises the scripted errors first, then answers."""

    def create(self, **kwargs):
        self.calls.append(kwargs)
        item = self.scripted.pop(0)
        if isinstance(item, Exception):
            raise item
        return item


def test_openai_provider_waits_out_a_rate_limit_and_keeps_the_turn():
    client = SimpleNamespace(responses=FlakyResponses([RateLimitError(), RateLimitError(retry_after="3"), fake_response([], text="ok")]))
    slept, usages = [], []
    result = OpenAIProvider(client, sleep=slept.append).run(make_request(on_usage=usages.append))
    assert result.text == "ok" and result.llm_calls == 1 and len(usages) == 1 and len(client.responses.calls) == 3    # a refused call is not a call of the turn
    assert sum(slept) == 15 + 3                                                                                       # the schedule, then the server's Retry-After


def test_openai_provider_does_not_wait_for_an_exhausted_quota_or_other_errors():
    for error in (RateLimitError("Error code: 429 - insufficient_quota"), RateLimitError("x", code="insufficient_quota"), RuntimeError("boom")):
        client = SimpleNamespace(responses=FlakyResponses([error, fake_response([], text="never")]))
        slept = []
        with pytest.raises(type(error)):
            OpenAIProvider(client, sleep=slept.append).run(make_request())
        assert slept == [] and len(client.responses.calls) == 1


def test_openai_provider_gives_up_after_the_last_wait_and_a_stop_ends_a_wait():
    client = SimpleNamespace(responses=FlakyResponses([RateLimitError() for _ in range(5)]))
    slept = []
    with pytest.raises(RateLimitError):
        OpenAIProvider(client, sleep=slept.append).run(make_request())
    assert sum(slept) == sum(forum_providers.RATE_LIMIT_WAITS) and len(client.responses.calls) == len(forum_providers.RATE_LIMIT_WAITS) + 1
    ticks = []
    def gate():
        ticks.append(1)
        if len(ticks) > 3:                  # the first check is the one before the call
            raise Cancelled()
    stopped = SimpleNamespace(responses=FlakyResponses([RateLimitError(), fake_response([], text="never")]))
    slept = []
    with pytest.raises(Cancelled):
        OpenAIProvider(stopped, sleep=slept.append).run(make_request(gate=gate))
    assert len(slept) == 2 and len(stopped.responses.calls) == 1


# --- Whole episodes on the mock world -----------------------------------------------------------------------------------------------------

needs_db = pytest.mark.skipif(not os.getenv("DATABASE_URL"), reason="Set local PostgreSQL environment to run integration tests")


@pytest.fixture(scope="module")
def forum_world(tmp_path_factory):
    if not os.getenv("DATABASE_URL"):
        pytest.skip("Set local PostgreSQL environment to run integration tests")
    from research0 import documents, embedding_jobs, extraction, mock_world, pairs, quality, release, signals
    from research0.demo import _planted_gold
    from research0.embeddings import HashEmbedder
    world = mock_world.create_world(n=200, seed=31)
    patch = pytest.MonkeyPatch()
    try:
        directory = tmp_path_factory.mktemp("forum") / "extraction"
        with world.connect() as c:
            ingestion_id, _ = documents.ingest(c, world.run_id, world.store)
            extraction.prepare(c, ingestion_id, directory)
            owner = {str(a): b for a, b in c.execute("SELECT version_id,canonical_id FROM paper_version")}
            lines = (directory / "requests.jsonl").read_text(encoding="utf-8").splitlines()
            (directory / "batch_output.jsonl").write_text("\n".join(mock_world.scripted_extraction_output(world, lines, owner)) + "\n", encoding="utf-8")
            extraction_run_id = uuid.UUID(extraction.collect_file(c, directory)["extraction_run_id"])
            documents.embed_chunks(c, ingestion_id, HashEmbedder())
            embedding_jobs.embed_records(c, extraction_run_id, HashEmbedder())
            pre = mock_world.make_preassignment(c, world, miss_rate=0.0)
            signal_run, _, _ = signals.build(c, extraction_run_id, HashEmbedder())
            signals.build_dossiers(c, signal_run)
            planted = _planted_gold(c, signal_run)
            for pair_id, kind in planted["decisions"].items():
                for who in ("Ana", "Ben"):
                    mock_world.decide_truth(c, pair_id, who, kind)
            spare = [r[0] for r in c.execute("SELECT pair_id FROM candidate_pair WHERE signal_run_id=%s ORDER BY pair_id", (signal_run,)) if r[0] not in planted["decisions"]][:1]
            for pair_id in spare:
                for who in ("Ana", "Ben"):
                    pairs.decide(c, pair_id, who, "unresolved", "cannot tell from the dossier", UNCLEAR)
            patch.setattr(quality, "evaluate", lambda *a, **k: [quality.verdict(g["id"], quality.passing_value(g)) for g in quality.GATES])
            release_id, *_ = release.build(c, world.run_id, "forum-test", pre, None, ingestion_id, extraction_run_id, signal_run)
            release.sign_off(c, release_id, "Ana")
            c.commit()
        yield SimpleNamespace(world=world, release_id=release_id, ingestion_id=ingestion_id, pre=pre, signal_run=signal_run)
    finally:
        patch.undo()
        world.drop()


@pytest.fixture
def fw(forum_world):
    yield forum_world
    with forum_world.world.connect() as c:           # episodes are committed (the runner uses its own connections): clear them between tests
        for table in ("forum_proposal_score", "forum_proposal_evaluation", "forum_proposal", "question", "forum_analysis_file", "forum_analysis", "forum_message", "forum_agent", "forum_episode"):
            c.execute(f"DELETE FROM {table}")
        c.execute("DELETE FROM retrieval_hit")
        c.execute("DELETE FROM retrieval_run")
        c.execute("DELETE FROM agent_profile WHERE agent_id LIKE 'forum:%'")
        c.commit()


def make_episode(fw, condition, **overrides):
    overrides.setdefault("trigger", "missing_mechanism")        # a fixed trigger with no extra demand, so that a test does not depend on what the seed draws
    with fw.world.connect() as c:
        return forum.create_episode(c, fw.release_id, config(condition, **overrides), "Tester")


def run_episode(fw, episode_id, provider=None, stop=None):
    runner = forum.Runner(episode_id, provider or ScriptedProvider(), app_url=fw.world.url, stop=stop)
    return runner.run()


def messages(fw, episode_id, **where):
    with fw.world.connect() as c:
        rows = c.execute("SELECT seq,agent_key,kind,audience,round,content,data FROM forum_message WHERE episode_id=%s ORDER BY seq", (episode_id,)).fetchall()
    keys = ("seq", "agent", "kind", "audience", "round", "content", "data")
    return [dict(zip(keys, r)) for r in rows if all(dict(zip(keys, r))[k] == v for k, v in where.items())]


@needs_db
def test_the_open_encounter_with_a_construct_problem_hands_over_reasons_not_scores_and_archives_trigger_opener_and_theme(fw):
    episode = make_episode(fw, "C", trigger="construct_problem", protocol={"rounds": 2, "lead": "cto"}, seed=11, focus="digital readiness and engagement")
    assert run_episode(fw, episode) == "completed"
    log = messages(fw, episode)
    kinds = [m["kind"] for m in log]
    assert kinds[0] == "note" and kinds.count("question") == 1
    trigger = next(m for m in log if m["kind"] == "system" and m["data"].get("role") == "trigger")
    assert trigger["data"]["trigger"] == "construct_problem" and trigger["audience"] == "all" and "Construct problem" in trigger["content"]
    premises = next(m for m in log if m["kind"] == "system" and m["data"].get("role") == "premises")
    assert premises["data"]["pairs"] and "decided by human reviewers" in premises["content"] and "Nothing tells you to separate or to integrate" in premises["content"]
    assert "reviewers_rationale" in premises["content"] and "s_sem_op" not in premises["content"]               # the reasons of the decision, not the similarity scores
    opening = next(m for m in log if m["kind"] == "system" and m["data"].get("role") == "opening")
    posts = [m for m in log if m["kind"] == "message" and m["data"].get("step") != "Proposal critique"]
    assert opening["data"]["opener"] == "cto" and log.index(opening) < log.index(posts[0])
    assert [m["agent"] for m in posts] == ["cto", "tim", "cto", "tim"] and all(m["audience"] == "all" for m in posts)       # the opener first, every round
    assert all(m["audience"] in ("cto", "tim") for m in log if m["kind"] in ("reflection", "tool_call", "tool_result"))   # reasoning stays private
    steps = [m["data"]["step"] for m in log if m["kind"] == "system" and m["data"].get("role") == "moderator"]
    assert steps == [proto.STEPS_V4_ENCOUNTER[0][0], proto.STEPS_V4_ENCOUNTER[-1][0]]
    with fw.world.connect() as c:
        row = c.execute("SELECT agents,rq,content,evidence,evaluation,ideation_trigger,numeric_seed,opener,opening_theme,alternative_puzzles,conceptual_signature "
                        "FROM question WHERE episode_id=%s", (episode,)).fetchone()
        assert row[0] == ["cto", "tim"] and row[1].startswith("[SIMULATION]") and row[4] == {}
        assert row[2]["_validation"]["unknown_sources"] == [] and row[3] and all(s["verified"] for s in row[3])
        assert (row[5], row[6], row[7]) == ("construct_problem", 11, "cto") and row[8]["mode"] == "opener_theme" and row[8]["agent"] == "cto" and row[8]["text"] == posts[0]["content"]
        assert row[9] == row[2]["alternative_puzzles"] and set(row[10]) == set(proto.SIGNATURE_FIELDS) and row[2]["_validation"]["trigger"] == "construct_problem"
        assert c.execute("SELECT status FROM forum_episode WHERE episode_id=%s", (episode,)).fetchone()[0] == "completed"
        tools = {m["data"]["tool"] for m in log if m["kind"] == "tool_call"}
    assert "list_diagnoses" in tools                                                                                   # the dossiers can be listed and read in the reflection


@needs_db
def test_every_retrieval_is_logged_under_the_episodes_agent_profile_and_follows_the_community_view(fw):
    episode = make_episode(fw, "C", protocol={"rounds": 1})
    assert run_episode(fw, episode) == "completed"
    with fw.world.connect() as c:
        runs = c.execute("SELECT agent,scope FROM retrieval_run WHERE agent LIKE %s ORDER BY created_at", (f"forum:{episode}:%",)).fetchall()
        assert runs and {a for a, _ in runs} == {f"forum:{episode}:cto", f"forum:{episode}:tim"}
        assert all(s["community"] == a.rsplit(":", 1)[1].upper() for a, s in runs)                     # a specialist reads only its community's view
        assert c.execute("SELECT count(*) FROM retrieval_hit h JOIN retrieval_run r USING (retrieval_run_id) WHERE r.agent LIKE %s", (f"forum:{episode}:%",)).fetchone()[0] > 0


@needs_db
def test_a_b_episode_made_with_both_specialists_before_14_october_can_be_read_but_not_run(fw):
    episode = make_episode(fw, "B", protocol={"rounds": 1})
    with fw.world.connect() as c:          # what such an episode looks like: both specialists in its frozen protocol, each with an agent row
        c.execute("""UPDATE forum_episode SET config = jsonb_set(jsonb_set(config, '{protocol,reasoners}', '["cto","tim"]'), '{protocol,reasoner_choice}', '"both"') WHERE episode_id=%s""", (episode,))
        c.execute("INSERT INTO forum_agent(episode_id,agent_key,label,community,role,profile_id,settings) SELECT episode_id,'tim','TIM specialist','TIM',role,%s,settings "
                  "FROM forum_agent WHERE episode_id=%s", (forum.profile_id(episode, "tim"), episode))
        c.commit()
        assert [a["key"] for a in forum.episode_detail(c, episode)["agents"]] == ["cto", "tim"]
    with pytest.raises(ValueError, match="made with both specialists.*can be read, not run"):
        forum.Runner(episode, ScriptedProvider(), app_url=fw.world.url)


@needs_db
def test_in_condition_b_only_the_chosen_specialist_takes_part_and_spends_the_whole_budget(fw):
    for name in ("cto", "tim"):
        episode = make_episode(fw, "B", protocol={"rounds": 2, "reasoner": name})
        assert run_episode(fw, episode) == "completed"
        log = messages(fw, episode)
        assert {m["agent"] for m in log if m["agent"]} == {name}                                          # the other specialist never speaks, searches or writes
        assert [m["agent"] for m in log if m["kind"] == "question"] == [name]
        with fw.world.connect() as c:
            assert [r[0] for r in c.execute("SELECT agent_key FROM forum_agent WHERE episode_id=%s", (episode,)).fetchall()] == [name]
            runs = c.execute("SELECT agent,scope FROM retrieval_run WHERE agent LIKE %s", (f"forum:{episode}:%",)).fetchall()
            assert runs and {a for a, _ in runs} == {f"forum:{episode}:{name}"} and all(s["community"] == name.upper() for _, s in runs)
            assert c.execute("SELECT agents FROM question WHERE episode_id=%s", (episode,)).fetchone()[0] == [name]
            assert set(forum.usage(c, episode)["by_agent"]) == {name}
            detail = forum.episode_detail(c, episode)
        assert [a["key"] for a in detail["agents"]] == [name] and detail["config"]["protocol"]["reasoners"] == [name]
        runner = forum.Runner(episode, ScriptedProvider(), app_url=fw.world.url)
        try:
            assert runner.caps["tokens"] == runner.budgets["max_total_tokens"] and runner.caps["passages"] == runner.budgets["max_passages"]
            assert [a["key"] for a in runner.agents] == [name]
        finally:
            runner.agent_con.close(); runner.app.close()
    with pytest.raises(ValueError, match="Unknown agent"):          # settings for the specialist that does not take part are a mistake, not something to ignore
        with fw.world.connect() as c:
            forum.create_episode(c, fw.release_id, config("B", protocol={"reasoner": "cto"}, agents={"tim": {}}), "Tester")


@needs_db
def test_condition_c_is_a_conversation_that_opens_on_a_random_theme_and_swaps_random_items(fw):
    def run_c(seed, **extra):
        episode = make_episode(fw, "C", protocol={"rounds": 2, "lead": "cto", **extra.pop("protocol", {})}, seed=seed, encounter={"theme_mode": "random_passages", "items_per_round": 2}, **extra)
        assert run_episode(fw, episode) == "completed"
        return episode, messages(fw, episode)

    episode, log = run_c(77)
    theme = next(m for m in log if m["kind"] == "system" and m["data"].get("role") == "theme")
    assert theme["audience"] == "all" and theme["data"]["opener"] == "cto" and theme["data"]["from_community"] == "CTO" and theme["data"]["hits"]
    assert not [m for m in log if m["kind"] == "system" and m["data"].get("role") == "premises"]                          # no diagnosis is handed over
    posts = [m for m in log if m["kind"] == "message" and m["data"].get("step") != "Proposal critique"]
    assert [m["agent"] for m in posts] == ["cto", "tim", "cto", "tim"] and all(m["audience"] == "all" for m in posts)        # they answer each other, the opener first
    assert log.index(theme) < log.index(posts[0])
    swaps = [m for m in log if m["kind"] == "encounter"]
    assert all(m["audience"] == "all" for m in swaps)                                                                      # the other agent sees what is brought in
    assert {(m["agent"], m["data"]["from_community"]) for m in swaps} == {("cto", "CTO"), ("tim", "TIM")}                   # each brings items of its OWN community
    assert [(m["agent"], m["round"]) for m in swaps] == [("tim", 1), ("cto", 2), ("tim", 2)]                               # the opener's first items are the opening theme
    kinds = {h["section"].split()[0] for m in swaps for h in m["data"]["hits"]} | {h["section"].split()[0] for h in theme["data"]["hits"]}
    assert "construct" in kinds and len(kinds) > 1                                                                         # passages and construct records are both drawn
    with fw.world.connect() as c:
        row = c.execute("SELECT agents FROM question WHERE episode_id=%s", (episode,)).fetchone()
        assert row[0] == ["cto", "tim"]                                                                                    # one joint question
    shape = lambda lg: [(m["kind"], m["agent"], [h["title"] for h in m["data"].get("hits", [])]) for m in lg if m["kind"] in ("encounter", "system")]
    assert shape(log) == shape(run_c(77)[1]) and shape(log) != shape(run_c(78)[1])                                          # the draw follows the seed
    opener_tim = run_c(77, protocol={"lead": "tim"})[1]
    assert [m["agent"] for m in opener_tim if m["kind"] == "message" and m["data"].get("step") != "Proposal critique"][0] == "tim"
    assert next(m for m in opener_tim if m["data"].get("role") == "theme")["data"]["from_community"] == "TIM"


@needs_db
def test_condition_c_retrieved_theme_opens_on_a_starting_point_and_its_closest_studies_and_the_seed_picks_the_opener(fw):
    def run_c(seed, **extra):
        episode = make_episode(fw, "C", protocol={"rounds": 2, **extra.pop("protocol", {})}, seed=seed, encounter={"theme_mode": "rag_cluster"}, **extra)
        assert run_episode(fw, episode) == "completed"
        with fw.world.connect() as c:
            cfg = c.execute("SELECT config FROM forum_episode WHERE episode_id=%s", (episode,)).fetchone()[0]
        return episode, cfg, messages(fw, episode)

    episode, cfg, log = run_c(77)
    lead = cfg["protocol"]["lead"]
    assert cfg["protocol"]["lead_choice"] == "random" and cfg["encounter"]["theme_mode"] == "rag_cluster" and cfg["encounter"]["items_per_round"] == 0
    theme = next(m for m in log if m["kind"] == "system" and m["data"].get("role") == "theme")
    hits = theme["data"]["hits"]
    assert theme["data"]["mode"] == "rag_cluster" and theme["data"]["opener"] == lead and theme["data"]["from_community"] == lead.upper() and theme["audience"] == "all"
    assert 1 < len(hits) <= 5 and theme["data"]["starting_point"] == hits[0]["id"]                      # a starting point and the studies closest to it
    assert len({h["title"] for h in hits[1:]}) == len(hits[1:]) and theme["data"]["counts"]["retrievals"] == 1       # at most one passage per paper, and the search was counted
    assert not [m for m in log if m["kind"] == "encounter"]                                              # random items per round are off after a retrieved theme
    posts = [m for m in log if m["kind"] == "message" and m["data"].get("step") != "Proposal critique"]
    assert [m["agent"] for m in posts] == [lead, "tim" if lead == "cto" else "cto"] * 2 and log.index(theme) < log.index(posts[0])
    steps = [m["data"]["step"] for m in log if m["kind"] == "system" and m["data"].get("role") == "moderator"]
    assert steps[0] == proto.STEPS_THEME[0][0] and steps[-1] == "Refinement"
    with fw.world.connect() as c:
        assert c.execute("SELECT agents FROM question WHERE episode_id=%s", (episode,)).fetchone()[0] == ["cto", "tim"]
    shape = lambda lg: [[h["title"] for h in m["data"]["hits"]] for m in lg if m["kind"] == "system" and m["data"].get("role") == "theme"]
    assert shape(log) == shape(run_c(77)[2])                                                            # the seed fixes the starting point and so the studies
    pinned = {name: run_c(77, protocol={"lead": name}) for name in ("cto", "tim")}                       # the person can choose who opens
    for name, (_, pcfg, plog) in pinned.items():
        ptheme = next(m for m in plog if m["kind"] == "system" and m["data"].get("role") == "theme")
        assert pcfg["protocol"]["lead_choice"] == name and ptheme["data"]["opener"] == name and ptheme["data"]["from_community"] == name.upper()
        assert [m["agent"] for m in plog if m["kind"] == "message" and m["data"].get("step") != "Proposal critique"][0] == name


@needs_db
def test_condition_c_can_still_open_on_random_passages_with_a_random_opener(fw):
    episode = make_episode(fw, "C", protocol={"rounds": 1}, seed=5, encounter={"theme_mode": "random_passages"})
    assert run_episode(fw, episode) == "completed"
    with fw.world.connect() as c:
        cfg = c.execute("SELECT config FROM forum_episode WHERE episode_id=%s", (episode,)).fetchone()[0]
    theme = next(m for m in messages(fw, episode) if m["kind"] == "system" and m["data"].get("role") == "theme")
    assert cfg["encounter"]["items_per_round"] == 3 and theme["data"]["mode"] == "random_passages" and theme["data"]["opener"] == cfg["protocol"]["lead"]


@needs_db
def test_conditions_0_and_a_read_the_union_and_only_condition_0_lacks_the_database(fw):
    base = make_episode(fw, "0", protocol={"rounds": 1})
    assert run_episode(fw, base) == "completed"
    generalist = make_episode(fw, "A", protocol={"rounds": 1})
    assert run_episode(fw, generalist) == "completed"
    with fw.world.connect() as c:
        for episode in (base, generalist):
            scopes = [s for (s,) in c.execute("SELECT scope FROM retrieval_run WHERE agent LIKE %s", (f"forum:{episode}:%",))]
            assert scopes and all(s["community"] == ["CTO", "TIM"] for s in scopes)
    tools = lambda e: {m["data"]["tool"] for m in messages(fw, e, kind="tool_call")}
    assert tools(base) == {"search_corpus"} and {"construct_overview", "construct_network"} <= tools(generalist) and "list_diagnoses" not in tools(generalist)
    assert not [m for m in messages(fw, base) if m["data"].get("pairs") or m["data"].get("diagnoses")]


@needs_db
def test_a_construct_problem_mixes_every_kind_of_relationship_and_the_same_seed_gives_every_condition_the_same_papers(fw):
    """Matched opportunities: with the same seed A and C get the same dossiers, and condition 0 gets the raw passages of the same papers, with no label or relationship."""
    episodes = {c: make_episode(fw, c, trigger="construct_problem", protocol={"rounds": 1}, diagnoses={"limit": 4}, seed=21) for c in ("0", "A", "C")}
    for episode in episodes.values():
        assert run_episode(fw, episode) == "completed"
    premises = {c: next(m for m in messages(fw, e, kind="system") if m["data"].get("role") == "premises") for c, e in episodes.items()}
    cards = premises["C"]["data"]["pairs"]
    assert {c["category"] for c in cards} >= {"conceptual_jingle", "conceptual_jangle", "unresolved"} and len(cards) == 4                    # every kind, in turn: nothing says separate or integrate
    with fw.world.connect() as c:
        by_id = {p["id"]: p["pair_id"] for p in forum.diagnosis_pool(c, fw.release_id)}
    pair_ids = lambda cs: sorted(by_id[x["id"]] for x in cs)
    assert pair_ids(premises["A"]["data"]["pairs"]) == pair_ids(cards)
    raw = premises["0"]
    assert raw["data"]["matched"] and sorted(raw["data"]["pair_ids"]) == pair_ids(cards) and raw["data"]["hits"]
    assert "pairs" not in raw["data"] and not any(x["label_a"] in raw["content"].split("\n\n")[0] for x in cards)      # raw evidence only: no label, no decision
    assert all(h["id"].startswith("E") for h in raw["data"]["hits"]) and not {"list_diagnoses", "read_dossier"} & {m["data"]["tool"] for m in messages(fw, episodes["0"], kind="tool_call")}


@needs_db
def test_in_b_the_specialist_gets_the_construct_problems_of_its_own_community_in_its_own_feed(fw):
    episode = make_episode(fw, "B", trigger="construct_problem", protocol={"rounds": 1, "reasoner": "tim"}, diagnoses={"limit": 2})
    assert run_episode(fw, episode) == "completed"
    handed = [m for m in messages(fw, episode, kind="system") if m["data"].get("role") == "premises"]
    assert [m["audience"] for m in handed] == ["tim"]
    runner = forum.Runner(episode, ScriptedProvider(), app_url=fw.world.url)
    try:
        runner.prepare()
        for m in handed:
            assert {x["id"] for x in m["data"]["pairs"]} <= set(runner.visible[m["audience"]])          # only pairs that touch its own community
        assert runner.premises_of("tim") == {x["id"] for m in handed for x in m["data"]["pairs"]}
    finally:
        runner.agent_con.close(); runner.app.close()


@needs_db
def test_the_budget_stops_the_rounds_but_the_question_is_still_written(fw):
    episode = make_episode(fw, "C", protocol={"rounds": 4}, budgets={"max_llm_calls": 4}, proposal={"enabled": False})
    assert run_episode(fw, episode) == "budget_exhausted"
    log = messages(fw, episode)
    assert any(m["kind"] == "note" and "limit on model calls" in m["content"] for m in log) and [m for m in log if m["kind"] == "question"]
    with fw.world.connect() as c:
        status, detail = c.execute("SELECT status,status_detail FROM forum_episode WHERE episode_id=%s", (episode,)).fetchone()
        assert status == "budget_exhausted" and "model calls" in detail
        assert forum.usage(c, episode)["llm_calls"] <= 4


@needs_db
def test_passage_and_search_budgets_are_enforced_and_counted(fw):
    episode = make_episode(fw, "A", protocol={"rounds": 3}, budgets={"max_passages": 3, "max_retrieval_calls": 2})
    assert run_episode(fw, episode) == "completed"
    with fw.world.connect() as c:
        used = forum.usage(c, episode)
    assert used["passages"] <= 3 and used["retrievals"] <= 2
    refused = [m for m in messages(fw, episode, kind="tool_result") if m["data"].get("error")]
    assert refused and "budget" in refused[0]["data"]["error"]


@needs_db
def test_a_stop_request_ends_the_episode_as_stopped_without_a_question(fw):
    episode = make_episode(fw, "A", protocol={"rounds": 2})
    stop = threading.Event()
    stop.set()
    assert run_episode(fw, episode, stop=stop) == "stopped"
    assert not messages(fw, episode, kind="question")
    with fw.world.connect() as c:
        assert c.execute("SELECT status,status_detail FROM forum_episode WHERE episode_id=%s", (episode,)).fetchone() == ("stopped", "Stopped by a person")


class InventingProvider(ScriptedProvider):
    """Cites evidence nobody retrieved, in the public message and in the final question."""
    def run(self, request):
        result = super().run(request)
        if request.task == "speak":
            result.text += " See also [E99]."
        if request.task == "synthesize":
            data = json.loads(result.text)
            data["sources"] = data["sources"] + ["E99"]
            data["analyses"] = [{"status": "obtained", "description": "d", "data": "x", "limitations": "none"}]
            result.text = json.dumps(data)
        return result


@needs_db
def test_citations_that_do_not_exist_are_flagged_not_hidden(fw):
    episode = make_episode(fw, "C", protocol={"rounds": 1})
    assert run_episode(fw, episode, InventingProvider()) == "completed"
    post = messages(fw, episode, kind="message")[0]
    assert {"id": "E99", "valid": False} in post["data"]["citations"]
    with fw.world.connect() as c:
        content, evidence = c.execute("SELECT content,evidence FROM question WHERE episode_id=%s", (episode,)).fetchone()
    assert content["_validation"]["unknown_sources"] == ["E99"] and any("obtained" in w for w in content["_validation"]["warnings"])
    assert {"id": "E99", "kind": "unknown", "verified": False} in evidence


class CutOffProvider(ScriptedProvider):
    def run(self, request):
        result = super().run(request)
        if request.task == "synthesize":
            result.finish, result.text = "incomplete", result.text[:40]
        return result


@needs_db
def test_a_truncated_question_is_not_stored_and_the_episode_says_it_failed(fw):
    episode = make_episode(fw, "A", protocol={"rounds": 1})
    assert run_episode(fw, episode, CutOffProvider()) == "failed"
    assert not messages(fw, episode, kind="question")
    with fw.world.connect() as c:
        assert "No valid research question" in c.execute("SELECT status_detail FROM forum_episode WHERE episode_id=%s", (episode,)).fetchone()[0]
        assert c.execute("SELECT count(*) FROM question").fetchone()[0] == 0


@needs_db
def test_an_unvalidated_release_can_host_an_episode_but_it_is_tagged_everywhere_and_the_embedding_model_must_have_vectors(fw):
    from research0 import release
    blocked = None
    try:
        with fw.world.connect() as c:
            blocked, *_ = release.build(c, fw.world.run_id, "unsigned-for-forum", fw.pre, None, fw.ingestion_id)
            c.commit()
            assert release.status(c, blocked) == "candidate"          # gates pass in this module, nobody signed it off: not validated
            episode = forum.create_episode(c, blocked, config("0", protocol={"rounds": 1}, title="On an unsigned release"), "Tester")
            stored = c.execute("SELECT config FROM forum_episode WHERE episode_id=%s", (episode,)).fetchone()[0]
            assert stored["release"] == {"label": "unsigned-for-forum", "status": "candidate", "validated": False, "community_basis": "batch_v2"}
            ok = forum.create_episode(c, fw.release_id, config("0", protocol={"rounds": 1}, title="On the validated one"), "Tester")
            assert c.execute("SELECT config->'release'->>'validated' FROM forum_episode WHERE episode_id=%s", (ok,)).fetchone()[0] == "true"
            with pytest.raises(ValueError, match="no vectors"):
                forum.create_episode(c, fw.release_id, config("A", retrieval={"embedding_model": "openai:text-embedding-3-small"}), "Tester")
            with pytest.raises(ValueError, match="Enter your name"):
                forum.create_episode(c, fw.release_id, config("A"), " ")
            with pytest.raises(ValueError, match="no signal run"):
                forum.create_episode(c, blocked, config("0", trigger="construct_problem"), "Tester")     # no signals: no construct relationship for that trigger
            with pytest.raises(ValueError, match="no extraction run"):
                forum.create_episode(c, blocked, config("A"), "Tester")                    # nor an extraction: A, B and C have no construct database to start from
            listing = {e["title"]: e["release"] for e in forum.list_episodes(c, [fw.release_id, blocked])}
            assert listing["On an unsigned release"]["validated"] is False and listing["On the validated one"]["validated"] is True
        assert run_episode(fw, episode) == "completed"                                       # the agents really read the unvalidated release
        with fw.world.connect() as c:
            content = c.execute("SELECT content FROM question WHERE episode_id=%s", (episode,)).fetchone()[0]
            assert content["_validation"]["release"]["status"] == "candidate" and "not validated" in content["_validation"]["warnings"][0]
            assert forum.episode_detail(c, episode)["config"]["release"]["validated"] is False
    finally:
        with fw.world.connect() as c:
            for table in ("forum_proposal_score", "forum_proposal_evaluation", "forum_proposal", "question", "forum_analysis_file", "forum_analysis"):
                c.execute(f"DELETE FROM {table}")
            c.execute("DELETE FROM forum_message"); c.execute("DELETE FROM forum_agent"); c.execute("DELETE FROM forum_episode")
            c.execute("DELETE FROM retrieval_hit"); c.execute("DELETE FROM retrieval_run"); c.execute("DELETE FROM agent_profile WHERE agent_id LIKE 'forum:%'")
            blocked and c.execute("DELETE FROM release WHERE release_id=%s", (blocked,))
            c.commit()


@needs_db
def test_the_construct_problem_trigger_is_refused_when_no_relationship_was_decided_and_says_when_the_seed_picked_it(fw, monkeypatch):
    monkeypatch.setattr(forum, "diagnosis_pool", lambda con, release_id, provisional=False, pipeline_hints=False: [])
    drawn = next(s for s in range(500) if proto.resolve_trigger("random", s)["key"] == "construct_problem")
    with fw.world.connect() as c:
        with pytest.raises(ValueError, match="no construct relationship decided by people") as chosen:
            forum.create_episode(c, fw.release_id, config("A", trigger="construct_problem"), "Tester")
        assert "the seed picked it" not in str(chosen.value)
        with pytest.raises(ValueError, match="the seed picked it"):
            forum.create_episode(c, fw.release_id, config("A", seed=drawn), "Tester")
        forum.create_episode(c, fw.release_id, config("A", trigger="anomaly"), "Tester")              # any other trigger needs no relationship
        c.rollback()


@needs_db
def test_a_diagnosis_decided_by_one_reviewer_counts_only_when_the_episode_allows_it_and_the_question_says_so(fw, monkeypatch):
    from research0 import pairs
    with fw.world.connect() as c:
        taken = {r[0] for r in c.execute("SELECT pair_id FROM pair_decision")}
        lone = next(pid for pid, in c.execute("SELECT d.pair_id FROM pair_dossier d JOIN candidate_pair p USING (pair_id) WHERE p.signal_run_id=%s ORDER BY d.pair_id", (fw.signal_run,)) if pid not in taken)
        contested = next(pid for pid, in c.execute("SELECT d.pair_id FROM pair_dossier d JOIN candidate_pair p USING (pair_id) WHERE p.signal_run_id=%s ORDER BY d.pair_id", (fw.signal_run,)) if pid not in taken and pid != lone)
        pairs.decide(c, lone, "Cleo", "measurement_divergence", "only one opinion", UNCLEAR)
        pairs.decide(c, contested, "Cleo", "distinct", "one says distinct", UNCLEAR)
        pairs.decide(c, contested, "Dan", "related_but_distinct", "the other says related", UNCLEAR)
        c.commit()
    try:
        with fw.world.connect() as c:
            strict, loose = forum.diagnosis_pool(c, fw.release_id), forum.diagnosis_pool(c, fw.release_id, provisional=True)
            assert str(lone) not in {p["pair_id"] for p in strict}
            assert {p["pair_id"]: p["provisional"] for p in loose}[str(lone)] is True
            assert str(contested) not in {p["pair_id"] for p in loose}                        # a disagreement is never a premise
            assert not any(p["provisional"] for p in strict) and len(loose) == len(strict) + 1
            with pytest.raises(ValueError, match="provisional"):
                forum.create_episode(c, fw.release_id, config("C", trigger="construct_problem", diagnoses={"provisional": "yes"}), "Tester")
            assert forum.create_episode(c, fw.release_id, config("C", trigger="construct_problem"), "Tester")
            assert c.execute("SELECT config->'diagnoses'->>'provisional' FROM forum_episode").fetchone()[0] == "false"
        real = forum.diagnosis_pool            # the lone pair is the only relationship the agents get, so the premise of the question is provisional
        monkeypatch.setattr(forum, "diagnosis_pool", lambda con, rid, provisional=False, pipeline_hints=False: [p for p in real(con, rid, provisional, pipeline_hints) if p["provisional"]])
        episode = make_episode(fw, "C", trigger="construct_problem", protocol={"rounds": 1}, diagnoses={"provisional": True})
        assert run_episode(fw, episode) == "completed"
        with fw.world.connect() as c:
            content = c.execute("SELECT content FROM question WHERE episode_id=%s", (episode,)).fetchone()[0]
            assert "single reviewer" in content["_validation"]["warnings"][0] and "exploratory" in content["_validation"]["warnings"][0]
            assert [e["provisional"] for e in forum.list_episodes(c, [fw.release_id]) if e["episode_id"] == str(episode)] == [True]
    finally:
        with fw.world.connect() as c:
            c.execute("DELETE FROM pair_decision WHERE pair_id = ANY(%s)", ([lone, contested],))
            c.commit()


@needs_db
def test_pairs_labelled_only_by_the_signals_can_start_a_discussion_but_say_so_everywhere(fw, monkeypatch):
    with fw.world.connect() as c:
        decisions_before = c.execute("SELECT count(*) FROM pair_decision").fetchone()[0]
        decided = {r[0] for r in c.execute("SELECT pair_id FROM pair_decision")}
        ids = forum.hint_pair_ids(c, fw.release_id)
        assert set(ids) == {"conceptual_jingle", "conceptual_jangle", "unresolved"} and all(len(found) <= forum.HINT_PAIRS_PER_CATEGORY for found in ids.values())
        flat = [p for found in ids.values() for p in found]
        assert flat and not decided & set(flat)                                     # a pair a person decided on never comes back as a hint
        strict = forum.diagnosis_pool(c, fw.release_id)
        assert not any(p["pipeline"] for p in strict)
        c.execute("DELETE FROM pair_dossier WHERE pair_id = ANY(%s)", (flat,))      # the signal run only has dossiers for the pairs somebody sampled
        c.commit()
        assert [p["pair_id"] for p in forum.diagnosis_pool(c, fw.release_id, pipeline_hints=True)] == [p["pair_id"] for p in strict]
        assert forum.prepare_hint_dossiers(c, fw.release_id) == len(flat) and forum.prepare_hint_dossiers(c, fw.release_id) == 0     # built once, idempotent
        loose = forum.diagnosis_pool(c, fw.release_id, pipeline_hints=True)
        added = [p for p in loose if p["pipeline"]]
        assert len(loose) == len(strict) + len(flat) and {p["pair_id"] for p in added} == {str(p) for p in flat}
        assert all(p["relation"] == "not_reviewed" and p["flag"] == p["category"] and not p["provisional"] for p in added)
        assert [p["id"] for p in loose] == [f"P{n}" for n in range(1, len(loose) + 1)]
        with pytest.raises(ValueError, match="pipeline_hints"):
            forum.create_episode(c, fw.release_id, config("C", trigger="construct_problem", diagnoses={"pipeline_hints": 1}), "Tester")
        assert forum.create_episode(c, fw.release_id, config("C", trigger="construct_problem"), "Tester")
        assert c.execute("SELECT config->'diagnoses'->>'pipeline_hints' FROM forum_episode").fetchone()[0] == "false"
    real = forum.diagnosis_pool          # only the signal-labelled pairs are left, so the premise of the question is unreviewed
    monkeypatch.setattr(forum, "diagnosis_pool", lambda con, rid, provisional=False, pipeline_hints=False: [p for p in real(con, rid, provisional, pipeline_hints) if p["pipeline"]])
    episode = make_episode(fw, "C", trigger="construct_problem", protocol={"rounds": 1}, diagnoses={"pipeline_hints": True})
    assert run_episode(fw, episode) == "completed"
    handed = next(m for m in messages(fw, episode, kind="system") if m["data"].get("role") == "premises")
    assert "NO PERSON HAS REVIEWED IT" in handed["content"] and "EXCEPTION to rule 3" in handed["content"] and "decided by human reviewers" not in handed["content"].split("\n\n")[1]
    assert all(card["pipeline"] and card["relation"] == "not_reviewed" for card in handed["data"]["pairs"])
    with fw.world.connect() as c:
        content = c.execute("SELECT content FROM question WHERE episode_id=%s", (episode,)).fetchone()[0]
        assert "NOT decided by any person" in content["_validation"]["warnings"][0] and "exploratory" in content["_validation"]["warnings"][0]
        assert [e["pipeline_hints"] for e in forum.list_episodes(c, [fw.release_id]) if e["episode_id"] == str(episode)] == [True]
        assert c.execute("SELECT count(*) FROM pair_decision").fetchone()[0] == decisions_before     # reading the hints decides nothing on anybody's behalf


@needs_db
def test_agents_connect_read_only_and_cannot_touch_the_workspace(fw):
    episode = make_episode(fw, "A")
    runner = forum.Runner(episode, ScriptedProvider(), app_url=fw.world.url)
    try:
        with pytest.raises(psycopg.errors.ReadOnlySqlTransaction):
            runner.agent_con.execute("DELETE FROM forum_message")
    finally:
        runner.agent_con.close(); runner.app.close()


@needs_db
def test_listing_and_detail_are_plain_json_and_episodes_with_equal_budgets_are_comparable(fw):
    first, second = make_episode(fw, "0", protocol={"rounds": 1}), make_episode(fw, "C", protocol={"rounds": 1})
    other = make_episode(fw, "A", protocol={"rounds": 1}, budgets={"max_passages": 7})
    run_episode(fw, second)
    with fw.world.connect() as c:
        listing = forum.list_episodes(c, [fw.release_id])
        detail = forum.episode_detail(c, second)
        json.dumps(listing), json.dumps(detail)
        assert {e["episode_id"] for e in listing} == {str(first), str(second), str(other)}
        assert [m["seq"] for m in detail["messages"]] == sorted(m["seq"] for m in detail["messages"])
        assert forum.episode_detail(c, second, after=detail["messages"][2]["seq"])["messages"][0]["seq"] == detail["messages"][3]["seq"]       # polling returns only what is new
        assert [a["key"] for a in detail["agents"]] == ["cto", "tim"] and detail["cost_usd"] is None                                         # the stand-in has no price: tokens only
        assert [e["episode_id"] for e in forum.comparable(c, second)] == [str(first)]
        assert forum.mark_orphans(c) == 0


@needs_db
def test_an_episode_left_running_by_a_stopped_server_is_marked_failed(fw):
    episode = make_episode(fw, "A")
    with fw.world.connect() as c:
        c.execute("UPDATE forum_episode SET status='running' WHERE episode_id=%s", (episode,))
        assert forum.mark_orphans(c) == 1
        assert c.execute("SELECT status FROM forum_episode WHERE episode_id=%s", (episode,)).fetchone()[0] == "failed"
        c.rollback()


# --- The HTTP layer ---------------------------------------------------------------------------------------------------------------------

def http(port, method, path, body=None, headers=None):
    import urllib.error
    import urllib.request
    request = urllib.request.Request(f"http://127.0.0.1:{port}{path}", method=method, data=json.dumps(body).encode() if body is not None else None,
                                     headers={"Content-Type": "application/json", **(headers or {})})
    try:
        with urllib.request.urlopen(request) as response:
            raw = response.read()
            return response.status, (json.loads(raw) if response.headers["Content-Type"].startswith("application/json") else raw.decode())
    except urllib.error.HTTPError as exc:
        return exc.code, json.loads(exc.read())


@pytest.fixture
def server(fw, monkeypatch):
    from research0.forum_server import serve
    monkeypatch.setenv("DATABASE_URL", fw.world.url)
    monkeypatch.delenv("AGENT_DATABASE_URL", raising=False)
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    started = []

    def start(read_only=False):
        srv, run_id = serve(str(fw.world.run_id), 0, read_only=read_only)
        threading.Thread(target=srv.serve_forever, daemon=True).start()
        started.append(srv)
        return srv.server_address[1], str(run_id)
    yield start
    for srv in started:
        srv.shutdown()
        srv.server_close()


def wait_for(port, episode_id, states, seconds=60):
    import time
    end = time.time() + seconds
    while time.time() < end:
        status, detail = http(port, "GET", f"/api/forum/episode?id={episode_id}")
        if detail["status"] in states:
            return detail
        time.sleep(0.2)
    raise AssertionError("episode did not reach " + str(states))


@needs_db
def test_page_config_and_release_endpoints(fw, server):
    port, run_id = server()
    status, html = http(port, "GET", "/")
    assert status == 200 and "Discovery forum" in html and run_id in html and "/*__BOOT__*/null" not in html and '"read_only": false' in html
    status, cfg = http(port, "GET", "/api/forum/config")
    assert status == 200 and [c["id"] for c in cfg["conditions"]] == list("0ABCDEF") and cfg["providers"]["openai"]["available"] is False
    assert cfg["design"] == list("0ABC") and [c["id"] for c in cfg["conditions"] if c["legacy"]] == list("DEF")       # D, E, F only to read old episodes
    assert [t["key"] for t in cfg["triggers"]] == list(proto.TRIGGERS) and cfg["defaults"]["encounter"]["theme_mode"] == "opener_theme"
    assert [m["key"] for m in cfg["theme_modes"] if m["auxiliary"]] == list(proto.AUXILIARY_THEME_MODES)
    assert cfg["limits"]["rounds"] == [1, 12] and cfg["defaults"]["budgets"]["max_total_tokens"] == 300_000 and cfg["steps"][0]["title"] == "Search by the trigger"
    assert [c["key"] for c in cfg["scoring"]["criteria"]] == ["novelty", "methodology", "impact"] and "rubric" not in cfg      # the blind evaluator is gone (18 Oct 2026)
    status, rel = http(port, "GET", "/api/forum/releases")
    assert [r["status"] for r in rel["releases"]] == ["validated"]
    status, info = http(port, "GET", f"/api/forum/release?id={rel['releases'][0]['release_id']}")
    assert status == 200 and info["error"] is None and {"conceptual_jingle", "conceptual_jangle", "unresolved"} <= set(info["diagnoses"]) and info["embedding_models"][0]["model_id"] == "hash-rp-v1"
    assert http(port, "GET", "/api/forum/release?id=" + str(uuid.uuid4()))[0] == 400
    assert http(port, "GET", "/api/forum/episode?id=not-a-uuid")[0] == 400


@needs_db
def test_an_episode_is_created_run_in_the_background_and_polled(fw, server):
    port, run_id = server()
    release_id = http(port, "GET", "/api/forum/releases")[1]["releases"][0]["release_id"]
    body = {"run_id": run_id, "release_id": release_id, "created_by": "Tester", "start": True, "config": config("C", protocol={"rounds": 2}, title="From the page")}
    status, out = http(port, "POST", "/api/forum/episode/create", body)
    assert status == 200 and out["started"] is True
    detail = wait_for(port, out["episode_id"], {"completed", "failed", "budget_exhausted"})
    assert detail["status"] == "completed" and detail["title"] == "From the page" and len(detail["questions"]) == 1 and detail["usage"]["tokens"] > 0
    first = detail["messages"][-3]["seq"]
    status, later = http(port, "GET", f"/api/forum/episode?id={out['episode_id']}&after={first}")
    assert [m["seq"] for m in later["messages"]] == [m["seq"] for m in detail["messages"] if m["seq"] > first]
    status, listing = http(port, "GET", "/api/forum/episodes")
    assert [e["title"] for e in listing["episodes"]] == ["From the page"] and listing["job"]["running"] is None
    # an episode that has run cannot be started again, and a mistaken request is refused with a reason
    assert http(port, "POST", "/api/forum/episode/start", {"run_id": run_id, "episode_id": out["episode_id"]})[0] == 400
    status, err = http(port, "POST", "/api/forum/episode/create", {**body, "config": config("C", budgets={"max_total_tokens": 10})})
    assert status == 400 and "max_total_tokens" in err["error"]
    assert http(port, "POST", "/api/forum/episode/create", {**body, "run_id": str(uuid.uuid4())})[0] == 409
    assert http(port, "POST", "/api/forum/episode/create", body, {"Origin": "http://evil.example"})[0] == 403


@needs_db
def test_a_running_episode_can_be_stopped_and_only_one_runs_at_a_time(fw, server, monkeypatch):
    monkeypatch.setenv("FORUM_SCRIPTED_DELAY", "2")
    port, run_id = server()
    release_id = http(port, "GET", "/api/forum/releases")[1]["releases"][0]["release_id"]
    make = lambda: http(port, "POST", "/api/forum/episode/create", {"run_id": run_id, "release_id": release_id, "created_by": "T", "start": True, "config": config("A", protocol={"rounds": 3})})
    status, first = make()
    assert status == 200
    status, second = make()                                   # the episode is created, but it cannot start while another runs
    assert status == 400 and "Another episode is running" in second["error"]
    status, mid = http(port, "GET", f"/api/forum/episode?id={first['episode_id']}")
    assert mid["job"]["running"] == first["episode_id"] and mid["status"] in ("running", "created")
    assert http(port, "POST", "/api/forum/episode/stop", {"run_id": run_id, "episode_id": first["episode_id"]})[0] == 200
    assert wait_for(port, first["episode_id"], {"stopped"})["status_detail"] == "Stopped by a person"
    assert http(port, "POST", "/api/forum/episode/stop", {"run_id": run_id, "episode_id": first["episode_id"]})[0] == 400


@needs_db
def test_web_sources_get_citable_ids_and_the_question_argues_its_timeliness(fw):
    """With internet open every page a search returns gets a W-id, the agents can cite it, and the timeliness check counts it (the stand-in returns one page per reflection)."""
    episode = make_episode(fw, "C", provider="openai", external_access="open", protocol={"rounds": 2})
    assert run_episode(fw, episode) == "completed"
    with fw.world.connect() as c:
        detail = forum.episode_detail(c, episode)
    web = [w for m in detail["messages"] if m["kind"] == "reflection" for w in (m["data"] or {}).get("web", [])]
    assert [w["id"] for w in web] == [f"W{n}" for n in range(1, len(web) + 1)] and len(web) == 4 and all(w["url"].startswith("https://example.org/") for w in web)
    question = detail["questions"][0]
    timeliness, warnings = question["content"]["timeliness"], question["content"]["_validation"]["warnings"]
    assert set(timeliness) == set(proto.TIMELINESS_FIELDS) and "[W" in timeliness["recent_evidence"]
    assert any(s["kind"] == "web" and s["verified"] and s["url"].startswith("https://example.org/") for s in question["evidence"])
    assert not any("no web source supports" in w or "against the corpus only" in w for w in warnings)
    offline = make_episode(fw, "C", protocol={"rounds": 2})               # no internet: the question says its novelty was checked against the corpus only
    assert run_episode(fw, offline) == "completed"
    with fw.world.connect() as c:
        q = forum.episode_detail(c, offline)["questions"][0]
    assert any("against the corpus only" in w for w in q["content"]["_validation"]["warnings"]) and not any(s["kind"] == "web" for s in q["evidence"])


@needs_db
def test_a_simulated_episode_made_before_the_removal_can_be_read_but_not_started(fw, server, monkeypatch):
    port, run_id = server()
    release_id = http(port, "GET", "/api/forum/releases")[1]["releases"][0]["release_id"]
    status, out = http(port, "POST", "/api/forum/episode/create", {"run_id": run_id, "release_id": release_id, "created_by": "T", "config": config("A", protocol={"rounds": 1})})
    assert status == 200 and out["started"] is False
    monkeypatch.setattr(proto, "PROVIDERS", {"openai": None})                       # the application as it is, without the tests' stand-in
    status, err = http(port, "POST", "/api/forum/episode/start", {"run_id": run_id, "episode_id": out["episode_id"]})
    assert status == 400 and "simulation was removed" in err["error"]
    assert http(port, "GET", f"/api/forum/episode?id={out['episode_id']}")[0] == 200
    status, err = http(port, "POST", "/api/forum/episode/create", {"run_id": run_id, "release_id": release_id, "created_by": "T", "config": config("A")})
    assert status == 400 and "simulation was removed" in err["error"]
    status, cfg = http(port, "GET", "/api/forum/config")
    assert list(cfg["providers"]) == ["openai"]
    # creating an OpenAI episode sends nothing: only starting it asks for the acknowledgement
    status, out = http(port, "POST", "/api/forum/episode/create", {"run_id": run_id, "release_id": release_id, "created_by": "T", "config": {"condition": "A", "provider": "openai"}})
    assert status == 200 and out["started"] is False
    status, err = http(port, "POST", "/api/forum/episode/start", {"run_id": run_id, "episode_id": out["episode_id"]})
    assert status == 400 and "costs money" in err["error"]


@needs_db
def test_openai_needs_a_key_and_an_explicit_acknowledgement_and_the_read_only_deployment_cannot_write(fw, server, monkeypatch):
    port, run_id = server()
    release_id = http(port, "GET", "/api/forum/releases")[1]["releases"][0]["release_id"]
    body = {"run_id": run_id, "release_id": release_id, "created_by": "T", "start": True, "config": {"condition": "A", "provider": "openai"}}
    status, err = http(port, "POST", "/api/forum/episode/create", body)
    assert status == 400 and "costs money" in err["error"]
    status, err = http(port, "POST", "/api/forum/episode/create", {**body, "acknowledge": "send-text-to-openai"})
    assert status == 400 and "OPENAI_API_KEY" in err["error"]
    with fw.world.connect() as c:
        assert c.execute("SELECT count(*) FROM forum_episode").fetchone()[0] == 0              # nothing was created by the refused requests
    ro_port, _ = server(read_only=True)
    status, page = http(ro_port, "GET", "/")
    assert status == 200 and '"read_only": true' in page
    assert http(ro_port, "GET", "/api/forum/episodes")[0] == 200
    assert http(ro_port, "POST", "/api/forum/episode/create", body)[0] == 403


@needs_db
def test_the_forum_is_its_own_service_and_the_dashboard_knows_nothing_of_it(fw, server, monkeypatch):
    from research0.dashboard_server import serve as dashboard_serve
    port, run_id = server()
    assert http(port, "GET", "/healthz")[0] == 200 and http(port, "GET", "/api/snapshot")[0] == 404               # no dashboard routes on the forum's port
    assert http(port, "GET", "/", headers={"Host": "evil.example"})[0] == 403
    monkeypatch.setenv("FORUM_ALLOWED_HOSTS", "forum.example.org")
    from research0.forum_server import serve
    srv, _ = serve(run_id, 0)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    try:
        assert http(srv.server_address[1], "GET", "/", headers={"Host": "forum.example.org"})[0] == 200
        assert http(srv.server_address[1], "GET", "/", headers={"Host": "dashboard.example.org"})[0] == 403
    finally:
        srv.shutdown(); srv.server_close()
    dash, _ = dashboard_serve(run_id, 0)
    threading.Thread(target=dash.serve_forever, daemon=True).start()
    try:
        for path in ("/forum", "/api/forum/config", "/api/forum/episodes"):
            assert http(dash.server_address[1], "GET", path)[0] == 404
        assert http(dash.server_address[1], "POST", "/api/forum/episode/create", {"run_id": run_id})[0] == 404
    finally:
        dash.shutdown(); dash.server_close()


# --- Embedding jobs, the construct-record search and the v1 fallback ---------------------------------------------------------------------

@needs_db
def test_record_texts_cover_the_four_kinds_and_embedding_is_idempotent_and_resumable(fw):
    from research0 import embedding_jobs as jobs
    from research0.embeddings import HashEmbedder
    with fw.world.connect() as c:
        run = c.execute("SELECT extraction_run_id FROM extraction_run").fetchone()[0]
        texts = {k: jobs.record_texts(c, run, k) for k in jobs.KINDS}
        assert all(texts[k] for k in jobs.KINDS) or not texts["attribute"]                     # attributes exist only where the extraction gave one
        assert texts["mention"][0][3].startswith("Construct: ") and "Evidence:" in texts["mention"][0][3]
        assert texts["claim"][0][3].startswith("Claim: ")
        assert c.execute("SELECT count(*) FROM record_embedding").fetchone()[0] == sum(len(v) for v in texts.values())      # the fixture embedded them all
        assert jobs.embed_records(c, run, HashEmbedder())["mention"]["embedded"] == 0                                         # nothing left: a rerun sends nothing
        c.execute("DELETE FROM record_embedding WHERE record_kind='claim'")
        c.commit()
        again = jobs.embed_records(c, run, HashEmbedder())                                                                    # resumes: only the deleted kind is redone
        assert again["claim"]["embedded"] == len(texts["claim"]) and again["mention"]["embedded"] == 0
        report = jobs.dry_run(c, fw.ingestion_id, run, "hash-rp-v1")
        assert report["sources"]["chunks (PDF text)"]["texts"] == 0 and report["total_texts"] == 0 and "tokenizer" in report


@needs_db
def test_search_records_returns_cited_records_and_respects_the_community_filter(fw):
    from research0 import embedding_jobs as jobs, rag
    from research0.embeddings import HashEmbedder
    emb = HashEmbedder()
    with fw.world.connect() as c:
        run = c.execute("SELECT extraction_run_id FROM extraction_run").fetchone()[0]
        _, manifest = rag.release_scope(c, fw.release_id)
        cto = [str(p) for p in rag.community_papers(c, manifest, "CTO")]
        hits = jobs.search_records(c, run, emb.model_id, emb.embed(["digital readiness"])[0], 5, ["mention"], cto)
        assert hits and all(h["record_kind"] == "mention" and h["canonical_id"] in cto and h["page"] and h["title"] for h in hits)
        assert [h["distance"] for h in hits] == sorted(h["distance"] for h in hits)
        assert jobs.search_records(c, run, emb.model_id, emb.embed(["x"])[0], 5, None, [])  == []


@needs_db
def test_agents_can_search_the_construct_database_and_cite_what_it_returns(fw):
    episode = make_episode(fw, "A", protocol={"rounds": 1})
    assert run_episode(fw, episode) == "completed"
    results = [m for m in messages(fw, episode, kind="tool_result") if m["data"].get("tool") == "search_constructs"]
    assert results and results[0]["data"]["hits"][0]["section"].startswith("construct ") and results[0]["data"]["hits"][0]["id"].startswith("E")
    base = make_episode(fw, "0", protocol={"rounds": 1})
    assert run_episode(fw, base) == "completed"
    assert not [m for m in messages(fw, base, kind="tool_call") if m["data"]["tool"] == "search_constructs"]                   # the baseline has no database


@needs_db
def test_the_construct_map_and_network_aggregate_the_claims_of_a_view_with_their_moderators(fw):
    from research0 import forum_constructs as fcon, rag
    with fw.world.connect() as c:
        manifest = rag.release_scope(c, fw.release_id, True)[1]
        run = manifest["runs"]["extraction_run_id"]
        papers = sorted({str(p) for k in ("CTO", "TIM") for p in rag.community_papers(c, manifest, k, True)})
        linked = fcon.overview(c, run, papers, sort="links", limit=5)
        assert linked["rows"] and linked["claims_in_view"] > 0 and [r["links"] for r in linked["rows"]] == sorted((r["links"] for r in linked["rows"]), reverse=True)
        thin = fcon.overview(c, run, papers, sort="thin", limit=5)["rows"]
        assert all(r["papers"] >= 2 for r in thin) and [r["links"] for r in thin] == sorted(r["links"] for r in thin)
        top = linked["rows"][0]["construct"]
        net = fcon.network(c, run, papers, top.upper())                                  # labels are matched normalized
        assert net["found"] and net["construct"] == top and net["papers"] >= 1 and net["antecedents"] + net["outcomes"] + net["other_links"]
        assert all(e["example"][0] == "claim" and e["papers"] >= 1 for g in ("antecedents", "outcomes", "other_links") for e in net[g])
        moderator = next(r["construct"] for r in fcon.overview(c, run, papers, limit=10_000)["rows"] if r["as_modifier"].get("moderator"))
        assert fcon.network(c, run, papers, moderator)["as_modifier"][0]["kind"] == "moderator"      # the mock world plants moderated claims
        missing = fcon.network(c, run, papers, top + " zzz")
        assert missing["found"] is False and top in [x["construct"] for x in missing["closest"]]
        assert fcon.overview(c, run, papers[:1])["papers_in_view"] <= 1                     # only the papers of the view count


@needs_db
def test_without_the_construct_problem_trigger_a_b_and_c_explore_the_database_and_never_see_a_relationship(fw):
    from research0 import forum_proposal as fprop
    assert "dossier" in fprop.draft_task(4, True) and not any("dossier" in t for t in (fprop.draft_task(4, False), fprop.critique_task(False), fprop.shorten_task(5, 4, False)))
    for condition, rounds in (("A", 2), ("B", 1), ("C", 2)):
        episode = make_episode(fw, condition, protocol={"rounds": rounds})
        assert run_episode(fw, episode) == "completed"
        log = messages(fw, episode)
        assert not [m for m in log if m["data"].get("pairs") or m["data"].get("diagnoses") or m["data"].get("role") in ("theme", "premises")]   # no dossier, no drawn theme
        tools = [m for m in log if m["kind"] == "tool_call"]
        assert not {"list_diagnoses", "read_dossier"} & {m["data"]["tool"] for m in tools}
        networks = [m for m in log if m["kind"] == "tool_result" and m["data"].get("tool") == "construct_network"]
        assert networks and all(h["id"].startswith("E") and h["section"].startswith("construct ") for m in networks for h in m["data"]["hits"])
        assert {m["agent"] for m in networks} == {"generalist"} if condition == "A" else {"cto", "tim"}
        with fw.world.connect() as c:
            cfg, version = c.execute("SELECT config,prompt_version FROM forum_episode WHERE episode_id=%s", (episode,)).fetchone()
        assert cfg["start"] == "constructs" and version == proto.PROMPT_VERSION_V4 and cfg["trigger"]["key"] == "missing_mechanism"
        if condition == "C":
            steps = [m["data"]["step"] for m in log if m["kind"] == "system" and m["data"].get("role") == "moderator"]
            assert steps == [proto.STEPS_V4_ENCOUNTER[0][0], proto.STEPS_V4_ENCOUNTER[-1][0]] and cfg["encounter"]["theme_mode"] == "opener_theme"
        if condition == "A":
            posts = [m["data"]["step"] for m in log if m["kind"] == "message"]
            assert posts == [proto.STEPS_V4_SOLO[0][0], proto.STEPS_V4_SOLO[-1][0]]


@needs_db
def test_the_auxiliary_construct_opening_of_c_still_runs_its_own_chain(fw):
    episode = make_episode(fw, "C", protocol={"rounds": 2}, encounter={"theme_mode": "constructs"})
    assert run_episode(fw, episode) == "completed"
    steps = [m["data"]["step"] for m in messages(fw, episode, kind="system") if m["data"].get("role") == "moderator"]
    assert steps == [proto.STEPS_CONSTRUCTS[0][0], proto.STEPS_CONSTRUCTS[-1][0]] and not [m for m in messages(fw, episode, kind="system") if m["data"].get("role") == "opening"]


@needs_db
def test_an_isolated_specialist_on_a_cross_community_trigger_is_flagged_for_separate_analysis(fw):
    episode = make_episode(fw, "B", trigger="translation", protocol={"rounds": 1, "reasoner": "cto"})
    assert run_episode(fw, episode) == "completed"
    with fw.world.connect() as c:
        content = c.execute("SELECT content FROM question WHERE episode_id=%s", (episode,)).fetchone()[0]
    assert content["_validation"]["cross_community"] is True and "analyse this question apart" in content["_validation"]["warnings"][0]


@needs_db
def test_a_legacy_episode_can_be_read_but_never_run(fw):
    with fw.world.connect() as c:
        cfg = proto.normalize_config(config("C"), MODELS)
        episode = uuid.uuid4()
        c.execute("INSERT INTO forum_episode(episode_id,release_id,condition,title,provider,seed,config,config_sha256,budgets_sha256,prompt_version,created_by) "
                  "VALUES (%s,%s,'D','Old jingle episode','scripted',1,%s,'x','y','forum-v6','Tester')", (episode, fw.release_id, json.dumps({**cfg, "condition": "D"})))
        c.commit()
        listed = [e for e in forum.list_episodes(c, [fw.release_id]) if e["episode_id"] == str(episode)]
        assert listed[0]["legacy"] is True and listed[0]["condition_name"] == "Jingle reasoning"
    with pytest.raises(ValueError, match="removed from the design"):
        forum.Runner(episode, ScriptedProvider(), app_url=fw.world.url)


@needs_db
def test_v1_suggestions_are_a_fallback_only_when_asked_and_never_override_a_v2_answer(fw):
    from research0 import confirmation
    with fw.world.connect() as c:
        run = fw.world.run_id
        base = confirmation.memberships(c, run)
        pid = c.execute("INSERT INTO preassignment_run(preassignment_run_id,run_id,source,model,prompt_version,scope_version,codebook_sha256) VALUES (gen_random_uuid(),%s,'v1_suggestion','m','p','s','0') RETURNING preassignment_run_id", (run,)).fetchone()[0]
        papers = [r[0] for r in c.execute("SELECT canonical_id FROM canonical_paper WHERE run_id=%s ORDER BY canonical_id", (run,))]
        known = {cid for cid, _ in base}
        fresh = next(p for p in papers if p not in known)
        for community in ("CTO", "TIM"):
            c.execute("INSERT INTO community_preassignment(preassignment_run_id,canonical_id,community,label,rationale,text_basis) VALUES (%s,%s,%s,'yes','r','abstract')", (pid, fresh, community))
        one_known = next(iter(base))
        c.execute("INSERT INTO community_preassignment(preassignment_run_id,canonical_id,community,label,rationale,text_basis) VALUES (%s,%s,%s,%s,'r','abstract')",
                  (pid, one_known[0], one_known[1], "no" if base[one_known]["label"] == "yes" else "yes"))
        try:
            assert (fresh, "CTO") not in confirmation.memberships(c, run)                                                      # strict by default
            wide = confirmation.memberships(c, run, include_v1=True)
            assert wide[(fresh, "CTO")] == {"label": "yes", "status": "suggestion_v1", "stage": "metadata", "confidence": None}
            assert wide[one_known] == base[one_known]                                                                         # a v2 answer is never overridden
        finally:
            c.rollback()


def test_the_proposal_makes_the_puzzle_visible_first_argues_timeliness_and_lists_a_web_source_as_unchecked():
    from research0 import forum_proposal as fprop
    for dossiers in (True, False):
        draft = fprop.draft_task(4, dossiers)
        assert "remains consequential now" in draft and "[W1]" in draft and "new or timely" in fprop.critique_task(dossiers)
        assert "the theoretical puzzle first" in draft and "before any measure or estimator is named" in draft and "rival interpretations" in draft
        assert "names measures or estimators before the puzzle" in fprop.critique_task(dossiers)
    content = {"_validation": {"cited": ["W1", "W9"]}}
    lines = fprop.reference_lines(content, {"W1": {"kind": "web", "title": "A 2026 study", "url": "https://example.org/a"}})
    assert lines == ["[W1] A 2026 study. Web page found by the agents' search, not checked by the system: https://example.org/a"]


# --- Batches: the four conditions under the same settings, repeated --------------------------------------------------------------------

def test_a_batch_expands_one_form_into_five_arms_per_repetition_with_matched_seeds():
    form = {"provider": "scripted", "seed": 100, "trigger": "random", "agent": {"max_message_tokens": 900}, "protocol": {"rounds": 2, "lead": "tim", "reasoner": "tim"},
            "budgets": {"max_total_tokens": 300_000, "max_llm_calls": 61}}                               # odd numbers are fine: nothing is split any more
    plan = proto.batch_configs(form, 3)
    configs = [proto.normalize_config(cfg, MODELS) for _, _, cfg in plan]
    arms = [proto.arm_of(c["condition"], c["protocol"]) for c in configs]
    assert arms == ["0", "A", "B-cto", "B-tim", "C"] * 3 and [r for r, _, _ in plan] == [1] * 5 + [2] * 5 + [3] * 5
    assert [cfg["seed"] for _, _, cfg in plan] == [100] * 5 + [101] * 5 + [102] * 5               # vary: a new seed per repetition, the same within one
    assert {cfg["seed"] for _, _, cfg in proto.batch_configs(form, 3, "same")} == {100}
    for rep in range(3):
        five = configs[rep * 5:rep * 5 + 5]
        assert len({c["trigger"]["key"] for c in five}) == 1 and len({proto.budgets_checksum(c) for c in five}) == 1     # matched within a repetition
        assert all(a["max_message_tokens"] == 900 for c in five for a in c["agents"].values())           # one block for every agent
        assert five[4]["protocol"]["lead"] == "tim" and [set(c["agents"]) for c in five[2:4]] == [{"cto"}, {"tim"}]   # the lead only reaches C; B once per specialist
    assert len({proto.budgets_checksum(c) for c in configs}) == 1


@pytest.mark.parametrize("form, repetitions, seed_mode, message", [
    ({"condition": "A"}, 1, "vary", "leave the condition out"),
    ({}, 0, "vary", "repetitions must be between"),
    ({}, 51, "vary", "repetitions must be between"),
    ({}, 2, "sometimes", "seed_mode must be one of"),
    ({"seed": 2_147_483_647}, 2, "vary", "seed must be between"),
])
def test_a_batch_refuses_what_it_cannot_run(form, repetitions, seed_mode, message):
    with pytest.raises(ValueError, match=message):
        proto.batch_configs({"provider": "scripted", **form}, repetitions, seed_mode)


@needs_db
def test_a_batch_is_written_whole_or_not_at_all_and_freezes_its_place_in_every_episode(fw):
    form = {"provider": "scripted", "seed": 7, "trigger": "missing_mechanism", "protocol": {"rounds": 1}, "title": "Pilot"}
    with fw.world.connect() as c:
        with pytest.raises(ValueError, match="Repetition 1, condition 0 .*embedding model has no vectors"):     # refused before anything is written
            forum.create_batch(c, fw.release_id, {**form, "retrieval": {"embedding_model": "nobody-v1"}}, "Tester", 2)
        assert c.execute("SELECT count(*) FROM forum_episode").fetchone()[0] == 0
        batch_id, ids = forum.create_batch(c, fw.release_id, form, "Tester", 2)
        rows = c.execute("SELECT episode_id,condition,seed,title,budgets_sha256,config FROM forum_episode WHERE episode_id = ANY(%s)", (ids,)).fetchall()
    by_id = {r[0]: r for r in rows}
    ordered = [by_id[i] for i in ids]
    assert [r[1] for r in ordered] == list("0ABBC0ABBC") and [r[2] for r in ordered] == [7] * 5 + [8] * 5
    assert ordered[0][3] == "Pilot - rep 1/2 - 0 Baseline" and ordered[3][3] == "Pilot - rep 1/2 - B Isolated TIM" and len({r[4] for r in rows}) == 1
    assert [r[5]["batch"]["position"] for r in ordered] == list(range(1, 11)) and {r[5]["batch"]["id"] for r in rows} == {str(batch_id)}
    assert [r[5]["batch"]["arm"] for r in ordered[:5]] == ["0", "A", "B-cto", "B-tim", "C"]
    with fw.world.connect() as c:
        listed = forum.list_episodes(c, [fw.release_id])
        detail = forum.batch_detail(c, batch_id)
    assert all(e["batch"]["id"] == str(batch_id) for e in listed)
    assert detail["repetitions"] == 2 and detail["arms"] == ["0", "A", "B-cto", "B-tim", "C"] and detail["statuses"] == {"created": 10}


@needs_db
def test_a_batch_runs_its_episodes_one_after_the_other_and_can_be_stopped_and_resumed(fw, server, monkeypatch):
    port, run_id = server()
    release_id = http(port, "GET", "/api/forum/releases")[1]["releases"][0]["release_id"]
    body = {"run_id": run_id, "release_id": release_id, "created_by": "Tester", "repetitions": 1, "start": True,
            "config": {"provider": "scripted", "trigger": "missing_mechanism", "protocol": {"rounds": 1}, "proposal": {"enabled": False}}}
    status, out = http(port, "POST", "/api/forum/batch/create", body)
    assert status == 200 and out["started"] is True and len(out["episode_ids"]) == 5
    for episode_id in out["episode_ids"]:
        wait_for(port, episode_id, {"completed", "failed", "budget_exhausted"})
    status, detail = http(port, "GET", f"/api/forum/batch?id={out['batch_id']}")
    assert status == 200 and detail["statuses"] == {"completed": 5} and all(detail["by_arm"][a]["questions"] == 1 for a in detail["arms"])
    assert http(port, "POST", "/api/forum/batch/start", {"run_id": run_id, "batch_id": out["batch_id"]})[1]["error"].startswith("Every episode of this batch has already run")
    assert http(port, "GET", f"/api/forum/batch?id={uuid.uuid4()}")[0] == 400

    monkeypatch.setenv("FORUM_SCRIPTED_DELAY", "2")         # slow turns, so that the batch can be stopped while its first episode runs
    status, slow = http(port, "POST", "/api/forum/batch/create", {**body, "config": {**body["config"], "protocol": {"rounds": 3}}})
    assert status == 200
    assert http(port, "POST", "/api/forum/episode/create", {**body, "config": config("A")})[1]["error"].startswith("Another episode is running")
    wait_for(port, slow["episode_ids"][0], {"running"})
    assert http(port, "POST", "/api/forum/batch/stop", {"run_id": run_id, "batch_id": slow["batch_id"]})[0] == 200
    wait_for(port, slow["episode_ids"][0], {"stopped"})
    import time
    end = time.time() + 30
    while (http(port, "GET", f"/api/forum/batch?id={slow['batch_id']}")[1]["job"].get("batch") or {}).get("running") and time.time() < end:
        time.sleep(0.2)                                                                          # the batch's thread ends once its episode has stopped
    status, detail = http(port, "GET", f"/api/forum/batch?id={slow['batch_id']}")
    assert detail["statuses"] == {"stopped": 1, "created": 4} and not detail["job"]["batch"]["running"]     # the episodes not reached are left ready to run
    monkeypatch.setenv("FORUM_SCRIPTED_DELAY", "0")
    status, resumed = http(port, "POST", "/api/forum/batch/start", {"run_id": run_id, "batch_id": slow["batch_id"]})
    assert status == 200 and resumed["started"] == 4
    for episode_id in slow["episode_ids"][1:]:
        wait_for(port, episode_id, {"completed", "failed", "budget_exhausted"})
    assert http(port, "GET", f"/api/forum/batch?id={slow['batch_id']}")[1]["statuses"] == {"stopped": 1, "completed": 4}


def test_a_batch_with_two_workers_runs_two_real_episodes_at_once_and_finishes_them_all(fw, server, monkeypatch):
    import time
    monkeypatch.setenv("FORUM_BATCH_WORKERS", "2")
    monkeypatch.setenv("FORUM_SCRIPTED_DELAY", "0.3")          # a little time per turn, so that the episodes overlap
    port, run_id = server()
    release_id = http(port, "GET", "/api/forum/releases")[1]["releases"][0]["release_id"]
    body = {"run_id": run_id, "release_id": release_id, "created_by": "Tester", "repetitions": 1, "start": True,
            "config": {"provider": "scripted", "trigger": "missing_mechanism", "protocol": {"rounds": 1}, "proposal": {"enabled": False}}}
    status, out = http(port, "POST", "/api/forum/batch/create", body)
    assert status == 200 and out["started"] is True and len(out["episode_ids"]) == 5
    most_at_once, workers, end = 0, None, time.time() + 90
    while time.time() < end:
        job = http(port, "GET", f"/api/forum/batch?id={out['batch_id']}")[1]["job"]
        batch = job.get("batch") or {}
        most_at_once, workers = max(most_at_once, len(job.get("running_ids") or [])), batch.get("workers", workers)
        if batch and not batch["running"]:
            break
        time.sleep(0.1)
    detail = http(port, "GET", f"/api/forum/batch?id={out['batch_id']}")[1]
    assert detail["statuses"] == {"completed": 5} and all(detail["by_arm"][a]["questions"] == 1 for a in detail["arms"])      # the same result as one after the other
    assert workers == 2 and most_at_once == 2
    assert not detail["job"]["batch"]["running"] and detail["job"]["batch"]["done"] == 5 and detail["job"]["batch"]["halted"] is None

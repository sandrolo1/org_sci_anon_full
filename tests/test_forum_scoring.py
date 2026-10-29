"""The evaluation of a forum proposal: novelty, methodology and impact against their own nearest papers (forum_scoring.py). The judges are a scripted stand-in and the embeddings are
the offline lexical ones, so nothing here calls a model."""
import csv
import io
import json
import re
import uuid

import numpy as np
import pytest

from research0 import embeddings, forum, forum_cards as fcards, forum_scoring as fscore, paper_cards, scoring_corpora, scoring_io, scoring_judges
from research0.forum_providers import TurnResult

from test_forum import fw, forum_world, http, make_episode, messages, needs_db, run_episode, server   # noqa: F401

EMBEDDER = embeddings.get_embedder("hash-rp-v1")
TOPICS = ["platform governance and complementor entry", "open innovation contests and solver diversity", "remote work and knowledge sharing in teams",
          "artificial intelligence adoption and task reallocation", "venture capital and startup scaling", "digital transformation and middle managers",
          "user communities and product innovation", "enterprise social media and accountability"]


# --- the rules computed by code -------------------------------------------------------------------------------------------------------------

def test_the_indicative_novelty_vote_follows_the_users_anchor_and_spreads_the_scale():
    assert fscore.novelty_indicative(0.50) == 8                                                # "a nearest neighbour at 0.5 is a good 8"
    assert [fscore.novelty_indicative(s) for s in (0.44, 0.56, 0.62, 0.68, 0.74, 0.80)] == [9, 7, 6, 5, 4, 3]
    assert fscore.novelty_indicative(0.10) == 10 and fscore.novelty_indicative(0.99) == 1         # kept within 1-10


def test_the_indicative_methodology_vote_rises_with_closeness_to_a_precedent():
    votes = [fscore.methodology_indicative(p) for p in (0.0, 0.25, 0.5, 0.75, 0.9, 1.0)]
    assert votes == sorted(votes) and votes[0] == 2 and votes[2] == 6 and votes[-1] == 9


def test_citations_are_read_as_a_percentile_within_the_year_of_publication():
    got = fscore.year_percentiles([("a", 2020, 0), ("b", 2020, 10), ("c", 2020, 10), ("d", 2020, 50), ("e", 2025, 3)])
    assert got["a"] == (0, 0.125) and got["b"] == got["c"] == (10, 0.5) and got["d"] == (50, 0.875)    # ties count half
    assert got["e"] == (3, 0.5)                                                                        # alone in its year: the middle, not the top


def test_the_similarity_indices_place_the_closest_paper_against_the_corpus():
    vectors = np.eye(4, dtype=np.float32)
    vectors[1] = [0.8, 0.6, 0, 0]
    dist = fscore.nearest_distribution(vectors)
    assert list(np.round(dist, 2)) == [0.0, 0.0, 0.8, 0.8]
    idx = fscore.similarity_indices([0.9, 0.5, 0.1], dist)
    assert idx["top1"] == 0.9 and idx["mean_topk"] == 0.5 and idx["k"] == 3 and idx["percentile"] == 1.0
    assert fscore.similarity_indices([0.5], dist)["percentile"] == 0.5


def test_each_judgement_reasons_before_it_votes_and_the_schemas_differ_where_the_criteria_do():
    for criterion in fscore.CRITERIA:
        props = list(fscore.judge_schema(criterion)["properties"])
        assert props[1] == "similarity_assessment" and props[-1] == "vote" and props.index("rationale") < props.index("vote")
    assert "success_condition" in fscore.judge_schema("impact")["properties"] and "relation_to_indicative_vote" not in fscore.judge_schema("impact")["properties"]
    assert "relation_to_indicative_vote" in fscore.judge_schema("novelty")["properties"]
    assert "Distance is GOOD" in fscore.SYSTEMS["novelty"] and "SUBSTANTIVE" in fscore.SYSTEMS["novelty"] and "not a rule" in fscore.SYSTEMS["novelty"]
    assert "Similarity is GOOD" in fscore.SYSTEMS["methodology"] and "CARDS ONLY" in fscore.SYSTEMS["methodology"]
    assert "mechanism_test" in fscore.judge_schema("methodology")["properties"] and "mechanism_test" not in fscore.judge_schema("novelty")["properties"]
    assert "mechanism_test" not in fscore.judge_schema("impact")["properties"]
    props = list(fscore.judge_schema("methodology")["properties"])
    assert props.index("comparisons") < props.index("mechanism_test") < props.index("rationale")           # the mechanism is judged before the vote
    assert "MECHANISM TEST" in fscore.SYSTEMS["methodology"] and "DEMONSTRATE" in fscore.SYSTEMS["methodology"] and "CAN BE CARRIED OUT" in fscore.SYSTEMS["methodology"]
    assert "POTENTIAL" in fscore.SYSTEMS["impact"] and "citations" in fscore.SYSTEMS["impact"].lower() and "CARDS ONLY" in fscore.SYSTEMS["impact"]


def answer(criterion, proposal_quote, quotes, vote=7):
    obj = {"input_id": "in_1", "similarity_assessment": {"substantive": "partly", "explanation": "Scripted."}, "rationale": "Scripted rationale.", "vote": vote,
           "comparisons": [{"neighbour_id": label, "neighbour_passage": quote, "proposal_passage": proposal_quote, "assessment": fscore.ASSESSMENTS[criterion][0],
                            "substantive": "yes", "explanation": "Scripted difference."} for label, quote in quotes]}
    if criterion == "impact":
        obj["success_condition"] = "The moderated effect emerges."
    else:
        obj["relation_to_indicative_vote"] = "Scripted: kept."
    if criterion == "methodology":
        obj["mechanism_test"] = {"mechanism": "Shared governance opens the platform.", "mechanism_passage": proposal_quote, "design_passage": proposal_quote, "verdict": "demonstrates",
                                 "rival_explanations": [], "explanation": "Scripted."}
    return obj


def hybrid_answer(proposal_quote, quotes, vote=7):
    """An answer of the hybrid novelty judge: every dimension of every comparison different, quoting `quotes` = [(label, passage of that candidate)] and the proposal."""
    def dimension(quote):
        return {"assessment": "different", "neighbour_passage": quote, "proposal_passage": proposal_quote, "explanation": "Scripted difference."}
    return {"input_id": "in_1", "similarity_assessment": {"substantive": "partly", "explanation": "Scripted."}, "rationale": "Scripted rationale.", "relation_to_indicative_vote": "Scripted: kept.",
            "vote": vote, "comparisons": [{"neighbour_id": label, **{d: dimension(quote) for d in fscore.DIMENSIONS}, "substantive": "yes", "explanation": "Scripted."} for label, quote in quotes]}


def test_the_hybrid_judgement_compares_question_mechanism_and_predictions_with_quotations():
    proposal = "ABSTRACT:\nPlatforms reorganise how firms coordinate innovation across partners."
    texts = {"V1": "Paper one studies platform governance and complementor entry.", "V2": "Paper two studies open innovation contests and solver diversity.",
             "V3": "Paper three studies remote work and knowledge sharing in teams."}
    quotes = [("V1", "studies platform governance and complementor"), ("V2", "open innovation contests and solver"), ("V3", "remote work and knowledge sharing")]
    good = hybrid_answer("Platforms reorganise how firms coordinate", quotes)
    check = lambda o, lexical_only=("V3",): fscore.validate_hybrid(o, input_id="in_1", proposal_text=proposal, neighbour_texts=texts, lexical_only=lexical_only)
    assert check(good) == []
    assert list(fscore.hybrid_schema()["properties"]["comparisons"]["items"]["properties"])[1:4] == ["question", "mechanism", "predictions"]
    assert any("at least 3" in e for e in check({**good, "comparisons": good["comparisons"][:2]}))
    assert any("found only by the lexical search" in e for e in check({**good, "comparisons": good["comparisons"][:2] + [good["comparisons"][0]]}))            # V3 is the lexical one and was not compared
    invented = {**good["comparisons"][0], "mechanism": {**good["comparisons"][0]["mechanism"], "neighbour_passage": "a sentence that no candidate ever wrote"}}
    assert any("comparisons[0].mechanism: neighbour_passage" in e for e in check({**good, "comparisons": [invented] + good["comparisons"][1:]}))
    wrong_text = {**good["comparisons"][0], "predictions": {**good["comparisons"][0]["predictions"], "proposal_passage": "words the proposal does not contain"}}
    assert any("predictions: proposal_passage" in e for e in check({**good, "comparisons": [wrong_text] + good["comparisons"][1:]}))
    silent = {**good["comparisons"][0], "predictions": {"assessment": "not_stated", "neighbour_passage": "", "proposal_passage": "", "explanation": "Neither abstract states a prediction."}}
    assert check({**good, "comparisons": [silent] + good["comparisons"][1:]}) == []                                  # an abstract that does not say it may leave the passage empty ...
    lying = {**silent, "predictions": {**silent["predictions"], "neighbour_passage": "a sentence that no candidate ever wrote"}}
    assert check({**good, "comparisons": [lying] + good["comparisons"][1:]}) != []                                    # ... but a quotation, when given, must be exact
    empty = {**good["comparisons"][0], "question": {**good["comparisons"][0]["question"], "neighbour_passage": ""}}
    assert any("comparisons[0].question: neighbour_passage" in e for e in check({**good, "comparisons": [empty] + good["comparisons"][1:]}))              # a stated dimension needs its passages
    assert any("input_id" in e for e in check({**good, "input_id": "other"})) and check({**good, "vote": 11}) != []
    assert "QUESTION" in fscore.SYSTEMS[fscore.HYBRID] and "MECHANISM" in fscore.SYSTEMS[fscore.HYBRID] and "PREDICTIONS" in fscore.SYSTEMS[fscore.HYBRID] and "lexical" in fscore.SYSTEMS[fscore.HYBRID]


def test_a_judgement_must_quote_both_texts_exactly_and_compare_with_at_least_two_neighbours():
    proposal = "ABSTRACT:\nPlatforms reorganise how firms coordinate innovation across partners."
    texts = {"V1": "Paper one studies platform governance and complementor entry.", "V2": "Paper two studies open innovation contests and solver diversity."}
    good = answer("novelty", "Platforms reorganise how firms coordinate", [("V1", "studies platform governance and complementor"), ("V2", "open innovation contests and solver")])
    check = lambda o: fscore.validate_judgement(o, criterion="novelty", input_id="in_1", proposal_text=proposal, neighbour_texts=texts)
    assert check(good) == []
    assert any("neighbour_passage" in e for e in check({**good, "comparisons": [{**good["comparisons"][0], "neighbour_passage": "a sentence no neighbour ever wrote"}] + good["comparisons"][1:]}))
    assert any("proposal_passage" in e for e in check({**good, "comparisons": [{**c, "proposal_passage": "words the proposal does not contain"} for c in good["comparisons"]]}))
    assert any("at least 2" in e for e in check({**good, "comparisons": good["comparisons"][:1]}))
    assert any("not a neighbour" in e for e in check({**good, "comparisons": [{**c, "neighbour_id": "V9"} for c in good["comparisons"]]}))
    assert any("input_id" in e for e in check({**good, "input_id": "other"}))
    assert check({**good, "vote": 11})                                                              # out of the scale


def test_the_methodology_judge_must_say_how_far_the_design_demonstrates_the_mechanism_and_quote_both_sides():
    proposal = ("ABSTRACT OF THE PROPOSAL:\nBoundary resources open the platform only when governance is shared.\n\nTHE PROPOSAL'S METHOD CARD:\n"
                "Design (inferred): A panel of platforms observes the sharing step in time. «panel of platforms observes the sharing»")
    texts = {"V1": "Paper one studies platform governance and complementor entry.", "V2": "Paper two studies open innovation contests and solver diversity."}
    good = answer("methodology", "Boundary resources open the platform only when", [("V1", "studies platform governance and complementor"), ("V2", "open innovation contests and solver")])
    test = good["mechanism_test"]
    check = lambda t: fscore.validate_judgement({**good, "mechanism_test": t}, criterion="methodology", input_id="in_1", proposal_text=proposal, neighbour_texts=texts)
    exact = {**test, "mechanism_passage": "Boundary resources open the platform only when governance is shared", "design_passage": "panel of platforms observes the sharing"}
    assert check(exact) == []
    assert any("mechanism_passage" in e for e in check({**exact, "mechanism_passage": "a mechanism the proposal never states anywhere"}))        # a quotation must be exact ...
    assert any("design_passage" in e for e in check({**exact, "design_passage": "a design element that is not in the card at all"}))
    assert any("design_passage" in e for e in check({**exact, "design_passage": ""}))                                                                 # ... and a stated mechanism needs both
    only = {**exact, "verdict": "consistent_only", "rival_explanations": []}
    assert any("rival_explanations" in e for e in check(only))                                                                                         # a design that only fits the mechanism names what it leaves open
    assert any("rival_explanations" in e for e in check({**only, "verdict": "partly_demonstrates", "rival_explanations": ["  "]}))
    assert check({**only, "rival_explanations": ["Selection of platforms that were already open."]}) == []
    silent = {**exact, "verdict": "mechanism_not_stated", "mechanism_passage": "", "design_passage": ""}
    assert check(silent) == []                                                                                                                         # nothing to quote when no mechanism is stated ...
    assert check({**silent, "mechanism_passage": "an invented mechanism that no text states"}) != []                                                  # ... but a quotation, when given, is checked
    assert any("explanation is empty" in e for e in check({**exact, "explanation": " "})) and any("mechanism is empty" in e for e in check({**exact, "mechanism": ""}))
    assert check({**exact, "verdict": "proves_it"}) != []                                                                                              # only the four verdicts
    assert fscore.validate_judgement({k: v for k, v in good.items() if k != "mechanism_test"}, criterion="methodology", input_id="in_1", proposal_text=proposal, neighbour_texts=texts)   # the field is required


# --- a whole evaluation, offline ----------------------------------------------------------------------------------------------------------

def card(kind, topic, i):
    return {**{f: {"text": f"{f.replace('_', ' ')} of a study on {topic}", "status": "extracted", "passages": [f"A passage of paper {i} about {topic}."]}
               for f in paper_cards.METHOD.fields if kind == "method"},
            **{f: {"text": f"{f.replace('_', ' ')} of a study on {topic}", "status": "extracted", "passages": [f"A passage of paper {i} about {topic}."]}
               for f in fscore.PAPER_PROFILES["impact"].fields if kind == "impact"}, "info_gaps": []}


def entry(i):
    return {"id": f"P{i}", "title": f"Paper {i} on {TOPICS[i]}", "journal": f"Journal {i % 3}", "journal_key": f"journal {i % 3}", "year": 2018 + i, "publication_date": f"{2018 + i}-12-31",
            "family_id": f"P{i}", "quartile": "", "ajg2024": "4" if i % 2 else "3", "doi": f"10.1/{i}", "cited_by": 10 * i, "citation_percentile": round(i / len(TOPICS), 3)}


def build_corpora(root):
    root.mkdir(parents=True, exist_ok=True)
    nov = root / "cor_test"
    nov.mkdir()
    abstracts = [f"This paper examines {t}. We theorise a mechanism and test it on a panel of firms, and we discuss what it changes for the literature on {t}." for t in TOPICS]
    np.save(nov / "vectors.npy", np.asarray(EMBEDDER.embed([f"Paper {i} on {t}\n{a}" for i, (t, a) in enumerate(zip(TOPICS, abstracts))]), dtype=np.float32))
    for i, a in enumerate(abstracts):
        scoring_io.append_jsonl(nov / "entries.jsonl", {**entry(i), "community": "CTO"})
        scoring_io.append_jsonl(nov / "texts.jsonl", {"id": f"P{i}", "text": scoring_corpora.ABSTRACT_CARD + a})
    scoring_io.write_json_once(nov / "corpus.json", {"schema": "anvur-corpus-v1", "corpus_id": "cor_test", "embedder": EMBEDDER.model_id, "entries": len(TOPICS), "journals": 3, "min_similarity": 0.5})
    made = {}
    for kind in ("method", "impact"):
        corpus = scoring_corpora.build_card_corpus([entry(i) for i in range(len(TOPICS))], {f"P{i}": card(kind, t, i) for i, t in enumerate(TOPICS)}, kind, EMBEDDER, root,
                                                   source={"test": kind})
        made[kind] = corpus.dir
    return {"novelty": nov, "methodology": made["method"], "impact": made["impact"]}


def write_config(path, k=3, lexical_k=0):
    cfg = {**scoring_judges.load_config("config/forum_scoring.json"), "run_live": False, "embedding": "hash-rp-v1", "k": k, "lexical_k": lexical_k, "concurrency": 1,
           "budget": {"max_live_calls": 0, "max_tokens": 0, "max_seconds": 0}}
    cfg["judges"] = [{**j, "provider": "synthetic"} for j in cfg["judges"]]
    path.write_text(json.dumps(scoring_judges.validate_config(cfg)), encoding="utf-8")
    return path


def proposal_content():
    abstract = ("Platforms reorganise how firms govern complementor entry. We expect that boundary resources open the platform only when governance is shared, and we discuss what this "
                "changes for the literature on platform governance and complementor entry.")
    method = card("method", "platform governance and complementor entry", 99)
    impact = card("impact", "platform governance and complementor entry", 99)
    return {"title": "Who opens the platform?", "abstract": abstract, "introduction": "Intro [E1].", "theoretical_background": "Background [E2].",
            "cards": {"method": fcards.record("method", "valid", source="x", card=method), "impact": fcards.record("impact", "valid", source="x", card=impact)}}


class Judge:
    """Answers the three judgements with quotations taken from the prompt it is shown. NOT a model. `bad` = {criterion: how many first answers per judge invent a quotation}."""
    name, synthetic, cache_identity = "synthetic", True, "scripted-forum-judge-1"

    def __init__(self, vote=7, bad=None, hybrid_vote=None):
        self.requests, self.vote, self.bad, self.seen, self.hybrid_vote = [], vote, dict(bad or {}), {}, hybrid_vote

    def run(self, request):
        self.requests.append(request)
        criterion = request.task.split("forum_score_", 1)[1]
        key = (criterion, request.model, request.hints.get("role"))
        self.seen[key] = self.seen.get(key, 0) + 1
        iid = re.search(r"INPUT_ID: (\S+)", request.prompt).group(1)
        proposal = request.prompt.split("<<<PROPOSAL\n", 1)[1].split("\n>>>END_PROPOSAL", 1)[0]
        mine = next(line for line in reversed(proposal.splitlines()) if len(line) > 40)[:70]
        blocks = re.findall(r"NEIGHBOUR: (V\d+)\n.*?\n(?:ABSTRACT|CARD):\n(.*?)(?=\n\nNEIGHBOUR: |\n\nYOUR PREVIOUS|\n*\Z)", request.prompt, re.S)
        chosen = blocks[:2]
        if criterion == fscore.HYBRID:           # the hybrid judge compares the two closest and one that only the lexical search brought
            found = dict(re.findall(r"NEIGHBOUR: (V\d+)\n.*?FOUND BY: ([^\n]*)", request.prompt, re.S))
            chosen += [b for b in blocks if "semantic" not in found[b[0]]][:1]
        quotes = [(label, text.strip().splitlines()[0][:70]) for label, text in chosen]
        if self.seen[key] <= self.bad.get(criterion, 0):
            quotes = [(label, "a sentence that no neighbour of this dossier ever wrote") for label, _ in quotes]
        if criterion == fscore.HYBRID:
            obj = {**hybrid_answer(mine, quotes, self.vote if self.hybrid_vote is None else self.hybrid_vote), "input_id": iid}
        else:
            obj = {**answer(criterion, mine, quotes, self.vote), "input_id": iid}
        text = json.dumps(obj)
        return TurnResult(text=text, model=request.model, resolved_model="scripted", llm_calls=1, input_tokens=len(request.prompt) // 4, output_tokens=len(text) // 4, latency_ms=1)


@pytest.fixture
def setup(tmp_path):
    where = {"config": write_config(tmp_path / "cfg.json"), "workspace": tmp_path / "ws", **build_corpora(tmp_path / "corpora")}
    cfg = scoring_judges.load_config(where["config"])
    return where, cfg


def score(where, cfg, judge, content=None, session="t1", lexical_k=0):
    judges = scoring_judges.Judges(where["workspace"], cfg, {r: judge for r in scoring_judges.JUDGE_ROLES}, session_id=session)
    corpora = fscore.load_corpora(where, where["workspace"])
    return fscore.score_proposal(judges, corpora, EMBEDDER, cfg["k"], "in_1", content or proposal_content(), lexical_k=lexical_k)


def test_three_votes_each_against_its_own_neighbours_with_all_the_similarity_indices(setup):
    where, cfg = setup
    judge = Judge(vote=7)
    criteria = score(where, cfg, judge)
    assert [r.task for r in judge.requests].count("forum_score_novelty") == 2 and len(judge.requests) == 6        # two judges per criterion
    for criterion in fscore.CRITERIA:
        c = criteria[criterion]
        assert c["status"] == "scored" and c["vote"] == 7.0 and c["votes"] == {"judge_1": 7, "judge_2": 7} and len(c["neighbours"]) == 3
        s = c["similarity"]
        assert s["k"] == 3 and s["top1"] >= s["mean_topk"] and 0 <= s["percentile"] <= 1 and s["corpus_median_nearest"] is not None
        assert [n["label"] for n in c["neighbours"]] == ["V1", "V2", "V3"] and "text" not in c["neighbours"][0]
    assert 1 <= criteria["novelty"]["indicative_vote"] <= 10 and 1 <= criteria["methodology"]["indicative_vote"] <= 10 and criteria["impact"]["indicative_vote"] is None
    assert criteria["impact"]["neighbours"][0]["cited_by"] is not None and criteria["impact"]["similarity"]["neighbours_citation_percentile_median"] is not None
    assert criteria["novelty"]["neighbours"][0]["title"].startswith("Paper 0")            # the closest abstract is the one on the same topic
    prompts = {r.task.split("_")[-1]: r.prompt for r in judge.requests}
    assert "INDICATIVE VOTE" in prompts["novelty"] and "INDICATIVE VOTE" in prompts["methodology"] and "INDICATIVE VOTE" not in prompts["impact"]
    assert "CITATIONS:" in prompts["impact"] and "PERCENTILE AMONG THE PAPERS OF" in prompts["impact"] and "CITATIONS:" not in prompts["methodology"]
    assert "THE PROPOSAL'S METHOD CARD" in prompts["methodology"] and "[E1]" not in prompts["novelty"]
    assert all(j["mechanism_test"]["verdict"] == "demonstrates" for j in criteria["methodology"]["judges"].values())       # stored with the answer, for the page
    assert all("mechanism_test" not in j for c in ("novelty", "impact") for j in criteria[c]["judges"].values())


def test_the_hybrid_judge_reads_the_closest_by_embedding_and_what_only_the_lexical_search_finds(setup):
    where, cfg = setup
    judge = Judge(vote=7, hybrid_vote=4)
    criteria = score(where, cfg, judge, session="h1", lexical_k=2)
    tasks = [r.task for r in judge.requests]
    assert tasks.count("forum_score_novelty_hybrid") == 2 and tasks.count("forum_score_novelty") == 2 and len(tasks) == 8          # four calls per judge: the three criteria and the hybrid one
    novelty, hybrid = criteria["novelty"], criteria["novelty"]["hybrid"]
    assert novelty["vote"] == 7.0 and hybrid["vote"] == 4.0 and hybrid["votes"] == {"judge_1": 4, "judge_2": 4} and hybrid["version"] == fscore.HYBRID_VERSION      # beside, not instead
    assert fscore._status(criteria) == "ok" and fscore._mean3({c: criteria[c]["vote"] for c in fscore.CRITERIA}) == 7.0                                          # the row and the mean ignore it
    assert len(hybrid["candidates"]) == 5 and [c["label"] for c in hybrid["candidates"]] == ["V1", "V2", "V3", "V4", "V5"]
    assert [c["id"] for c in hybrid["candidates"][:3]] == [n["id"] for n in novelty["neighbours"]]                                  # the first V1-V3 are the first judge's neighbours
    assert hybrid["lexical_only"] == ["V4", "V5"] and all(c["found_by"] == ["lexical"] and c["lexical_rank"] and c["semantic_rank"] > 3 for c in hybrid["candidates"][3:])
    assert all("text" not in c for c in hybrid["candidates"])
    prompts = {r.task: r.prompt for r in judge.requests}
    assert "FOUND BY: lexical search (rank" in prompts["forum_score_novelty_hybrid"] and "embedding rank" in prompts["forum_score_novelty_hybrid"] and "CANDIDATES:" in prompts["forum_score_novelty_hybrid"]
    assert "FOUND BY" not in prompts["forum_score_novelty"] and "CANDIDATES" not in prompts["forum_score_novelty"]                    # the first judge's dossier is what it was
    assert set(hybrid["judges"]["judge_1"]["comparisons"][0]) >= {"question", "mechanism", "predictions"}
    assert "hybrid" not in criteria["methodology"] and "hybrid" not in criteria["impact"]


def test_the_first_novelty_judge_is_not_asked_again_when_the_hybrid_judge_is_added(setup):
    where, cfg = setup
    first = Judge(vote=7)
    plain = score(where, cfg, first, session="p1")                                                                                 # as before: three criteria, no hybrid
    assert "hybrid" not in plain["novelty"] and len(first.requests) == 6
    second = Judge(vote=7, hybrid_vote=3)
    again = score(where, cfg, second, session="p2", lexical_k=2)
    assert [r.task for r in second.requests] == ["forum_score_novelty_hybrid"] * 2                                                  # the six other answers come from the register
    assert {c: again[c]["vote"] for c in fscore.CRITERIA} == {c: plain[c]["vote"] for c in fscore.CRITERIA}
    assert again["novelty"]["judges"] == plain["novelty"]["judges"] and again["novelty"]["hybrid"]["vote"] == 3.0


def test_a_refused_hybrid_answer_is_asked_for_again_and_a_failed_one_does_not_fail_the_evaluation(setup):
    where, cfg = setup
    judge = Judge(bad={"novelty_hybrid": 2})
    criteria = score(where, cfg, judge, session="h2", lexical_k=2)
    assert criteria["novelty"]["hybrid"]["status"] == "scored"
    retries = [r for r in judge.requests if r.task == "forum_score_novelty_hybrid" and "REFUSED BY THE CHECK" in r.prompt]
    assert retries and "neighbour_passage is not an exact piece" in retries[0].prompt
    hopeless = score(where, cfg, Judge(bad={"novelty_hybrid": 99}), session="h3", lexical_k=3)
    assert hopeless["novelty"]["hybrid"]["status"] == "failed" and hopeless["novelty"]["hybrid"]["vote"] is None and hopeless["novelty"]["hybrid"]["reason"]
    assert hopeless["novelty"]["status"] == "scored" and fscore._status(hopeless) == "ok"                                           # the first judge's evaluation stands


def test_the_two_novelty_judges_are_compared_on_the_proposals_both_voted():
    rows = [{"votes": {"novelty": 7.0}, "hybrid_vote": 5.0}, {"votes": {"novelty": 6.0}, "hybrid_vote": 6.0}, {"votes": {"novelty": 8.0}, "hybrid_vote": 7.5},
            {"votes": {"novelty": 8.0}, "hybrid_vote": None}, {"votes": {"novelty": None}, "hybrid_vote": 4.0}]
    got = fscore.novelty_comparison(rows)
    assert got == {"n": 3, "mean_first": 7.0, "mean_hybrid": 6.167, "mean_difference": -0.833, "mean_abs_difference": 0.833, "exact_share": 0.333, "within_one_share": 0.667}
    assert fscore.novelty_comparison([])["n"] == 0 and fscore.novelty_comparison([])["mean_difference"] is None
    doc = {"criteria": {"novelty": {"hybrid": {"vote": 5.5, "votes": {"judge_1": 5}, "lexical_only": ["V6", "V7"]}}}}
    assert fscore.hybrid_summary(doc) == {"vote": 5.5, "votes": {"judge_1": 5}, "lexical_only": 2}
    assert fscore.hybrid_summary({"criteria": {"novelty": {"vote": 7}}}) is None                                                    # an evaluation made before the hybrid judge
    assert fscore.hybrid_summary({"criteria": {"novelty": {"hybrid": {"status": "failed", "vote": None}}}}) is None


def test_lexical_k_is_an_optional_whole_number_and_the_real_configuration_turns_the_hybrid_judge_on():
    cfg = scoring_judges.load_config("config/forum_scoring.json")
    assert cfg["lexical_k"] == 5 and scoring_judges.validate_config({k: v for k, v in cfg.items() if k != "lexical_k"})["k"] == cfg["k"]
    for bad in (-1, 11, 2.5, True, "3"):
        with pytest.raises(scoring_judges.ConfigError, match="lexical_k"):
            scoring_judges.validate_config({**cfg, "lexical_k": bad})


def test_one_judge_scores_and_a_fixed_sample_of_the_proposals_gets_the_second(setup, monkeypatch):
    where, cfg = setup
    judge = Judge(vote=7)
    judges = scoring_judges.Judges(where["workspace"], cfg, {r: judge for r in scoring_judges.JUDGE_ROLES}, session_id="t-one")
    criteria = fscore.score_proposal(judges, fscore.load_corpora(where, where["workspace"]), EMBEDDER, cfg["k"], "in_1", proposal_content(), roles=("judge_1",))
    assert len(judge.requests) == 3 and {r.hints["role"] for r in judge.requests} == {"forum_score_novelty", "forum_score_methodology", "forum_score_impact"}
    for c in fscore.CRITERIA:
        assert criteria[c]["votes"] == {"judge_1": 7} and criteria[c]["vote"] == 7.0 and set(criteria[c]["judges"]) == {"judge_1"}
    ids = [str(uuid.UUID(int=i)) for i in range(400)]
    assert all(fscore.panel(i, 0.0) == ("judge_1",) for i in ids) and all(fscore.panel(i, 1.0) == ("judge_1", "judge_2") for i in ids)
    double = sum(len(fscore.panel(i, 0.2)) == 2 for i in ids)
    assert 50 <= double <= 110 and [fscore.panel(i, 0.2) for i in ids] == [fscore.panel(i, 0.2) for i in ids]     # about a fifth, and always the same proposals
    monkeypatch.setenv("FORUM_SCORING_DOUBLE_SHARE", "1.5")
    with pytest.raises(ValueError):
        fscore.double_share()


def test_the_agreement_of_the_two_judges_is_measured_where_both_voted():
    doc = {"criteria": {"novelty": {"votes": {"judge_1": 8, "judge_2": 8}}, "methodology": {"votes": {"judge_1": 7, "judge_2": 6}}, "impact": {"votes": {"judge_1": 4}}}}
    assert fscore.judge_gaps(doc) == {"novelty": 0, "methodology": 1}
    assert fscore.agreement([fscore.judge_gaps(doc), {}]) == {"n": 2, "exact_share": 0.5, "mean_gap": 0.5, "max_gap": 1}
    assert fscore.agreement([]) == {"n": 0, "exact_share": None, "mean_gap": None, "max_gap": None}


def test_the_cards_are_retrieved_on_the_card_and_nothing_else(setup):
    texts, why = fscore.proposal_texts(proposal_content())
    card_text = paper_cards.card_text(proposal_content()["cards"]["method"]["card"], paper_cards.METHOD)
    assert texts["methodology"][0] == card_text and "Platforms reorganise" not in texts["methodology"][0]           # the abstract is shown to the judge, never embedded for the card
    assert texts["novelty"][0].startswith("Who opens the platform?\nPlatforms reorganise") and why == {}


def test_a_refused_answer_is_asked_for_again_with_the_reasons(setup):
    where, cfg = setup
    judge = Judge(bad={"impact": 2})                     # the two attempts of the first call fail, the next one, with the reasons, passes
    criteria = score(where, cfg, judge, session="t2")
    assert criteria["impact"]["status"] == "scored"
    retries = [r for r in judge.requests if r.task == "forum_score_impact" and "REFUSED BY THE CHECK" in r.prompt]
    assert retries and "neighbour_passage is not an exact piece" in retries[0].prompt


def test_a_criterion_that_cannot_be_judged_says_why_and_the_others_still_are(setup):
    where, cfg = setup
    content = proposal_content()
    content["cards"]["impact"] = fcards.record("impact", "rejected", problems=["x"])
    criteria = score(where, cfg, Judge(), content, session="t3")
    assert criteria["impact"] == {"status": "not_scored", "reason": "the proposal has no valid impact card", "vote": None}
    assert criteria["novelty"]["status"] == criteria["methodology"]["status"] == "scored" and fscore._status(criteria) == "partial"
    missing = {**where, "methodology": where["workspace"] / "nothing"}
    criteria = score(missing, cfg, Judge(), session="t4")
    assert criteria["methodology"]["status"] == "not_scored" and "forum-score-corpus" in criteria["methodology"]["reason"]


def test_an_old_proposal_and_a_simulated_episode_are_not_scored(setup):
    where, _ = setup
    assert fscore.score_proposals(None, [], type("P", (), {"name": "scripted"})(), settings_=where) == {"skipped": "the episode does not run on the OpenAI provider"}
    texts, why = fscore.proposal_texts({"title": "Old", "introduction": "x", "methods": "y"})
    assert texts == {} and set(why) == set(fscore.CRITERIA)


# --- stored, shown, ranked (mock database) ---------------------------------------------------------------------------------------------

@needs_db
def test_the_evaluation_is_stored_shown_with_the_proposal_ranked_and_old_proposals_are_hidden(fw, setup, server):
    where, _ = setup
    episode = make_episode(fw, "A", protocol={"rounds": 1})
    assert run_episode(fw, episode) == "completed"
    notes = [m["content"] for m in messages(fw, episode, kind="note") if m["data"].get("role") == "evaluation"]
    assert notes == ["The proposals were not evaluated: the episode does not run on the OpenAI provider"]      # the scripted episodes cost nothing
    with fw.world.connect() as c:
        (pid,) = c.execute("SELECT proposal_id FROM forum_proposal WHERE episode_id=%s", (episode,)).fetchone()
        out = fscore.score_proposals(c, [pid], None, settings_=where, synthetic_provider=Judge(vote=6), embedder=EMBEDDER)
        assert out["stored"] == 1 and out["statuses"] == {"ok": 1, "partial": 0, "failed": 0}
        assert c.execute("SELECT status,novelty,methodology,impact,synthetic FROM forum_proposal_score WHERE proposal_id=%s", (pid,)).fetchone() == ("ok", 6, 6, 6, True)
        shown = forum.episode_detail(c, episode)["proposals"][0]["score"]
        assert shown["mean"] == 6.0 and shown["votes"] == {"novelty": 6.0, "methodology": 6.0, "impact": 6.0} and shown["similarity"]["novelty"]["top1"] is not None
        detail = fscore.detail(c, pid)
        assert detail["scored"] and set(detail["score"]["criteria"]) == set(fscore.CRITERIA) and detail["score"]["criteria"]["impact"]["judges"]["judge_1"]["success_condition"]
        c.execute("UPDATE forum_episode SET provider='openai' WHERE episode_id=%s", (episode,))      # the results rank real episodes only: this test episode stands in for one
        c.commit()
        board = fscore.leaderboard(c, fw.world.run_id)
        row = next(r for r in board["rows"] if r["proposal_id"] == str(pid))
        assert row["scored"] and row["mean"] == 6.0 and row["similarity"]["impact"]["neighbours_citation_percentile_median"] is not None
        exported = list(csv.DictReader(io.StringIO(fscore.export_csv(board["rows"]).decode("utf-8-sig"))))
        line = next(x for x in exported if x["proposal_id"] == str(pid))
        assert len(exported) == len(board["rows"]) and line["mean"] == "6.0" and line["vote_impact"] == "6.0" and line["sim_impact_neighbours_citation_percentile_median"] != ""
        stats = board["stats"]["A"]
        assert stats["scored"] == 1 and stats["votes"]["impact"]["mean"] == 6.0 and set(stats["similarity"]) == {k["key"] for k in board["similarity_keys"]}
        assert stats["similarity"]["methodology.percentile"]["n"] == 1 and fscore.pending(c, fw.world.run_id) == []
        port, run_id = server()
        status, page = http(port, "GET", "/api/forum/evaluations")
        assert status == 200 and any(r["proposal_id"] == str(pid) for r in page["rows"]) and page["job"] == {"running": False}
        status, err = http(port, "POST", "/api/forum/evaluation/run", {"run_id": run_id, "proposal_id": str(pid)})
        assert status == 400 and "OpenAI" in err["error"]                                                    # it costs money: never without the confirmation
        c.execute("UPDATE forum_proposal SET content = content - 'cards' WHERE proposal_id=%s", (pid,))       # as if it had been written in the earlier format
        c.commit()
        assert forum.episode_detail(c, episode)["proposals"] == [] and forum.episode_detail(c, episode)["hidden_proposals"] == 1
        board = fscore.leaderboard(c, fw.world.run_id)
        assert not any(r["proposal_id"] == str(pid) for r in board["rows"]) and board["counts"]["hidden_old_format"] >= 1
        assert forum.list_episodes(c, [fw.release_id])[0]["proposals"] == 0


@needs_db
def test_the_hybrid_vote_is_stored_beside_the_first_and_the_scored_proposals_without_it_are_found(fw, setup):
    where, _ = setup
    episode = make_episode(fw, "A", protocol={"rounds": 1})
    assert run_episode(fw, episode) == "completed"
    with fw.world.connect() as c:
        (pid,) = c.execute("SELECT proposal_id FROM forum_proposal WHERE episode_id=%s", (episode,)).fetchone()
        c.execute("UPDATE forum_episode SET provider='openai' WHERE episode_id=%s", (episode,))      # the results rank real episodes only: this test episode stands in for one
        c.commit()
        assert fscore.pending_hybrid(c, fw.world.run_id) == []                                           # nothing is scored yet: that is `pending`'s business
        out = fscore.score_proposals(c, [pid], None, settings_=where, synthetic_provider=Judge(vote=6), embedder=EMBEDDER)       # as before the hybrid judge: lexical_k 0
        assert out["stored"] == 1 and fscore.pending(c, fw.world.run_id) == [] and fscore.pending_hybrid(c, fw.world.run_id) == [pid]
        with_hybrid = {**where, "config": write_config(where["config"].with_name("cfg_hybrid.json"), lexical_k=2)}
        again = Judge(vote=6, hybrid_vote=4)
        out = fscore.score_proposals(c, [pid], None, settings_=with_hybrid, synthetic_provider=again, embedder=EMBEDDER)
        assert out["stored"] == 1 and {r.task for r in again.requests} == {"forum_score_novelty_hybrid"}        # only the hybrid judge is paid for: the other answers are in the register
        assert fscore.pending_hybrid(c, fw.world.run_id) == []
        assert c.execute("SELECT count(*) FROM forum_proposal_score WHERE proposal_id=%s", (pid,)).fetchone()[0] == 2           # a new row; the first stays on record
        assert c.execute("SELECT novelty,methodology,impact FROM forum_proposal_score WHERE proposal_id=%s ORDER BY created_at DESC LIMIT 1", (pid,)).fetchone() == (6, 6, 6)
        board = fscore.leaderboard(c, fw.world.run_id)
        row = next(r for r in board["rows"] if r["proposal_id"] == str(pid))
        assert row["votes"]["novelty"] == 6.0 and row["hybrid_vote"] == 4.0 and row["mean"] == 6.0
        stats = board["stats"]["A"]
        assert stats["novelty_hybrid"]["mean"] == 4.0 and stats["novelty_comparison"]["n"] == 1 and stats["novelty_comparison"]["mean_difference"] == -2.0
        shown = fscore.detail(c, pid)["summary"]["hybrid"]                                                # one judge or both, by the proposal's id: every vote is the scripted 4
        assert shown["vote"] == 4.0 and set(shown["votes"].values()) == {4} and shown["lexical_only"] == 2


def test_the_results_export_is_one_flat_row_per_proposal_with_empty_cells_for_missing_values():
    sim = {c: {"top1": 0.5, "mean_topk": 0.4, "percentile": 0.3} for c in fscore.CRITERIA}
    sim["impact"]["neighbours_citation_percentile_median"] = 0.7
    scored = {"proposal_id": "p1", "episode_id": "e1", "condition": "A", "trigger": "t", "cross_community": False, "web": True, "tools": False, "unvalidated": False, "budgets": "abc",
              "author": "cto", "pages": 3, "status": "final", "title": "A, \"quoted\" title", "episode_title": "ep", "scored": True, "score_status": "ok",
              "votes": {"novelty": 7.0, "methodology": 5.0, "impact": 6.0}, "mean": 6.0, "hybrid_vote": 4.0, "similarity": sim, "indicative": {"novelty": 8, "methodology": 5, "impact": 6},
              "judge_gaps": {"novelty": 1}, "at": "2026-10-29T10:00:00", "created_at": "2026-10-29T09:00:00"}
    pending = {**scored, "proposal_id": "p2", "scored": False, "score_status": None, "votes": {c: None for c in fscore.CRITERIA}, "mean": None, "hybrid_vote": None,
               "similarity": {c: {} for c in fscore.CRITERIA}, "indicative": {}, "judge_gaps": {}, "at": None}
    raw = fscore.export_csv([scored, pending])
    assert raw.startswith("﻿".encode("utf-8"))
    rows = list(csv.DictReader(io.StringIO(raw.decode("utf-8-sig"))))
    assert list(rows[0]) == list(fscore.EXPORT_FIELDS) and len(rows) == 2
    assert rows[0]["title"] == 'A, "quoted" title' and rows[0]["vote_novelty"] == "7.0" and rows[0]["vote_novelty_hybrid"] == "4.0" and rows[0]["judge_gap_novelty"] == "1"
    assert rows[0]["judge_gap_impact"] == "" and rows[0]["sim_impact_neighbours_citation_percentile_median"] == "0.7" and rows[0]["rubric_version"] == fscore.VERSION
    assert rows[1]["mean"] == rows[1]["vote_impact"] == rows[1]["sim_novelty_top1"] == rows[1]["rubric_version"] == rows[1]["scored_at"] == ""


def test_the_forum_evaluator_defaults_to_gpt6_luna_and_offers_the_benchmark_judge(monkeypatch):
    monkeypatch.delenv("FORUM_SCORING_CONFIG", raising=False)
    base = fscore.settings()
    assert fscore.settings_for(None) == fscore.settings_for("default") == base
    assert base["config"] == fscore.ROOT / "config" / "forum_scoring.json"                # what the automatic run, the CLI and the page use unless told otherwise
    default = scoring_judges.load_config(base["config"])
    assert {(j["model"]["id"], j["params"]["reasoning_effort"]) for j in default["judges"]} == {("gpt-6-luna", "medium")}
    gpt52 = fscore.settings_for("gpt52")
    assert gpt52["config"] == fscore.ROOT / "config" / "evaluation_pre2026.json" and gpt52["novelty"] == base["novelty"]
    cfg = scoring_judges.load_config(gpt52["config"])
    assert {j["model"]["id"] for j in cfg["judges"]} == {"gpt-5.2-2025-12-11"}
    assert [j["id"] for j in fscore.describe()["judges"]] == ["default", "gpt52"]
    with pytest.raises(ValueError):
        fscore.settings_for("gpt-4")

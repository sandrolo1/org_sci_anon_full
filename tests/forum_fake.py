"""The offline stand-in for a model, for the automatic tests only.

The scripted simulation was removed from the forum on 13 Oct 2026 (the user's request): the page, the API and the CLI run episodes only with a real model. The tests
still need whole episodes that call no model and cost nothing, so the stand-in lives here, outside the package, and `register()` (called by conftest.py) makes the
provider name "scripted" known to the protocol and to `get_provider` for the test session only. Nothing in `src/` can select it.
"""
import json
import os
import re
import time

from research0 import forum_protocol as proto
from research0 import forum_providers
from research0.forum_providers import TurnResult, _usage

NAME, MODEL = "scripted", "scripted-v1"
STOP = set("the of and to in a is that for with as on by this are be from or an it at which was were has have not can their these those into than then".split())


def register():
    proto.PROVIDERS[NAME] = MODEL
    forum_providers.FACTORIES[NAME] = lambda: ScriptedProvider(delay=float(os.environ.get("FORUM_SCRIPTED_DELAY", 0)))   # seconds per turn: a test can stop a running episode


def _terms(text, n=6):
    seen, out = set(), []
    for word in re.findall(r"[a-zA-Z][a-zA-Z-]{3,}", text or ""):
        w = word.lower()
        if w not in STOP and w not in seen:
            seen.add(w)
            out.append(w)
        if len(out) == n:
            break
    return out


class ScriptedProvider:
    """Deterministic stand-in for a model. It searches with words taken from the task, then writes short turns that quote what the tools returned.
    The text is templated: it exercises the forum, it says nothing about the literature. With web search on it returns one invented web source per reflection."""
    name = NAME

    def __init__(self, delay=0):
        self.delay = delay

    def run(self, request):
        started = time.monotonic()
        result = TurnResult(model=request.model, llm_calls=1)
        request.gate()
        for _ in range(int(self.delay * 4)):
            time.sleep(0.25)
            request.gate()                                                  # a Stop is honoured within a quarter of a second
        hints = request.hints
        who = hints.get("agent_label", "Agent")
        tag = "[SIMULATION] "
        if request.task == "reflect":
            tools = {t.name for t in request.tools}
            seen_hits, seen_pairs, network = [], [], None
            if "construct_overview" in tools and hints.get("round", 1) == 1:      # A, B, C: the work starts from the map of the constructs and the network of one of them
                rows = json.loads(request.run_tool("construct_overview", {"sort": "links", "limit": 5})).get("rows") or []
                if rows and "construct_network" in tools:
                    network = json.loads(request.run_tool("construct_network", {"construct": rows[0]["construct"]}))
                    seen_hits += network.get("records", [])
            if "list_diagnoses" in tools and hints.get("wants_diagnoses"):
                listed = json.loads(request.run_tool("list_diagnoses", {"limit": 3}))
                seen_pairs = listed.get("diagnoses", [])
                if seen_pairs and "read_dossier" in tools and hints.get("read_dossier"):
                    json.loads(request.run_tool("read_dossier", {"id": seen_pairs[0]["id"]}))
            if "search_corpus" in tools:
                query = " ".join(hints.get("search_terms") or _terms(hints.get("topic", ""))) or "construct definition measure"
                found = json.loads(request.run_tool("search_corpus", {"query": query, "k": 3}))
                seen_hits = found.get("hits", [])
                if seen_hits and hints.get("second_search"):
                    seen_hits += json.loads(request.run_tool("search_corpus", {"query": " ".join(_terms(seen_hits[0].get("text", ""), 4)), "k": 2})).get("hits", [])
            if "search_constructs" in tools and hints.get("round", 1) == 1:
                seen_hits += json.loads(request.run_tool("search_constructs", {"query": query, "k": 2})).get("records", [])
            ran = None
            if "run_python" in tools and hints.get("round", 1) == 1:
                description = next(t.description for t in request.tools if t.name == "run_python")
                table = "constructs.csv" if "data/constructs.csv" in description else "passages.csv"
                code = "\n".join(["import csv, collections",
                                  f"rows = list(csv.DictReader(open('data/{table}', encoding='utf-8')))",
                                  "key = 'label' if rows and 'label' in rows[0] else 'title'",
                                  f"print(len(rows), 'rows in {table}; most frequent', key + ':', collections.Counter(r[key] for r in rows).most_common(3))"])
                ran = json.loads(request.run_tool("run_python", {"purpose": f"count the rows of {table} and list its most frequent values", "code": code}))
            lines = [tag + f"{who} - private notes for round {hints.get('round', 1)}."]
            if ran and "run_id" in ran:
                lines.append(f"- [{ran['run_id']}] " + (ran["stdout"].strip()[:160] or ran["stderr"].strip()[-160:]))
            if network and network.get("found"):
                lines.append(f"Network of {network['construct']}: {len(network.get('antecedents', []))} antecedents, {len(network.get('outcomes', []))} outcomes. "
                             "Tentative question: which conditions change the link to its main outcome?")
            if seen_pairs:
                lines.append("Diagnoses on the table: " + "; ".join(f"[{p['id']}] {p['label_a']} vs {p['label_b']} ({p['category']})" for p in seen_pairs) + ".")
            for hit in seen_hits[:3]:
                lines.append(f"- [{hit['id']}] \"{hit['title']}\" ({hit.get('year')}), {hit['section']}: {hit['text'][:140].strip()}...")
            if not seen_hits and "search_corpus" in tools:
                lines.append("- The search returned nothing usable; I need a different query.")
            if request.web_search:
                result.sources = [{"url": f"https://example.org/recent-study-{who.split()[0].lower()}-{hints.get('round', 1)}", "title": f"A recent study found by {who} (2026)"}]
                lines.append("A web search found a study after the corpus on the same relation; it does not test the boundary condition.")
            lines.append("Still missing: the measurement evidence for the second construct, and any contrary result.")
            result.text = "\n".join(lines)
            result.reasoning_summary = f"{tag}Looked at the diagnosis, searched the corpus with the task's key words, kept the {len(seen_hits)} passages that mention the constructs."
        elif request.task == "proposal_critique":
            ids = re.findall(r"[EPAW]\d+", (re.search(r"Valid source ids: (.*)", request.prompt) or [None, ""])[1])[:2]
            result.text = (tag + f"{who} - critique: in each section I would ask for the evidence behind the main claim " + " ".join(f"[{i}]" for i in ids)
                           + ", for the boundary condition my community would insist on, and for a test that tells the two predictions apart.")
        elif request.task in ("card_method", "card_impact"):         # the cards of a proposal (forum_cards.py): every passage is an exact sentence of the text the prompt shows
            body = (re.search(r"<<<TESTO\n(.*?)\n>>>FINE_TESTO", request.prompt, re.S) or [None, ""])[1]
            sentences = [s.strip() for s in re.split(r"(?<=[.!?])\s+", body) if 25 <= len(s.strip()) <= 190]
            fields = (("question", "objective", "units_levels", "data_sample", "design", "measures", "procedure", "assumptions_controls") if request.task == "card_method"
                      else ("conversation", "prior_explanation", "claimed_change", "consequences", "scope_conditions"))
            card = {f: {"text": tag + f"Simulated {f.replace('_', ' ')}.", "status": "extracted" if i == 0 else "inferred", "passages": sentences[:1]} for i, f in enumerate(fields)}
            card["info_gaps"] = [tag + "Simulated: what the proposal leaves open."]
            result.text = json.dumps(card)
        elif request.task.startswith("proposal_"):
            ids = re.findall(r"[EPAW]\d+", (re.search(r"Valid source ids: (.*)", request.prompt) or [None, ""])[1])
            cite = lambda n: " ".join(f"[{i}]" for i in ids[n:n + 2])
            sentence = "The two traditions use the same label for different things, and only a study that fixes the unit of analysis and the context can show where the difference matters. "
            result.text = json.dumps({
                "title": tag + "Where does a shared label hide two mechanisms?",
                "abstract": tag + (sentence * 7),
                "introduction": tag + sentence + f"This proposal asks one circumscribed question and says what would count as an answer {cite(0)}.",
                "theoretical_background": tag + sentence + f"The diagnosis taken from the release is a premise with its uncertainty, not a verdict {cite(1)}. Each tradition predicts a different sign for the shared outcome. "
                                          f"What was run in this forum is cited where it is used {cite(2)}; everything else is only possible."})
        elif request.task == "speak":
            ids = list(dict.fromkeys(re.findall(r"\[([EPW]\d+)\]", hints.get("evidence_text", ""))))[:3]
            cites = " ".join(f"[{i}]" for i in ids)
            step = hints.get("step", "the current question")
            result.text = (f"{tag}{who} - on {step}: reading the evidence I collected {cites}, the two uses differ in definition and in the outcomes they are tied to, "
                           f"so if the diagnosis holds we should expect different mechanisms in each tradition. A distinguishing prediction: the relation to the shared outcome "
                           f"should change sign under the second definition. I would still want the missing measurement evidence before treating this as settled.").replace("  ", " ")
        else:   # synthesize
            ids = list(dict.fromkeys(re.findall(r"\[([EPW]\d+)\]", request.prompt)))
            e_ids = [i for i in ids if i.startswith("E")][:2] or []
            p_ids = [i for i in ids if i.startswith("P")][:1] or []
            w_ids = [i for i in ids if i.startswith("W")][:1] or []
            premise = hints.get("premise", "the diagnosis stays uncertain")
            ran_ok = re.findall(r"\[(A\d+)\] Python run by [^(]*\(finished\)", request.prompt)
            a_ids = ran_ok[:1]
            recent = " ".join(f"[{i}]" for i in e_ids[-1:] + w_ids)
            if "puzzle" in ((request.json_schema or {}).get("schema") or {}).get("properties", {}):       # the question of the proposal v4
                start = " ".join(f"[{i}]" for i in p_ids + e_ids[:1])
                result.text = json.dumps({
                    "rq": f"{tag}When does the shared outcome follow from one mechanism rather than the other?",
                    "trigger_evidence": f"The passages I read report the same outcome under two definitions {start}.", "puzzle": f"The literature assumes one mechanism; {premise}.",
                    "rival_interpretations": [{"interpretation": "One process, measured two ways.", "evidence": " ".join(f"[{i}]" for i in e_ids[:1])},
                                              {"interpretation": "Two processes that look alike.", "evidence": " ".join(f"[{i}]" for i in e_ids[1:2])}],
                    "alternative_puzzles": [{"puzzle": "Whether the effect is simply larger in big firms.", "why_discarded": "A variation of a settled result."}],
                    "mechanism": "Two different processes lead to the same observable outcome under different conditions.",
                    "predictions": [{"statement": "The effect on the shared outcome differs between the two operationalisations.", "would_support": "a moderated effect", "would_contradict": "equal effects"}],
                    "theoretical_contribution": "Explains why one outcome has two causal routes.",
                    "conceptual_signature": {"phenomenon": "shared outcome", "challenged_assumption": "one mechanism", "mechanism": "two routes", "explanatory_contrast": "route A versus route B",
                                             "level": "organization"},
                    "empirical_design": {"unit_of_analysis": "Organization", "context": "Digitally transforming firms", "measures": "the two scales", "timing": "two waves, one year apart",
                                         "design": "multi-group panel"},
                    "boundary_conditions": ["firm size", "industry digital intensity"], "uncertainty": "The starting evidence is a few passages.",
                    "timeliness": {"why_now": f"Firms are adopting the technology faster than the studies can follow {recent}.",
                                   "recent_evidence": f"The most recent studies I read still pool the two uses {recent}.",
                                   "already_answered": "I searched for the boundary condition in the corpus" + (" and on the web" if w_ids else "") + "; the closest study does not test it.",
                                   "check_scope": "The corpus up to 2025" + (" and a web search" if w_ids else "") + "."},
                    "argument_steps": [{"step": "Evidence", "claim": "The uses define the construct differently.", "sources": e_ids}],
                    "verification_plan": {"starting_point": "Re-read the papers behind the starting evidence.", "new_relation": "Test the moderated effect with a design that fixes the context."},
                    "analyses": ([{"status": "obtained", "description": "Row counts of the data tables, as printed by the run.", "data": "the tables of the release", "limitations": "Counts of what the papers say, not a prevalence.", "run_id": ran_ok[0]}] if ran_ok else [])
                                + [{"status": "proposed", "description": "Multi-group comparison of the two scales.", "data": "to be collected", "limitations": "No data were analysed in this simulation.", "run_id": ""}],
                    "sources": e_ids + p_ids + a_ids + w_ids})
                request.on_usage(_usage(len(request.prompt) // 4, len(result.text) // 4, 0))
                result.input_tokens, result.output_tokens = len(request.prompt) // 4, len(result.text) // 4
                result.latency_ms = int((time.monotonic() - started) * 1000)
                return result
            result.text = json.dumps({
                "rq": f"{tag}Under which boundary conditions does the relation between the two constructs differ, if {premise}?",
                "diagnosis_premise": premise, "uncertainty": "The diagnosis is a premise taken from the release; this question is conditional on it.",
                "implications": "If the constructs are distinct, results pooled under one label would average over two mechanisms.",
                "mechanism": "Two different processes lead to the same observable outcome under different conditions.",
                "predictions": [{"statement": "The effect on the shared outcome differs between the two operationalisations.", "would_support": "a moderated effect", "would_contradict": "equal effects"}],
                "unit_of_analysis": "Organization", "context": "Digitally transforming firms", "boundary_conditions": ["firm size", "industry digital intensity"],
                "contribution": "Shows where a shared label hides two mechanisms.",
                "timeliness": {"why_now": f"Firms are adopting the technology faster than the studies can follow {recent}.",
                               "recent_evidence": f"The most recent studies I read still pool the two uses {recent}.",
                               "already_answered": "I searched for the boundary condition in the corpus" + (" and on the web" if w_ids else "") + "; the closest study does not test it.",
                               "check_scope": "The corpus up to 2025" + (" and a web search" if w_ids else "") + "."},
                "argument_steps": [{"step": "Diagnosis and uncertainty", "claim": "The release marks the pair as a candidate.", "sources": p_ids},
                                   {"step": "Implication of the diagnosis", "claim": "The uses define the construct differently.", "sources": e_ids}],
                "verification_plan": {"diagnosis": "Compare the two scales on the same respondents.", "new_relation": "Test the moderated effect with a design that fixes the context."},
                "analyses": ([{"status": "obtained", "description": "Row counts of the data tables, as printed by the run.", "data": "the tables of the release", "limitations": "Counts of what the papers say, not a prevalence.", "run_id": ran_ok[0]}] if ran_ok else [])
                            + [{"status": "proposed", "description": "Multi-group comparison of the two scales.", "data": "to be collected", "limitations": "No data were analysed in this simulation.", "run_id": ""}],
                "sources": e_ids + p_ids + a_ids + w_ids})
        request.on_usage(_usage(len(request.prompt) // 4, len(result.text) // 4, 0))
        result.input_tokens, result.output_tokens = len(request.prompt) // 4, len(result.text) // 4
        result.latency_ms = int((time.monotonic() - started) * 1000)
        return result

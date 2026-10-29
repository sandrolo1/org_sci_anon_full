"""Phase 6: the signals of the proposal, candidate-pair retrieval and evidence dossiers.

Proposal v4 (13 Oct 2026, signals-v4): the meaning of a construct use and its operationalization are compared apart.
  S_def     proximity of the definitions and of the meaning profile (referent, defining features, exclusions, level, temporal form, theoretical role,
            scope; extraction v4) - what the construct IS;
  S_op      proximity of the items and measures (and the measurement attributes) - how it is MEASURED.
A disagreement between the two is informative (close measures under distinct definitions, or the reverse) and is never averaged away. S_sem-op, the
single signal of signals-v3 (definitions and measures embedded together), is still computed so that runs stay comparable. The hint of a v4 run is a
RETRIEVAL STRATUM of the proposal's pair sampling, never a verdict:
  same_label_divergent             same label, S_def low (where a conceptual jingle may be);
  different_label_close            different labels, S_def high (where a conceptual jangle may be);
  close_measures_distinct_meaning  different labels, S_op high while S_def is not (measurement overlap is not identity);
  context_shifted                  same label, S_def high, different contexts or populations;
  hard_negative                    same label, S_def high, same context: the label used consistently;
  unresolved                       S_def missing on one side (no definition and no meaning profile): the meaning cannot be compared;
  none                             everything else.
Since 14 Oct 2026 (user's decision) a use whose meaning cannot be compared is left out BEFORE the pairs are formed (param `require`, default "definition":
no definition and no meaning profile = no S_def), so a new run has no "unresolved" stratum. The left-out uses are counted in the run's params
(`excluded_nodes`): they are outside the jingle/jangle analysis for lack of a definition, not judged distinct. "none" keeps every use (the runs before that
date); "definition_and_measures" also needs measures (S_op).
The description of signals-v3 follows.

A *node* is one paper's use of one construct label (all its mentions with the same normalised label): a provisional identity, as the proposal
keeps "a provisional identity" when a link between mentions has not been approved. Two nodes are compared by

  S_sem-op  semantic proximity of the operationalisation: cosine of the embeddings of the definition, the items and the measures;
  C_cit     citation proximity: bibliographic coupling, the Jaccard overlap of the references the two papers cite (co-citation needs a corpus of
            citing papers that we do not have: it is not computed and nothing stands in for it);
  N_nom     similarity of the nomological profiles: antecedents are compared with antecedents, outcomes with outcomes, mediators with mediators and
            moderators with moderators, each role as a soft weighted Jaccard over the constructs it holds; a role missing on either side is skipped
            (missing data are not disagreement) and the signal is unavailable when no role can be compared;

plus name_sim (character n-gram cosine of the labels), which only says whether the two labels are the same. A signal whose input is missing is NULL.
The links between the neighbours of the two nodes are *proposed* by label similarity (threshold link_threshold) and are not approved identities; the
detail of every N_nom says so.

The triage hint (signals-v3, 11 Oct 2026) reads S_sem-op only and just labels the pairs for people; N_nom is still computed and stored, but it is
*not* used by the hint: on the first real extraction only 109 of 29,432 comparable pairs had N_nom > 0, because the neighbours of two papers almost never
match by label. It comes back when the links between neighbours are made better.
  jingle_candidate      same label, S_sem-op low (C_cit says whether the polysemy lives in one conversation or across two);
  jangle_candidate      different labels, S_sem-op high (high C_cit: terminological proliferation in one conversation; low: possible duplication
                        across communities);
  unresolved            S_sem-op is missing (no definition and no measure on at least one side): missing data are not "different";
  none                  everything else: the same label used consistently, or different labels with distant operationalisations.
There is no related_but_distinct hint any more: telling it from a jangle needed N_nom, and people decide it. A "none" pair is a candidate negative: the blind
sample includes them, so that people can say "two different constructs".
Nothing here, and no threshold, decides that two constructs are the same or different (the internal project notes principle 3). The thresholds are uncalibrated defaults:
the proposal calibrates them on development data before the audit and freezes them.
"""
import json
import logging
import re
from collections import defaultdict

import numpy as np

from .core import digest, stable_id
from .embeddings import cosine
from .extraction import normalize_label

LOG = logging.getLogger("research0")
VERSION = "signals-v4"
DEFAULTS = {"k_name": 8, "k_sem": 8, "k_same_label": 8, "k_same_label_close": 2, "k_op": 6, "name_hi": 0.8, "s_hi": 0.6, "s_lo": 0.3,
            "cit_hi": 0.05, "context_overlap": 0.2, "link_threshold": 0.8, "block": 1000, "require": "definition"}
REQUIREMENTS = ("none", "definition", "definition_and_measures")
def version_number(version):
    """4 for "signals-v4"; 0 when it cannot be read."""
    try:
        return int(str(version).rsplit("-v", 1)[1])
    except (IndexError, ValueError):
        return 0


HINTS_V3 = ("jingle_candidate", "jangle_candidate", "related_but_distinct", "unresolved", "none")
HINTS = ("same_label_divergent", "different_label_close", "close_measures_distinct_meaning", "context_shifted", "hard_negative", "unresolved", "none")
# The hint of a v4 stratum read as the conceptual class it looks for (the forum's signals-only option, the gates); the v3 hints keep their reading.
CLASS_OF_HINT = {"same_label_divergent": "conceptual_jingle", "different_label_close": "conceptual_jangle", "jingle_candidate": "conceptual_jingle",
                 "jangle_candidate": "conceptual_jangle"}
ROLES = ("antecedents", "outcomes", "mediators", "moderators", "associates")


def name_vectors(labels):
    from sklearn.feature_extraction.text import HashingVectorizer
    return HashingVectorizer(analyzer="char_wb", ngram_range=(3, 4), n_features=2 ** 18, alternate_sign=False, norm="l2").transform([f" {x} " for x in labels])


class LabelSimilarity:
    """Cached character n-gram cosine between construct labels."""

    def __init__(self, labels=()):
        self.index, self.matrix = {}, None
        self.add(labels)

    def add(self, labels):
        new = sorted(set(labels) - set(self.index))
        if new:
            import scipy.sparse as sp
            matrix = name_vectors(new)
            self.matrix = matrix if self.matrix is None else sp.vstack([self.matrix, matrix]).tocsr()
            self.index.update({label: len(self.index) + i for i, label in enumerate(new)})

    def between(self, a, b):
        self.add([a, b])
        return float((self.matrix[self.index[a]] @ self.matrix[self.index[b]].T).toarray()[0, 0])

    def block(self, labels_a, labels_b):
        self.add(list(labels_a) + list(labels_b))
        ia, ib = [self.index[x] for x in labels_a], [self.index[x] for x in labels_b]
        return (self.matrix[ia] @ self.matrix[ib].T).toarray()


# --- N_nom ----------------------------------------------------------------------------------------------------------------------------

def role_similarity(entries_a, entries_b, similarity, link_threshold):
    """Soft weighted Jaccard of two sets of neighbour constructs {label: {"sign":.., "status":..}}.
    Neighbours are linked one-to-one, best first, when their label similarity reaches link_threshold (a proposed link, not an approved identity)."""
    la, lb = sorted(entries_a), sorted(entries_b)
    sims = similarity.block(la, lb)
    pairs = sorted(((sims[i, j], i, j) for i in range(len(la)) for j in range(len(lb)) if sims[i, j] >= link_threshold), reverse=True)
    used_a, used_b, matched = set(), set(), []
    for sim, i, j in pairs:
        if i not in used_a and j not in used_b:
            used_a.add(i); used_b.add(j); matched.append((sim, la[i], lb[j]))
    weight = sum(sim for sim, _, _ in matched)
    value = weight / (len(la) + len(lb) - len(matched))
    signs = [(entries_a[x]["sign"], entries_b[y]["sign"]) for _, x, y in matched if entries_a[x]["sign"] not in (None, "unspecified") and entries_b[y]["sign"] not in (None, "unspecified")]
    return value, {"sizes": [len(la), len(lb)], "matched": [[x, y, round(sim, 3)] for sim, x, y in matched],
                   "sign_agreement": (sum(a == b for a, b in signs) / len(signs)) if signs else None,
                   "evidence_status": sorted(({entries_a[x]["status"] for x in la} | {entries_b[y]["status"] for y in lb}) - {None})}


def n_nom(profile_a, profile_b, similarity, link_threshold):
    """Mean of the role similarities over the roles both profiles have. Returns (value or None, detail)."""
    detail, values = {}, []
    for role in ROLES:
        a, b = profile_a.get(role), profile_b.get(role)
        if not a or not b:
            detail[role] = {"compared": False, "sizes": [len(a or {}), len(b or {})]}
            continue
        value, info = role_similarity(a, b, similarity, link_threshold)
        detail[role] = {"compared": True, "value": round(value, 4), **info}
        values.append(value)
    detail["basis"] = "links between neighbours proposed by label similarity; not approved identities"
    detail["roles_compared"] = len(values)
    return (sum(values) / len(values) if values else None), detail


def jaccard(a, b):
    if not a or not b:
        return None
    return len(a & b) / len(a | b)


def hint_for(name_sim, s_sem_op, p):
    """signals-v3 (kept for reading the earlier runs and their tests)."""
    if s_sem_op is None:
        return "unresolved"
    if name_sim >= p["name_hi"]:
        return "jingle_candidate" if s_sem_op <= p["s_lo"] else "none"     # the same label used consistently is not a candidate
    return "jangle_candidate" if s_sem_op >= p["s_hi"] else "none"


def contexts_differ(a, b, p):
    """Both uses report a context or population and their words barely overlap. Unknown on either side is not a difference."""
    if not a or not b:
        return False
    return len(a & b) / len(a | b) < p["context_overlap"]


def hint_v4(name_sim, s_def, s_op, context_a, context_b, p):
    """The retrieval stratum of a pair (signals-v4). It orders and samples the review; it decides nothing."""
    if s_def is None:
        return "unresolved"
    if name_sim >= p["name_hi"]:
        if s_def <= p["s_lo"]:
            return "same_label_divergent"
        if s_def >= p["s_hi"]:
            return "context_shifted" if contexts_differ(context_a, context_b, p) else "hard_negative"
        return "none"
    if s_def >= p["s_hi"]:
        return "different_label_close"
    if s_op is not None and s_op >= p["s_hi"]:
        return "close_measures_distinct_meaning"
    return "none"


def citation_reading(c_cit, p):
    if c_cit is None:
        return None
    return "same_conversation" if c_cit >= p["cit_hi"] else "separate_conversations"


# --- Nodes ----------------------------------------------------------------------------------------------------------------------------

def _add(seq, text):
    if text and text not in seq:
        seq.append(text)


def load_nodes(con, extraction_run_id):
    """Nodes with three texts: `def_text` (definitions and meaning profile, for S_def), `measure_text` (measures and measurement attributes, for S_op) and
    `op_text` (definitions and measures together, the S_sem-op of signals-v3); `context` is the words of the reported contexts and populations."""
    from .extraction import CONTEXT_KINDS, MEANING_KINDS, OPERATIONAL_KINDS
    rows = con.execute("SELECT mention_id,canonical_id,label_normalized,definition_text FROM construct_mention WHERE extraction_run_id=%s ORDER BY canonical_id,label_normalized,char_start",
                       (extraction_run_id,)).fetchall()
    nodes = {}
    for mention_id, canonical_id, label, definition in rows:
        node = nodes.setdefault((canonical_id, label), {"canonical_id": canonical_id, "label": label, "mentions": [], "op": [], "defs": [], "measures": [],
                                                        "context": set(), "profile": defaultdict(dict)})
        node["mentions"].append(mention_id)
        _add(node["op"], definition)
        _add(node["defs"], definition)
    mention_node = {m: key for key, n in nodes.items() for m in n["mentions"]}
    for mention_id, name, items, scale in con.execute(
            "SELECT m.mention_id,s.name,s.items_text,s.scale FROM construct_measure s JOIN construct_mention m USING (mention_id) WHERE m.extraction_run_id=%s ORDER BY s.char_start", (extraction_run_id,)):
        text = " ".join(x for x in (name, items, scale) if x)
        _add(nodes[mention_node[mention_id]]["op"], text)
        _add(nodes[mention_node[mention_id]]["measures"], text)
    for mention_id, kind, value in con.execute(
            "SELECT a.mention_id,a.kind,a.value FROM extraction_attribute a JOIN construct_mention m USING (mention_id) WHERE m.extraction_run_id=%s AND a.basis<>'missing' "
            "ORDER BY a.mention_id,a.kind,a.char_start", (extraction_run_id,)):
        node = nodes[mention_node[mention_id]]
        if kind in MEANING_KINDS:
            _add(node["defs"], value)          # the content only: field names shared by every node would make any two of them look alike
        elif kind in OPERATIONAL_KINDS:
            _add(node["measures"], value)
        elif kind in CONTEXT_KINDS:
            node["context"] |= set(normalize_label(value).split())
    claims = con.execute("SELECT claim_id,subject_mention_id,object_mention_id,relation,direction,evidence_status FROM nomological_claim WHERE extraction_run_id=%s", (extraction_run_id,)).fetchall()
    modifiers = defaultdict(list)
    for claim_id, kind, mention_id in con.execute(
            "SELECT m.claim_id,m.kind,m.mention_id FROM claim_modifier m JOIN nomological_claim c USING (claim_id) WHERE c.extraction_run_id=%s AND m.mention_id IS NOT NULL", (extraction_run_id,)):
        modifiers[claim_id].append((kind, mention_id))
    for claim_id, subject, obj, relation, sign, status in claims:
        a, b = mention_node.get(subject), mention_node.get(obj)
        if not a or not b:
            continue
        entry = lambda label: {"sign": sign, "status": status}
        if relation == "predicts":
            nodes[a]["profile"]["outcomes"][b[1]] = entry(b[1])
            nodes[b]["profile"]["antecedents"][a[1]] = entry(a[1])
        elif relation == "correlates":
            nodes[a]["profile"]["associates"][b[1]] = entry(b[1])
            nodes[b]["profile"]["associates"][a[1]] = entry(a[1])
        for kind, mention_id in modifiers.get(claim_id, []):
            m = mention_node.get(mention_id)
            if m and kind in ("mediator", "moderator"):
                role = kind + "s"
                nodes[a]["profile"][role][m[1]] = entry(m[1])
                nodes[b]["profile"][role][m[1]] = entry(m[1])
    for node in nodes.values():
        node["op_text"] = " ".join(node["op"]) or None
        node["def_text"] = " ".join(node["defs"]) or None
        node["measure_text"] = " ".join(node["measures"]) or None
        node["profile"] = {role: entries for role, entries in node["profile"].items() if entries}
    return nodes


def load_references(con, canonical_ids):
    refs = {}
    for cid, raw in con.execute("SELECT c.canonical_id,r.raw_fields->>'References' FROM canonical_paper c JOIN export_record r ON r.record_id=c.representative_record_id WHERE c.canonical_id = ANY(%s)", (list(canonical_ids),)):
        refs[cid] = {x for x in (normalize_label(part) for part in re.split(r"\s*;\s*", raw or "")) if x}
    return refs


# --- Retrieval of candidate pairs -----------------------------------------------------------------------------------------------------

def candidate_pairs(nodes, vectors, p, def_vectors=None, measure_vectors=None):
    """{(key_a, key_b): set(reasons)}. Never pairs two nodes of the same paper. Channels: similar names; on S_sem-op (signals-v3) close operationalisations
    under another name, same name with the most distant ones and with the closest ones; and, when the v4 vectors are given, the same three on S_def
    (definition_neighbor, same_label_divergent_definition, same_label_close_definition) and close measures under another name (measure_neighbor) -
    the strata of the proposal's pair sampling."""
    keys = list(nodes)
    index = {k: i for i, k in enumerate(keys)}
    pairs = defaultdict(set)

    def add(i, j, reason):
        if keys[i][0] == keys[j][0]:
            return
        a, b = (i, j) if i < j else (j, i)
        pairs[(a, b)].add(reason)

    labels = sorted({k[1] for k in keys})
    label_nodes = defaultdict(list)
    for k in keys:
        label_nodes[k[1]].append(index[k])
    if len(labels) > 1:
        matrix = name_vectors(labels)
        for start in range(0, len(labels), p["block"]):
            sims = (matrix[start:start + p["block"]] @ matrix.T).toarray()
            for row, label in enumerate(labels[start:start + p["block"]]):
                sims[row, start + row] = -1
                for other in np.argsort(-sims[row])[:p["k_name"]]:
                    if sims[row, other] <= 0:
                        break
                    for i in label_nodes[label]:
                        for j in label_nodes[labels[other]][:p["k_name"]]:
                            add(i, j, "name_neighbor")
    label_code = {l: n for n, l in enumerate(labels)}
    paper_code = {c: n for n, c in enumerate(sorted({k[0] for k in keys}, key=str))}

    def channels(vecs, specs):
        """specs: [(other label?, sign, k, reason)]: for every node, the k other-paper nodes with the highest sign * similarity, among those with the same
        label (other label False) or another label (True)."""
        have = [i for i, k in enumerate(keys) if vecs.get(k) is not None]
        if not have:
            return
        matrix = np.vstack([vecs[keys[i]] for i in have])
        lab = np.array([label_code[keys[i][1]] for i in have])
        pap = np.array([paper_code[keys[i][0]] for i in have])
        for start in range(0, len(have), p["block"]):
            rows = slice(start, start + p["block"])
            sims = matrix[rows] @ matrix.T
            other_paper = pap[rows, None] != pap[None, :]
            same_label = lab[rows, None] == lab[None, :]
            for other_label, sign, k, reason in specs:
                scores = np.where(other_paper & (~same_label if other_label else same_label), sign * sims, -np.inf)
                k = min(k, scores.shape[1])
                if k < 1:
                    continue
                top = np.argpartition(-scores, k - 1, axis=1)[:, :k] if k < scores.shape[1] else np.argsort(-scores, axis=1)
                for row in range(scores.shape[0]):
                    for col in top[row]:
                        if np.isfinite(scores[row, col]):
                            add(have[start + row], have[col], reason)
    channels(vectors, [(True, 1, p["k_sem"], "operationalization_neighbor"), (False, -1, p["k_same_label"], "same_label_divergent"),
                       (False, 1, p["k_same_label_close"], "same_label_close")])
    if def_vectors is not None:
        channels(def_vectors, [(True, 1, p["k_sem"], "definition_neighbor"), (False, -1, p["k_same_label"], "same_label_divergent_definition"),
                               (False, 1, p["k_same_label_close"], "same_label_close_definition")])
    if measure_vectors is not None:
        channels(measure_vectors, [(True, 1, p["k_op"], "measure_neighbor")])
    return {(keys[a], keys[b]): reasons for (a, b), reasons in pairs.items()}


def build(con, extraction_run_id, embedder, params=None):
    p = {**DEFAULTS, **(params or {})}
    if p["require"] not in REQUIREMENTS:
        raise ValueError("require must be one of: " + ", ".join(REQUIREMENTS))
    signal_run_id = stable_id("signal-run", extraction_run_id, embedder.model_id, VERSION, json.dumps(p, sort_keys=True))
    existing = con.execute("SELECT node_count,pair_count FROM signal_run WHERE signal_run_id=%s", (signal_run_id,)).fetchone()
    if existing:
        return signal_run_id, {"nodes": existing[0], "pairs": existing[1]}, True
    nodes = load_nodes(con, extraction_run_id)
    if not nodes:
        raise ValueError("The extraction run has no construct mentions")
    loaded = len(nodes)
    if p["require"] != "none":
        nodes = {k: n for k, n in nodes.items() if n["def_text"] and (p["require"] == "definition" or n["measure_text"])}
    if not nodes:
        raise ValueError(f"No construct use of the extraction run meets require={p['require']}")
    excluded = loaded - len(nodes)
    embedder.register(con)

    def embed(field):
        keyed, out = [k for k, n in nodes.items() if n[field]], dict.fromkeys(nodes)
        if keyed:
            for k, v in zip(keyed, embedder.embed([nodes[k][field] for k in keyed])):
                out[k] = v
        return out
    vectors, def_vectors, measure_vectors = embed("op_text"), embed("def_text"), embed("measure_text")
    refs = load_references(con, {k[0] for k in nodes})
    found = candidate_pairs(nodes, vectors, p, def_vectors, measure_vectors)
    similarity = LabelSimilarity({k[1] for k in nodes} | {l for n in nodes.values() for entries in n["profile"].values() for l in entries})
    node_id = lambda k: stable_id("node", signal_run_id, k[0], k[1])
    rows = []
    for (ka, kb), reasons in found.items():
        a, b = (ka, kb) if node_id(ka) < node_id(kb) else (kb, ka)
        name_sim = similarity.between(a[1], b[1])
        between = lambda vecs: cosine(vecs[a], vecs[b]) if vecs.get(a) is not None and vecs.get(b) is not None else None
        s_sem, s_def, s_op = between(vectors), between(def_vectors), between(measure_vectors)
        c_cit = jaccard(refs.get(a[0], set()), refs.get(b[0], set()))
        n_value, n_detail = n_nom(nodes[a]["profile"], nodes[b]["profile"], similarity, p["link_threshold"])
        pair_id = stable_id("pair", signal_run_id, node_id(a), node_id(b))
        rows.append((pair_id, signal_run_id, node_id(a), node_id(b), name_sim, s_sem, s_def, s_op, c_cit, n_value, sorted(reasons),
                     hint_v4(name_sim, s_def, s_op, nodes[a]["context"], nodes[b]["context"], p), citation_reading(c_cit, p), json.dumps(n_detail)))
    with con.transaction():
        con.execute("INSERT INTO signal_run(signal_run_id,extraction_run_id,model_id,version,params,node_count,pair_count) VALUES (%s,%s,%s,%s,%s::jsonb,%s,%s)",
                    (signal_run_id, extraction_run_id, embedder.model_id, VERSION, json.dumps({**p, "excluded_nodes": excluded}), len(nodes), len(rows)))
        with con.cursor() as cur:
            cur.executemany("INSERT INTO construct_node(signal_run_id,node_id,canonical_id,label_normalized,mention_count,mention_ids) VALUES (%s,%s,%s,%s,%s,%s)",
                            [(signal_run_id, node_id(k), k[0], k[1], len(n["mentions"]), n["mentions"]) for k, n in nodes.items()])
            cur.executemany("INSERT INTO candidate_pair(pair_id,signal_run_id,node_a,node_b,name_sim,s_sem_op,s_def,s_op,c_cit,n_nom,retrieval_reasons,hint,c_cit_reading,n_nom_detail) "
                            "VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s::jsonb)", rows)
    return signal_run_id, {"nodes": len(nodes), "excluded_nodes": excluded, "pairs": len(rows)}, False


# --- Dossiers ---------------------------------------------------------------------------------------------------------------------------

DOSSIER_VERSION = "dossier-v3"
# Appendix B of the proposal v4: define, operationalize, compare propositions, confirm - then one class of the codebook (pairs.CLASSES).
QUESTIONS = ["Define (before looking at the measures): what is the referent of each use, which properties are necessary or distinguishing, what is excluded? Compare level, "
             "temporal form, theoretical role and scope under matched conditions.",
             "Operationalize: do the items and indicators represent those definitions - and the same ones? Different items or contexts alone are a measurement or scope "
             "question, not a conceptual jingle.",
             "Substitute: if the two meanings were separated, or the two labels integrated, which central proposition, explanation or prediction would change under "
             "comparable conditions? A high correlation or shared items alone do not make a jangle.",
             "Confirm: what nomological, psychometric, experimental or qualitative evidence supports the reading, and what are the counterexamples and the contrary evidence?",
             "Decide one class: conceptual jingle, conceptual jangle, measurement divergence, scope variation, empirical redundancy, part-whole, related but distinct, "
             "distinct, same meaning (no fallacy) or unresolved. A conceptual jingle or jangle needs a meaning-level argument and a theoretical consequence. A merge is "
             "your decision; nothing in this dossier proposes it."]


def node_evidence(con, signal_run_id, node_id):
    canonical_id, label, mention_ids = con.execute("SELECT canonical_id,label_normalized,mention_ids FROM construct_node WHERE signal_run_id=%s AND node_id=%s", (signal_run_id, node_id)).fetchone()
    title, year, doi = con.execute("SELECT title,publication_year,doi FROM canonical_paper WHERE canonical_id=%s", (canonical_id,)).fetchone()
    mentions = [{"mention_id": str(m), "role": role, "surface_form": surface, "definition": definition, "quote": quote, "page": page, "char_start": cs, "char_end": ce}
                for m, role, surface, definition, quote, page, cs, ce in con.execute(
                    "SELECT mention_id,role,surface_form,definition_text,quote,page,char_start,char_end FROM construct_mention WHERE mention_id = ANY(%s) ORDER BY char_start", (mention_ids,))]
    measures = [{"name": n, "items": i, "scale": s, "quote": q, "page": pg} for n, i, s, q, pg in con.execute(
        "SELECT name,items_text,scale,quote,page FROM construct_measure WHERE mention_id = ANY(%s) ORDER BY char_start", (mention_ids,))]
    attributes = [{"kind": k, "basis": b, "value": v, "quote": q, "page": pg} for k, b, v, q, pg in con.execute(
        "SELECT kind,basis,value,quote,page FROM extraction_attribute WHERE mention_id = ANY(%s) ORDER BY kind,char_start", (mention_ids,))]
    from .extraction import MEANING_KINDS, OPERATIONAL_KINDS
    meaning = [a for a in attributes if a["kind"] in MEANING_KINDS]          # compared first (define), then the operationalization
    operational = [a for a in attributes if a["kind"] in OPERATIONAL_KINDS]
    claims = [{"as": "subject" if side == "s" else "object", "other_label": other, "relation": rel, "sign": sign, "evidence_status": status, "quote": quote, "page": page}
              for side, other, rel, sign, status, quote, page in con.execute(
                  "SELECT 's',o.label_normalized,c.relation,c.direction,c.evidence_status,c.quote,c.page FROM nomological_claim c JOIN construct_mention o ON o.mention_id=c.object_mention_id WHERE c.subject_mention_id = ANY(%s) "
                  "UNION ALL SELECT 'o',s.label_normalized,c.relation,c.direction,c.evidence_status,c.quote,c.page FROM nomological_claim c JOIN construct_mention s ON s.mention_id=c.subject_mention_id WHERE c.object_mention_id = ANY(%s)",
                  (mention_ids, mention_ids))]
    return {"label": label, "paper": {"canonical_id": str(canonical_id), "title": title, "year": year, "doi": doi}, "mentions": mentions, "meaning_profile": meaning,
            "measures": measures, "operationalization": operational, "attributes": attributes, "claims": claims}


def candidate_relation(a, b, name_sim, s, c, n, reading, s_def=None, s_op=None):
    """The data side of the proposal's 'Pair comparison' (Appendix B): what the data can say, never the answer to the substitution test."""
    names = lambda side: {re.sub(r"\W+", " ", m["name"].lower()).strip() for m in side["measures"]}
    contrary = [x for side, other in ((a, b), (b, a)) for x in side["claims"] if x["relation"] in ("same_as_claimed", "distinct_from_claimed") and other["label"] == x["other_label"]]
    return {"same_label": a["label"] == b["label"], "name_similarity": name_sim, "semantic_similarity_definition": s_def, "semantic_similarity_measures": s_op,
            "semantic_similarity_operationalization": s, "same_scale": bool(names(a) & names(b)) if names(a) and names(b) else None,
            "definition_and_measures_disagree": None if s_def is None or s_op is None else abs(s_def - s_op) >= 0.3,
            "nomological_network_similarity": n, "shared_citations": c, "citation_reading": reading, "contrary_evidence": [{"quote": x["quote"], "page": x["page"], "relation": x["relation"]} for x in contrary]}


def build_dossiers(con, signal_run_id, pair_ids=None, hints=None, limit=None):
    """Evidence-only dossier per pair: both uses with their quotes, pages and offsets, the signals and their availability, and the questions for the reviewer."""
    where, params = "", {"run": signal_run_id, "limit": limit}
    if pair_ids is not None:
        where, params["ids"] = "AND pair_id = ANY(%(ids)s)", list(pair_ids)
    if hints:
        where, params["hints"] = where + " AND hint = ANY(%(hints)s)", list(hints)
    pairs = con.execute(
        "SELECT pair_id,node_a,node_b,name_sim,s_sem_op,c_cit,n_nom,retrieval_reasons,hint,c_cit_reading,n_nom_detail,s_def,s_op FROM candidate_pair p WHERE signal_run_id=%(run)s " + where +
        " AND NOT EXISTS (SELECT 1 FROM pair_dossier d WHERE d.pair_id=p.pair_id) ORDER BY pair_id LIMIT %(limit)s", params).fetchall()
    created = 0
    for pair_id, a_id, b_id, name_sim, s, c, n, reasons, hint, reading, detail, s_def, s_op in pairs:
        a, b = node_evidence(con, signal_run_id, a_id), node_evidence(con, signal_run_id, b_id)
        content = {"pair_id": str(pair_id), "version": DOSSIER_VERSION, "signal_run_id": str(signal_run_id),
                   "signals": {"name_sim": name_sim, "s_def": s_def, "s_op": s_op, "s_sem_op": s, "c_cit": c, "n_nom": n,
                               "c_cit_basis": "bibliographic coupling (Jaccard of cited references); co-citation not computed",
                               "n_nom_detail": detail, "unavailable": [k for k, v in (("s_def", s_def), ("s_op", s_op), ("s_sem_op", s), ("c_cit", c), ("n_nom", n)) if v is None]},
                   "retrieval_reasons": reasons, "triage_hint": hint,
                   "notice": "Signals and the hint order the review queue; they are not a verdict. Equivalence of constructs is decided by people.",
                   "candidate_relation": candidate_relation(a, b, name_sim, s, c, n, reading, s_def, s_op), "a": a, "b": b, "questions": QUESTIONS}
        blob = json.dumps(content, sort_keys=True, ensure_ascii=False)
        con.execute("INSERT INTO pair_dossier(pair_id,dossier_version,content,content_sha256) VALUES (%s,%s,%s::jsonb,%s) ON CONFLICT DO NOTHING", (pair_id, DOSSIER_VERSION, blob, digest(blob.encode())))
        created += 1
    return created

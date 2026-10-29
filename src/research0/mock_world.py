"""A synthetic world for tests and demos. NOTHING HERE IS REAL DATA and nothing is scientific evidence.

Everything lives in one throw-away PostgreSQL schema named mock_<hex> (the real `public` schema is never written) and, for the
PDFs, in memory. Remove it with `World.drop()`, `research0 mock-clean`, or by deleting this file and the tests that import it.

The corpus has four invented topics with known membership, planted constructs (a jingle, a jangle and a partial overlap, each
with a known truth), tiny generated PDFs that contain the planted sentences, and a scripted stand-in for the LLM that answers
only with what is really in the text it receives.
"""
import datetime as dt
import hashlib
import json
import os
import random
import re
import textwrap
import uuid
from dataclasses import dataclass, field
from urllib.parse import quote

import psycopg
from psycopg.types.json import Jsonb

from .core import norm_title, stable_id

PREFIX = "mock_"

TOPICS = {
    "platform_work": dict(cto=True, tim=False, terms=[
        "algorithmic management", "gig workers", "ride-hailing platforms", "worker autonomy", "rating systems", "digital surveillance",
        "task allocation", "platform labor", "workplace monitoring", "crowdwork", "performance dashboards", "worker resistance"],
        keywords=["algorithmic management", "platform work", "gig economy"]),
    "open_innovation": dict(cto=False, tim=True, terms=[
        "open innovation", "R&D alliances", "patent portfolios", "technology licensing", "inbound knowledge flows", "innovation ecosystems",
        "startup acquisitions", "technology roadmaps", "product development cycles", "university partnerships", "R&D spillovers", "technology scouting"],
        keywords=["open innovation", "technology management", "R&D"]),
    "digital_transformation": dict(cto=True, tim=True, terms=[
        "digital transformation", "enterprise platforms", "digital capabilities", "data analytics adoption", "organizational change",
        "cloud migration", "digital business models", "communication technologies", "innovation management", "digital ecosystems",
        "technology adoption", "collaboration tools"],
        keywords=["digital transformation", "technology adoption", "organizational change"]),
    "nursing_burnout": dict(cto=False, tim=False, terms=[
        "nursing staff", "hospital wards", "emotional exhaustion", "shift schedules", "patient safety", "clinical supervision",
        "compassion fatigue", "night shifts", "staffing ratios", "bedside care", "ward managers", "medication errors"],
        keywords=["nursing", "burnout", "hospital"]),
}
CONNECT = ["shapes", "is associated with", "moderates the effect of", "is rarely examined together with", "interacts with"]
OPENERS = ["This study examines how {a} {c} {b} in contemporary organizations.",
           "Drawing on survey and interview data, we investigate whether {a} {c} {b}.",
           "We develop a framework in which {a} {c} {b}.",
           "Using a longitudinal design, the paper asks how {a} {c} {b}."]
CLOSERS = ["Findings suggest boundary conditions that earlier work overlooked.", "Implications for theory and practice are discussed.",
           "The results qualify prior assumptions about {a}.", "We outline directions for future research on {b}."]

# Planted constructs: (label, sense) -> how a paper defines, measures and relates it. The truth about which pairs are
# a jingle (same label, different construct), a jangle (different labels, same construct) or a partial overlap is below.
BANK = {
    ("work engagement", "w"): dict(role="focal", definition="Work engagement is a positive, fulfilling state of vigor, dedication and absorption in one's job.",
                                   measure=("UWES-9", "nine items covering vigor, dedication and absorption", "seven-point frequency scale")),
    ("engagement", "employee"): dict(role="focal", definition="Engagement is defined as the investment of an employee's energy in role performance at work.",
                                     measure=("Employee Engagement Survey", "twelve items on energy invested in role performance", "five-point agreement scale")),
    ("engagement", "customer"): dict(role="focal", definition="Engagement is defined as the number of clicks, likes and shares that a user generates on a platform.",
                                     measure=("platform interaction logs", "counts of clicks, likes and shares per user per week", "behavioral count")),
    ("digital readiness", "d"): dict(role="focal", definition="Digital readiness is the organizational capacity to adopt and use digital tools, resting on skills, infrastructure and leadership support.",
                                     measure=("Digital Capacity Index", "eight items on skills, infrastructure and leadership support for digital tools", "five-point agreement scale")),
    ("technology preparedness", "t"): dict(role="focal", definition="Technology preparedness is the organizational capacity to adopt and use digital technologies, resting on skills, infrastructure and leadership support.",
                                           measure=("Technology Capacity Scale", "eight items on skills, infrastructure and leadership support for digital technologies", "five-point agreement scale")),
    ("absorptive capacity", "a"): dict(role="focal", definition="Absorptive capacity is the ability of a firm to recognize the value of external knowledge, assimilate it and apply it to commercial ends.",
                                       measure=("Absorptive Capacity Scale", "ten items on recognizing, assimilating and applying external knowledge", "seven-point agreement scale")),
    ("knowledge integration capability", "k"): dict(role="focal", definition="Knowledge integration capability is the ability of a firm to combine specialized knowledge held by different units and apply it to commercial ends.",
                                                    measure=("Integration Capability Scale", "nine items on combining specialized knowledge across units", "seven-point agreement scale")),
    ("algorithmic control", "c"): dict(role="antecedent", definition="Algorithmic control is the use of automated systems to direct, evaluate and discipline workers.",
                                       measure=("Algorithmic Control Scale", "six items on automated direction, evaluation and discipline", "five-point agreement scale")),
    ("job performance", "j"): dict(role="outcome", definition="Job performance is the extent to which an employee fulfils the formal requirements of the role.",
                                   measure=("supervisor ratings", "four items rated by the direct supervisor", "seven-point rating scale")),
    ("firm innovation performance", "i"): dict(role="outcome", definition="Firm innovation performance is the share of annual revenue generated by products introduced in the last three years.",
                                               measure=("annual reports", "revenue share of new products", "ratio")),
}
# The referent of each use (meaning profile, extraction v4): what the construct is about, inferred from its definition sentence.
REFERENTS = {("work engagement", "w"): "an employee's experience of the job", ("engagement", "employee"): "employees and the energy they invest in work roles",
             ("engagement", "customer"): "platform users and the clicks, likes and shares they generate online",
             ("digital readiness", "d"): "an organization's capacity to adopt and use digital tools", ("technology preparedness", "t"): "an organization's capacity to adopt and use digital technologies",
             ("absorptive capacity", "a"): "a firm's use of knowledge from outside it", ("knowledge integration capability", "k"): "a firm's combination of knowledge held by its own units",
             ("algorithmic control", "c"): "automated systems that manage workers", ("job performance", "j"): "an employee's fulfilment of role requirements",
             ("firm innovation performance", "i"): "a firm's revenue from new products"}
TRUTH_PAIRS = {  # unordered label/sense keys -> truth relation
    frozenset([("engagement", "employee"), ("engagement", "customer")]): "jingle",
    frozenset([("digital readiness", "d"), ("technology preparedness", "t")]): "jangle",
    frozenset([("absorptive capacity", "a"), ("knowledge integration capability", "k")]): "related_but_distinct",
}
# The planted truth as a decision of the pair codebook v2: (class, comparison checklist, theoretical consequence).
TRUTH_DECISIONS = {
    "jingle": ("conceptual_jingle", {"same_referent": "no", "compatible_properties": "no", "substitution_changes": "yes"},
               "Investment of energy in a role and counts of platform interactions predict different outcomes: pooled, they would average two mechanisms."),
    "jangle": ("conceptual_jangle", {"same_referent": "yes", "compatible_properties": "yes", "substitution_changes": "no"},
               "Antecedents and outcomes found for one label apply to the other, so the two literatures can be read together."),
    "related_but_distinct": ("related_but_distinct", {"same_referent": "no", "compatible_properties": "unclear", "substitution_changes": "yes"}, ""),
}


def decide_truth(con, pair_id, validator, kind, rationale="scripted from the planted truth", **extra):
    """Records the planted truth of a pair as a person's decision (tests and the demo only)."""
    from . import pairs
    cls, checklist, consequence = TRUTH_DECISIONS[kind]
    pairs.decide(con, pair_id, validator, cls, rationale, checklist, consequence, **extra)
PLANS = {
    "platform_work": [[("work engagement", "w"), ("algorithmic control", "c"), ("job performance", "j")],
                      [("engagement", "employee"), ("algorithmic control", "c")],
                      [("digital readiness", "d"), ("firm innovation performance", "i")]],
    "open_innovation": [[("absorptive capacity", "a"), ("firm innovation performance", "i")],
                        [("technology preparedness", "t"), ("firm innovation performance", "i")]],
    "digital_transformation": [[("engagement", "customer"), ("digital readiness", "d"), ("firm innovation performance", "i")],
                               [("knowledge integration capability", "k"), ("firm innovation performance", "i")],
                               [("technology preparedness", "t"), ("firm innovation performance", "i")]],
}
LEVELS = {"work engagement": "individual", "engagement#employee": "individual", "engagement#customer": "individual", "digital readiness": "organizational",
          "technology preparedness": "organizational", "absorptive capacity": "organizational", "knowledge integration capability": "organizational",
          "algorithmic control": "individual", "job performance": "individual", "firm innovation performance": "organizational"}
FILLER = ["Prior research has produced mixed evidence on this question.", "We collected data over eighteen months from several organizations.",
          "The sample included respondents from different sectors and hierarchical levels.", "All analyses controlled for tenure, size and sector.",
          "Interviews were recorded, transcribed and coded by two researchers.", "Reliability coefficients exceeded conventional thresholds.",
          "These patterns were stable across robustness checks.", "Several limitations should be acknowledged."]


def make_pdf(pages):
    """A minimal valid PDF with one text line per list item. For tests only."""
    out, offsets = [b"%PDF-1.4\n"], []

    def add(body):
        offsets.append(sum(len(x) for x in out))
        out.append(f"{len(offsets)} 0 obj\n".encode() + body + b"\nendobj\n")

    escape = lambda s: s.replace("\\", "\\\\").replace("(", "\\(").replace(")", "\\)")
    add(b"<< /Type /Catalog /Pages 2 0 R >>")
    add(f"<< /Type /Pages /Kids [{' '.join(f'{4 + 2 * i} 0 R' for i in range(len(pages)))}] /Count {len(pages)} >>".encode())
    add(b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica /Encoding /WinAnsiEncoding >>")
    for i, lines in enumerate(pages):
        data = ("BT /F1 10 Tf 12 TL 50 790 Td " + " ".join(f"({escape(line)}) Tj T*" for line in lines) + " ET").encode("cp1252", "replace")
        add(f"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 595 842] /Resources << /Font << /F1 3 0 R >> >> /Contents {5 + 2 * i} 0 R >>".encode())
        add(b"<< /Length %d >>\nstream\n" % len(data) + data + b"\nendstream")
    xref = sum(len(x) for x in out)
    out.append(b"xref\n0 %d\n0000000000 65535 f \n" % (len(offsets) + 1) + b"".join(b"%010d 00000 n \n" % o for o in offsets)
               + b"trailer\n<< /Size %d /Root 1 0 R >>\nstartxref\n%d\n%%%%EOF\n" % (len(offsets) + 1, xref))
    return b"".join(out)


class MemoryStore:
    """Stands in for S3Store: PDFs and nothing else, in memory."""
    location = "mock://fulltext/"
    prefix = "fulltext/"

    def __init__(self):
        self.objects = {}

    def put(self, key, data, sha256=None):
        self.objects[key] = data
        return f"mock://{key}"

    def get(self, uri):
        return self.objects[uri.removeprefix("mock://")]


@dataclass
class MockPaper:
    canonical_id: uuid.UUID
    topic: str
    title: str
    abstract: str
    author_keywords: str
    references: list
    year: int
    doi: str
    eid: str
    plan: list = field(default_factory=list)   # planted constructs, empty if the paper has no PDF
    truth: dict = field(default_factory=dict)  # {"CTO": bool, "TIM": bool}
    status: str = "retained"
    perimeter_flags: list = field(default_factory=list)
    pdf: bytes | None = None
    planted: list = field(default_factory=list)  # what the PDF really says (for the scripted LLM and for checks)


def _sentence_pool(rng, topic):
    terms = TOPICS[topic]["terms"]
    a, b = rng.sample(terms, 2)
    return a, b


def _abstract(rng, topic):
    a, b = _sentence_pool(rng, topic)
    first = rng.choice(OPENERS).format(a=a, b=b, c=rng.choice(CONNECT))
    middle = (f"The analysis centers on {rng.choice(TOPICS[topic]['terms'])} and {rng.choice(TOPICS[topic]['terms'])}, "
              f"and it draws on evidence about {rng.choice(TOPICS[topic]['terms'])} collected in several organizations over a number of years.")
    return " ".join([first, middle, rng.choice(FILLER), rng.choice(CLOSERS).format(a=a, b=b)])


def _claim_sentences(rng, plan):
    claims = []
    if len(plan) == 3:   # first predicts third; the second moderates that relationship (a modifier of the claim, not a claim of its own)
        a, b, c = plan
        direction = rng.choice(["positive", "negative"])
        word = "positively" if direction == "positive" else "negatively"
        sentence = f"{a[0].capitalize()} {word} predicted {c[0]} (beta = .{rng.randint(11, 48)}), and {b[0]} moderated this relationship."
        claims.append(dict(subject=a, object=c, relation="predicts", direction=direction, sentence=sentence, quote=sentence,
                           modifier=dict(kind="moderator", construct=b, quote=f"{b[0]} moderated this relationship")))
        return claims
    for i, subject in enumerate(plan):
        for obj in plan[i + 1:]:
            direction = rng.choice(["positive", "negative"])
            word = "positively" if direction == "positive" else "negatively"
            sentence = f"{subject[0].capitalize()} {word} predicted {obj[0]} (beta = .{rng.randint(11, 48)})."
            claims.append(dict(subject=subject, object=obj, relation="predicts", direction=direction, sentence=sentence, quote=sentence))
    return claims


def _paper_sections(rng, paper):
    """Sections with sentences; planted facts are recorded on the paper so tests know the truth."""
    plan = paper.plan
    intro = [rng.choice(FILLER) for _ in range(3)] + [f"{rng.choice(TOPICS[paper.topic]['terms']).capitalize()} has attracted growing attention."]
    theory, method, results = [], [], []
    planted = {"constructs": [], "measures": [], "claims": [], "attributes": []}
    for key in plan:
        spec = BANK[key]
        theory.append(spec["definition"])
        level = LEVELS.get(key[0] + "#" + key[1], LEVELS.get(key[0], "individual"))
        level_sentence = f"{key[0].capitalize()} is studied at the {level} level."
        theory.append(level_sentence)
        planted.setdefault("attributes", []).append(dict(construct=key, kind="level", basis="extracted", value=f"{level} level", quote=level_sentence))
        if key in REFERENTS:
            planted["attributes"].append(dict(construct=key, kind="referent", basis="inferred", value=REFERENTS[key], quote=spec["definition"]))
        name, items, scale = spec["measure"]
        sentence = f"{key[0].capitalize()} was measured with the {name}, which has {items} on a {scale}."
        method.append(sentence)
        planted["constructs"].append(dict(label=key[0], sense=key[1], role=spec["role"], definition=spec["definition"], quote=spec["definition"]))
        planted["measures"].append(dict(construct=key, name=name, items=items, scale=scale, quote=sentence))
        theory.append(rng.choice(FILLER))
    for claim in _claim_sentences(rng, plan):
        results.append(claim["sentence"])
        planted["claims"].append(claim)
    results.append(rng.choice(FILLER))
    discussion = [rng.choice(FILLER) for _ in range(3)]
    paper.planted = planted
    refs = [("References", paper.references)]
    return [("Abstract", [paper.abstract]), ("1. Introduction", intro), ("2. Theory", theory), ("3. Method", method + [rng.choice(FILLER)]),
            ("4. Results", results), ("5. Discussion", discussion)], refs


def render_pdf(rng, paper):
    sections, refs = _paper_sections(rng, paper)
    lines = [paper.title, ""]
    for heading, sentences in sections:
        lines += [heading] + textwrap.wrap(" ".join(sentences), 92) + [""]
    lines += ["References"] + [ln for r in refs[0][1] for ln in textwrap.wrap(r, 92)]
    pages = [lines[i:i + 52] for i in range(0, len(lines), 52)]
    return make_pdf(pages)


def build_papers(n=240, pdfs=True, seed=11):
    rng = random.Random(seed)
    pools = {t: [f"Author{t[0].upper()}{i}, A. ({2000 + i % 20}). Study {t}-{i}. Journal of Mock Studies." for i in range(40)] for t in TOPICS}
    shared = [f"Common, C. ({2005 + i}). Common reference {i}. Mock Review." for i in range(30)]
    topics = list(TOPICS)
    weights = [0.28, 0.28, 0.24, 0.20]
    papers, plans_used = [], {t: 0 for t in PLANS}
    for i in range(n):
        topic = rng.choices(topics, weights)[0]
        spec = TOPICS[topic]
        abstract = _abstract(rng, topic) if rng.random() > 0.05 else "[No abstract available]"
        refs_from = pools[topic] if topic != "digital_transformation" else pools["platform_work"][:20] + pools["open_innovation"][:20]
        refs = rng.sample(refs_from, 12) + rng.sample(shared, 2) if topic != "nursing_burnout" or rng.random() > 0.1 else []
        title = f"{rng.choice(spec['terms']).capitalize()} and {rng.choice(spec['terms'])} in organizations ({i})"
        paper = MockPaper(canonical_id=stable_id("mock-paper", seed, i), topic=topic, title=title, abstract=abstract,
                          author_keywords="; ".join(spec["keywords"]), references=refs, year=2015 + i % 11,
                          doi=f"10.9999/mock.{seed}.{i}", eid=f"2-s2.0-mock{seed}{i:05d}", truth={"CTO": spec["cto"], "TIM": spec["tim"]})
        if pdfs and topic in PLANS and plans_used[topic] < 12:
            paper.plan = PLANS[topic][plans_used[topic] % len(PLANS[topic])]
            plans_used[topic] += 1
            paper.pdf = render_pdf(rng, paper)
        papers.append(paper)
    papers[3].status = "review_needed"
    papers[5].status = "review_needed"
    papers[7].perimeter_flags = ["year_outside_2015_2025"]
    return papers


@dataclass
class World:
    schema: str
    url: str
    run_id: uuid.UUID
    papers: list
    store: MemoryStore
    _owner: psycopg.Connection = None

    def connect(self):
        return psycopg.connect(self.url)

    def eligible(self):
        return [p for p in self.papers if p.status == "retained" and not p.perimeter_flags]

    def paper(self, canonical_id):
        return next(p for p in self.papers if p.canonical_id == canonical_id)

    def drop(self):
        drop_schema(self.schema)


def url_for(schema, base=None):
    base = base or os.environ["DATABASE_URL"]
    return f"{base}{'&' if '?' in base else '?'}options={quote(f'-csearch_path={schema},public')}"


def drop_schema(schema):
    if not re.fullmatch(PREFIX + r"[0-9a-f]{8,32}", schema):
        raise ValueError("Refusing to drop a schema that is not a mock schema")
    with psycopg.connect(os.environ["DATABASE_URL"], autocommit=True) as con:
        con.execute(f'DROP SCHEMA IF EXISTS "{schema}" CASCADE')


def clean_all():
    """Drop every mock_* schema (leftovers of an interrupted test run). Returns the names dropped."""
    with psycopg.connect(os.environ["DATABASE_URL"], autocommit=True) as con:
        names = [r[0] for r in con.execute("SELECT nspname FROM pg_namespace WHERE nspname ~ %s", (r"^mock_[0-9a-f]{8,32}$",))]
        for name in names:
            con.execute(f'DROP SCHEMA "{name}" CASCADE')
    return names


def create_world(n=240, pdfs=True, seed=11):
    from .migrate import apply_migrations
    schema = PREFIX + uuid.uuid4().hex[:12]
    with psycopg.connect(os.environ["DATABASE_URL"], autocommit=True) as admin:
        admin.execute(f'CREATE SCHEMA "{schema}"')
    world = None
    try:
        papers = build_papers(n, pdfs, seed)
        store = MemoryStore()
        with psycopg.connect(url_for(schema)) as con:
            apply_migrations(con)
            run_id = _populate(con, papers, store, seed)
        world = World(schema, url_for(schema), run_id, papers, store)
        return world
    except BaseException:
        drop_schema(schema)
        raise


def _populate(con, papers, store, seed):
    qid = stable_id("mock-query", seed)
    bid = stable_id("mock-batch", seed)
    run_id = stable_id("mock-run", seed)
    con.execute("INSERT INTO query_version(query_id,source_version,query_text,query_sha256) VALUES (%s,'mock-v1','MOCK QUERY',%s)", (qid, "0" * 64))
    con.execute("INSERT INTO export_batch(batch_id,query_id,original_uri,original_sha256,original_filename,exported_on,provenance,record_count) "
                "VALUES (%s,%s,'mock://none',%s,'mock.csv',%s,'MOCK DATA - not a Scopus export',%s)", (bid, qid, "0" * 64, dt.date(2026, 1, 1), len(papers)))
    con.execute("INSERT INTO catalog_run(run_id,query_id,dedup_version,batch_ids,input_record_count,canonical_count,summary) VALUES (%s,%s,'mock',%s,%s,%s,%s)",
                (run_id, qid, [bid], len(papers), len(papers),
                 Jsonb({"mock": True, "dedup_version": "mock", "input_records": len(papers), "canonical_papers": len(papers), "records_linked_to_existing_paper": 0,
                        "review_needed_papers": sum(p.status == "review_needed" for p in papers), "perimeter_flagged_papers": sum(bool(p.perimeter_flags) for p in papers), "stats": {}})))
    for n, p in enumerate(papers, 1):
        record_id = stable_id("mock-record", p.canonical_id)
        raw = {"Title": p.title, "Abstract": p.abstract, "Author Keywords": p.author_keywords, "Index Keywords": "", "Year": str(p.year),
               "DOI": p.doi, "EID": p.eid, "References": "; ".join(p.references), "Language of Original Document": "English"}
        con.execute("INSERT INTO export_record(record_id,batch_id,row_number,doi,eid,title,title_normalized,authors,publication_year,source_title,document_type,source_type,raw_fields) "
                    "VALUES (%s,%s,%s,%s,%s,%s,%s,'Mock A.',%s,'Journal of Mock Studies','Article','Journal',%s)",
                    (record_id, bid, n, p.doi, p.eid, p.title, norm_title(p.title), p.year, Jsonb(raw)))
        con.execute("INSERT INTO canonical_paper(canonical_id,run_id,representative_record_id,doi,eid,all_dois,all_eids,title,authors,publication_year,source_title,document_type,"
                    "member_count,batch_count,status,perimeter_flags) VALUES (%s,%s,%s,%s,%s,%s,%s,%s,'Mock A.',%s,'Journal of Mock Studies','Article',1,1,%s,%s)",
                    (p.canonical_id, run_id, record_id, p.doi, p.eid, [p.doi], [p.eid], p.title, p.year, p.status, p.perimeter_flags))
        if p.pdf:
            sha = hashlib.sha256(p.pdf).hexdigest()
            uri = store.put(f"fulltext/{p.canonical_id}/{sha}.pdf", p.pdf, sha)
            con.execute("INSERT INTO fulltext_file(file_id,run_id,canonical_id,origin,s3_uri,sha256,size_bytes,acquired_by) VALUES (%s,%s,%s,'manual_upload',%s,%s,%s,'mock')",
                        (stable_id("mock-file", p.canonical_id), run_id, p.canonical_id, uri, sha, len(p.pdf)))
    eligible = [p for p in papers if p.status == "retained" and not p.perimeter_flags]
    for community in ("CTO", "TIM"):
        members = [p for p in eligible if p.truth[community]][:6]
        for i, p in enumerate(members, 1):
            con.execute("INSERT INTO catalog_seed_result(run_id,seed_key,community,benchmark_version,seed_status,outcome,match_method,canonical_ids) "
                        "VALUES (%s,%s,%s,'mock-seeds','eligible','found','doi',%s)", (run_id, f"MOCK-{community}-{i}", community, [p.canonical_id]))
    return run_id


# --- The scripted stand-in for the external model (only used by tests and `research0 demo`) -----------------------------------

def _batch_line(custom_id, payload):
    body = {"output": [{"type": "message", "content": [{"type": "output_text", "text": json.dumps(payload)}]}]}
    return json.dumps({"custom_id": custom_id, "response": {"status_code": 200, "body": body}})


def scripted_preassignment_output(world, request_lines, fabricate_for=None, drop_paper=None):
    """Batch output lines for community-trinary-v2 requests, answering from the planted topics (CTO/TIM truth).

    fabricate_for: paper id for which the 'yes' quote is invented (the collector must reject it).
    drop_paper: paper id left out of its request (the collector must reject the whole request)."""
    out = []
    for line in request_lines:
        request = json.loads(line)
        papers = json.loads(request["body"]["input"])
        answers = []
        for paper in papers:
            truth = world.paper(uuid.UUID(paper["record_id"])).truth
            sides = {}
            for community in ("CTO", "TIM"):
                if truth[community]:
                    quote = "this text never appears" if paper["record_id"] == str(fabricate_for) else paper["abstract"].split(".")[0]
                    sides[community.lower()] = dict(label="yes", confidence=0.9, evidence_field="abstract", evidence_quote=quote, rationale="The abstract states the topic.")
                else:
                    sides[community.lower()] = dict(label="no", confidence=0.8, evidence_field="none", evidence_quote="", rationale="The abstract concerns another field.")
            answers.append({"record_id": paper["record_id"], **sides})
        if drop_paper:
            answers = [a for a in answers if a["record_id"] != str(drop_paper)]
        out.append(_batch_line(request["custom_id"], {"assignments": answers}))
    return out


def scripted_extraction_output(world, request_lines, owner_of, invent_in=None, garble_claim_in=None):
    """Batch output lines for construct-extraction-v3-paper requests. Like an honest model it reports only what the paper really says:
    the planted sentences that occur in the sections it receives. owner_of maps version_id -> canonical_id.

    invent_in: version id of the paper where a construct with an invented quote is added (must be rejected).
    garble_claim_in: version id of the paper where a claim quote is altered (must be rejected)."""
    out = []
    for line in request_lines:
        request = json.loads(line)
        sent = json.loads(request["body"]["input"])
        sections, paper = [part["text"] for part in sent["sections"]], world.paper(owner_of[sent["paper_id"]])
        constructs, measures, claims = {}, [], []

        def add(key, label, role, definition, quote):
            if key not in constructs:
                constructs[key] = dict(label=label, surface_form=label, role=role, definition=definition, quote=quote)
        for c in paper.planted.get("constructs", []):
            if any(c["quote"] in part for part in sections):
                constructs[c["label"] + "#" + c["sense"]] = dict(label=c["label"], surface_form=c["label"], role=c["role"], definition=c["definition"], quote=c["quote"])
        for m in paper.planted.get("measures", []):
            if any(m["quote"] in part for part in sections):
                label = m["construct"][0]
                add(label + "#" + m["construct"][1], label, "measured", "", m["quote"])
                measures.append(dict(construct_label=label, name=m["name"], items=m["items"], scale=m["scale"], quote=m["quote"]))
        attributes = []
        for a in paper.planted.get("attributes", []):
            if any(a["quote"] in part for part in sections):
                add(a["construct"][0] + "#" + a["construct"][1], a["construct"][0], "mentioned", "", a["quote"])
                attributes.append(dict(construct_label=a["construct"][0], kind=a["kind"], basis=a["basis"], value=a["value"], quote=a["quote"]))
        for c in paper.planted.get("claims", []):
            if any(c["quote"] in part for part in sections):
                involved = [c["subject"], c["object"]] + ([c["modifier"]["construct"]] if c.get("modifier") else [])
                for key in involved:
                    if not any(v["label"] == key[0] for v in constructs.values()):
                        constructs[key[0] + "#" + key[1]] = dict(label=key[0], surface_form=key[0], role="mentioned", definition="", quote=c["quote"])
                quote = c["quote"].replace("positively", "strongly") if sent["paper_id"] == str(garble_claim_in) else c["quote"]
                modifiers = [dict(kind=c["modifier"]["kind"], construct_label=c["modifier"]["construct"][0], value=c["modifier"]["construct"][0], quote=c["modifier"]["quote"])] if c.get("modifier") else []
                claims.append(dict(subject_label=c["subject"][0], object_label=c["object"][0], relation=c["relation"], direction=c["direction"], evidence_status="supported",
                                   confidence=0.8, quote=quote, modifiers=modifiers))
        listed = list(constructs.values())
        if sent["paper_id"] == str(invent_in):
            listed.append(dict(label="phantom construct", surface_form="phantom construct", role="focal", definition="", quote="The phantom construct is never mentioned."))
        out.append(_batch_line(request["custom_id"], {"constructs": listed, "measures": measures, "claims": claims, "attributes": attributes}))
    return out


# --- Scripted human work, for tests and the demo ---------------------------------------------------------------------------------

def make_preassignment(con, world, miss_rate=0.1, seed=1):
    """Pre-assignment equal to the planted truth, except for a few deliberate misses so that 'no' hides some members."""
    run = uuid.uuid4()
    con.execute("INSERT INTO preassignment_run(preassignment_run_id,run_id,source,model,prompt_version,scope_version,codebook_sha256) VALUES (%s,%s,'batch_v2','m','p','s','0')", (run, world.run_id))
    rng = random.Random(seed)
    for p in world.eligible():
        for community in ("CTO", "TIM"):
            label = "yes" if p.truth[community] else "no"
            confident = True
            if p.truth[community] and rng.random() < miss_rate:
                label, confident = "no", False
            if p.abstract == "[No abstract available]":
                label = "uncertain"
            con.execute("INSERT INTO community_preassignment(preassignment_run_id,canonical_id,community,label,rationale,text_basis,evidence_quote) VALUES (%s,%s,%s,%s,'r','abstract','q')",
                        (run, p.canonical_id, community, label))
            con.execute("UPDATE community_preassignment SET confidence=%s WHERE preassignment_run_id=%s AND canonical_id=%s AND community=%s",
                        (0.92 if confident else 0.6, run, p.canonical_id, community))
    return run


def mock_audit(con, world, n_total=120, annotator_error_every=7, seed=4):
    """Pre-assignment, stratified sample, two blind annotators (the second makes planted mistakes), an adjudicator. Returns (preassignment_run_id, audit_id)."""
    from . import audit
    pre = make_preassignment(con, world)
    audit_id, _, _ = audit.create_sample(con, world.run_id, pre, n_total=n_total, min_per_stratum=8, min_excluded=15, seed=seed)
    truth = {p.canonical_id: p.truth for p in world.papers}
    papers = audit.blind_papers(con, audit_id, "Ana")
    for who, every in (("Ana", 0), ("Ben", annotator_error_every)):
        for i, paper in enumerate(papers):
            cid = uuid.UUID(paper["id"])
            for community in ("CTO", "TIM"):
                label = "yes" if truth[cid][community] else "no"
                if every and (i + (community == "TIM")) % every == 0:
                    label = "no" if label == "yes" else "yes"
                audit.rate(con, audit_id, cid, community, who, label)
    for item in audit.adjudication_items(con, audit_id):
        cid = uuid.UUID(item["id"])
        audit.adjudicate(con, audit_id, cid, item["community"], "Chi", "yes" if truth[cid][item["community"]] else "no")
    return pre, audit_id


def scripted_confirmation_output(world, request_lines, fabricate_for=None):
    """Batch output for community-fulltext-v1: the planted topic decides the membership; a 'yes' quotes the first sentence of the first excerpt."""
    out = []
    for line in request_lines:
        request = json.loads(line)
        paper = json.loads(request["body"]["input"])[0]
        truth = world.paper(uuid.UUID(paper["record_id"])).truth
        first = paper["excerpts"][0]
        sides = {}
        for community in ("CTO", "TIM"):
            if truth[community]:
                quote = "this sentence is not in the paper" if paper["record_id"] == str(fabricate_for) else first["text"].split(".")[0]
                sides[community.lower()] = dict(label="yes", confidence=0.9, evidence_chunk_id=first["chunk_id"], evidence_quote=quote, rationale="The full text is about this field.")
            else:
                sides[community.lower()] = dict(label="no", confidence=0.85, evidence_chunk_id="", evidence_quote="", rationale="The full text is about another field.")
        out.append(_batch_line(request["custom_id"], {"assignments": [{"record_id": paper["record_id"], **sides}]}))
    return out


def scripted_proposal_output(request_lines, break_for=None):
    """Batch output for pair-proposal-v1, from the planted labels: the same label twice is read as a jingle when the definitions differ, otherwise a jangle."""
    out = []
    for line in request_lines:
        request = json.loads(line)
        data = json.loads(request["body"]["input"])
        a, b = data["a"]["label"], data["b"]["label"]
        kind = None
        for pair, relation in TRUTH_PAIRS.items():
            if {a, b} == {key[0] for key in pair}:
                kind = relation
        cls = TRUTH_DECISIONS[kind][0] if kind else "unresolved"
        body = dict(decision_class="maybe" if request["custom_id"] == str(break_for) else cls, substitution_test="Replacing one meaning with the other changes the predicted outcome.",
                    why_not_mere_variation="The definitions name different referents, not only different items.", rationale="Compared the quoted definitions and measures.",
                    contrary_evidence="", evidence_needed="A psychometric comparison.")
        out.append(_batch_line(request["custom_id"], body))
    return out


def planted_gold_items(con, signal_run_id):
    """Gold pair items built from the planted truth (the class of the pair codebook v2), for tests and the demo."""
    items = []
    for ca, la, cb, lb in con.execute(
            "SELECT a.canonical_id,a.label_normalized,b.canonical_id,b.label_normalized FROM candidate_pair p JOIN construct_node a ON a.signal_run_id=p.signal_run_id AND a.node_id=p.node_a "
            "JOIN construct_node b ON b.signal_run_id=p.signal_run_id AND b.node_id=p.node_b WHERE p.signal_run_id=%s", (signal_run_id,)):
        for pair, kind in TRUTH_PAIRS.items():
            if {la, lb} == {key[0] for key in pair}:
                items.append({"canonical_id_a": str(ca), "label_a": la, "canonical_id_b": str(cb), "label_b": lb, "class": TRUTH_DECISIONS[kind][0]})
    return items

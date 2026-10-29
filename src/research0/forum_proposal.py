"""The research proposal an episode writes once its question is settled: an abstract, an introduction and a theoretical background, in at most `page_limit` pages
(4 by default), as a PDF.

Since 18 Oct 2026 (the user's decision) the document holds ONLY these three parts, each written in depth; the methods and the expected outcomes of the earlier proposals are no longer
text of the proposal. They are written afterwards as two cards in the structure of the cards of the published papers (forum_cards.py), so they can be compared with those.

Nothing here calls a model or touches the database. The orchestrator (forum.py) drafts the text with the agents; this module holds what is fixed around it:
the structured shape the model must answer in, the prompts, the checks on the answer, the reference list and the PDF.

Two rules keep the document honest. (1) The reference list is NOT written by the model: it is built from the evidence ids the text cites ([E3] a passage or a
construct record, [P1] a construct-problem dossier, [A2] a code run), looked up in what the episode actually retrieved or ran, so a reference cannot be invented; an id that does not
exist is flagged. (2) The page limit is measured, not estimated: the PDF is rendered and its pages counted; a proposal that is too long is shortened by the model,
and one that is still too long is kept, marked `over_limit`, and says so, never silently cut.
"""
import io
import re
import unicodedata
from xml.sax.saxutils import escape

from . import forum_protocol as proto

SECTIONS = (("introduction", "Introduction", 0.35), ("theoretical_background", "Theoretical background", 0.65))
ABSTRACT_RESERVE = 280                                  # words of the printed page budget kept for the abstract; the two sections share the rest
ABSTRACT_MIN_WORDS, ABSTRACT_MAX_WORDS = 60, 300    # the abstract: what the originality check compares with the abstracts of the published papers (before 18 Oct 2026: 60 to 250)
ABSTRACT_SHALLOW_WORDS = 160                         # below this the abstract is kept but flagged (the guide asks for 200 to 260); only MIN raises, because a raise loses the whole draft
WORDS_PER_PAGE = 560            # body text per page of the allowed total, once the title, the notice and the references have taken their share
MIN_SECTION_WORDS = 40
SHALLOW_SHARE = 0.6             # a section below this share of its word budget is flagged as shallow: the proposal is meant to be deep, so a thin one must say so

PROPOSAL_SCHEMA = proto._object({"title": proto._string(), "abstract": proto._string(), **{key: proto._string() for key, _, _ in SECTIONS}})
ABSTRACT_GUIDE = ("Also write the `abstract` (200 to 260 words); it is printed first and counts in the page limit, and the budget of the two sections below already leaves room for it. Write it as "
                  "the abstract of a published paper in this field, in depth: the puzzle and purpose, the rival interpretations, the explanation or mechanism and how it differs from the "
                  "closest existing accounts, the propositions or competing predictions and the EXPECTED results - what each leads you to expect, stated as expectations ('we expect', "
                  "'the theory predicts'), never as results obtained, with no invented numbers - the empirical approach in one clause only, and the theoretical contribution and "
                  "implications. It must read without the references: no evidence ids. ")


def word_budget(pages):
    """(words the printed pages hold, words for each of the two sections). The abstract is printed too, so the sections share what is left after ABSTRACT_RESERVE."""
    total = pages * WORDS_PER_PAGE
    return total, {key: int((total - ABSTRACT_RESERVE) * share) for key, _, share in SECTIONS}


def _guidance(pages):
    total, per = word_budget(pages)
    return (ABSTRACT_GUIDE + f"At most {pages} printed pages in all, which is about {sum(per.values())} words for the two sections together "
            f"(introduction {per['introduction']}, theoretical background {per['theoretical_background']}); use the room, because depth is what is wanted. "
            "Plain paragraphs separated by a blank line; a list only when it really is one (lines starting with '- ').")


CITE_RULE = ("Cite the evidence behind every claim about the literature inline as [E3], [P1], [A2] or [W1] (a web source), using ONLY ids from the list you are given; do NOT write a reference list, it is built "
             "from the ids you cite. Do not invent papers, authors or numbers. State what is still unknown rather than smoothing it over.")
# Proposal v5 (18 Oct 2026, the user's decision): abstract, introduction and theoretical background only, each in depth. The leading question still makes the theoretical puzzle
# visible before any measure or estimator is named. The premise is the evidence the trigger led to, or, with the construct-problem trigger, the dossier people decided.
# The design and the expected results are NOT written here: the author writes them afterwards as a methodology card and an impact card (forum_cards.py).
CONTENT_GUIDE = (
    "introduction (in depth: the whole case for the puzzle, not a summary of it): the theoretical puzzle first - what is observed or assumed, in which settings, why it is puzzling "
    "given what the field takes for granted, and the rival interpretations the puzzle allows - then why it remains consequential now (what has changed recently, shown with the most "
    "recent studies and their years, and web sources if any), what the check of earlier answers covered and which part, if any, is already answered and by whom, what stays open and "
    "why the existing accounts cannot close it, the research question in one short sentence that states the explanatory tension, before any measure or estimator is named, and what "
    "answering it would be worth to each community. "
    "theoretical_background (in depth: the argument the whole proposal rests on): the constructs and how each tradition defines, bounds and operationalises them; {premise}; the "
    "state of the debate, with the strongest version of each rival interpretation reasoned from the cited literature; the mechanism, step by step, with its boundary conditions and "
    "the level at which it operates; the propositions or competing predictions that follow from each interpretation, each one saying what it implies that the others do not, so "
    "that evidence could tell them apart; the theoretical contribution - what changes in what is explained, how or why, not a new combination of variables; and the objections a "
    "sceptical reader from each community would raise, with your reply. Say which analyses were really run in this forum (cite their run ids) and which are only possible. "
    "There is NO methods section and NO expected-outcomes section in this document: the design that would test the propositions and what each result would change are written "
    "afterwards, on two separate cards, so do not spend words on data, samples, measures or estimators here. ")
PREMISES = {True: "the construct-problem dossier as a premise decided by people, with their rationale, the contrary evidence and its uncertainty",
            False: "the evidence or observation the question starts from, with how far it can be trusted"}


def draft_task(pages, dossiers=False):
    return ("RESEARCH PROPOSAL. The work above settled on the research question below. Write the proposal as the JSON requested: a title, an abstract, an introduction and a "
            "theoretical background. " + CONTENT_GUIDE.format(premise=PREMISES[bool(dossiers)]) + _guidance(pages) + " " + CITE_RULE)


def critique_task(dossiers=False):
    return ("PEER CRITIQUE. Your colleague drafted the proposal below. In at most 250 words, give for each of its three parts (abstract, introduction, theoretical background) the one "
            f"weakness that matters most: a claim with no valid evidence id, a step beyond what {'the dossier' if dossiers else 'the evidence'} allows, a leading question that names "
            "measures or estimators before the puzzle is visible, a single explanation with no rival weighed, a rival interpretation that is named but not argued, a mechanism "
            "asserted without its steps, competing predictions that no evidence could tell apart, a claim that the question is new or timely that the evidence does not show (old "
            "studies only, or no check of what came after), or a boundary condition that your community's literature would insist on. Judge depth too: say where the argument stays "
            "at the surface. Do not rewrite the proposal.")


def revise_task(pages):
    return ("REVISION. Revise your draft in the light of the critique: keep what holds, fix what does not, and where you disagree say so in the text with a reason. "
            "Answer with the same JSON. " + _guidance(pages) + " " + CITE_RULE)


def shorten_task(pages_now, limit, dossiers=False):
    total, per = word_budget(limit)
    return (f"TOO LONG. Set in print, this proposal runs {pages_now} pages and the limit is {limit}. Cut it to about {int(total * 0.92)} words without changing its puzzle, its question, its "
            f"{'dossier premise' if dossiers else 'starting evidence'}, its rival interpretations, its mechanism or its predictions: drop repetition and secondary detail first. "
            "Keep the evidence ids that support what stays. Answer with the same JSON. " + _guidance(limit))


_EXPECTS = re.compile(r"\b(?:expect\w*|predict\w*|anticipat\w*|hypothes\w*|should|would|likely|will|we propose to|aims? to)\b", re.I)
_OBTAINED = re.compile(r"\b(?:we found|we show(?:ed)? that|results (?:show|showed|indicate|indicated|reveal|revealed|suggest|suggested)|findings (?:show|showed|indicate|indicated|reveal|revealed|suggest|suggested)|our results|our findings|we observe[d]?)\b", re.I)


def words(text):
    return len((text or "").split())


def validate_proposal(obj, known_ids, pages=None):
    """(content, warnings). The text is kept as written; the warnings say what a reader should not take on trust. Raises ValueError if it is not a proposal. With `pages`, a section
    well under its word budget is flagged as shallow (it is kept: depth is asked for, not enforced by rejecting the draft)."""
    if not isinstance(obj, dict):
        raise ValueError("the answer is not a proposal")
    for key, label, _ in SECTIONS:
        if words(obj.get(key)) < MIN_SECTION_WORDS:
            raise ValueError(f"the section {label!r} is missing or has fewer than {MIN_SECTION_WORDS} words")
    if not str(obj.get("title", "")).strip():
        raise ValueError("the proposal has no title")
    if words(obj.get("abstract")) < ABSTRACT_MIN_WORDS:
        raise ValueError(f"the abstract is missing or has fewer than {ABSTRACT_MIN_WORDS} words")
    text = "\n".join(str(obj[k]) for k, _, _ in SECTIONS)
    cited = proto.cited_ids(text)
    unknown = [i for i in cited if i not in known_ids]
    warnings = []
    if unknown:
        warnings.append("The text cites ids that the episode never retrieved, was never given or never ran: " + ", ".join(unknown))
    if not cited:
        warnings.append("The proposal cites no evidence.")
    if words(obj["abstract"]) > ABSTRACT_MAX_WORDS:
        warnings.append(f"The abstract has {words(obj['abstract'])} words; it should be about 200 to 260.")
    elif words(obj["abstract"]) < ABSTRACT_SHALLOW_WORDS:
        warnings.append(f"The abstract has only {words(obj['abstract'])} words; it should be about 200 to 260, in depth.")
    if pages:
        for key, label, _ in SECTIONS:
            budget = word_budget(pages)[1][key]
            if words(obj[key]) < SHALLOW_SHARE * budget:
                warnings.append(f"The {label.lower()} has {words(obj[key])} words against a budget of about {budget}: it is shallower than the proposal is meant to be.")
    if proto.cited_ids(str(obj["abstract"])):
        warnings.append("The abstract cites evidence ids; it should read without the references.")
    if not _EXPECTS.search(str(obj["abstract"])):
        warnings.append("The abstract does not state any expected result ('we expect', 'predicts'...).")
    if _OBTAINED.search(str(obj["abstract"])):
        warnings.append("The abstract words a result as obtained ('we found', 'results show'...); a proposal can only state what it expects.")
    return {**{k: obj[k] for k in ("title", "abstract", *[s[0] for s in SECTIONS])}, "_validation": {"unknown_ids": unknown, "cited": cited, "warnings": warnings,
                                                                                                     "words": {k: words(obj[k]) for k, _, _ in SECTIONS},
                                                                                                     "abstract_words": words(obj["abstract"])}}, warnings


# --- References ----------------------------------------------------------------------------------------------------------------------------

def reference_lines(content, sources):
    """One line per cited id that exists. `sources` is {id: {"kind": passage|construct|diagnosis|analysis|web, ...}} from the episode."""
    lines = []
    for i in content["_validation"]["cited"]:
        s = sources.get(i)
        if not s:
            continue
        if s["kind"] == "diagnosis":
            lines.append(f"[{i}] Construct relationship in the release ({s['category']}, {s['basis']}): {s['labels'][0]} vs {s['labels'][1]}.")
        elif s["kind"] == "analysis":
            lines.append(f"[{i}] Python analysis run in this forum: {s['purpose']} (code sha256 {s['code_sha256'][:10]}; {'finished' if s['ok'] else 'FAILED'}).")
        elif s["kind"] == "web":
            lines.append(f"[{i}] {s['title'] or '(untitled web page)'}. Web page found by the agents' search, not checked by the system: {s['url']}")
        else:
            authors = format_authors(s.get("authors"))
            year = f"({s['year']})" if s.get("year") else "(n.d.)"
            journal = f" {s['journal'].strip()}." if (s.get("journal") or "").strip() else ""
            first, last = (s.get("pages") or [None, None])[:2]
            pages = "" if not first else f" Cited at p. {first}." if last in (None, first) else f" Cited at pp. {first}-{last}."
            what = " Construct record." if s["kind"] == "construct" else ""
            doi = f" doi:{s['doi']}" if s.get("doi") else ""
            lines.append(f"[{i}] {authors + ' ' if authors else ''}{year}. {(s['title'] or '(untitled)').strip().rstrip('.')}.{journal}{pages}{what}{doi}")
    return lines


MAX_AUTHORS = 3     # more than this -> the first one and "et al.": the references count towards the page limit, and one record has 738 authors


def format_authors(authors):
    """Scopus writes 'Gao F.; Chen J.; Lee K.'. Returns 'Gao F., Chen J. & Lee K.', or 'Gao F. et al.' beyond MAX_AUTHORS; '' when unknown."""
    names = [a.strip() for a in (authors or "").split(";") if a.strip() and a.strip().lower() not in ("[no author name available]", "anonymous")]
    if not names:
        return ""
    if len(names) > MAX_AUTHORS:
        return f"{names[0]} et al."
    return names[0] if len(names) == 1 else f"{', '.join(names[:-1])} & {names[-1]}"


# --- The PDF -------------------------------------------------------------------------------------------------------------------------------

REPLACEMENTS = {"≥": ">=", "≤": "<=", "→": "->", "←": "<-", "−": "-", "≈": "~", "≠": "!=", "′": "'", "∗": "*", " ": " ", " ": " ", "​": ""}


def pdf_text(text):
    """The standard PDF fonts cover Windows-1252 only: what falls outside it is spelled out with ASCII (ligatures, arrows, comparison signs) or becomes '?'."""
    out = []
    for ch in text or "":
        if ch in REPLACEMENTS:
            out.append(REPLACEMENTS[ch])
            continue
        try:
            ch.encode("cp1252")
            out.append(ch)
        except UnicodeEncodeError:
            out.append(unicodedata.normalize("NFKD", ch).encode("ascii", "ignore").decode() or "?")
    return "".join(out)


def _blocks(text):
    for block in re.split(r"\n\s*\n", (text or "").strip()):
        lines = [ln.strip() for ln in block.splitlines() if ln.strip()]
        if lines and all(ln[:2] in ("- ", "* ", "• ") or ln.startswith("•") for ln in lines):
            for ln in lines:
                yield "bullet", ln.lstrip("-*• ").strip()
        elif lines:
            yield "text", " ".join(lines)


def render_pdf(content, meta, references):
    """(pdf bytes, pages). `meta`: {"condition", "authors" (labels), "episode", "notice" (None or a short exploratory notice), "date"}."""
    from pypdf import PdfReader
    from reportlab.lib.enums import TA_JUSTIFY
    from reportlab.lib.pagesizes import A4
    from reportlab.lib.styles import ParagraphStyle
    from reportlab.lib.units import cm
    from reportlab.platypus import Paragraph, SimpleDocTemplate, Spacer

    body = ParagraphStyle("body", fontName="Times-Roman", fontSize=10.5, leading=12.6, alignment=TA_JUSTIFY, spaceAfter=4)
    bullet = ParagraphStyle("bullet", parent=body, leftIndent=14, bulletIndent=4, spaceAfter=2)
    heading = ParagraphStyle("heading", fontName="Times-Bold", fontSize=12, leading=14, spaceBefore=8, spaceAfter=3)
    title = ParagraphStyle("title", fontName="Times-Bold", fontSize=15, leading=18, spaceAfter=3)
    small = ParagraphStyle("small", fontName="Times-Italic", fontSize=8.5, leading=10, spaceAfter=3)
    notice = ParagraphStyle("notice", parent=small, fontName="Times-Bold", textColor="#9a1b1b")
    refs = ParagraphStyle("refs", fontName="Times-Roman", fontSize=8.5, leading=10, leftIndent=12, firstLineIndent=-12, spaceAfter=1)

    def footer(canvas, doc):
        canvas.saveState()
        canvas.setFont("Times-Roman", 8.5)
        canvas.drawCentredString(A4[0] / 2, 1.1 * cm, f"{pdf_text(meta.get('episode', ''))}  -  page {doc.page}")
        canvas.restoreState()

    p = lambda text, style: Paragraph(escape(pdf_text(text)), style)
    story = [p(content["title"], title),
             p(f"Condition {meta['condition']}  |  {', '.join(meta['authors'])}  |  written by language-model agents of the discovery forum, not by the authors of the papers cited  |  {meta['date']}", small)]
    if meta.get("notice"):
        story.append(p(meta["notice"], notice))
    if str(content.get("abstract") or "").strip():                      # proposals written before 17 Oct 2026 have none
        story.append(p("Abstract", heading))
        story += [p(text, body) for _, text in _blocks(content["abstract"])]
    for number, (key, label, _) in enumerate(SECTIONS, 1):
        story.append(p(f"{number}. {label}", heading))
        for kind, text in _blocks(content[key]):
            story.append(Paragraph(escape(pdf_text(text)), bullet, bulletText="-") if kind == "bullet" else p(text, body))
    if references:
        story += [p("References (the ids cited above; each exists in the episode's record)", heading)] + [p(r, refs) for r in references]
    buffer = io.BytesIO()
    SimpleDocTemplate(buffer, pagesize=A4, leftMargin=2 * cm, rightMargin=2 * cm, topMargin=1.8 * cm, bottomMargin=1.9 * cm, title=pdf_text(content["title"]),
                      author="Discovery forum (language-model agents)", creator="research0").build(story, onFirstPage=footer, onLaterPages=footer)
    data = buffer.getvalue()
    return data, len(PdfReader(io.BytesIO(data)).pages)

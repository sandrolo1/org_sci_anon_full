"""Phase 5, first half: from a stored PDF to checkable text and chunks.

Every stored fulltext_file is parsed into a paper_version: the normalised full text, where each page starts, and a SHA-256 of that text.
Sections are found from headings; the reference list is kept in the text but never chunked. Chunks are made of whole sentences,
about 500-900 tokens, never cross a section, overlap only by one or two sentences inside a section, and remember page, character
offsets (so that `full_text[char_start:char_end] == text` always holds), hash and language.

A PDF without a text layer becomes status 'no_text_layer': its full text stays unavailable (it is not "irrelevant"); no OCR is done here.
"""
import bisect
import hashlib
import json
import logging
import re
from collections import Counter

from .core import stable_id

LOG = logging.getLogger("research0")
PARSER = "pypdf"
CHUNKER_VERSION = "chunk-v1"
MIN_TOKENS, MAX_TOKENS, OVERLAP_TOKENS, OVERLAP_SENTENCES = 500, 900, 60, 2
MIN_TEXT_CHARS = 200

SECTION_WORDS = [  # (canonical name, pattern)
    ("abstract", r"abstract|summary"), ("introduction", r"introduction|background"),
    ("theory", r"theor\w*(?: background| framework| development)?|literature review|hypothes\w+(?: development)?|conceptual framework"),
    ("method", r"methods?|methodology|data and methods?|research design|materials and methods|sample and (?:data|measures)|measures"),
    ("results", r"results?|findings|analysis|empirical results"), ("discussion", r"discussion|implications"),
    ("conclusion", r"conclusions?|concluding remarks|limitations(?: and future research)?"),
    ("references", r"references|bibliography|works cited"), ("appendix", r"appendix(?: [a-z0-9])?|supplementary material"),
    ("acknowledgements", r"acknowledg(?:e)?ments?")]
NUMBER = r"(?:(?:\d{1,2}(?:\.\d{1,2}){0,2}|[IVX]{1,4})\.?\s+)?"
CANON = [(name, re.compile(rf"^{NUMBER}(?:{pattern})[\s.:]*$", re.I)) for name, pattern in SECTION_WORDS]
NUMBERED = re.compile(r"^\d{1,2}(?:\.\d{1,2}){0,2}\.?\s+[A-Z][^.!?]{2,70}$")
ABBREVIATIONS = {"e.g", "i.e", "et al", "al", "fig", "figs", "vs", "cf", "no", "nos", "eq", "eqs", "approx", "dr", "prof", "mr", "mrs", "ms", "inc", "ltd", "co",
                 "etc", "p", "pp", "vol", "ed", "eds", "sec", "ch", "u.s", "st", "jr", "sr"}
STOPWORDS = {
    "en": "the of and to in a is that for with as on by this are be from or an it at which was were has have not can".split(),
    "it": "il di e che la per un in una è con non sono le dei del della come più si da gli nel anche".split(),
    "de": "der die und das in den von zu mit ist nicht für auf ein eine dem sich des auch als wird".split(),
    "fr": "le la les de des et en un une est que pour dans qui sur par pas avec ce sont au du plus".split(),
    "es": "el la los de que y en un una es por con para las del se no al como más su lo".split()}


def detect_language(text):
    words = re.findall(r"[^\W\d_]+", (text or "").lower())[:3000]
    if len(words) < 30:
        return "und"
    counts = Counter(words)
    scores = {lang: sum(counts[w] for w in vocabulary) / len(words) for lang, vocabulary in STOPWORDS.items()}
    best = max(scores, key=scores.get)
    second = sorted(scores.values())[-2]
    return best if scores[best] >= 0.12 and scores[best] > 1.3 * second else "und"


def estimate_tokens(text):
    return max(1, round(len(text.split()) * 1.33))


# --- PDF to normalised text ----------------------------------------------------------------------------------------------

def extract_pages(data):
    import io
    from pypdf import PdfReader
    reader = PdfReader(io.BytesIO(data))
    if reader.is_encrypted:
        raise ValueError("The PDF is encrypted")
    return [clean_text(page.extract_text() or "") for page in reader.pages]


def clean_text(text):
    """Some real PDFs yield NUL bytes (PostgreSQL text columns reject them) and lone UTF-16 surrogates (they cannot be encoded as UTF-8, so the text
    could not be hashed or stored). Both are dropped here, before any offset is computed, so every later offset refers to the stored text."""
    return text.replace("\x00", "").encode("utf-8", "ignore").decode("utf-8")


def heading_of(line):
    """(canonical section name, True) if the line is a heading, else (None, False)."""
    line = line.strip()
    # A heading starts with a capital (after an optional number): the wrapped tail "acknowledged." of a sentence is not one.
    if not line or len(line) > 80 or not re.match(r"^(?:(?:\d{1,2}(?:\.\d{1,2}){0,2}|[IVX]{1,4})\.?\s+)?[A-Z]", line):
        return None, False
    for name, pattern in CANON:
        if pattern.match(line):
            return name, True
    if NUMBERED.match(line):
        return "other", True
    return None, False


def running_lines(pages_lines):
    """Lines repeated at the top or bottom of at least half of the pages (running headers, footers) and bare page numbers."""
    if len(pages_lines) < 3:
        return set()
    counts = Counter()
    for lines in pages_lines:
        edges = {ln for ln in (lines[:2] + lines[-2:])}
        counts.update(edges)
    return {ln for ln, n in counts.items() if n >= max(3, len(pages_lines) / 2)}


def build_text(pages):
    """Returns (full_text, page_offsets, sections). sections: [{name, heading, start, body_start, end}] covering the whole text."""
    pages_lines = [[ln.strip() for ln in page.splitlines() if ln.strip()] for page in pages]
    noise = running_lines(pages_lines)
    out, page_offsets, sections = [], [], []
    length = 0

    def tail():
        return out[-1][-1:] if out else ""

    for lines in pages_lines:
        page_offsets.append(length)
        for line in lines:
            if line in noise or re.fullmatch(r"(?:page\s*)?\d{1,4}(?:\s*(?:of|/)\s*\d{1,4})?", line, re.I):
                continue
            name, is_heading = heading_of(line)
            if is_heading:
                piece = ("\n" if length and tail() != "\n" else "") + line + "\n"
                start = length + (1 if piece.startswith("\n") else 0)
                if sections:
                    sections[-1]["end"] = start
                out.append(piece)
                length += len(piece)
                sections.append({"name": name, "heading": line, "start": start, "body_start": length, "end": None})
                continue
            if not sections:
                sections.append({"name": "front", "heading": "", "start": 0, "body_start": 0, "end": None})
            if length == 0 or tail() == "\n":
                piece = line
            elif tail() == "-" and line[:1].islower():
                piece = line  # a hyphen at the end of a line is kept ("five-point", "organi-zation"): no character of the text is ever deleted
            else:
                piece = " " + line
            out.append(piece)
            length += len(piece)
    full = "".join(out)
    if sections:
        sections[-1]["end"] = len(full)
    return full, page_offsets, sections


# --- Chunking -------------------------------------------------------------------------------------------------------------

def sentence_spans(text, start, end):
    """Sentence spans (absolute offsets) of text[start:end]. A full stop after an abbreviation or an initial does not end a sentence."""
    spans, cursor = [], start
    for m in re.finditer(r"[.!?][\"')\]]*\s+(?=[\"'(\[]?[A-Z0-9])", text[start:end]):
        stop = start + m.start() + 1 + len(re.match(r"[\"')\]]*", text[start + m.start() + 1:]).group(0))
        before = re.search(r"([A-Za-z.]+)\.$", text[cursor:start + m.start() + 1])
        word = (before.group(1) if before else "").lower().rstrip(".")
        if text[start + m.start()] == "." and (word in ABBREVIATIONS or (len(word) == 1 and word.isalpha())):
            continue
        spans.append((cursor, stop))
        cursor = start + m.end()
    if cursor < end and text[cursor:end].strip():
        spans.append((cursor, end))
    return [(s, e) for s, e in spans if text[s:e].strip()]


def split_long(text, s, e, max_tokens):
    """A sentence longer than the budget is cut at spaces so that every piece fits."""
    pieces, words, cursor, count = [], list(re.finditer(r"\S+", text[s:e])), s, 0
    for m in words:
        count += 1
        if round(count * 1.33) >= max_tokens:
            pieces.append((cursor, s + m.end()))
            cursor, count = s + m.end() + 1, 0
    if cursor < e and text[cursor:e].strip():
        pieces.append((cursor, e))
    return pieces


def chunk_section(text, section, min_tokens=MIN_TOKENS, max_tokens=MAX_TOKENS, overlap_tokens=OVERLAP_TOKENS, overlap_sentences=OVERLAP_SENTENCES):
    """Chunks of one section as (start, end) spans. The last chunk may be shorter than min_tokens; nothing crosses the section."""
    sentences = []
    for s, e in sentence_spans(text, section["body_start"], section["end"]):
        sentences.extend(split_long(text, s, e, max_tokens) if estimate_tokens(text[s:e]) > max_tokens else [(s, e)])
    chunks, current, total = [], [], 0
    for sentence in sentences:
        size = estimate_tokens(text[sentence[0]:sentence[1]])
        if current and total + size > max_tokens:
            chunks.append((current[0][0], current[-1][1]))
            keep, kept = [], 0
            for prev in reversed(current):
                prev_size = estimate_tokens(text[prev[0]:prev[1]])
                if len(keep) >= overlap_sentences or kept + prev_size > overlap_tokens:
                    break
                keep.insert(0, prev)
                kept += prev_size
            if kept + size > max_tokens:
                keep, kept = [], 0
            current, total = keep, kept
        current.append(sentence)
        total += size
    if current and (not chunks or current[-1][1] > chunks[-1][1]):
        chunks.append((current[0][0], current[-1][1]))
    return chunks


def make_chunks(full_text, page_offsets, sections, **limits):
    chunks = []
    for section in sections:
        if section["name"] == "references" or section["end"] <= section["body_start"]:
            continue
        for start, end in chunk_section(full_text, section, **limits):
            text = full_text[start:end]
            chunks.append({"ordinal": len(chunks), "section": section["name"], "char_start": start, "char_end": end, "text": text,
                           "text_sha256": hashlib.sha256(text.encode()).hexdigest(), "token_count": estimate_tokens(text),
                           "page_start": page_of(page_offsets, start), "page_end": page_of(page_offsets, max(start, end - 1)),
                           "language": detect_language(text)})
    return chunks


def page_of(page_offsets, offset):
    return max(1, bisect.bisect_right(page_offsets, offset))


def parse_pdf(data):
    """Returns a dict ready to store: status, detail, page_count, language, full_text, page_offsets, sections, chunks."""
    try:
        return _parse(data)
    except Exception as exc:  # a corrupt or unsupported PDF, or one the parser cannot handle: reported for this paper, never guessed, never stops the run
        return {"status": "failed", "detail": f"{type(exc).__name__}: {str(exc)[:200]}"}


WORD_NOTE = "Parsed from a Word document (python-docx {version}): Word has no fixed pages, so page_start/page_end are 1 for every chunk and mean nothing; footnotes, text boxes and images are not read."


def extract_docx(data):
    """The body of a .docx as one 'page' of lines: paragraphs and table rows in document order."""
    import io
    import docx
    document = docx.Document(io.BytesIO(data))
    lines = []
    for block in document.iter_inner_content():
        if hasattr(block, "rows"):
            lines += [" | ".join(c.text.strip() for c in row.cells if c.text.strip()) for row in block.rows]
        else:
            lines.append(block.text)
    return [clean_text("\n".join(lines))], docx.__version__ if hasattr(docx, "__version__") else "?"


def parse_document(data, uri=""):
    """Dispatch on what the stored file is. A PDF is parsed as before; a .docx is read from its XML; a legacy .doc is kept but not read."""
    extension = uri.rsplit(".", 1)[-1].lower() if "." in uri.rsplit("/", 1)[-1] else "pdf"
    if extension == "docx":
        try:
            return _parse(data, word=True)
        except Exception as exc:
            return {"status": "failed", "detail": f"{type(exc).__name__}: {str(exc)[:200]}"}
    if extension == "doc":
        return {"status": "failed", "detail": "Legacy .doc (binary Word) is stored but not read; save it as .docx or PDF and upload it again. The full text stays unavailable."}
    return parse_pdf(data)


def _parse(data, word=False):
    if word:
        pages, version = extract_docx(data)
    else:
        pages = extract_pages(data)
    full, offsets, sections = build_text(pages)
    if len(full.strip()) < MIN_TEXT_CHARS:
        return {"status": "no_text_layer", "detail": "No usable text layer (scanned PDF?); OCR is not performed, the full text stays unavailable.", "page_count": len(pages)}
    return {"status": "parsed", "detail": WORD_NOTE.format(version=version) if word else None, "page_count": None if word else len(pages), "language": detect_language(full),
            "full_text": full, "page_offsets": offsets, "sections": sections, "chunks": make_chunks(full, offsets, sections)}


# --- Database -----------------------------------------------------------------------------------------------------------------

def start_run(con, run_id):
    params = {"min_tokens": MIN_TOKENS, "max_tokens": MAX_TOKENS, "overlap_tokens": OVERLAP_TOKENS, "overlap_sentences": OVERLAP_SENTENCES}
    import pypdf
    ingestion_id = stable_id("ingestion", run_id, PARSER, pypdf.__version__, CHUNKER_VERSION, json.dumps(params, sort_keys=True))
    con.execute("INSERT INTO ingestion_run(ingestion_run_id,run_id,parser,parser_version,chunker_version,params) VALUES (%s,%s,%s,%s,%s,%s::jsonb) ON CONFLICT DO NOTHING",
                (ingestion_id, run_id, PARSER, pypdf.__version__, CHUNKER_VERSION, json.dumps(params)))
    return ingestion_id


def ingest(con, run_id, store, limit=None, canonical_ids=None):
    """Parse every stored PDF of the run that this ingestion run has not parsed yet, except those of excluded papers. Returns (ingestion_run_id, counts)."""
    ingestion_id = start_run(con, run_id)
    where, params = "", {"run": run_id, "ing": ingestion_id, "limit": limit}
    if canonical_ids is not None:
        where, params["ids"] = "AND f.canonical_id = ANY(%(ids)s)", list(canonical_ids)
    files = con.execute(
        "SELECT f.file_id,f.canonical_id,f.s3_uri,f.sha256 FROM fulltext_file f WHERE f.run_id=%(run)s " + where +
        # papers taken out of the full-text work (fulltext_exclusion, latest decision wins) are not parsed
        " AND NOT coalesce((SELECT x.excluded FROM fulltext_exclusion x WHERE x.run_id=f.run_id AND x.canonical_id=f.canonical_id ORDER BY x.decided_at DESC, x.exclusion_id DESC LIMIT 1), false)" +
        " AND NOT EXISTS (SELECT 1 FROM paper_version v WHERE v.file_id=f.file_id AND v.ingestion_run_id=%(ing)s) ORDER BY f.canonical_id,f.created_at LIMIT %(limit)s", params).fetchall()
    counts = Counter()
    for file_id, canonical_id, uri, sha in files:
        data = store.get(uri)
        if hashlib.sha256(data).hexdigest() != sha:
            parsed = {"status": "failed", "detail": "Stored file does not match its recorded SHA-256"}
        else:
            parsed = parse_document(data, uri)
        version_id = stable_id("paper-version", ingestion_id, file_id)
        text = parsed.get("full_text")
        with con.transaction():
            con.execute("INSERT INTO paper_version(version_id,ingestion_run_id,canonical_id,file_id,status,detail,page_count,language,full_text,page_offsets,text_sha256) "
                        "VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)",
                        (version_id, ingestion_id, canonical_id, file_id, parsed["status"], parsed.get("detail"), parsed.get("page_count"), parsed.get("language"), text,
                         parsed.get("page_offsets"), hashlib.sha256(text.encode()).hexdigest() if text else None))
            with con.cursor() as cur:
                cur.executemany("INSERT INTO document_chunk(chunk_id,version_id,ordinal,section,page_start,page_end,char_start,char_end,text,text_sha256,token_count,language) "
                                "VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)",
                                [(stable_id("chunk", version_id, c["ordinal"]), version_id, c["ordinal"], c["section"], c["page_start"], c["page_end"], c["char_start"], c["char_end"],
                                  c["text"], c["text_sha256"], c["token_count"], c["language"]) for c in parsed.get("chunks", [])])
        counts[parsed["status"]] += 1
    return ingestion_id, dict(counts)


def embed_chunks(con, ingestion_run_id, embedder, batch=500):
    from .embeddings import to_vector_literal
    embedder.register(con)
    rows = con.execute("SELECT c.chunk_id,c.text FROM document_chunk c JOIN paper_version v USING (version_id) WHERE v.ingestion_run_id=%s "
                       "AND NOT EXISTS (SELECT 1 FROM chunk_embedding e WHERE e.chunk_id=c.chunk_id AND e.model_id=%s) ORDER BY c.chunk_id", (ingestion_run_id, embedder.model_id)).fetchall()
    for i in range(0, len(rows), batch):
        part = rows[i:i + batch]
        vectors = embedder.embed([t for _, t in part])
        with con.cursor() as cur:
            cur.executemany("INSERT INTO chunk_embedding(chunk_id,model_id,embedding) VALUES (%s,%s,%s::vector) ON CONFLICT DO NOTHING",
                            [(cid, embedder.model_id, to_vector_literal(v)) for (cid, _), v in zip(part, vectors)])
    return len(rows)

import hashlib

import pytest

from research0 import documents as docs
from research0.mock_world import make_pdf


def test_language_detection():
    assert docs.detect_language("The effect of the platform on the work of the employees is that it can be seen in the data " * 4) == "en"
    assert docs.detect_language("Il lavoro dei dipendenti non è più quello che era e la tecnologia è entrata nel lavoro come una forza " * 3) == "it"
    assert docs.detect_language("short") == "und"


def test_headings_are_recognised_in_common_forms():
    for line, name in [("Abstract", "abstract"), ("1. Introduction", "introduction"), ("2 Theoretical background", "theory"), ("III. Methods", "method"),
                       ("Data and Methods", "method"), ("4. Results", "results"), ("5. Discussion", "discussion"), ("References", "references"),
                       ("6. Limitations and future research", "conclusion"), ("3. Research setting and sample", "other")]:
        assert docs.heading_of(line) == (name, True), line
    for line in ("The results were mixed.", "", "x" * 120, "2019"):
        assert docs.heading_of(line)[1] is False


def test_sentences_do_not_break_on_abbreviations_and_initials():
    text = "Smith et al. (2019) found this, e.g. in J. Brown's data. A second sentence follows (see Fig. 2). Third one? Yes. Last."
    spans = docs.sentence_spans(text, 0, len(text))
    parts = [text[s:e] for s, e in spans]
    assert parts[0].startswith("Smith et al. (2019)") and parts[0].endswith("data.")
    assert parts[1].endswith("Fig. 2).") and parts[2] == "Third one?" and parts[-1] == "Last."
    assert "".join(parts).replace(" ", "") == text.replace(" ", "")


def long_section(sentences):
    body = " ".join(f"Sentence number {i} says that organizations adopt tools, and workers react to them in several ways." for i in range(sentences))
    text = "Heading\n" + body
    return text, {"name": "results", "heading": "Heading", "start": 0, "body_start": 8, "end": len(text)}


def test_chunks_respect_size_overlap_and_offsets():
    text, section = long_section(400)
    chunks = docs.chunk_section(text, section)
    assert len(chunks) > 3
    sizes = [docs.estimate_tokens(text[s:e]) for s, e in chunks]
    assert all(size <= docs.MAX_TOKENS for size in sizes) and all(size >= docs.MIN_TOKENS for size in sizes[:-1])
    for (s1, e1), (s2, e2) in zip(chunks, chunks[1:]):
        assert s2 > s1 and e2 > e1
        assert s2 < e1 and (e1 - s2) < 400                    # a small overlap, never a big one
        assert text[e1 - 1] in ".?!" and text[s2].isupper()    # whole sentences only
    assert chunks[0][0] == section["body_start"] and chunks[-1][1] == len(text)


def test_an_oversized_sentence_is_cut_and_nothing_is_lost():
    text = "Heading\n" + " ".join(["word"] * 2000) + "."
    section = {"name": "results", "heading": "Heading", "start": 0, "body_start": 8, "end": len(text)}
    chunks = docs.chunk_section(text, section)
    assert all(docs.estimate_tokens(text[s:e]) <= docs.MAX_TOKENS for s, e in chunks) and len(chunks) >= 3
    assert chunks[-1][1] == len(text)


def test_pdf_to_text_sections_pages_and_chunks(world):
    paper = next(p for p in world.papers if p.pdf)
    parsed = docs.parse_pdf(paper.pdf)
    assert parsed["status"] == "parsed" and parsed["page_count"] >= 1 and parsed["language"] == "en"
    full, sections = parsed["full_text"], parsed["sections"]
    assert [s["name"] for s in sections] == ["front", "abstract", "introduction", "theory", "method", "results", "discussion", "references"]
    assert all(s["end"] >= s["body_start"] for s in sections) and sections[0]["start"] == 0 and sections[-1]["end"] == len(full)
    for planted in paper.planted["constructs"]:            # the definition sentence survived line wrapping verbatim
        assert planted["quote"] in full
    for claim in paper.planted["claims"]:
        assert claim["quote"] in full
    chunks = parsed["chunks"]
    assert chunks and all(c["section"] != "references" for c in chunks)
    for c in chunks:
        assert full[c["char_start"]:c["char_end"]] == c["text"] and hashlib.sha256(c["text"].encode()).hexdigest() == c["text_sha256"]
        assert 1 <= c["page_start"] <= c["page_end"] <= parsed["page_count"]
    assert [c["ordinal"] for c in chunks] == list(range(len(chunks)))
    assert not any("Author" in c["text"] and "Journal of Mock Studies" in c["text"] for c in chunks)   # reference list is not chunked


def test_hyphenation_running_headers_and_page_numbers():
    pages = [["Running title", "1. Introduction", "Technol-", "ogy changes the organi-", "zation of work.", "1"],
             ["Running title", "Work continues here as it did.", "2"],
             ["Running title", "A third page of text.", "3"]]
    full, offsets, sections = docs.build_text(["\n".join(p) for p in pages])
    assert "Technol-ogy changes the organi-zation of work. Work continues here as it did. A third page of text." in full  # hyphens are kept: nothing is deleted
    assert "Running title" not in full and "\n1\n" not in full
    assert len(offsets) == 3 and offsets[0] == 0 and offsets[0] <= offsets[1] <= offsets[2]


def test_pdf_without_text_and_garbage_are_reported_not_guessed():
    blank = docs.parse_pdf(make_pdf([[" "]]))
    assert blank["status"] == "no_text_layer" and "OCR" in blank["detail"]
    broken = docs.parse_pdf(b"%PDF-1.4 this is not really a pdf")
    assert broken["status"] == "failed"


def test_nul_bytes_from_real_pdfs_are_dropped_before_offsets_are_computed(monkeypatch):
    class FakePage:
        def extract_text(self):
            return "1. Introduction\nTechno\x00logy changes work.\x00"

    class FakeReader:
        def __init__(self, stream):
            self.is_encrypted, self.pages = False, [FakePage()]

    monkeypatch.setattr("pypdf.PdfReader", FakeReader)
    pages = docs.extract_pages(b"unused")
    assert pages == ["1. Introduction\nTechnology changes work."]


def test_ingestion_is_idempotent_checks_integrity_and_embeds(world, con):
    run, counts = docs.ingest(con, world.run_id, world.store)
    n_pdf = sum(1 for p in world.papers if p.pdf)
    assert counts == {"parsed": n_pdf}
    again, counts2 = docs.ingest(con, world.run_id, world.store)
    assert again == run and counts2 == {}
    assert con.execute("SELECT count(*) FROM paper_version WHERE ingestion_run_id=%s", (run,)).fetchone()[0] == n_pdf
    from research0.embeddings import HashEmbedder
    new = docs.embed_chunks(con, run, HashEmbedder())
    assert new == con.execute("SELECT count(*) FROM document_chunk").fetchone()[0] > n_pdf
    assert docs.embed_chunks(con, run, HashEmbedder()) == 0


def test_a_tampered_file_is_not_parsed(world, con):
    paper = next(p for p in world.papers if p.pdf)
    uri = con.execute("SELECT s3_uri FROM fulltext_file WHERE canonical_id=%s", (paper.canonical_id,)).fetchone()[0]
    original = world.store.objects[uri.removeprefix("mock://")]
    world.store.objects[uri.removeprefix("mock://")] = original + b"\n% tampered"
    try:
        _, counts = docs.ingest(con, world.run_id, world.store, canonical_ids=[paper.canonical_id])
    finally:
        world.store.objects[uri.removeprefix("mock://")] = original
    assert counts == {"failed": 1}
    assert "SHA-256" in con.execute("SELECT detail FROM paper_version WHERE canonical_id=%s", (paper.canonical_id,)).fetchone()[0]


def test_surrogates_and_nul_are_dropped_and_one_bad_pdf_does_not_stop_the_parse(monkeypatch):
    assert docs.clean_text("a\x00b \ud835 c") == "ab  c"
    monkeypatch.setattr(docs, "extract_pages", lambda data: ["1. Introduction\n" + "Sentence about work. " * 60])
    monkeypatch.setattr(docs, "make_chunks", lambda *a, **k: (_ for _ in ()).throw(RuntimeError("chunker broke")))
    result = docs.parse_pdf(b"unused")
    assert result["status"] == "failed" and "RuntimeError" in result["detail"]


def test_docx_is_read_in_document_order_with_tables_and_marked_as_pageless():
    import io
    import docx
    document = docx.Document()
    document.add_paragraph("Abstract")
    document.add_paragraph("We study how workers react to platform tools. " * 8)
    table = document.add_table(rows=1, cols=2)
    table.rows[0].cells[0].text, table.rows[0].cells[1].text = "Construct", "Measure"
    document.add_paragraph("References")
    document.add_paragraph("Smith (2019). A paper.")
    buffer = io.BytesIO()
    document.save(buffer)
    parsed = docs.parse_document(buffer.getvalue(), "s3://b/fulltext/x/abc.docx")
    assert parsed["status"] == "parsed" and parsed["page_count"] is None and "no fixed pages" in parsed["detail"]
    text = parsed["full_text"]
    assert text.index("Abstract") < text.index("We study") < text.index("Construct | Measure") < text.index("References")
    assert {c["page_start"] for c in parsed["chunks"]} == {1} and all(text[c["char_start"]:c["char_end"]] == c["text"] for c in parsed["chunks"])


def test_legacy_doc_and_corrupt_docx_fail_with_a_reason_and_pdf_is_unchanged():
    legacy = docs.parse_document(b"\xd0\xcf\x11\xe0", "s3://b/fulltext/x/abc.doc")
    assert legacy["status"] == "failed" and ".docx" in legacy["detail"]
    broken = docs.parse_document(b"not a zip", "s3://b/fulltext/x/abc.docx")
    assert broken["status"] == "failed" and broken["detail"]
    assert docs.parse_document(b"not a pdf", "s3://b/fulltext/x/abc.pdf")["status"] == "failed"

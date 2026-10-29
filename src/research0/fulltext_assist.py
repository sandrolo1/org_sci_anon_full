"""Hand-assisted full text: a person passes the bot-protection test in a real browser, this module does the rest.

`assist_page` writes a local HTML list of the papers still without a PDF, with the links to open.
`watch` imports the PDFs the person saves into a folder, matching each to a paper only on strong evidence
(its ID as file name, its DOI in the PDF metadata or first page, or its full title on the first pages).
A file that matches nothing, or more than one paper, is left untouched and reported: never guessed.
"""
import html
import json
import logging
import re
import shutil
import time
import urllib.parse
import uuid
from pathlib import Path

from . import fulltext

LOG = logging.getLogger("research0")
DOI_RE = re.compile(r"10\.\d{4,9}/[^\s\"'<>]+", re.I)
MIN_TITLE_WORDS = 6
MAX_PAGES = 2
MAX_TEXT = 30000
STABLE_SECONDS = 2.0


def _norm(text):
    return " ".join(re.findall(r"[a-z0-9]+", (text or "").lower()))


def _clean_doi(value):
    return value.strip().rstrip(".,;:)]}>\"'").lower()


def papers_to_fetch(con, run_id):
    """Papers tried and still without a file, as identification records. Walls first: they are the ones nothing else can fetch."""
    rows = [fulltext._missing_row(r) for r in con.execute(fulltext.MISSING_SQL, fulltext._selection_params(run_id))]
    rows.sort(key=lambda r: r["reason_code"] != "blocked_by_site")
    return [{**r, "doi_key": _clean_doi(r["doi"]) if r["doi"] else None, "title_key": _norm(r["title"])} for r in rows]


# --- The list for the person ------------------------------------------------------------------------------------------

def assist_page(papers, folder):
    """A self-contained local page (no scripts, no external resources). Links open in a new tab, one paper at a time."""
    esc = html.escape
    items = []
    for n, p in enumerate(papers, 1):
        links = []
        for label, url in (("Open via DOI", p["doi_url"]), ("Open access page", fulltext._web_url(p["open_access_page"]))):
            if url:
                links.append(f'<a href="{esc(url, quote=True)}" target="_blank" rel="noopener noreferrer">{label}</a>')
        items.append(f"<tr><td>{n}</td><td>{esc(str(p['year'] or ''))}</td><td>{esc(p['title'] or '')}<br><small>{esc(p['source'] or '')}</small></td>"
                     f"<td>{esc(p['reason'])}</td><td>{' · '.join(links) or 'no link: search the title'}</td></tr>")
    return f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<meta http-equiv="Content-Security-Policy" content="default-src 'none'; style-src 'unsafe-inline'">
<title>PDFs to fetch by hand</title>
<style>
:root {{ --bg:#fff; --fg:#1b1f23; --muted:#5b6570; --line:#d8dee4; --link:#0b5cad; }}
@media (prefers-color-scheme: dark) {{ :root {{ --bg:#14171a; --fg:#e6e9ec; --muted:#9aa5b1; --line:#2d343b; --link:#6cb2ff; }} }}
body {{ background:var(--bg); color:var(--fg); font:15px/1.45 system-ui,sans-serif; margin:0 auto; max-width:1100px; padding:16px; }}
table {{ border-collapse:collapse; width:100%; }} th,td {{ border-bottom:1px solid var(--line); padding:6px 8px; text-align:left; vertical-align:top; }}
small {{ color:var(--muted); }} a {{ color:var(--link); }} code {{ background:var(--line); padding:1px 5px; border-radius:3px; }}
</style></head><body>
<h1>PDFs to fetch by hand ({len(papers)})</h1>
<ol>
<li>Start the importer: <code>research0 fulltext-watch --run-id &lt;id&gt; --folder &lt;folder&gt; --uploader "&lt;your name&gt;"</code></li>
<li>Open a paper below in your browser (signed in through your library if needed), pass the site's check, and save the PDF into <code>{esc(str(folder))}</code>. The file name does not matter.</li>
<li>The importer attaches each PDF to its paper when the DOI or title in the file identifies it, and says so when it cannot.</li>
</ol>
<table><thead><tr><th>#</th><th>Year</th><th>Paper</th><th>Why not automatic</th><th>Links</th></tr></thead>
<tbody>
{chr(10).join(items)}
</tbody></table></body></html>
"""


# --- Matching a saved file to a paper ---------------------------------------------------------------------------------

def read_pdf(data):
    """(metadata text, first-page text) of a PDF; ('', '') when it cannot be read, which then just means no evidence."""
    import io
    from pypdf import PdfReader
    try:
        reader = PdfReader(io.BytesIO(data))
        info = reader.metadata or {}
        meta = " ".join(str(v) for v in info.values() if v)
        try:
            xmp = reader.xmp_metadata
            meta += " " + " ".join(str(v) for v in (getattr(xmp, "dc_identifier", None) or []) if v) if xmp else ""
        except Exception:  # damaged XMP is no evidence, not an error
            pass
        text = " ".join((page.extract_text() or "") for page in reader.pages[:MAX_PAGES])[:MAX_TEXT]
        return meta, text
    except Exception as exc:  # pypdf raises many types on damaged files
        LOG.info(json.dumps({"event": "fulltext_assist_unreadable", "error": str(exc)[:200]}))
        return "", ""


def identify(filename, data, papers):
    """(paper, how) or (None, reason). A paper is chosen only if exactly one candidate is supported by the evidence."""
    stem = Path(filename).stem
    try:
        wanted = str(uuid.UUID(stem))
        hit = [p for p in papers if p["id"] == wanted]
        return (hit[0], "file name is the paper ID") if hit else (None, "file name is a paper ID that is not in the missing list")
    except ValueError:
        pass
    meta, text = read_pdf(data)
    name = re.sub(r"^(10\.\d{4,9})_", r"\1/", stem)
    by_doi = {}
    for source, haystack in (("PDF metadata", meta), ("file name", name), ("first page", text)):
        for found in DOI_RE.findall(haystack):
            found = _clean_doi(found)
            for p in papers:
                if p["doi_key"] and (found == p["doi_key"] or found.startswith(p["doi_key"]) and not found[len(p["doi_key"])].isalnum()):
                    by_doi.setdefault(p["id"], (p, source))
    if len(by_doi) == 1:
        paper, source = next(iter(by_doi.values()))
        return paper, f"DOI in {source}"
    if len(by_doi) > 1:
        # Several missing papers' DOIs on the first pages: a reference list or a review, not evidence about this file.
        titled = [p for p, _ in by_doi.values() if len(p["title_key"].split()) >= MIN_TITLE_WORDS and p["title_key"] in _norm(text)]
        if len(titled) == 1:
            return titled[0], "DOI and full title on the first pages"
        return None, f"{len(by_doi)} missing papers' DOIs appear in the file"
    haystack = _norm(meta + " " + text)
    titled = [p for p in papers if len(p["title_key"].split()) >= MIN_TITLE_WORDS and p["title_key"] in haystack]
    if len(titled) == 1:
        return titled[0], "full title on the first pages"
    if len(titled) > 1:
        return None, f"{len(titled)} missing papers' titles appear in the file"
    return None, "no DOI or full title of a missing paper found in the file"


# --- The folder -------------------------------------------------------------------------------------------------------

def scan_folder(con, store, run_id, folder, uploader, papers, seen=None):
    """One pass: import what is identified, move it to <folder>/imported/. Returns one report row per file looked at."""
    folder = Path(folder)
    seen = seen if seen is not None else {}
    report = []
    for path in sorted(folder.glob("*.pdf")):
        try:
            stat = path.stat()
        except OSError:
            continue
        key = (stat.st_size, stat.st_mtime_ns)
        if seen.get(path.name) == key or time.time() - stat.st_mtime < STABLE_SECONDS:
            continue  # already judged unchanged, or the browser may still be writing it
        seen[path.name] = key
        try:
            data = path.read_bytes()
            fulltext.check_pdf(data)
        except (OSError, ValueError) as exc:
            report.append({"file": path.name, "status": "rejected", "reason": str(exc)})
            continue
        paper, how = identify(path.name, data, papers)
        if not paper:
            report.append({"file": path.name, "status": "unmatched", "reason": how})
            continue
        try:
            out = fulltext.save_upload(con, store, run_id, uuid.UUID(paper["id"]), data, path.name, uploader)
        except ValueError as exc:
            report.append({"file": path.name, "status": "rejected", "reason": str(exc)})
            continue
        done = folder / "imported"
        done.mkdir(exist_ok=True)
        target = done / path.name
        if target.exists():
            target = done / f"{path.stem}.{out['sha256'][:8]}{path.suffix}"
        shutil.move(str(path), str(target))
        seen.pop(path.name, None)
        papers[:] = [p for p in papers if p["id"] != paper["id"]]
        report.append({"file": path.name, "status": "imported", "paper": paper["id"], "title": paper["title"], "matched_by": how,
                      "already_stored": out["already_stored"]})
    return report


def watch(con, store, run_id, folder, uploader, *, once=False, interval=3.0, emit=print):
    """Import the folder now and, unless `once`, keep doing it until interrupted."""
    Path(folder).mkdir(parents=True, exist_ok=True)
    papers, seen, total = papers_to_fetch(con, run_id), {}, {"imported": 0, "unmatched": 0, "rejected": 0}
    try:
        while True:
            for row in scan_folder(con, store, run_id, folder, uploader, papers, seen):
                total[row["status"]] += 1
                emit(json.dumps({"event": "fulltext_assist_" + row["status"], **row}, ensure_ascii=False))
            if once:
                break
            time.sleep(interval)
    except KeyboardInterrupt:
        pass
    return {**total, "still_missing": len(papers)}


def write_page(con, run_id, folder, output):
    papers = papers_to_fetch(con, run_id)
    Path(output).write_text(assist_page(papers, folder), encoding="utf-8", newline="")
    return len(papers)

"""Import of a collection of full texts that comes with its own index (the OrgSci Drive collection of 17 Oct 2026).

Each index row names the catalog paper (`record_id` = canonical_id), the file, its SHA-256, the source URL and the version. Files are PDFs or Word
documents; they enter as manual uploads (a person brought them), attributed to `uploader` plus the collection folder they came from, because the
folder says whether a reuse license was reported or redistribution permission is still to be established.

Without `execute` nothing is written. A paper that already has a different file in the database is left alone (the stored file stays the valid one);
a paper the collection supplies with two different files is held back too, so that a person chooses which one counts.
"""
import csv
import hashlib
import json
import logging
import uuid
from collections import Counter, defaultdict
from pathlib import Path

from . import fulltext

LOG = logging.getLogger("research0")
VERSIONS = (("accepted", ("accepted author manuscript", "acceptedversion")),
            ("published", ("published version", "publishedversion", "published journal article")),
            ("submitted", ("submitted manuscript", "submittedversion", "preprint")))


def version_label(text):
    """The index states versions in free text; only the three values the database accepts are mapped, anything else stays unknown (NULL)."""
    t = (text or "").strip().lower()
    for label, prefixes in VERSIONS:
        if t.startswith(prefixes):
            return label
    return None


def read_index(path):
    with open(path, encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def plan(con, run_id, folder, index_path, uploader):
    """One entry per index row: what would happen to it and why."""
    rows = read_index(index_path)
    per_paper = Counter(r["record_id"] for r in rows)
    stored = defaultdict(set)
    for canonical_id, sha in con.execute("SELECT canonical_id::text, sha256 FROM fulltext_file WHERE run_id=%s", (run_id,)):
        stored[canonical_id].add(sha)
    catalog = {str(r[0]) for r in con.execute("SELECT canonical_id FROM canonical_paper WHERE run_id=%s AND canonical_id = ANY(%s)",
                                              (run_id, [uuid.UUID(r["record_id"]) for r in rows]))}
    entries = []
    for row in rows:
        path = Path(folder) / row["path"]
        entry = {"record_id": row["record_id"], "path": row["path"], "format": row["format"], "sha256": row["sha256"], "action": None, "reason": None}
        entries.append(entry)
        bucket = Path(row["path"]).parts[1] if len(Path(row["path"]).parts) > 2 else ""
        entry.update(acquired_by=f"{uploader} (Drive OrgSci/{bucket})", version=version_label(row["version"]), source_url=row["url"] or None)
        if row["record_id"] not in catalog:
            entry.update(action="rejected", reason="paper not in this catalog run")
        elif not path.is_file():
            entry.update(action="rejected", reason="file not found in the folder")
        elif hashlib.sha256(path.read_bytes()).hexdigest() != row["sha256"]:
            entry.update(action="rejected", reason="file does not match the SHA-256 in the index")
        elif row["sha256"] in stored.get(row["record_id"], ()):
            entry.update(action="already_stored", reason="same bytes already stored for this paper")
        elif stored.get(row["record_id"]):
            entry.update(action="held", reason="paper already has a different file in the database")
        elif per_paper[row["record_id"]] > 1:
            entry.update(action="held", reason="the collection supplies several different files for this paper")
        else:
            entry["action"] = "import"
    return entries


def import_collection(con, store, run_id, folder, index_path, uploader, execute=False):
    entries = plan(con, run_id, folder, index_path, uploader)
    if execute:
        for entry in entries:
            if entry["action"] != "import":
                continue
            try:
                out = fulltext.save_upload(con, store, run_id, uuid.UUID(entry["record_id"]), (Path(folder) / entry["path"]).read_bytes(), Path(entry["path"]).name,
                                           entry["acquired_by"], source_url=entry["source_url"], version=entry["version"], allow_word=True)
            except ValueError as exc:
                entry.update(action="rejected", reason=str(exc))
                continue
            entry["action"] = "already_stored" if out["already_stored"] else "imported"
            entry["file_id"] = out["file_id"]
    counts = Counter((e["format"], e["action"]) for e in entries)
    report = {"mode": "execute" if execute else "dry-run", "run_id": str(run_id), "rows": len(entries),
              "by_action": dict(Counter(e["action"] for e in entries)),
              "by_format_and_action": {f"{fmt}/{action}": n for (fmt, action), n in sorted(counts.items())},
              "entries": entries}
    LOG.info(json.dumps({"event": "fulltext_collection", "mode": report["mode"], "by_action": report["by_action"]}))
    return report

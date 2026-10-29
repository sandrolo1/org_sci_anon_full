"""The source register of the proposal (Appendix A): which journals form the perimeter, and on what evidence.

The proposal's rule: a journal qualifies when (i) it is graded 4 or 4* in the Chartered Association of Business Schools Academic Journal Guide 2024 or is in
the UT Dallas list of 24 journals, and (ii) at least one article or review published in 2015-2025 is verified as substantively within the CTO or TIM domain
(recorded with its DOI or EID, the community and the date). A journal from the AIS Senior Scholars list may be added as a separately flagged CTO supplement,
and a title that is below AJG 4 and not in UTD24 must never be reported as meeting the AJG/UTD criterion. This module records and checks those facts; it
fills nothing in: every field is typed by a person who looked at the source.
"""
import csv
import datetime as dt
import re
from pathlib import Path

from .core import norm_doi, norm_eid

COLUMNS = ["source_id", "version", "exact_title", "issn", "eissn", "ajg_grade", "utd24", "ais_supplement", "community", "verified_doi", "verified_eid",
           "verified_on", "verified_by", "title_variants", "notes"]
AJG_GRADES = ("1", "2", "3", "4", "4*")
COMMUNITIES = ("CTO", "TIM", "both")
ISSN = re.compile(r"^[0-9]{4}-[0-9]{3}[0-9X]$")


def query_srcids(text):
    """SRCIDs of a Scopus query, in order, without repeats."""
    seen = []
    for x in re.findall(r"SRCID\((\d+)\)", text):
        if x not in seen:
            seen.append(x)
    return seen


def write_template(query_path, output):
    """A CSV with one row per SRCID of the query and every other field empty, for a person to fill after checking each source in Scopus."""
    ids = query_srcids(Path(query_path).read_text(encoding="utf-8"))
    if not ids:
        raise ValueError("No SRCID found in the query")
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(COLUMNS)
        writer.writerows([x] + [""] * (len(COLUMNS) - 1) for x in ids)
    return len(ids)


def _flag(value, name):
    value = (value or "").strip().lower()
    if value in ("", "false", "no", "0"):
        return False
    if value in ("true", "yes", "1"):
        return True
    raise ValueError(f"{name} must be true or false")


def parse_row(raw):
    """One CSV row -> the values to store, or ValueError saying what is wrong. Nothing is guessed or defaulted except the two yes/no flags (false)."""
    row = {k: (raw.get(k) or "").strip() for k in COLUMNS}
    if not re.fullmatch(r"[0-9]+", row["source_id"]):
        raise ValueError(f"{row['source_id'] or '(empty)'}: source_id must be the numeric Scopus Source ID")
    where = f"source {row['source_id']}"
    if not row["version"] or not row["exact_title"]:
        raise ValueError(f"{where}: version and exact_title are required")
    for field in ("issn", "eissn"):
        if row[field] and not ISSN.match(row[field]):
            raise ValueError(f"{where}: {field} must look like 1234-567X")
    if not (row["issn"] or row["eissn"]):
        raise ValueError(f"{where}: give the ISSN, the eISSN or both")
    if row["ajg_grade"] and row["ajg_grade"] not in AJG_GRADES:
        raise ValueError(f"{where}: ajg_grade must be one of {', '.join(AJG_GRADES)} or empty")
    if row["community"] not in COMMUNITIES:
        raise ValueError(f"{where}: community must be CTO, TIM or both")
    doi, eid = norm_doi(row["verified_doi"]), norm_eid(row["verified_eid"])
    if not (doi or eid):
        raise ValueError(f"{where}: a verified CTO/TIM paper is required (verified_doi or verified_eid)")
    try:
        verified_on = dt.date.fromisoformat(row["verified_on"])
    except ValueError:
        raise ValueError(f"{where}: verified_on must be a date like 2026-10-10") from None
    if not row["verified_by"]:
        raise ValueError(f"{where}: verified_by is required")
    try:
        utd24, ais = _flag(row["utd24"], "utd24"), _flag(row["ais_supplement"], "ais_supplement")
    except ValueError as exc:
        raise ValueError(f"{where}: {exc}") from None
    return {"version": row["version"], "source_id": row["source_id"], "exact_title": row["exact_title"], "issn": row["issn"] or None, "eissn": row["eissn"] or None,
            "ajg_grade": row["ajg_grade"] or None, "utd24": utd24, "ais_supplement": ais, "community": row["community"], "verified_doi": doi, "verified_eid": eid,
            "verified_on": verified_on, "verified_by": row["verified_by"], "title_variants": [x.strip() for x in row["title_variants"].split(";") if x.strip()],
            "notes": row["notes"]}


def eligibility(row):
    """'eligible' (AJG 4/4* or UTD24), 'ais_supplement' (separately flagged, never reported as meeting the AJG/UTD criterion) or 'not_eligible'."""
    if row["ajg_grade"] in ("4", "4*") or row["utd24"]:
        return "eligible"
    return "ais_supplement" if row["ais_supplement"] else "not_eligible"


def import_register(con, path):
    """Insert the rows of a CSV. All or nothing; the same row again is a no-op; a different row for an existing (version, source) is refused."""
    with Path(path).open(encoding="utf-8-sig", newline="") as stream:
        reader = csv.DictReader(stream)
        missing = [c for c in COLUMNS if c not in (reader.fieldnames or [])]
        if missing:
            raise ValueError("Required columns: " + ", ".join(COLUMNS))
        raws = [r for r in reader if any((v or "").strip() for v in r.values())]
    rows = [parse_row(r) for r in raws]
    keys = [(r["version"], r["source_id"]) for r in rows]
    if len(keys) != len(set(keys)):
        raise ValueError("The file lists the same source twice in one version")
    inserted = 0
    with con.transaction():
        for r in rows:
            existing = con.execute("SELECT exact_title,issn,eissn,ajg_grade,utd24,ais_supplement,community,verified_doi,verified_eid,verified_on,verified_by,title_variants,notes "
                                   "FROM source_register WHERE version=%s AND source_id=%s", (r["version"], r["source_id"])).fetchone()
            values = (r["exact_title"], r["issn"], r["eissn"], r["ajg_grade"], r["utd24"], r["ais_supplement"], r["community"], r["verified_doi"], r["verified_eid"],
                      r["verified_on"], r["verified_by"], r["title_variants"], r["notes"])
            if existing:
                if tuple(existing) != values:
                    raise ValueError(f"Source {r['source_id']} already exists in version {r['version']!r} with different content; create a new version")
                continue
            con.execute("INSERT INTO source_register(version,source_id,exact_title,issn,eissn,ajg_grade,utd24,ais_supplement,community,verified_doi,verified_eid,verified_on,verified_by,"
                        "title_variants,notes) VALUES (" + ",".join(["%s"] * 15) + ")", (r["version"], r["source_id"], *values))
            inserted += 1
    return {"rows": len(rows), "inserted": inserted}


def check(con, version, query_text):
    """Compare the register of one version with the SRCIDs of the Scopus query. `ready` is true only when every SRCID of the query has a register row and
    every row is eligible or a flagged AIS supplement. A missing source triggers the documented source-level check before the query is finalised."""
    rows = {r[0]: dict(zip(("source_id", "exact_title", "ajg_grade", "utd24", "ais_supplement", "community"), r)) for r in con.execute(
        "SELECT source_id,exact_title,ajg_grade,utd24,ais_supplement,community FROM source_register WHERE version=%s", (version,))}
    wanted = query_srcids(query_text)
    status = {sid: eligibility(r) for sid, r in rows.items()}
    counts = {}
    for s in status.values():
        counts[s] = counts.get(s, 0) + 1
    problems = []
    missing = [x for x in wanted if x not in rows]
    extra = [x for x in rows if x not in wanted]
    if missing:
        problems.append(f"{len(missing)} SRCIDs of the query have no register row: verify each source before the query is final")
    if extra:
        problems.append(f"{len(extra)} register rows are not in the query: the query or the register is out of date")
    if counts.get("not_eligible"):
        problems.append(f"{counts['not_eligible']} sources meet neither the AJG 4/4* nor the UTD24 criterion and are not flagged as AIS supplements")
    by_community = {}
    for r in rows.values():
        by_community[r["community"]] = by_community.get(r["community"], 0) + 1
    return {"version": version, "query_sources": len(wanted), "register_rows": len(rows), "status": counts, "by_community": by_community,
            "missing_in_register": missing, "not_in_query": extra, "not_eligible": [x for x, s in status.items() if s == "not_eligible"],
            "ais_supplements": [x for x, s in status.items() if s == "ais_supplement"], "problems": problems, "ready": not problems and bool(wanted)}

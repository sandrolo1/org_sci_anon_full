import csv
import hashlib
import json
import re
import unicodedata
import uuid
from pathlib import Path

from openpyxl import load_workbook


NAMESPACE = uuid.UUID("5df824e0-3fb8-466d-97cc-6c3fc84797e2")
SOURCE_COLUMNS = ["source_id", "journal_name", "inclusion_rationale", "active", "version", "validated_on", "validated_by"]
SEED_COLUMNS = ["seed_id", "community", "version", "doi", "eid", "title", "authors", "publication_year", "journal", "top_rank", "expert_rationale", "status", "retrieval_note", "validated_by", "validated_on"]


def digest(data):
    return hashlib.sha256(data).hexdigest()


def file_digest(path):
    sha = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            sha.update(block)
    return sha.hexdigest()


def stable_id(kind, *parts):
    return uuid.uuid5(NAMESPACE, kind + ":" + ":".join(str(p) for p in parts))


def norm_doi(value):
    value = str(value or "").strip().lower()
    return re.sub(r"^(https?://(dx\.)?doi\.org/|doi:\s*)", "", value) or None


def norm_eid(value):
    return str(value or "").strip().lower() or None


def norm_title(value):
    value = unicodedata.normalize("NFKD", str(value or "")).casefold()
    value = "".join(c for c in value if not unicodedata.combining(c))
    return " ".join(re.findall(r"[a-z0-9]+", value)) or None


def query_text(source_ids):
    ids = sorted(set(str(x).strip() for x in source_ids))
    if not ids or any(not re.fullmatch(r"[0-9]+", x) for x in ids):
        raise ValueError("At least one numeric, validated active source_id is required")
    return ("(" + " OR ".join(f"SRCID({x})" for x in ids) + ")\n"
            "AND PUBYEAR AFT 2014\nAND PUBYEAR BEF 2026\n"
            "AND (DOCTYPE(ar) OR DOCTYPE(re))\nAND SRCTYPE(j)")


def read_excel(path):
    book = load_workbook(path, read_only=True, data_only=True)
    sheet = book.active
    rows = sheet.iter_rows(values_only=True)
    try:
        headers = [str(v).strip() if v is not None else f"unnamed_{i}" for i, v in enumerate(next(rows), 1)]
    except StopIteration:
        raise ValueError("Empty workbook")
    if len(headers) != len(set(headers)):
        raise ValueError("Duplicate column headers would lose data")
    for row_number, row in enumerate(rows, 2):
        if any(v is not None and str(v).strip() for v in row):
            yield row_number, {key: str(value).strip() if value is not None else "" for key, value in zip(headers, row)}


def read_scopus(path, errors=None):
    if path.suffix.lower() in (".xlsx", ".xlsm"):
        yield from read_excel(path)
        return
    if path.suffix.lower() != ".csv":
        raise ValueError("Scopus export must be .csv, .xlsx, or .xlsm")
    with path.open(encoding="utf-8-sig", newline="") as stream:
        first = stream.readline()
        stream.seek(0)
        # Some spreadsheet exports wrap a complete comma-separated Scopus row
        # in a semicolon-separated row, followed by many empty cells.
        outer_header = next(csv.reader([first], delimiter=";"))
        wrapped = len(outer_header) > 1 and sum(bool(x) for x in outer_header) == 1 and len(next(csv.reader([outer_header[0]]))) > 1
        if wrapped:
            outer = csv.reader(stream, delimiter=";")
            def decoded():
                expected = None
                for row_number, row in enumerate(outer, 1):
                    while row and not row[-1]:
                        row.pop()
                    source = ";".join(row)
                    try:
                        inner = next(csv.reader([source], strict=True))
                    except csv.Error as exc:
                        if errors is None:
                            raise ValueError(f"Malformed CSV row {row_number}: {exc}") from exc
                        errors.append({"row_number": row_number, "reason": str(exc)})
                        yield None
                        continue
                    if expected is None:
                        expected = len(inner)
                    if len(inner) > expected and ',"' in source and not source.startswith('"'):
                        # Scopus sometimes leaves a comma in the first Authors
                        # field unquoted (for example a surname suffix).
                        split = source.index(',"')
                        repaired = '"' + source[:split].replace('"', '""') + '"' + source[split:]
                        inner = next(csv.reader([repaired], strict=True))
                    yield inner
            reader = decoded()
        else:
            reader = csv.reader(stream)
        try:
            headers = [v.strip() or f"unnamed_{i}" for i, v in enumerate(next(reader), 1)]
        except StopIteration:
            raise ValueError("Empty CSV")
        if len(headers) != len(set(headers)):
            raise ValueError("Duplicate column headers would lose data")
        for row_number, row in enumerate(reader, 2):
            if row is None:
                continue
            if not any(v.strip() for v in row):
                continue
            if len(row) != len(headers):
                message = f"CSV row {row_number} has {len(row)} fields; expected {len(headers)}"
                if errors is None:
                    raise ValueError(message)
                errors.append({"row_number": row_number, "reason": message})
                continue
            yield row_number, dict(zip(headers, row))


def mapped(raw):
    lower = {k.casefold().strip(): v for k, v in raw.items()}
    def get(*names):
        return next((lower[n] for n in names if lower.get(n)), "")
    year = get("year", "publication year")
    if year:
        year = int(float(year))
    return {"doi": norm_doi(get("doi")), "eid": norm_eid(get("eid")),
            "title": get("title", "document title"), "title_normalized": norm_title(get("title", "document title")),
            "authors": get("authors", "author names"), "publication_year": year or None,
            "source_title": get("source title", "source"), "source_id": get("source id", "sourceid") or None,
            "document_type": get("document type"), "source_type": get("source type")}


def reconcile(rows):
    missing_doi = [n for n, r in rows if not r["doi"]]
    missing_eid = [n for n, r in rows if not r["eid"]]
    groups = {}
    for n, r in rows:
        for key in (("doi", r["doi"]), ("eid", r["eid"]), ("title_year", (r["title_normalized"], r["publication_year"]) if r["title_normalized"] and r["publication_year"] else None)):
            if key[1]:
                groups.setdefault(key, []).append(n)
    duplicates = [{"method": kind, "value": value, "rows": nums} for (kind, value), nums in groups.items() if len(nums) > 1]
    return {"record_count": len(rows), "missing_doi_rows": missing_doi, "missing_eid_rows": missing_eid, "potential_duplicates": duplicates}


def match(seed, records):
    if seed["status"] == "out_of_scope":
        return "out_of_scope", None, [], "Excluded by human benchmark status"
    if seed["status"] in ("duplicate", "to_verify"):
        return "out_of_scope", None, [], "Not an eligible seed; requires human review"
    # Conflicting strong identifiers are never resolved through title fallback.
    strong_hits = {field: [r for r in records if r.get(field) == seed[field]] for field in ("doi", "eid") if seed.get(field)}
    if len(strong_hits) == 2 and all(strong_hits.values()) and {r["record_id"] for r in strong_hits["doi"]} != {r["record_id"] for r in strong_hits["eid"]}:
        both = {r["record_id"]: r for hits in strong_hits.values() for r in hits}
        return "ambiguous", "doi_eid", list(both), "DOI and EID point to different records"
    for field in ("doi", "eid"):
        value = seed.get(field)
        if value:
            candidates = strong_hits[field]
            if candidates:
                conflicting = [r for r in candidates if any(seed.get(other) and r.get(other) and seed[other] != r[other] for other in ("doi", "eid") if other != field)]
                if conflicting:
                    return "ambiguous", field, [r["record_id"] for r in candidates], "Multiple records or conflicting identifiers"
                if len(candidates) > 1:
                    for other in ("doi", "eid"):
                        values = {r.get(other) for r in candidates if r.get(other)}
                        if len(values) > 1:
                            return "ambiguous", field, [r["record_id"] for r in candidates], "Conflicting identifiers across export rows"
                    if not any(r.get("doi") or r.get("eid") for r in candidates):
                        return "ambiguous", field, [r["record_id"] for r in candidates], "Multiple unidentified rows"
                return "found", field, [r["record_id"] for r in candidates], None
    if seed.get("doi") or seed.get("eid"):
        # A bibliographic match with a different identifier needs review.
        pass
    title = norm_title(seed["title"])
    candidates = [r for r in records if r.get("title_normalized") == title and r.get("publication_year") == seed["publication_year"]]
    if len(candidates) == 1:
        r = candidates[0]
        if any(seed.get(field) and r.get(field) and seed[field] != r[field] for field in ("doi", "eid")):
            return "ambiguous", "title_year", [r["record_id"]], "Title/year agrees but identifier conflicts"
        return "found", "title_year", [r["record_id"]], None
    if candidates:
        return "ambiguous", "title_year", [r["record_id"] for r in candidates], "Multiple title/year candidates"
    return "missing", None, [], "Check source list, export completeness, year, document/source type, or bibliographic metadata"


def coverage(results):
    eligible = [r for r in results if r["seed_status"] == "eligible"]
    found = sum(r["outcome"] == "found" for r in eligible)
    denominator = len(eligible)
    return {"found": found, "eligible": denominator,
            "coverage": found / denominator if denominator else None,
            "target_met": found / denominator >= .95 if denominator else None,
            "out_of_scope": sum(r["seed_status"] == "out_of_scope" for r in results),
            "ambiguous": sum(r["outcome"] == "ambiguous" for r in eligible)}

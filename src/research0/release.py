"""Phase 7 (last part): the database release.

`build` evaluates every quality gate and records a release row with a manifest (what it contains, which runs and versions it comes from,
and a checksum of the manifest). The row is immutable. A release is

  blocked    at least one gate is not a pass (a failed gate, or a gate that cannot be evaluated yet): the run is kept, nothing is overwritten,
             the report says what to correct;
  candidate  every gate passes but nobody has signed it off;
  validated  every gate passes AND at least one person signed it off. Only a validated release is read by the discovery agents (Phase 8).
"""
import csv
import json
import logging
from pathlib import Path

from . import pairs as pair_module
from . import quality
from .core import digest, stable_id

LOG = logging.getLogger("research0")
REQUIRED_SIGNOFFS = 1
SCHEMA = "release-v1"


def _count(con, sql, *params):
    return con.execute(sql, params).fetchone()[0]


def membership_counts(con, run_id):
    """Counts as the proposal defines them: global statistics count distinct papers; statistics by membership count each paper in every community it belongs to;
    cross-community totals can weight a paper 1/k, k being its number of memberships. 'Belongs' means the effective label is yes (the full text wins over the metadata)."""
    from .confirmation import memberships
    effective = memberships(con, run_id)
    per_paper = {}
    for (cid, community), m in effective.items():
        per_paper.setdefault(cid, {})[community] = m["label"]
    out = {"papers_with_a_membership_label": len(per_paper), "by_community": {}, "fractional_1_over_k": {"CTO": 0.0, "TIM": 0.0}}
    for community in ("CTO", "TIM"):
        out["by_community"][community] = {label: sum(1 for p in per_paper.values() if p.get(community) == label) for label in ("yes", "no", "uncertain")}
    members = {cid: [c for c, label in p.items() if label == "yes"] for cid, p in per_paper.items()}
    out["distinct_papers_in_any_community"] = sum(1 for m in members.values() if m)
    out["in_both_communities"] = sum(1 for m in members.values() if len(m) == 2)
    for m in members.values():
        for c in m:
            out["fractional_1_over_k"][c] += 1 / len(m)
    out["status_of_memberships"] = {}
    for m in effective.values():
        out["status_of_memberships"][m["status"]] = out["status_of_memberships"].get(m["status"], 0) + 1
    return out


def model_card(con, ids):
    card = {"intended_use": "Pre-assign papers to CTO and TIM, extract construct evidence and propose candidate construct pairs for human adjudication. Not a decision maker.",
            "not_for": ["deciding that two constructs are the same or different", "excluding a paper for which text or an abstract is missing", "any use outside the declared journal and year perimeter"],
            "components": {}}
    if ids.get("preassignment_run_id"):
        card["components"]["preassignment"] = dict(zip(("source", "model", "prompt_version", "scope_version", "codebook_sha256"), con.execute(
            "SELECT source,model,prompt_version,scope_version,codebook_sha256 FROM preassignment_run WHERE preassignment_run_id=%s", (ids["preassignment_run_id"],)).fetchone()))
    if ids.get("extraction_run_id"):
        card["components"]["extraction"] = dict(zip(("model", "prompt_version", "schema_version"), con.execute(
            "SELECT model,prompt_version,schema_version FROM extraction_run WHERE extraction_run_id=%s", (ids["extraction_run_id"],)).fetchone()))
    if ids.get("signal_run_id"):
        row = con.execute("SELECT r.model_id,m.kind,m.dimensions,m.parameters,r.version,r.params FROM signal_run r JOIN embedding_model m USING (model_id) WHERE r.signal_run_id=%s", (ids["signal_run_id"],)).fetchone()
        card["components"]["signals"] = dict(zip(("embedding_model", "embedding_kind", "dimensions", "embedding_parameters", "version", "thresholds_and_parameters"), row))
        card["components"]["signals"]["note"] = "Thresholds are targets set before the audit and frozen; they are not results."
    card["known_limits"] = ["The signals (S_def, S_op, C_cit, N_nom) and the retrieval strata only select pairs for review; no threshold establishes conceptual identity or difference.",
                            "Citation proximity is bibliographic coupling only; co-citation needs a corpus of citing papers that is not available.",
                            "The links between the neighbours of two constructs in N_nom are proposed by label similarity, not approved identities.",
                            "A paper without a usable abstract is 'uncertain', and a PDF without a text layer stays unavailable (no OCR)."]
    return card


def manifest_for(con, run_id, ids, operational_window=None):
    from . import community_v2
    m = {"schema": SCHEMA, "catalog_run_id": str(run_id), "runs": {k: (str(v) if v else None) for k, v in ids.items()}}
    m["catalog"] = {"canonical_papers": _count(con, "SELECT count(*) FROM canonical_paper WHERE run_id=%s", run_id),
                    "eligible_papers": _count(con, "SELECT count(*) FROM canonical_paper WHERE run_id=%s AND status='retained' AND cardinality(perimeter_flags)=0", run_id)}
    first, last = con.execute("SELECT min(publication_year),max(publication_year) FROM canonical_paper WHERE run_id=%s AND status='retained' AND cardinality(perimeter_flags)=0", (run_id,)).fetchone()
    m["time_perimeter"] = {"catalog_years": [first, last], "operational_window_for_full_text": list(operational_window) if operational_window else None,
                           "papers_in_window": None if not operational_window else _count(con, "SELECT count(*) FROM canonical_paper WHERE run_id=%s AND status='retained' AND cardinality(perimeter_flags)=0 "
                                                                                          "AND publication_year BETWEEN %s AND %s", run_id, *operational_window)}
    m["full_text_status"] = dict(con.execute("SELECT s.status,count(*) FROM paper_status s JOIN canonical_paper c USING (canonical_id) WHERE c.run_id=%s AND c.status='retained' AND cardinality(c.perimeter_flags)=0 GROUP BY 1", (run_id,)).fetchall())
    m["counts"] = membership_counts(con, run_id)
    m["model_card"] = model_card(con, ids)
    if ids.get("audit_id"):
        from . import audit
        try:
            m["audit"] = {"design": con.execute("SELECT design FROM audit_sample WHERE audit_id=%s", (ids["audit_id"],)).fetchone()[0],
                          "agreement": audit.agreement(con, ids["audit_id"]), "estimates": audit.estimates(con, ids["audit_id"]),
                          "confusion_matrices": audit.confusion_matrices(con, ids["audit_id"]), "breakdowns": audit.breakdowns(con, ids["audit_id"])}
        except ValueError as exc:
            m["audit"] = {"unavailable": str(exc)}
    if ids.get("preassignment_run_id"):
        m["preassignment"] = community_v2.summary(con, ids["preassignment_run_id"])
        m["preassignment_run"] = dict(zip(("source", "model", "prompt_version", "scope_version", "codebook_sha256"), con.execute(
            "SELECT source,model,prompt_version,scope_version,codebook_sha256 FROM preassignment_run WHERE preassignment_run_id=%s", (ids["preassignment_run_id"],)).fetchone()))
    m["fulltext"] = {"files": _count(con, "SELECT count(*) FROM fulltext_file WHERE run_id=%s", run_id),
                     "unvalidated_files": _count(con, "SELECT count(*) FROM fulltext_file WHERE run_id=%s AND validation_status='unvalidated'", run_id)}
    if ids.get("ingestion_run_id"):
        versions = dict(con.execute("SELECT status,count(*) FROM paper_version WHERE ingestion_run_id=%s GROUP BY 1", (ids["ingestion_run_id"],)).fetchall())
        m["documents"] = {"versions": versions, "chunks": _count(con, "SELECT count(*) FROM document_chunk c JOIN paper_version v USING (version_id) WHERE v.ingestion_run_id=%s", ids["ingestion_run_id"]),
                          "parser": dict(zip(("parser", "parser_version", "chunker_version"), con.execute("SELECT parser,parser_version,chunker_version FROM ingestion_run WHERE ingestion_run_id=%s", (ids["ingestion_run_id"],)).fetchone()))}
    if ids.get("extraction_run_id"):
        e = ids["extraction_run_id"]
        m["extraction"] = {"mentions": _count(con, "SELECT count(*) FROM construct_mention WHERE extraction_run_id=%s", e),
                           "measures": _count(con, "SELECT count(*) FROM construct_measure s JOIN construct_mention m USING (mention_id) WHERE m.extraction_run_id=%s", e),
                           "claims": _count(con, "SELECT count(*) FROM nomological_claim WHERE extraction_run_id=%s", e),
                           "model": dict(zip(("model", "prompt_version", "schema_version"), con.execute("SELECT model,prompt_version,schema_version FROM extraction_run WHERE extraction_run_id=%s", (e,)).fetchone()))}
    if ids.get("signal_run_id"):
        s = ids["signal_run_id"]
        m["signals"] = {"nodes": _count(con, "SELECT count(*) FROM construct_node WHERE signal_run_id=%s", s), "candidate_pairs": _count(con, "SELECT count(*) FROM candidate_pair WHERE signal_run_id=%s", s),
                        "availability": dict(zip(("s_def", "s_op", "s_sem_op", "c_cit", "n_nom"), con.execute(
                            "SELECT count(s_def),count(s_op),count(s_sem_op),count(c_cit),count(n_nom) FROM candidate_pair WHERE signal_run_id=%s", (s,)).fetchone())),
                        "retrieval_strata": dict(con.execute("SELECT hint,count(*) FROM candidate_pair WHERE signal_run_id=%s GROUP BY 1", (s,)).fetchall()),
                        "adjudication": pair_module.counts(con, s),
                        "embedding": dict(zip(("model_id", "kind", "version"), con.execute("SELECT r.model_id,m.kind,r.version FROM signal_run r JOIN embedding_model m USING (model_id) WHERE r.signal_run_id=%s", (s,)).fetchone()))}
        if m["signals"]["embedding"]["kind"] == "lexical":
            m["signals"]["warning"] = "S_def, S_op and S_sem-op were computed with a lexical embedding: they measure shared words, not meaning."
    return m


def build(con, run_id, label, preassignment_run_id=None, audit_id=None, ingestion_run_id=None, extraction_run_id=None, signal_run_id=None,
          gold_extraction=None, gold_pairs=None, gold_ratings=None, operational_window=None):
    label = (label or "").strip()
    if not label:
        raise ValueError("A release label is required")
    gates = quality.evaluate(con, run_id, audit_id, extraction_run_id, signal_run_id, gold_extraction, gold_pairs, gold_ratings)
    ids = {"preassignment_run_id": preassignment_run_id, "audit_id": audit_id, "ingestion_run_id": ingestion_run_id, "extraction_run_id": extraction_run_id,
           "signal_run_id": signal_run_id, "gold_extraction": gold_extraction, "gold_pairs": gold_pairs, "gold_ratings": gold_ratings}
    manifest = manifest_for(con, run_id, ids, operational_window)
    manifest["gates"] = [{k: g[k] for k in ("id", "threshold", "value", "status")} for g in gates]
    blob = json.dumps(manifest, sort_keys=True, ensure_ascii=False, default=str)
    sha = digest(blob.encode())
    release_id = stable_id("release", run_id, label, sha)
    passed = all(g["status"] == "pass" for g in gates)
    created = con.execute("INSERT INTO release(release_id,run_id,label,gates_passed,gates,manifest,manifest_sha256) VALUES (%s,%s,%s,%s,%s::jsonb,%s::jsonb,%s) ON CONFLICT DO NOTHING RETURNING 1",
                          (release_id, run_id, label, passed, json.dumps(gates, default=str), blob, sha)).fetchone()
    return release_id, bool(created), passed, gates, manifest


def status(con, release_id):
    row = con.execute("SELECT gates_passed FROM release WHERE release_id=%s", (release_id,)).fetchone()
    if not row:
        raise ValueError("Unknown release")
    if not row[0]:
        return "blocked"
    signed = con.execute("SELECT count(DISTINCT validator) FROM release_signoff WHERE release_id=%s", (release_id,)).fetchone()[0]
    return "validated" if signed >= REQUIRED_SIGNOFFS else "candidate"


def sign_off(con, release_id, validator, note=""):
    validator = (validator or "").strip()
    if not validator or len(validator) > 80:
        raise ValueError("The validator's name is required")
    if status(con, release_id) == "blocked":
        raise ValueError("A blocked release cannot be signed off: correct the failing gates and build a new release")
    con.execute("INSERT INTO release_signoff(release_id,validator,note) VALUES (%s,%s,%s)", (release_id, validator, note or ""))


def latest_validated(con, run_id):
    for (release_id,) in con.execute("SELECT release_id FROM release WHERE run_id=%s ORDER BY created_at DESC", (run_id,)).fetchall():
        if status(con, release_id) == "validated":
            return release_id
    return None


def write_report(con, release_id, directory):
    row = con.execute("SELECT label,gates_passed,gates,manifest,manifest_sha256,created_at FROM release WHERE release_id=%s", (release_id,)).fetchone()
    label, passed, gates, manifest, sha, created = row
    state = status(con, release_id)
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    (directory / "manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False, sort_keys=True) + "\n", encoding="utf-8")
    lines = [f"# Release {label} ({release_id})", "", f"Status: **{state}** | manifest sha256 `{sha}` | created {created:%Y-%m-%d %H:%M} UTC", ""]
    if state == "blocked":
        lines += ["The release is suspended: at least one gate is not a pass. The run is kept; correct the cause and build a new release (nothing here is overwritten).", ""]
    elif state == "candidate":
        lines += ["All gates pass. The release becomes validated when a person signs it off.", ""]
    lines += ["| Gate | Threshold | Value | Status | Detail |", "|---|---:|---:|---|---|"]
    for g in gates:
        value = "n/a" if g["value"] is None else f"{g['value']:.3f}"
        lines.append(f"| {g['name']} | {g['threshold']:.2f} | {value} | {g['status']} | {g['detail']} |")
    signed = con.execute("SELECT validator,signed_at,note FROM release_signoff WHERE release_id=%s ORDER BY signed_at", (release_id,)).fetchall()
    lines += ["", "## Sign-offs", ""] + ([f"- {v} ({t:%Y-%m-%d}) {n}" for v, t, n in signed] or ["- none"])
    (directory / "release_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    sid = manifest.get("runs", {}).get("signal_run_id")
    if sid:
        pair_module.export_settled(con, sid, directory / "construct_relations.csv")
        write_registry(con, sid, directory / "construct_registry.csv")
    write_memberships(con, manifest["catalog_run_id"], directory / "memberships.csv")
    (directory / "model_card.json").write_text(json.dumps(manifest.get("model_card", {}), indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    (directory / "counts.json").write_text(json.dumps({k: manifest.get(k) for k in ("counts", "time_perimeter", "full_text_status")}, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    if manifest.get("audit") and "estimates" in manifest["audit"]:
        from . import audit
        audit.write_report(manifest["audit"]["estimates"], manifest["audit"]["agreement"], directory / "audit_report.md")
    return directory


def write_registry(con, signal_run_id, path):
    """The construct registry of the release: every construct use (a provisional identity per paper and label) with its definition and its meaning profile, and,
    apart, its operationalization (measures and measurement attributes), with the page of the evidence. Identities are not merged here; the decided, typed
    relations between uses are in construct_relations.csv."""
    import csv
    from .extraction import MEANING_KINDS, OPERATIONAL_KINDS
    with Path(path).open("w", encoding="utf-8", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(["node_id", "paper", "label", "definition", "definition_page", "meaning_profile", "measures", "operationalization", "other_attributes", "status"])
        for node_id, paper, label, mention_ids in con.execute("SELECT node_id,canonical_id,label_normalized,mention_ids FROM construct_node WHERE signal_run_id=%s ORDER BY label_normalized,canonical_id", (signal_run_id,)).fetchall():
            definition = con.execute("SELECT definition_text,page FROM construct_mention WHERE mention_id = ANY(%s) AND definition_text IS NOT NULL ORDER BY char_start LIMIT 1", (mention_ids,)).fetchone() or (None, None)
            measures = "; ".join(r[0] for r in con.execute("SELECT name FROM construct_measure WHERE mention_id = ANY(%s) ORDER BY char_start", (mention_ids,)))
            attributes = con.execute("SELECT kind,basis,value FROM extraction_attribute WHERE mention_id = ANY(%s) ORDER BY kind", (mention_ids,)).fetchall()
            text = lambda kinds, keep: "; ".join(f"{k}={v or 'MISSING'} ({b})" for k, b, v in attributes if (k in kinds) == keep)
            writer.writerow([node_id, paper, label, definition[0], definition[1], text(MEANING_KINDS, True), measures, text(OPERATIONAL_KINDS, True),
                             "; ".join(f"{k}={v or 'MISSING'} ({b})" for k, b, v in attributes if k not in MEANING_KINDS and k not in OPERATIONAL_KINDS), "provisional"])


def write_memberships(con, run_id, path):
    import csv
    from .confirmation import memberships
    with Path(path).open("w", encoding="utf-8", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(["canonical_id", "community", "label", "status", "stage", "confidence"])
        for (cid, community), m in sorted(memberships(con, run_id).items(), key=lambda x: (str(x[0][0]), x[0][1])):
            writer.writerow([cid, community, m["label"], m["status"], m["stage"], m["confidence"]])

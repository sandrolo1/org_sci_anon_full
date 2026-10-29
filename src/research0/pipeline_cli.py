"""Command line for Phases 2-8 (clusters, pre-assignment, audit, documents, extraction, signals, release, retrieval) and the mock world.

Nothing here calls an external model by itself. The steps that cost money or send text to OpenAI are the existing explicit ones:
`community-submit --run <dir>` submits a prepared request directory (pre-assignment or extraction) to the Batch API, and
`preassign-collect --remote` / `extract-collect --remote` fetch the result.
"""
import csv
import json
import os
import sys
import uuid
from pathlib import Path

DEFAULT_SCOPE_V2 = Path("config/community_scope_v2.json")
DEFAULT_SCOPE_V1 = Path("config/community_scope_v1.json")
# the classes of the pair codebook v2 (pairs.CLASSES), spelled out so that building the parser imports nothing heavy
PAIR_CLASSES = ("conceptual_jingle", "conceptual_jangle", "measurement_divergence", "scope_variation", "empirical_redundancy", "part_whole", "related_but_distinct",
                "distinct", "same_meaning", "unresolved")


def register(sub):
    def cmd(name, *args, **kwargs):
        p = sub.add_parser(name, **kwargs)
        for flags, options in args:
            p.add_argument(*([flags] if isinstance(flags, str) else flags), **options)
        return p

    run = ("--run-id", {"required": True})
    cmd("embed-papers", run, ("--model", {}), ("--limit", {"type": int}))
    cmd("cluster", run, ("--k", {"type": int, "default": 40}), ("--seed", {"type": int, "default": 7}), ("--model", {}), ("--limit", {"type": int}),
        ("--output", {"type": Path}), ("--preassignment-run", {}))
    cmd("preassign-prepare", run, ("--output", {"type": Path, "required": True}), ("--scope", {"type": Path, "default": DEFAULT_SCOPE_V2}))
    cmd("preassign-collect", ("--run-dir", {"type": Path, "required": True}), ("--scope", {"type": Path, "default": DEFAULT_SCOPE_V2}), ("--remote", {"action": "store_true"}))
    cmd("preassign-import-v1", run, ("--assignments-dir", {"type": Path, "required": True}), ("--scope", {"type": Path, "default": DEFAULT_SCOPE_V1}))
    cmd("audit-sample", run, ("--preassignment-run", {"required": True}), ("--size", {"type": int, "default": 300}), ("--min-per-stratum", {"type": int, "default": 20}),
        ("--min-excluded", {"type": int, "default": 60}), ("--seed", {"type": int, "default": 20261010}))
    cmd("audit-export", ("--audit-id", {"required": True}), ("--output", {"type": Path, "required": True}))
    cmd("audit-import", ("--audit-id", {"required": True}), ("--csv", {"type": Path, "required": True}), ("--validator", {"required": True}),
        ("--adjudication", {"action": "store_true"}))
    cmd("audit-report", ("--audit-id", {"required": True}), ("--output", {"type": Path, "required": True}), ("--expected-kappa", {"type": float, "default": 0.8}))
    cmd("ingest", run, ("--limit", {"type": int}), ("--years", {}))
    cmd("embed-chunks", ("--ingestion-run-id", {"required": True}), ("--model", {}))
    cmd("extract-prepare", ("--ingestion-run-id", {"required": True}), ("--output", {"type": Path, "required": True}))
    cmd("extract-collect", ("--run-dir", {"type": Path, "required": True}), ("--remote", {"action": "store_true"}),
        ("--direct", {"action": "store_true", "help": "write straight into the live tables; by default the run goes to staging and waits for extract-promote"}))
    cmd("extract-promote", ("--extraction-run-id", {}), ("--by", {}), ("--note", {"default": ""}),
        ("--delta", {"action": "store_true", "help": "for a run already promoted: copy only the staged rows the live tables lack, under a new delta run"}),
        ("--execute", {"action": "store_true", "help": "without it, only list the staged runs or say what would be copied"}))
    cmd("signals-build", ("--extraction-run-id", {"required": True}), ("--model", {}),
        ("--require", {"choices": ("definition", "definition_and_measures", "none"), "default": "definition", "help": "construct uses left out before pairing: without a definition or meaning profile (default), also without measures, or none"}),
        ("--corpus-tag", {"help": "free label kept in the run's params; it gives a new signal_run_id when the rows of the extraction run changed after an earlier build (a rebuild with the same inputs is otherwise 'already built')"}))
    cmd("dossiers-build", ("--signal-run-id", {"required": True}), ("--hint", {"action": "append"}), ("--targeted", {"action": "store_true", "help": "the strata the cascade targets (proposals.TARGETED_HINTS) instead of --hint"}),
        ("--limit", {"type": int}), ("--pair-sample-id", {}))
    cmd("pairs-export-blind", ("--signal-run-id", {"required": True}), ("--output", {"type": Path, "required": True}), ("--hint", {"action": "append"}), ("--limit", {"type": int}))
    answers = {"required": True, "choices": ("yes", "no", "unclear")}
    cmd("pair-decide", ("--pair-id", {"required": True}), ("--validator", {"required": True}), ("--class", {"required": True, "dest": "decision_class", "choices": PAIR_CLASSES}),
        ("--same-referent", answers), ("--compatible-properties", answers), ("--substitution-changes", answers), ("--consequence", {"default": ""}),
        ("--counterexample", {"default": ""}), ("--rationale", {"required": True}), ("--final", {"action": "store_true"}))
    cmd("pairs-report", ("--signal-run-id", {"required": True}), ("--output", {"type": Path}))
    cmd("gold-import", ("--kind", {"required": True, "choices": ("extraction", "pairs", "ratings")}), ("--file", {"type": Path, "required": True}),
        ("--label", {"required": True}), ("--provided-by", {"required": True}))
    cmd("release-build", run, ("--label", {"required": True}), ("--operational-window", {}), ("--preassignment-run", {}), ("--audit-id", {}), ("--ingestion-run-id", {}), ("--extraction-run-id", {}),
        ("--signal-run-id", {}), ("--gold-extraction", {}), ("--gold-pairs", {}), ("--gold-ratings", {}), ("--output", {"type": Path}))
    cmd("release-signoff", ("--release-id", {"required": True}), ("--validator", {"required": True}), ("--note", {"default": ""}))
    cmd("rag-search", ("--release-id", {"required": True}), ("--query", {"required": True}), ("-k", {"type": int, "default": 10}), ("--agent", {"default": "cli"}),
        ("--section", {"action": "append"}), ("--model", {}))
    cmd("confirm-prepare", run, ("--ingestion-run-id", {"required": True}), ("--output", {"type": Path, "required": True}), ("--scope", {"type": Path, "default": DEFAULT_SCOPE_V2}), ("--years", {}))
    cmd("confirm-collect", ("--run-dir", {"type": Path, "required": True}), ("--scope", {"type": Path, "default": DEFAULT_SCOPE_V2}))
    cmd("extraction-sample", run, ("--preassignment-run", {"required": True}), ("--seed", {"type": int, "default": 20261010}))
    targeted = ("--targeted", {"action": "store_true", "help": "the strata the cascade targets (proposals.TARGETED_HINTS) instead of --hint"})
    cmd("proposal-prepare", ("--signal-run-id", {"required": True}), ("--output", {"type": Path, "required": True}), ("--hint", {"action": "append"}), targeted, ("--limit", {"type": int}))
    cmd("proposal-export", ("--signal-run-id", {"required": True}), ("--output", {"type": Path, "required": True}),
        ("--class", {"action": "append", "dest": "classes", "help": "model classes to export (default conceptual jingle and jangle)"}))
    cmd("proposal-collect", ("--run-dir", {"type": Path, "required": True}), ("--remote", {"action": "store_true", "help": "fetch the submitted Batch's output first"}))
    cmd("pair-sample", ("--signal-run-id", {"required": True}), ("--probabilistic", {"type": int, "default": 200}), ("--challenge", {"type": int, "default": 100}), ("--seed", {"type": int, "default": 20261010}), ("--allocation", {"choices": ("equal", "proportional"), "default": "equal"}),
        ("--design", {"choices": ("stratified", "cascade", "review"), "default": "stratified", "help": "cascade: the pairs the model promotes, plus audit samples of those it rejects and of those never targeted; review: every pair the model calls a conceptual jingle or jangle (an enriched set, not the validation)"}),
        ("--hint", {"action": "append", "help": "cascade: the targeted strata (default proposals.TARGETED_HINTS)"}), ("--audit", {"type": int, "default": 50, "help": "cascade: pairs per audit group"}),
        ("--promoted", {"type": int, "help": "cascade: at most this many promoted pairs (default all)"}))
    cmd("pair-sample-report", ("--pair-sample-id", {"required": True}))
    cmd("rag-index", ("--model", {}), ("--kind", {"default": "hnsw", "choices": ("hnsw", "ivfflat")}))
    cmd("source-register-template", ("--query", {"type": Path, "default": Path("config/scopus_query_v1.txt")}), ("--output", {"type": Path, "default": Path("templates/source_register_from_query_v1.csv")}))
    cmd("source-register-import", ("csv", {"type": Path}))
    cmd("source-register-check", ("--version", {"required": True}), ("--query", {"type": Path, "default": Path("config/scopus_query_v1.txt")}))
    cmd("mock-clean")
    cmd("demo", ("--keep", {"action": "store_true"}), ("--papers", {"type": int, "default": 240}))
    cmd("embed-corpus", ("--ingestion-run-id", {"required": True}), ("--extraction-run-id", {"required": True}), ("--model", {"default": "openai:text-embedding-3-small"}),
        ("--only", {"choices": ("chunks", "records", "both"), "default": "both"}), ("--dry-run", {"action": "store_true", "help": "count what would be sent; call nothing"}),
        ("--acknowledge-send-to-openai", {"action": "store_true", "help": "required for an openai model: the texts are sent to OpenAI and cost money"}))
    cmd("forum-serve", ("--run-id", {"required": True}), ("--port", {"type": int, "default": 8770}))
    cmd("forum-score", ("--run-id", {"required": True, "help": "catalog run"}), ("--proposal-id", {"help": "evaluate this proposal again (a new row); without it, every proposal that has no score yet"}),
        ("--hybrid-missing", {"action": "store_true", "help": "instead of the proposals with no score: those already scored whose novelty has no hybrid-judge vote (a new row each; the other judges' answers are already in the register, so only the hybrid judge is paid for)"}),
        ("--acknowledge-send-to-openai", {"action": "store_true", "help": "required: the abstract and the cards are sent to OpenAI (embeddings, one judge, two for a sample of the proposals) and cost money"}))
    cmd("forum-score-corpus", ("--run-id", {"required": True, "help": "catalog run whose paper cards (method, impact) or CTO/TIM assignments (novelty) are used"}),
        ("--kind", {"choices": ("novelty", "method", "impact"), "required": True, "help": "novelty: the abstracts of the Scopus exports; method, impact: the paper cards"}),
        ("--scopus", {"type": Path, "nargs": "+", "help": "novelty: the Scopus export CSVs (default: every CSV in inputs/scopus)"}),
        ("--assignments", {"type": Path, "help": "novelty: assignments.csv of the CTO/TIM Batch (default: reports/community_dashboard/<run-id>/assignments.csv)"}),
        ("--catalog", {"type": Path, "default": Path("reports/catalog_2026-10-09/catalog.csv"), "help": "novelty: the canonical catalog, to link the assignments to Scopus EIDs"}),
        ("--model", {"default": "openai:text-embedding-3-small", "help": "must be the embedding model of the scoring configuration"}),
        ("--acknowledge-send-to-openai", {"action": "store_true", "help": "required for an openai model: the texts are sent to OpenAI and cost money"}))
    cmd("forum-run", ("--release-id", {"required": True}), ("--config", {"type": Path, "required": True, "help": "JSON file with the episode configuration (see the Settings tab of an episode)"}),
        ("--by", {"required": True}), ("--create-only", {"action": "store_true"}),
        ("--acknowledge-send-to-openai", {"action": "store_true", "help": "required for provider openai: the prompts, with passages of the corpus, are sent to OpenAI and cost money"}))
    cmd("forum-batch", ("--release-id", {"required": True}),
        ("--config", {"type": Path, "required": True, "help": "JSON file with the common settings: an episode configuration without condition, with one 'agent' block for every agent"}),
        ("--repetitions", {"type": int, "default": 1, "help": "how many times the five arms 0, A, B (CTO), B (TIM), C are run"}),
        ("--seed-mode", {"choices": ("vary", "same"), "default": "vary", "help": "vary: seed, seed+1, ... per repetition; same: the same seed every repetition"}),
        ("--by", {"required": True}), ("--create-only", {"action": "store_true"}),
        ("--acknowledge-send-to-openai", {"action": "store_true", "help": "required for provider openai: the prompts, with passages of the corpus, are sent to OpenAI and cost money"}))


COMMANDS = {"embed-papers", "cluster", "preassign-prepare", "preassign-collect", "preassign-import-v1", "audit-sample", "audit-export", "audit-import", "audit-report", "ingest",
            "embed-chunks", "extract-prepare", "extract-collect", "extract-promote", "signals-build", "dossiers-build", "pairs-export-blind", "pair-decide", "pairs-report", "gold-import",
            "release-build", "release-signoff", "rag-search", "mock-clean", "demo", "confirm-prepare", "confirm-collect", "extraction-sample", "proposal-prepare", "proposal-collect", "proposal-export",
            "pair-sample", "pair-sample-report", "rag-index", "source-register-template", "source-register-import", "source-register-check", "forum-run", "forum-batch", "forum-score", "forum-score-corpus", "forum-serve", "embed-corpus"}


def _uuid(value):
    return uuid.UUID(value) if value else None


def run(args, database):
    """Executes one command; returns the dict to print."""
    c = args.command
    if c == "source-register-template":
        from . import source_register
        return {"event": "source_register_template", "sources": source_register.write_template(args.query, args.output), "output": str(args.output),
                "next": "Fill every other column after checking each source in Scopus, AJG 2024 and the UTD24 list; nothing is pre-filled."}
    if c == "mock-clean":
        from . import mock_world
        return {"event": "mock_schemas_dropped", "schemas": mock_world.clean_all()}
    if c == "demo":
        from .demo import run_demo
        return run_demo(args.papers, args.keep)
    if c == "forum-serve":
        from .forum_server import serve
        server, run_id = serve(args.run_id, args.port)
        print(json.dumps({"event": "forum_serving", "run_id": str(run_id), "url": f"http://127.0.0.1:{server.server_address[1]}/"}), flush=True)
        try:
            server.serve_forever()
        except KeyboardInterrupt:
            server.server_close()
        return {"event": "forum_stopped"}
    if c == "rag-search":
        from . import rag
        from .embeddings import get_embedder
        url = os.environ.get("AGENT_DATABASE_URL") or os.environ["DATABASE_URL"]
        with rag.reader_connection(url) as con:
            hits = rag.search(con, _uuid(args.release_id), args.query, get_embedder(args.model), k=args.k, sections=args.section, agent=args.agent)
        return {"event": "rag_search", "hits": [{k: h[k] for k in ("rank", "score", "title", "year", "section", "pages", "chunk_id", "text")} for h in hits]}
    if c == "preassign-collect" and not args.remote:
        from . import community_v2
        with database() as con:
            return {"event": "preassignments_collected", **community_v2.collect_file(con, args.run_dir, args.scope)}
    with database() as con:
        return _with_database(args, con)


def _with_database(args, con):
    from . import audit, community_v2, documents, explore, extraction, pairs, quality, release, signals
    from .embeddings import get_embedder
    c = args.command
    if c == "embed-papers":
        new, existing = explore.embed_papers(con, _uuid(args.run_id), get_embedder(args.model, offline_default=True), args.limit)
        return {"event": "papers_embedded", "new": new, "already_embedded": existing}
    if c == "cluster":
        embedder = get_embedder(args.model, offline_default=True)
        cluster_run_id, summary, already = explore.cluster(con, _uuid(args.run_id), embedder, args.k, args.seed, args.limit)
        out = {"event": "clustered", "cluster_run_id": str(cluster_run_id), "already_built": already, "papers": summary["papers"], "k": summary["k"], "warning": summary["warning"]}
        if args.output:
            out["report"] = str(explore.write_report(con, cluster_run_id, args.output, _uuid(args.preassignment_run)))
        return out
    if c == "preassign-prepare":
        manifest = community_v2.prepare(con, _uuid(args.run_id), args.output, args.scope)
        return {"event": "preassignment_prepared", "directory": str(args.output), "eligible_papers": manifest["eligible_papers"], "requests": manifest["request_count"],
                "not_sent_uncertain": len(manifest["insufficient"]), "next": f"research0 community-submit --run {args.output}   (this sends text to OpenAI and costs money)"}
    if c == "preassign-collect":
        return {"event": "preassignments_collected", **community_v2.collect_remote(con, args.run_dir, args.scope)}
    if c == "preassign-import-v1":
        pre_id, n = community_v2.import_v1_suggestions(con, _uuid(args.run_id), args.assignments_dir, args.scope)
        return {"event": "v1_suggestions_imported", "preassignment_run_id": str(pre_id), "rows": n, "summary": community_v2.summary(con, pre_id)}
    if c == "audit-sample":
        audit_id, design, existed = audit.create_sample(con, _uuid(args.run_id), _uuid(args.preassignment_run), args.size, args.min_per_stratum, args.min_excluded, args.seed)
        return {"event": "audit_sample", "audit_id": str(audit_id), "already_existed": existed, "design": design}
    if c == "audit-export":
        papers = audit.blind_papers(con, _uuid(args.audit_id), "")
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with args.output.open("w", encoding="utf-8-sig", newline="") as stream:
            writer = csv.writer(stream)
            writer.writerow(["canonical_id", "title", "year", "abstract", "author_keywords", "cto", "tim", "note"])
            writer.writerows([p["id"], p["title"], p["year"], p["abstract"], p["author_keywords"], "", "", ""] for p in papers)
        return {"event": "audit_exported", "papers": len(papers), "output": str(args.output), "fill": "cto and tim with yes, no or uncertain"}
    if c == "audit-import":
        audit_id, n = _uuid(args.audit_id), 0
        with args.csv.open(encoding="utf-8-sig", newline="") as stream:
            for row in csv.DictReader(stream):
                for community in ("CTO", "TIM"):
                    label = (row.get(community.lower()) or "").strip().lower()
                    if not label:
                        continue
                    if args.adjudication:
                        audit.adjudicate(con, audit_id, _uuid(row["canonical_id"]), community, args.validator, label, row.get("note", ""))
                    else:
                        audit.rate(con, audit_id, _uuid(row["canonical_id"]), community, args.validator, label, row.get("note", ""))
                    n += 1
        return {"event": "audit_imported", "rows": n, "validator": args.validator, "kind": "adjudication" if args.adjudication else "rating"}
    if c == "audit-report":
        audit_id = _uuid(args.audit_id)
        agreement = audit.agreement(con, audit_id, args.expected_kappa)
        estimate = audit.estimates(con, audit_id)
        args.output.mkdir(parents=True, exist_ok=True)
        path = audit.write_report(estimate, agreement, args.output / "audit_report.md")
        return {"event": "audit_report", "report": str(path), "disagreements": agreement["disagreements"]}
    if c == "ingest":
        from .fulltext import S3Store
        from .fulltext import parse_years, papers_in_years
        years = parse_years(args.years)
        ingestion_id, counts = documents.ingest(con, _uuid(args.run_id), S3Store(), args.limit, papers_in_years(con, _uuid(args.run_id), years) if years else None)
        return {"event": "documents_ingested", "ingestion_run_id": str(ingestion_id), "counts": counts}
    if c == "embed-chunks":
        # embedding_jobs.embed_chunks, not documents.embed_chunks: it clips a text longer than the model accepts and commits after every batch, so one oversized chunk (the PDFs of
        # 18 Oct 2026 had some) no longer fails the API call and throws away the whole run. It is also the function `embed-corpus` uses for the chunks.
        from . import embedding_jobs
        done = embedding_jobs.embed_chunks(con, _uuid(args.ingestion_run_id), get_embedder(args.model), lambda event: print(json.dumps(event), flush=True))
        return {"event": "chunks_embedded", "new": done["embedded"], "tokens": done["tokens"]}
    if c == "extract-prepare":
        manifest = extraction.prepare(con, _uuid(args.ingestion_run_id), args.output)
        return {"event": "extraction_prepared", "directory": str(args.output), "requests": manifest["request_count"],
                "next": f"research0 community-submit --run {args.output}   (this sends text to OpenAI and costs money)"}
    if c == "extract-collect":
        staged = not args.direct          # by default a new run waits in staging for a person's promotion (the user's rule, 13 Oct 2026)
        result = extraction.collect_remote(con, args.run_dir, staged) if args.remote else extraction.collect_file(con, args.run_dir, staged=staged)
        return {"event": "extraction_collected", **result, **({"next": f"research0 extract-promote --extraction-run-id {result['extraction_run_id']} --by <name> [--execute]"} if staged else {})}
    if c == "extract-promote":
        if not args.extraction_run_id:
            return {"event": "staged_extraction_runs", "runs": extraction.staged_runs(con)}
        out = (extraction.promote_delta if args.delta else extraction.promote)(con, _uuid(args.extraction_run_id), args.by, args.note, args.execute)
        return {"event": "extraction_promoted" if out["executed"] else "extraction_promotion_preview", **out}
    if c == "signals-build":
        signal_run_id, counts, existed = signals.build(con, _uuid(args.extraction_run_id), get_embedder(args.model), {"require": args.require, **({"corpus_tag": args.corpus_tag} if args.corpus_tag else {})})
        return {"event": "signals_built", "signal_run_id": str(signal_run_id), "already_built": existed, **counts}
    if c == "dossiers-build":
        ids = None
        if args.pair_sample_id:
            ids = [r[0] for r in con.execute("SELECT pair_id FROM pair_sample_item WHERE pair_sample_id=%s", (_uuid(args.pair_sample_id),))]
            if not ids:
                raise ValueError("Unknown pair sample, or it has no pairs")
        from .proposals import TARGETED_HINTS
        return {"event": "dossiers_built", "created": signals.build_dossiers(con, _uuid(args.signal_run_id), pair_ids=ids, hints=TARGETED_HINTS if args.targeted else args.hint, limit=args.limit)}
    if c == "pairs-export-blind":
        return {"event": "pairs_exported", "pairs": pairs.export_blind(con, _uuid(args.signal_run_id), args.output, args.hint, args.limit), "output": str(args.output)}
    if c == "pair-decide":
        pairs.decide(con, _uuid(args.pair_id), args.validator, args.decision_class, args.rationale,
                     {"same_referent": args.same_referent, "compatible_properties": args.compatible_properties, "substitution_changes": args.substitution_changes},
                     args.consequence, args.counterexample, args.final)
        return {"event": "pair_decided", "pair_id": args.pair_id}
    if c == "pairs-report":
        out = {"event": "pairs_report", **pairs.counts(con, _uuid(args.signal_run_id))}
        if args.output:
            out["settled_exported"] = pairs.export_settled(con, _uuid(args.signal_run_id), args.output)
        return out
    if c == "gold-import":
        gold_id = quality.store_gold(con, args.kind, args.label, json.loads(args.file.read_text(encoding="utf-8")), args.provided_by)
        return {"event": "gold_imported", "gold_id": str(gold_id), "kind": args.kind}
    if c == "release-build":
        release_id, created, passed, gates, _ = release.build(
            con, _uuid(args.run_id), args.label, _uuid(args.preassignment_run), _uuid(args.audit_id), _uuid(args.ingestion_run_id), _uuid(args.extraction_run_id),
            _uuid(args.signal_run_id), _uuid(args.gold_extraction), _uuid(args.gold_pairs), _uuid(args.gold_ratings),
            __import__("research0.fulltext", fromlist=["parse_years"]).parse_years(args.operational_window))
        out = {"event": "release_built", "release_id": str(release_id), "created": created, "status": release.status(con, release_id),
               "gates": [{k: g[k] for k in ("id", "value", "threshold", "status")} for g in gates]}
        if args.output:
            out["report"] = str(release.write_report(con, release_id, args.output))
        return out
    if c == "release-signoff":
        release.sign_off(con, _uuid(args.release_id), args.validator, args.note)
        return {"event": "release_signed", "release_id": args.release_id, "status": release.status(con, _uuid(args.release_id))}
    if c == "source-register-import":
        from . import source_register
        return {"event": "source_register_imported", **source_register.import_register(con, args.csv)}
    if c == "source-register-check":
        from . import source_register
        return {"event": "source_register_checked", **source_register.check(con, args.version, args.query.read_text(encoding="utf-8"))}
    if c == "confirm-prepare":
        from . import confirmation
        from .fulltext import parse_years, papers_in_years
        years = parse_years(args.years)
        manifest = confirmation.prepare(con, _uuid(args.run_id), _uuid(args.ingestion_run_id), args.output, args.scope, papers_in_years(con, _uuid(args.run_id), years) if years else None)
        return {"event": "confirmation_prepared", "directory": str(args.output), "papers": manifest["request_count"],
                "next": f"research0 community-submit --run {args.output}   (this sends text to OpenAI and costs money)"}
    if c == "confirm-collect":
        from . import confirmation
        return {"event": "confirmation_collected", **confirmation.collect_file(con, args.run_dir, args.scope)}
    if c == "extraction-sample":
        ids, design = audit.extraction_validation_sample(con, _uuid(args.run_id), _uuid(args.preassignment_run), args.seed)
        return {"event": "extraction_validation_sample", "papers": [str(i) for i in ids], "design": design}
    if c == "proposal-prepare":
        from . import proposals
        manifest = proposals.prepare(con, _uuid(args.signal_run_id), args.output, proposals.TARGETED_HINTS if args.targeted else args.hint, args.limit)
        return {"event": "proposals_prepared", "directory": str(args.output), "pairs": manifest["request_count"], "estimated_tokens": manifest["estimated_tokens"],
                "next": f"research0 community-submit --run {args.output}   (this sends text to OpenAI and costs money)"}
    if c == "proposal-export":
        from . import pairs, proposal_export
        return {"event": "proposals_exported", **proposal_export.export_xlsx(con, _uuid(args.signal_run_id), args.output, tuple(args.classes or pairs.POSITIVE))}
    if c == "proposal-collect":
        from . import proposals
        return {"event": "proposals_collected", **(proposals.collect_remote(con, args.run_dir) if args.remote else proposals.collect_file(con, args.run_dir))}
    if c == "pair-sample":
        from . import pairsample
        if args.design == "review":
            sample_id, design, existed = pairsample.create_review_set(con, _uuid(args.signal_run_id))
        elif args.design == "cascade":
            sample_id, design, existed = pairsample.create_cascade(con, _uuid(args.signal_run_id), args.hint, n_audit=args.audit, n_promoted=args.promoted, seed=args.seed)
        else:
            sample_id, design, existed = pairsample.create(con, _uuid(args.signal_run_id), args.probabilistic, args.challenge, args.seed, allocation=args.allocation)
        return {"event": "pair_sample", "pair_sample_id": str(sample_id), "already_existed": existed, "design": design}
    if c == "pair-sample-report":
        from . import pairsample
        return {"event": "pair_sample_report", **pairsample.report(con, _uuid(args.pair_sample_id))}
    if c == "rag-index":
        from . import rag
        embedder = get_embedder(args.model)
        embedder.register(con)
        return {"event": "vector_index", "index": rag.create_vector_index(con, embedder.model_id, args.kind)}
    if c == "embed-corpus":
        from . import embedding_jobs
        if args.dry_run:
            return {"event": "embedding_dry_run", **embedding_jobs.dry_run(con, _uuid(args.ingestion_run_id), _uuid(args.extraction_run_id), args.model)}
        if args.model.startswith("openai:") and not args.acknowledge_send_to_openai:
            raise ValueError("an openai model sends the texts to OpenAI and costs money: add --acknowledge-send-to-openai (use --dry-run first to see what would be sent)")
        embedder = get_embedder(args.model)
        log = lambda event: print(json.dumps(event), flush=True)
        out = {"event": "corpus_embedded", "model": embedder.model_id, "dimensions": embedder.dimensions}
        if args.only in ("chunks", "both"):
            out["chunks"] = embedding_jobs.embed_chunks(con, _uuid(args.ingestion_run_id), embedder, log)
        if args.only in ("records", "both"):
            out["records"] = embedding_jobs.embed_records(con, _uuid(args.extraction_run_id), embedder, log=log)
        return out
    if c == "forum-score":
        from . import forum_scoring
        from .forum_providers import get_provider
        if args.proposal_id:
            ids = [_uuid(args.proposal_id)]
        elif args.hybrid_missing:
            ids = forum_scoring.pending_hybrid(con, _uuid(args.run_id))
        else:
            ids = forum_scoring.pending(con, _uuid(args.run_id))
        if not ids:
            return {"event": "forum_proposals_scored", "stored": 0, "note": "no proposal of the current format waits for a score" if not args.hybrid_missing else "every scored proposal has its hybrid novelty vote"}
        if not args.acknowledge_send_to_openai:
            raise ValueError("the abstract and the cards of the proposals are sent to OpenAI and this costs money: add --acknowledge-send-to-openai")
        if not os.environ.get("OPENAI_API_KEY"):
            raise ValueError("OPENAI_API_KEY is not set")
        return {"event": "forum_proposals_scored", **forum_scoring.score_proposals(con, ids, get_provider("openai"))}
    if c == "forum-score-corpus":
        from . import forum_scoring, scoring_corpora
        if args.model.startswith("openai:") and not args.acknowledge_send_to_openai:
            raise ValueError("an openai model sends the texts to OpenAI and costs money: add --acknowledge-send-to-openai")
        if args.kind == "novelty":
            scopus = args.scopus or sorted(Path("inputs/scopus").glob("*.csv"))
            assignments = args.assignments or Path("reports/community_dashboard") / args.run_id / "assignments.csv"
            corpus = scoring_corpora.build_abstract_corpus(scopus, forum_scoring.CORPORA, forum_scoring.GRADES, get_embedder(args.model),
                                                           log=lambda m: print(m, file=sys.stderr, flush=True), assignments=(args.catalog, assignments))
            return {"event": "forum_score_corpus_built", "kind": "novelty", "directory": str(corpus.dir), **{k: corpus.manifest[k] for k in ("corpus_id", "entries", "embedder", "journals")}}
        corpus = forum_scoring.build_card_corpus(con, _uuid(args.run_id), args.kind, get_embedder(args.model))
        return {"event": "forum_score_corpus_built", "kind": args.kind, "directory": str(corpus.dir), **{k: corpus.manifest[k] for k in ("corpus_id", "entries", "without_card", "embedder", "journals")}}
    if c == "forum-run":
        from . import forum
        config = json.loads(args.config.read_text(encoding="utf-8"))
        if config.get("provider", "openai") == "openai":      # the only provider since the simulation was removed (13 Oct 2026)
            if not args.acknowledge_send_to_openai:
                raise ValueError("provider openai sends the prompts, with passages of the corpus, to OpenAI and costs money: add --acknowledge-send-to-openai")
            if not os.environ.get("OPENAI_API_KEY"):
                raise ValueError("OPENAI_API_KEY is not set")
        episode_id = forum.create_episode(con, _uuid(args.release_id), config, args.by)
        con.commit()
        if args.create_only:
            return {"event": "forum_episode_created", "episode_id": str(episode_id)}
        status = forum.Runner(episode_id).run()
        return {"event": "forum_episode_finished", "episode_id": str(episode_id), "status": status, "usage": forum.usage(con, episode_id)}
    if c == "forum-batch":
        from . import forum
        config = json.loads(args.config.read_text(encoding="utf-8"))
        if config.get("provider", "openai") == "openai":
            if not args.acknowledge_send_to_openai:
                raise ValueError("provider openai sends the prompts, with passages of the corpus, to OpenAI and costs money: add --acknowledge-send-to-openai")
            if not os.environ.get("OPENAI_API_KEY"):
                raise ValueError("OPENAI_API_KEY is not set")
        batch_id, ids = forum.create_batch(con, _uuid(args.release_id), config, args.by, args.repetitions, args.seed_mode)
        if args.create_only:
            return {"event": "forum_batch_created", "batch_id": str(batch_id), "episode_ids": [str(i) for i in ids]}
        done = []
        for episode_id in ids:                  # one after the other, like the page; a failed episode does not stop the others here (Ctrl+C does)
            status = forum.Runner(episode_id).run()
            done.append({"episode_id": str(episode_id), "status": status})
            print(json.dumps({"event": "forum_batch_progress", "done": len(done), "total": len(ids), **done[-1]}), flush=True)
        return {"event": "forum_batch_finished", "batch_id": str(batch_id), "results": done}
    raise ValueError(f"Unknown command {c}")

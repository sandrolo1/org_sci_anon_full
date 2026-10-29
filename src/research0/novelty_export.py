"""Export a frozen catalog and optional forum questions for offline novelty retrieval.

Requires DATABASE_URL. Queries use a repeatable-read, read-only transaction.
The resulting JSON may contain licensed abstracts; share it securely.
"""

import argparse
import datetime as dt
import json
import os
from pathlib import Path


def export_snapshot(con, run_id, cutoff_year, episode_id=None):
    """Collect a consistent, read-only export from the existing schema."""
    con.execute("SET TRANSACTION ISOLATION LEVEL REPEATABLE READ READ ONLY")
    run = con.execute("SELECT run_id,query_id,batch_ids,created_at FROM catalog_run WHERE run_id=%s", (run_id,)).fetchone()
    if not run:
        raise ValueError(f"catalog run not found: {run_id}")
    rows = con.execute(
        "SELECT c.canonical_id,c.doi,c.eid,c.title,c.authors,c.publication_year,"
        "c.source_title,c.status,c.perimeter_flags,r.raw_fields->>'Abstract' "
        "FROM canonical_paper c JOIN export_record r ON r.record_id=c.representative_record_id "
        "WHERE c.run_id=%s ORDER BY c.canonical_id", (run_id,)).fetchall()
    papers = [{"id": str(pid), "doi": doi, "eid": eid, "title": title,
               "authors": authors, "year": year, "source": source,
               "status": status, "perimeter_flags": flags,
               "eligible": status == "retained" and not flags, "abstract": abstract}
              for pid, doi, eid, title, authors, year, source, status, flags, abstract in rows]
    catalog = {"catalog_run_id": str(run[0]), "query_id": str(run[1]),
               "batch_ids": [str(value) for value in run[2]],
               "catalog_created_at": run[3].isoformat(),
               "exported_at": dt.datetime.now(dt.timezone.utc).isoformat(),
               "literature_cutoff_year": cutoff_year, "papers": papers}
    questions = None
    if episode_id is not None:
        if not con.execute("SELECT 1 FROM forum_episode WHERE episode_id=%s", (episode_id,)).fetchone():
            raise ValueError(f"forum episode not found: {episode_id}")
        question_rows = con.execute(
            "SELECT question_id,content FROM question WHERE episode_id=%s ORDER BY question_id",
            (episode_id,)).fetchall()
        questions = [{"id": str(question_id), "content": content}
                     for question_id, content in question_rows]
    return catalog, questions


def export_scored_proposals(con, episode_id):
    """Export proposal abstracts and their latest real score document for an offline audit.

    Call this in the same read-only transaction as ``export_snapshot``. Older
    proposals have no abstract or three-metric score and remain visible as
    missing data rather than being silently excluded.
    """
    rows = con.execute(
        "SELECT p.proposal_id,p.content,s.document FROM forum_proposal p "
        "LEFT JOIN LATERAL (SELECT document FROM forum_proposal_score "
        "WHERE proposal_id=p.proposal_id AND synthetic=false "
        "ORDER BY created_at DESC,score_id DESC LIMIT 1) s ON TRUE "
        "WHERE p.episode_id=%s AND p.status='ok' ORDER BY p.proposal_id",
        (episode_id,)).fetchall()
    return [{"id": str(pid), "title": content.get("title"),
             "abstract": content.get("abstract"), "score": score}
            for pid, content, score in rows]


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--literature-cutoff-year", type=int, required=True)
    parser.add_argument("--catalog-output", type=Path, required=True)
    parser.add_argument("--episode-id", help="Optional single forum episode to export")
    parser.add_argument("--questions-output", type=Path)
    parser.add_argument("--scored-proposals-output", type=Path,
                        help="Optional latest real three-metric score documents for the episode")
    args = parser.parse_args(argv)
    if (args.episode_id is None) != (args.questions_output is None):
        parser.error("--episode-id and --questions-output must be used together")
    if args.scored_proposals_output and not args.episode_id:
        parser.error("--scored-proposals-output requires --episode-id")
    if args.literature_cutoff_year < 1900:
        parser.error("literature cutoff year must be at least 1900")
    if not os.environ.get("DATABASE_URL"):
        parser.error("DATABASE_URL is required")
    import psycopg
    with psycopg.connect(os.environ["DATABASE_URL"]) as con:
        catalog, questions = export_snapshot(con, args.run_id, args.literature_cutoff_year, args.episode_id)
        scored_proposals = export_scored_proposals(con, args.episode_id) if args.scored_proposals_output else None
    args.catalog_output.parent.mkdir(parents=True, exist_ok=True)
    args.catalog_output.write_text(json.dumps(catalog, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    if questions is not None:
        args.questions_output.parent.mkdir(parents=True, exist_ok=True)
        args.questions_output.write_text(json.dumps(questions, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    if scored_proposals is not None:
        args.scored_proposals_output.parent.mkdir(parents=True, exist_ok=True)
        args.scored_proposals_output.write_text(json.dumps(scored_proposals, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"Exported {len(catalog['papers'])} catalog papers and {len(questions or [])} questions")


if __name__ == "__main__":
    main()

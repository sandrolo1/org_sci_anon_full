# Evaluation atlas

## Reviewer edition without private data

For an anonymous, reproducible Stage 1 walkthrough, run `npm ci` and then
`npm run reviewer:dev` from this directory. The separate reviewer edition
uses four invented papers, computes candidate conceptual relations and exact
quote checks, and displays a verified aggregate of 30 real forum episodes.
It does not use the private exports, database, AWS, or an API key. Run
`npm run reviewer:check` for its evidence gates and `npm run reviewer:build`
for a distributable static build. The Stage 1 documents are in [stage1/](../stage1/).

**Not runnable from this package.** The rest of this file describes the atlas, which needs private exports that are not included here. It is a **private, read-only workbench** for the evaluator's
exports and an offline retrieval control. The proposal view displays 38 proposals, their three model scores,
cited paper comparisons, hybrid novelty checks, and integrity flags. The
2026 paper benchmark view displays the latest available paper evaluation export, its
score differences by journal grade, and paper-level date checks. It uses
Helvetica and sans-serif fallbacks. The existing Python forum remains the
production application; this view has no live database link.

The **Retrieval checks** view reads the aggregate-only
`reports/evaluator_export_2026-10-20/known_answer_retrieval.json` report. The
offline check samples articles already in the 2015–2025 novelty index and asks
whether BM25 recovers each article from text extracted from that article.
Its severe word-deletion controls stress lexical dependence. They do **not**
test paraphrases, source coverage outside the index, or novelty judgment
accuracy. A separate frozen-vector self-check, when present, tests only whether
each stored paper vector points back to its own indexed paper; it is not a
query-embedding or semantic rescue test. The report is generated separately with
`PYTHONPATH=src python3 -m research0.novelty_known_answer --export /path/to/private.zip --output reports/evaluator_export_2026-10-20/known_answer_retrieval.json`
from the repository root. `npm run prepare:data`
copies only allowlisted aggregate counts and checksums to ignored
`web/public/data/retrieval.json`; it never copies paper text, IDs, or raw
queries. If the local source ZIP is available, its hash is checked against
the report. If the report is absent, the view explains how to prepare it.

## Run locally

From `web/`:

```sh
npm ci
npm run dev
```

Open the localhost address printed by Vite. `npm run dev` regenerates the
private data file from
`inputs/evaluator_export_2026-10-20/raw/proposals_evaluations_export_2026-10-20.zip`
and runs the quote and source integrity audit first. It needs the copied ZIP
and the versioned paper cards in `data/paper_cards_2026-10-19/`. No AWS account,
database password, or model API key is needed. To use another export, run
`python3 scripts/prepare_data.py --export /absolute/path/to/export.zip` before
`npx vite --host 127.0.0.1`.

The 2026 benchmark view chooses the latest matched private CSV/JSON export in
`inputs/evaluation_2026/exports/`; the 80-paper and 125-paper exports were
copied from the evaluator's attachments and are excluded from Git. `npm run prepare:data`
checks that their paper IDs, grades and scores agree, then writes
`public/data/benchmark.json`. The view records the source filenames, export
timestamp, and SHA-256 hashes so a reviewer can identify exactly which frozen
files are on screen. Its review queue isolates missing scores and dates recorded
after the export, and the criterion table shows the actual number of available
scores in each group. If the matching private archive ZIP is present in
`inputs/evaluation_2026/`, the view also reports manuscript versions and PDF
origins. If the exports are unavailable, the proposal view still
works and the benchmark view explains how to prepare it. To choose different
files, run `python3 scripts/prepare_benchmark.py --csv /path/to/file.csv --json /path/to/file.json`.
The benchmark attachment does not include the full per-paper judge rationales,
so the paper register shows score records, not claimed evidence passages.

The generated `public/data/evaluations.json`, `public/data/benchmark.json`,
`public/data/retrieval.json`,
`dist/`, and `node_modules/` are
ignored by Git. The JSON and built app contain unpublished proposal text and
paper excerpts: keep them private. The app labels model scores as exploratory.
Its integrity checks establish source traceability and retrieval sensitivity;
they do not validate novelty, methodology, or impact judgments. The source
corpus covers a 2015–2025 CTO/TIM subset and does not guarantee full prior-art
coverage. The 2026 journal-tier comparison is analyzed in the
benchmark assessment (not included in this package).

## Review workflow

The page starts with six follow-up cases: three hybrid judgments failed their
quote check, and three completed hybrid judgments changed the novelty vote by
at least two points. Use the four queue cards to switch between follow-up,
failed, shifted, and all proposals; filter by condition or search titles and
abstracts. Each proposal has a novelty comparison showing primary and hybrid
reasoning beside their respective paper sets. The methodology view marks
which core plan fields were inferred by an agent.

**Download follow-up CSV** saves the six private cases with exact stored
failure reasons, score differences, and proposed next actions. It is generated
locally at `public/data/followup_queue.csv` and excluded from Git. Share it
only through an approved private channel. The aggregate actions and caveats
are in [the evaluator's handoff note](../docs/evaluation/EVALUATOR_HANDOFF.md).

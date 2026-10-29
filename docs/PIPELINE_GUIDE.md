# Research Intelligence: Scopus Coverage and Community Assignment

This project records a human retrieval benchmark for two independent Academy of Management communities: Communication, Digital Technology, and Organization (CTO) and Technology and Innovation Management (TIM). A paper may belong to both communities. Phase 0 measures whether validated benchmark papers occur in the Scopus exports. Phase 1 merges the exports of one query into a canonical catalog with a logged deduplication. The phase numbers in this file follow the internal project plan (not part of this package) (the original plan); the sections below named "Phase 2: human review" and "Model-assisted CTO and TIM decisions (prototype)" keep their historical titles. Phases 2-8 of the plan are implemented and tested on synthetic data (see "Phases 2-8" below); none of them has been approved or run on real data except the local clustering.

## Local setup

Requires Python 3.11+, Docker, and Docker Compose. In PowerShell:

```powershell
Copy-Item .env.example .env
docker compose up -d
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -e ".[test]"
research0 init
pytest -q
```

`research0` reads simple `KEY=VALUE` entries from `.env` in the current directory at startup. Environment variables already set in the terminal take precedence. Run commands from the project directory so the intended `.env` is used.

Docker Compose uses SeaweedFS for local S3-compatible storage and demonstration credentials. Shared environments should use managed secrets and TLS.

## Phase 0: human benchmark and export coverage

Fill the empty templates in `templates/` and place completed copies under `inputs/`. Source IDs must be verified Scopus IDs. CTO and TIM benchmark files remain independent; the same paper may appear in both. Use a new version for each correction rather than changing previously validated records. Seed statuses are `eligible`, `out_of_scope`, `duplicate`, and `to_verify`.

```powershell
research0 sources-import inputs/sources/source_list.csv
research0 benchmark-import inputs/benchmarks/benchmark_cto.csv
research0 benchmark-import inputs/benchmarks/benchmark_tim.csv
research0 query sources-2026-10-08-v1
```

Run the generated query in Scopus and preserve original `.csv` or `.xlsx` exports under `inputs/scopus/`. The query uses active source IDs, publication years 2015–2025, article/review document types, and journal source type. Inspect and import an export with its actual provenance:

```powershell
research0 scopus-inspect inputs/scopus/export.csv
research0 scopus-import inputs/scopus/export.csv --query-id <query_id> --export-date 2026-10-08 --provenance "Manual Scopus export"
research0 report --batch-id <batch_id> --benchmark-version benchmark-v1 --output reports/coverage
```

The import stores a checksum and an original copy in S3-compatible storage. The coverage report matches by normalized DOI, then EID, then exact normalized title plus year. Conflicting identifiers and multiple candidates are marked ambiguous. Only human-validated `eligible` seeds count toward coverage. With zero eligible seeds, coverage is `N/A`.

To measure retrieval across multiple original exports from the same query without importing them into the database:

```powershell
research0 report-files --benchmark-cto inputs/benchmarks/benchmark_cto.csv --benchmark-tim inputs/benchmarks/benchmark_tim.csv --exports inputs/scopus/export_a.csv inputs/scopus/export_b.csv --output reports/coverage
```

This report preserves file provenance and checksums. Rows with the same identifier across exports count as one recovered seed but remain distinct source rows. If the source list has not been validated, the report measures retrieval in the supplied files; it does not certify the completeness of the Scopus query. For synthetic fixtures, run `python examples/make_synthetic.py`. These examples are not scientific evidence.

### Source register

The proposal (Appendix A) asks for more than the seven columns of `source_list`: for each retained journal, its exact title, ISSN and eISSN, Scopus Source ID, AJG 2024 grade or UTD24 status, the DOI or EID of at least one verified CTO/TIM paper, the community, and the verification date. That lives in the table `source_register` (migration 016), beside `source_list`, which is unchanged. A journal qualifies when it is AJG 4/4* or in the UTD24 list **and** has a verified paper; a journal from the AIS Senior Scholars list can be added only as a separately flagged supplement (`ais_supplement`), and a title below AJG 4 and outside UTD24 is never reported as meeting the criterion.

```powershell
research0 source-register-template        # templates/source_register_from_query_v1.csv: the 34 SRCIDs of the query, every other field empty
# a person fills it after checking each source in Scopus, the AJG guide and the UTD24 list; nothing is inferred
research0 source-register-import templates/source_register_from_query_v1.csv   # all or nothing, idempotent; a changed row needs a new version
research0 source-register-check --version <version>   # SRCIDs of the query without a row, rows not in the query, sources that are not eligible; `ready` only if none
```

The register is append-only per version, and `research0 migrate` copies it with the rest.

## Phase 1: canonical catalog and deduplication

All export batches registered under one query form one logical catalog. Export records are never changed or deleted; each one is linked to a canonical paper and every decision is written to `dedup_log`. Policy (`dedup-v1`):

1. Same EID: linked. A DOI disagreement is linked but flagged.
2. Same DOI: linked only if at most one side has an EID. The same DOI on two different EIDs stays separate and is flagged.
3. Fallback on normalized title, year, authors and source, only for titles of at least four words, never across two different DOIs and never for records already flagged.

Flagged papers have status `review_needed`; nothing is merged by an embedding, a model, or a score. Records outside 2015–2025 or outside article/review get a `perimeter_flags` entry and stay in the catalog.

```powershell
research0 init                                   # applies migration 003
research0 query-register --label scopus-2026-10-08 [--text-file query.txt]
research0 scopus-import inputs/scopus/<file>.csv --query-id <query_id> --export-date 2026-10-08 --provenance "Manual Scopus export"   # once per export
research0 catalog-build --query-id <query_id>
research0 catalog-report --run-id <run_id> --output reports/catalog --benchmark-cto inputs/benchmarks/confirmed_2026-10-09/benchmark_cto.csv --benchmark-tim inputs/benchmarks/confirmed_2026-10-09/benchmark_tim.csv
```

Without `--text-file`, `query-register` stores the text as "UNRECORDED" rather than inventing it. `catalog-build` is idempotent: the run ID derives from the query, the batches and the dedup version, so rerunning returns the existing run. A new rule set needs a new `DEDUP_VERSION`, which creates a new run and keeps the old one. Import is slow (about 5 minutes for 17,000 rows) because it inserts row by row.

`catalog-report` writes `catalog_report.md` (PRISMA-style flow, counts, seed check), `catalog.csv`, `dedup_log.csv`, `counts_by_journal.csv`, `counts_by_year.csv`, `counts_by_batch.csv` and `seed_check.csv`. Seeds are matched against the catalog; a title/year ambiguity is resolved only when all candidates belong to one retained canonical paper. For missing seeds the report shows the closest catalog title as a review aid, not as a match.

Result on the two real exports (reported run `8c05a3dc-3dc2-55e0-a8f0-0c2f423b927e`): 26,862 records, 665 linked by EID across the two files, 26,197 canonical papers, 52 papers flagged for review (24 same title/authors/year/source with different DOIs, mostly SSRN versus journal DOIs; 2 same DOI on two EIDs), 0 perimeter flags. Seeds recovered: CTO 49/50, TIM 48/50; three eligible seeds (CTO-011, TIM-005, TIM-006) are not in the exports.

Manual checklist:

- [ ] Open `catalog_report.md`: input records equal the sum of the batch counts, and canonical papers equal records minus links.
- [ ] Review `dedup_log.csv` rows with `flagged_for_review`; decide each pair outside the pipeline.
- [ ] Spot-check 10 `merged` rows by EID in Scopus.
- [ ] Check the three missing seeds in Scopus: wrong year, journal outside the source list, or not indexed.
- [ ] Rerun `catalog-build` and confirm `already_built` is true.

Limits: the export has no Source ID or Source type column, so SRCID and SRCTYPE cannot be verified from the file, and the source list is still unvalidated. Journal counts use the Source title string. The two exports are assumed to come from one query because the query text was not recorded.

## Validation dashboard

`dashboard-build` turns one catalog run into a single static page for human validators: key figures, seed retrieval, flow, batches, years, journals, the Scopus query, and the items that need a decision (unrecovered seeds, duplicate conflicts, unvalidated source list, journal-name variants, papers without DOI). Each item has a decision list and a note. Decisions stay in the validator's browser and are downloaded as CSV; they are not yet imported into the database.

The dashboard now includes a searchable human review queue, open/decided/discussion/conflict filters, progress statistics, and the phase gates. Discussion and conflicting decisions do not count as completed work. Enter a validator name before recording a decision. In live mode decisions remain append-only in PostgreSQL; in the static page they remain in the browser until exported.

For a complete interface preview using the real catalog counts and pilot titles, but **simulated** review decisions and CTO/TIM suggestions:

```powershell
research0 dashboard-demo --snapshot reports/dashboard_2026-10-09/snapshot.json --pilot-run reports/community_pilot_2026-10-09 --output reports/dashboard_demo
```

Open `reports/dashboard_demo/index.html`. Its yellow banner and pilot warning identify all simulated values. The demo uses separate browser storage and never writes to PostgreSQL. The CTO/TIM pilot tab lets a human independently review both communities, add a note, and download a review CSV. When `community-collect` has produced real `assignments.csv`, pass `--assignments reports/community-pilot/assignments.csv` to preview the actual model suggestions instead. The exported human review still needs a project decision before any full-corpus model run.

## Phase 2: human review

Generate a review queue and a source-ID worksheet from the Phase 1 dashboard snapshot, without changing the catalog or PostgreSQL:

```powershell
research0 phase2-report --snapshot reports/dashboard_2026-10-09/snapshot.json --output reports/phase2_2026-10-09
```

The current packet has 32 open items: 3 unrecovered seeds, 26 deduplication conflicts, 1 unvalidated source list, 1 journal-name variant and 1 group of 14 papers without DOI. The worksheet has 34 SRCIDs from the recorded query. Fill its journal name, rationale, active flag, version, date and validator only after checking each source in Scopus. `review_queue.csv` includes the evidence and allowed decisions for each item.

To incorporate decisions downloaded from the static dashboard, add `--decisions path/to/decisions.csv`. The command checks the catalog run and item IDs, takes each validator's latest decision, and flags disagreements or discussion. It does not apply identity merges, change benchmark eligibility or mark the source list validated. In live dashboard mode, decisions are already stored in PostgreSQL; rebuild the snapshot first to include them. Phase 3 needs human sign-off after this review.

```powershell
research0 dashboard-build --run-id <run_id> --output reports/dashboard
```

The page embeds only the statistics and the records involved in open questions, not the whole catalog. It is generated from PostgreSQL, so rebuild it after every catalog change.

### Live mode, connected to the database

```powershell
research0 init                                  # applies migration 004 (validation_decision)
research0 dashboard-serve --run-id <run_id> --port 8765
```

The live dashboard also has a **Community assignment** tab. Run `research0 init` first to apply migration 005, set `OPENAI_API_KEY`, then press **Start assignment** for the selected catalog run. This prepares the eligible canonical papers and submits them to the OpenAI Batch API. Press **Refresh assignment status** to collect responses. The page counts CTO only, TIM only, both, neither, unassigned eligible papers, and papers excluded because of unresolved identity or perimeter flags. A submission is stored under `reports/community_dashboard/<run_id>/`; it is never submitted a second time automatically.

The live page polls Batch status every 15 seconds while it is active. The progress bar counts completed plus failed Batch requests out of the reported total; each request contains up to 10 papers. During input validation, when OpenAI has not reported request totals yet, the bar is indeterminate. Papers become "assigned" only after their validated output is collected. The dashboard server logs submission and each status check with request and paper counts to its terminal.

If a Batch expires or completes with missing responses, first collect its available output, then prepare a retry from only the requests without validated assignments:

```powershell
research0 community-collect --run reports/community_dashboard/<run_id> --scope config/community_scope_v1.json
research0 community-retry-prepare --run reports/community_dashboard/<run_id>
# Review the reported retry directory and requests.jsonl before explicitly submitting it:
research0 community-submit --run reports/community_dashboard/<run_id>/retry_001
research0 community-collect --run reports/community_dashboard/<run_id>/retry_001 --scope config/community_scope_v1.json
```

`community-retry-prepare` refuses an active Batch, keeps the original assignments, and writes no API request. After a retry, restart the dashboard server so it loads the current code; it then combines validated assignments from all attempts and reports progress for the latest Batch. If every paper is assigned, it reports that no retry is needed. A failed input validation needs manual inspection rather than an automatic retry of the same input.

Set the expected Cohen κ threshold (default 0.80) and press **Randomize 100 papers** to store one random sample in PostgreSQL. The button does not wait for the Batch: the sample is drawn from all eligible papers in the submitted Batch, not only those already labelled, so it stays random. Annotation can start immediately; the comparison with the Batch appears in **Calculate agreement** once every sampled paper has a collected Batch label. Each of two named annotators opens **Load my blind papers** and independently records binary CTO and TIM ratings. The blind view shows paper text without model suggestions or the other annotator's ratings. When both have rated every sampled paper, **Calculate agreement** reports observed agreement and Cohen's kappa separately for CTO and TIM, compared with that threshold. A changed rating is appended to history, with the latest rating used. The sample and threshold are fixed when created. If a Batch request fails, the eligible papers left unassigned remain visible in the statistics; inspect `batch_errors.jsonl` and the Batch run before interpreting the sample.

After both annotators finish, **Calculate agreement** reports Cohen's kappa for annotator A versus B, A versus Batch, and B versus Batch, separately for CTO and TIM. **Resolve disagreements** shows the two human ratings and paper text, but no Batch labels; an adjudicator records a final human CTO/TIM decision for each disputed paper. Once all disagreements are resolved, the agreement report also includes final human decision versus Batch. Equal human ratings become the final decision automatically. Run `research0 init` to apply migration 006 before using adjudication, then restart the dashboard server. The existing Batch does not need to be resubmitted.

Open http://127.0.0.1:8765/. The server reads the catalog from `DATABASE_URL` and saves each decision to `validation_decision` as the validator chooses it (name required). It listens on the loopback interface only, rejects other Host and Origin headers, and never updates a row: a change or withdrawal is a new row, and the latest row per item and validator counts. Statistics are computed once at first load (about 4 seconds on 26,000 papers); decisions are re-read on every request. `--run-id` is required because a run is an immutable version.

To use a database in AWS, create a managed PostgreSQL with pgvector, apply the migrations with `research0 init`, and set `DATABASE_URL` to it with `?sslmode=require`. The catalog must be imported there too (`scopus-import`, `catalog-build`).


`templates/source_list_from_query_v1.csv` lists the 34 SRCIDs of `config/scopus_query_v1.txt`. A human must add the journal name, rationale, version, date and validator; nothing there is inferred.

## Model-assisted CTO and TIM decisions (prototype, planned Phase 3)

The English [community scope configuration](../config/community_scope_v1.json) paraphrases the official [CTO domain statement](https://www.aom.org/divisions-and-interest-groups/communication-digital-technology-and-organization/) and [TIM domain statement](https://www.aom.org/divisions-and-interest-groups/technology-and-innovation-management/). Each paper receives `cto` and `tim`, each exactly `0` or `1`. Both may be `1` or both may be `0`. A short English rationale accompanies each decision. The 50 benchmark papers for each community are retrieval seeds, not a negative training set.

`community-prepare` merges rows with the same EID while preserving source file and row provenance. It quarantines DOI values associated with different EIDs for manual identity review. It sends title, abstract, and keywords to `gpt-6-luna` in groups of 10 papers through structured Responses requests. The Batch API can process those requests asynchronously. The manifest records export and scope checksums, model and prompt versions, and the expected paper IDs for every request. A response is accepted only when every expected ID has exactly one valid pair of binary decisions.

Prepare a reproducible 100-paper pilot locally, without an API call:

```powershell
research0 community-prepare --exports inputs/scopus/export_a.csv inputs/scopus/export_b.csv --scope config/community_scope_v1.json --output reports/community-pilot --limit 100
```

After reviewing the community scope and setting `OPENAI_API_KEY`, submit and collect the pilot:

```powershell
research0 community-submit --run reports/community-pilot
research0 community-collect --run reports/community-pilot --scope config/community_scope_v1.json
```

`community-collect` saves validated decisions in `assignments.csv` and any Batch API errors in `batch_errors.jsonl`. Repeat collection while the asynchronous batch is running. After reviewing the pilot, prepare a new run without `--limit` for the full corpus. The `reports/` directory is excluded from version control.

## Full text (PDF) acquisition

The live dashboard has a **Full text** tab. Run `research0 init` first to apply migration 007 and set `UNPAYWALL_EMAIL` in `.env` (a real address: Unpaywall asks for a contact on every request), then restart `dashboard-serve`.

**Where everything goes.** Every PDF, downloaded or attached by hand, is stored in the same S3 bucket as the Scopus exports (`S3_BUCKET`; locally SeaweedFS), under `fulltext/<canonical_id>/<sha256>.pdf`. PostgreSQL keeps one row per file in `fulltext_file` (URI, SHA-256, size, origin `open_access_download` or `manual_upload`, source URL, who acquired it, validation status `unvalidated`) and one row per attempt in `fulltext_attempt`. Both tables are append-only. Nothing is stored on the server's disk.

**Automatic download.** Choose which papers (the 100-paper blind sample, the papers the Batch labels CTO or TIM once collected, or all eligible papers), optionally a maximum, and press **Download open-access PDFs** (**locked since 26 Oct 2026**: the page is hidden and the server refuses the download until an admin mode exists; use `research0 fulltext-fetch`). The DOI is looked up in Unpaywall, then OpenAlex if no PDF was downloaded, including when Unpaywall is unavailable, then Semantic Scholar (its `openAccessPdf`) if there is still no PDF and `SEMANTIC_SCHOLAR_API_KEY` is set. Only locations reported as open access are fetched. For each provider the resolver tries up to 12 distinct direct links, then up to 12 landing pages; it follows article PDF metadata, repository download links and PDF viewers for at most two HTML hops, with a total budget of 40 distinct document requests per paper shared between the providers. Links to supplements and pages with a different DOI are excluded. Transient HTTP errors and network failures have up to three attempts with backoff and `Retry-After`; refusals such as HTTP 403 are not retried within a request. Every connection, including redirects and links discovered in HTML, must resolve to public internet addresses. Only PDF bytes are stored (header check, 60 MB limit).

The successful copy retains its version (`published`, `accepted` manuscript or `submitted` preprint; empty when unknown), license and resolved download URL. Files already stored are skipped. Old unsuccessful attempts from earlier resolvers (Unpaywall only, Unpaywall+OpenAlex, `oa-pages-v2`) are rechecked once by `oa-pages-v3`; lookup and download failures remain retryable. Semantic Scholar sends its key in the `x-api-key` header, only to its API (never to PDF hosts, not even across a redirect). The key allows 1 request per second across all endpoints, so the downloader sends at most one Semantic Scholar request every 1.1 s per process, whatever the number of workers: papers that reach this third step go at about one per second. Do not run the dashboard download and `fulltext-fetch` at the same time with the same key. A rejected key (HTTP 401/403) or a persistent HTTP 429 is recorded as `lookup_failed` (retried in a later run), and 20 in a row stop the run. `OPENALEX_API_KEY` is optional for basic use; a [free key raises the API budget](https://help.openalex.org/api/authentication/). It is recommended for large runs. CLI equivalent: `research0 fulltext-fetch --run-id <id> --scope sample --limit 20`.

**Measuring the 80–90% target.** At completion, the CLI and dashboard report PDF coverage of the entire selected group, including papers not attempted because of the run limit, and the number of additional PDFs needed to reach 80% and 90%. This is a measured target, not a guaranteed open-access retrieval rate. Institutional access is not automatically inherited from your browser: PDFs obtained through the library can be attached or bulk-imported as below. Restart the dashboard after updating the code so it uses the new resolver.

**Papers to get by hand.** The same tab lists every paper that was tried and has no PDF, with the reason (not open access, no DOI, no direct PDF link, publisher refused, ...), a link to the DOI and, when Unpaywall knows one, the open-access page. Save the PDF from the publisher (with your institutional access), then press **Choose file** and **Attach PDF** on that paper's row; no renaming needed. Enter your name in *Validator name* first. **Download this list (CSV)** (or `research0 fulltext-missing --run-id <id> --output list.csv`) gives the same list with a `suggested_filename` column. For many files, save them into one folder named `<suggested_filename>` and run `research0 fulltext-import --run-id <id> --folder <folder> --uploader "<your name>"`. Importing the same file twice is recognised by its checksum and not stored twice. A collection that comes with an index (columns `record_id` = paper ID, `path`, `format`, `sha256`, `url`, `version`; PDF or Word) is imported with `research0 fulltext-import-collection --run-id <id> --folder <extracted folder> --index <index.csv> --uploader "<your name>" --output <report.json>`; without `--execute` it only reports. A paper that already has a different file, or that the collection supplies with several files, is held back. Word files (.docx) are stored with their own extension and read by `ingest` (no page numbers); a legacy .doc is stored but not read.

**Bot protection (`blocked_by_site`).** Many repositories and publishers (Wiley, Sage, Taylor & Francis, SSRN, most university repositories) answer automated requests with a Cloudflare, AWS WAF or "Human Verification" test instead of the PDF, even for open-access papers. The resolver recognises these walls (header or challenge page on a refusal status), records them as `blocked_by_site`, does not repeat them within a request, and never tries to pass them: they are final, not retried, and appear in the by-hand list with the reason "bot protection". A plain refusal or a network error stays `download_failed` and is retried in the next run. Requires migration 018 (`research0 init`). For figshare, whose pages are behind such a wall, the resolver reads the public figshare API instead; embargoed and confidential records and supplementary files are not fetched.

**Assisted download (passing the bot-protection test yourself).** `research0 fulltext-assist --run-id <id> --folder <folder>` writes `reports/fulltext_assist.html`: every paper without a PDF, walls first, with links to its DOI and open-access page (a local page, no scripts). Start `research0 fulltext-watch --run-id <id> --folder <folder> --uploader "<your name>"` (Ctrl+C to stop; `--once` imports what is there and stops). Open each paper in your browser, signed in through your library if needed, and save the PDF into the folder; the file name does not matter. The importer reads the file and attaches it to a paper only on strong evidence: the paper ID as file name, its DOI in the PDF metadata, the file name or the first page, or its full title (at least six words) on the first two pages. A file that matches nothing, or several papers (a reference list, a review), is left in the folder and reported as `unmatched`; a file that is not a PDF (a saved challenge page) as `rejected`. Imported files are stored like any upload (`manual_upload`, your name) and moved to `<folder>/imported/`. Files modified in the last two seconds are skipped, so a download in progress is not read.

A paper with no PDF stays *unavailable*; it is never treated as irrelevant. Downloading does not parse or validate the text: that is the next step and needs approval.

## Phases 2-8: from the pre-assignment to the validated release

The source for what each step must do is the research proposal (the Stage 1 proposal in [stage1/](../stage1/)); the internal project plan (not part of this package) lists what is aligned and what is still open. Everything below is implemented and tested on a synthetic world (next section). **None of it replaces a human step**: labels are suggestions to audit, signals order a queue, a model's proposal on a pair is hidden until the reviewer has decided, and a release is validated only by gates plus a person's sign-off. The commands that send text to OpenAI and cost money are never run by these commands; they print the explicit `community-submit` line you run yourself.

Run `research0 init` first (migrations 008-015).

**Embeddings.** They are computed on what the proposal says: the textual evidence of constructs (definition, items, measures) for S_sem-op, and the chunks of the full texts for retrieval. The default is OpenAI `text-embedding-3-small` (`EMBEDDING_MODEL=openai:<model>` to change it; text goes to OpenAI and costs money). `hash-rp-v1` is an offline *lexical* stand-in used by the tests, the demo and the optional abstract-level exploration below.

**Optional: exploratory clustering of abstracts** (offline, free, derived). The proposal does not require it: clustering of subtopics "can be introduced after selection and approved by the experts". `research0 embed-papers --run-id <id>` then `research0 cluster --run-id <id> --k 40 --output reports/phase2_clusters`. With the lexical stand-in the clusters follow shared words, not meaning.

**Phase 3: pre-assignment from metadata** (the proposal's `paper_dig`: table `community_preassignment`; prompt `community-trinary-v2`, codebook [config/community_scope_v2.json](../config/community_scope_v2.json))

```powershell
research0 preassign-prepare --run-id <run_id> --output reports/preassign_v2     # writes requests, calls nothing
research0 community-submit --run reports/preassign_v2                              # EXPLICIT: sends text to OpenAI, costs money
research0 preassign-collect --run-dir reports/preassign_v2 --remote
research0 preassign-import-v1 --run-id <run_id> --assignments-dir reports/community_dashboard/<run_id>   # the binary Batch labels, as suggestions only
```

Each community independently gets yes / no / uncertain with a quoted evidence, a rationale, a confidence and the codebook version. A "yes" must quote words that occur in the title, abstract or keywords; an absent or too-short abstract (under 40 words) is "uncertain" and is not sent. **The Batch already running uses prompt v1 (binary 0/1, no evidence, no uncertain): it is a different, reduced thing.** Its labels import as `source = v1_suggestion`. The v2 codebook has the community texts of v1 and a new decision rule only; the proposal also wants inclusions, exclusions, positive examples and confusable cases written by the experts, and a calibration set of 50-100 papers before the pipeline is frozen. The 100-paper blind sample can serve as that calibration set: the audit never draws from it.

**Phase 4: stratified audit.** Strata are the predicted membership (the pair of labels) and a confidence band; every non-empty stratum gets a minimum and the strata excluded from both communities a larger one, so the false-negative rate can be estimated.

```powershell
research0 audit-sample --run-id <run_id> --preassignment-run <id> --size 400 --min-per-stratum 15 --min-excluded 40
# annotate in the dashboard tab "Audit (Phase 4)" (blind: text only), or offline:
research0 audit-export --audit-id <id> --output audit_ana.csv ; research0 audit-import --audit-id <id> --csv audit_ana.csv --validator Ana
research0 audit-import --audit-id <id> --csv decisions.csv --validator Chi --adjudication
research0 audit-report --audit-id <id> --output reports/audit
research0 fulltext-fetch --run-id <run_id> --scope candidates --years 2023-2025 --limit 20   # full text only for candidates and uncertain cases, inside the operational window
```

Two annotators rate blind, agreement is computed before discussion, and a third person settles the disagreements (adjudication does not change the agreement). The report gives kappa and alpha, design-based estimates with intervals (precision, recall, macro-F1, screening recall including the excluded papers, the multi-label set metrics, calibration error, estimated missed members among the "no" papers), the confusion matrices with numerators and denominators, and breakdowns by period and source.

**Phase 5: documents, full-text confirmation and extraction**

```powershell
research0 ingest --run-id <run_id> --years 2023-2025   # parse every stored PDF into text, sections, pages, chunks (no OCR: a scanned PDF stays unavailable)
research0 embed-chunks --ingestion-run-id <id>
research0 confirm-prepare --run-id <run_id> --ingestion-run-id <id> --output reports/confirm   # the full text confirms, modifies or rejects the preliminary membership
research0 community-submit --run reports/confirm ; research0 confirm-collect --run-dir reports/confirm
research0 extract-prepare --ingestion-run-id <id> --output reports/extraction
research0 community-submit --run reports/extraction                                 # EXPLICIT: costs money
research0 extract-collect --run-dir reports/extraction --remote
research0 extraction-sample --run-id <run_id> --preassignment-run <id>               # 10% of each community, at least 100, for the in-depth check
```

Extraction sends one request per paper (its sections without references, acknowledgements and appendices), not one per chunk: the model keeps one label per construct across the whole paper and joins an acronym or synonym to it only when the authors state the equivalence (stored as an `acronym`/`synonym` attribute with the sentence that says so); a quote must sit inside one sent section, and the chunk of each record is derived from its offset. Chunks are whole sentences (about 500-900 tokens), never cross a section, and satisfy `full_text[char_start:char_end] == text`. Every extracted construct, measure, claim and attribute carries a verbatim quote found in its chunk; others are rejected and counted. Each attribute (level, dimensionality, informant, reliability, validity, mechanism, boundary condition, context, ...) is *extracted*, *inferred* or *missing*. A claim has relation, sign, evidence status (proposed, tested, supported, not supported) and confidence; mediators and moderators are constructs attached to the claim they qualify. People validate or reject records through `decision_event` (append-only).

**Staging of an extraction run (migration 026).** `research0 extract-collect --run-dir <dir> [--remote]` writes a new run into the `staged_*` twin tables, which neither the releases nor the discovery agents read (the `research_agent` role has no access). `research0 extract-promote` lists the staged runs; `research0 extract-promote --extraction-run-id <id> --by <name>` says what would be copied and `--execute` copies the run into the live tables in one transaction and records the promotion (the staged rows stay). `extract-collect --direct` writes straight into the live tables as before.

**Phase 6: the signals and candidate pairs** (`research0 signals-build`, `research0 dossiers-build`). A node is one paper's use of one construct label (a provisional identity). Since `signals-v4` (proposal v4, 13 Oct 2026) the meaning and the measures are compared apart: **S_def** is the embedding similarity of the definitions and the meaning profile (referent, defining features, exclusions, level, temporal form, theoretical role, scope: extraction v4), **S_op** that of the items and measures; a disagreement between the two is informative. The hint is a **retrieval stratum** of the proposal's pair sampling, never a verdict: same label with diverging meanings, different labels with close meanings, close measures under distinct meanings, same meaning in a shifted context, hard negative (one label, one meaning), unresolved (no meaning to compare), none. Dossiers (`dossier-v3`) show the meaning profile apart from the operationalization and ask define, operationalize, substitute, confirm. What follows describes the earlier `signals-v3`, still readable. **S_sem-op**: embedding similarity of definition, items and measures. **C_cit**: bibliographic coupling (Jaccard of the references both papers cite); co-citation needs a corpus of citing papers and is not computed. **N_nom**: typed profiles (antecedents with antecedents, outcomes with outcomes, mediators, moderators), each role a soft weighted Jaccard; a missing role is skipped, not counted as disagreement; sign and evidence status are reported apart. Links between neighbours are *proposed* by label similarity and are not approved identities. A signal without its input is empty (NULL), never zero. Since `signals-v3` (11 Oct 2026) the triage hint reads **S_sem-op only**: jingle candidate = same label and S_sem-op low, jangle candidate = different labels and S_sem-op high, unresolved = S_sem-op missing, none = anything else (a candidate negative). N_nom is still computed and stored but not used: on the first real extraction it was 0 for almost every pair, because the neighbours of two papers rarely match by label. There is no `related_but_distinct` hint any more (people decide it). The thresholds are uncalibrated defaults to be calibrated and frozen before the audit. The dashboard tab "Construct pairs" opens with the statistics of the run (availability and distribution of the signals, the hint, how the pairs were found, review progress).

**Phase 7: adjudication, proposals, gold sets, gates, release**

```powershell
research0 proposal-prepare --signal-run-id <id> --output reports/proposals ; research0 community-submit --run reports/proposals ; research0 proposal-collect --run-dir reports/proposals   # step 08a (costs money)
research0 pair-sample --signal-run-id <id> ; research0 pair-sample-report --pair-sample-id <id>   # probabilistic sample (equal pairs per stratum, so "none" pairs and positives are both there) and challenge set, never mixed
research0 dossiers-build --signal-run-id <id> --pair-sample-id <id>                              # the evidence dossiers of the sample's pairs; the dashboard's blind queue serves only these, in a fixed shuffled order
# the cascade (signals -> model -> people): the model reads only the targeted strata, people get what it promotes plus an audit of what it rejects
research0 dossiers-build --signal-run-id <id> --targeted ; research0 proposal-prepare --signal-run-id <id> --output reports/proposals --targeted   # the manifest estimates the tokens
research0 community-submit --run reports/proposals ; research0 proposal-collect --run-dir reports/proposals                                       # costs money
research0 pair-sample --signal-run-id <id> --design cascade [--audit 50] [--promoted N] ; research0 dossiers-build --signal-run-id <id> --pair-sample-id <id>
research0 pair-sample-report --pair-sample-id <id>                                              # with a cascade sample: the recall of signals + model per conceptual class
research0 pairs-export-blind --signal-run-id <id> --output pairs_blind.jsonl                    # for independent labelling, without signals or hints
# decide in the dashboard tab "Construct pairs", or: research0 pair-decide --pair-id <id> --validator Ana --class conceptual_jingle --same-referent no --compatible-properties no --substitution-changes yes --consequence "..." --rationale "..."
research0 gold-import --kind pairs --file gold_pairs.json --label v1 --provided-by Ana           # formats: templates/gold_*_template.json
research0 release-build --run-id <run_id> --label r1 --preassignment-run <id> --audit-id <id> --ingestion-run-id <id> --extraction-run-id <id> --signal-run-id <id> --gold-pairs <id> --operational-window 2023-2025 --output reports/release_r1
research0 release-signoff --release-id <id> --validator Ana
```

A decision follows the pair codebook v2 of the proposal v4: one *class* (conceptual jingle, conceptual jangle, measurement divergence, scope variation, empirical redundancy, part-whole, related but distinct, distinct, same meaning (no fallacy), unresolved) with the answers of the comparison (same referent? compatible defining properties? does substituting the labels, or separating the meanings, change a central proposition? yes / no / unclear), the theoretical consequence and a counterexample. A conceptual jingle needs the same label, properties that are not compatible and a separation that changes a proposition; a conceptual jangle different labels, the same referent, compatible properties and a substitution that preserves the propositions; both need the consequence written down. Decisions taken before with relation + flag are kept as history (`pair-codebook-v1`) and not counted: those pairs are decided again. The model's proposal on a pair (step 08a, `pair-proposal-v2`: substitution test, then one class of the same codebook) is stored apart and shown to a reviewer only after that reviewer's own decision. The jingle/jangle gate holds separately for each positive class (its value is the lower of the two) and reports how the near negatives were flagged. The gates are those of the proposal's tables (screening recall, human agreement with its action bands, membership macro-F1, multi-label Jaccard/micro/macro-F1, extraction F1 / complete records / definition-and-measure precision, jingle/jangle recall-precision-F1 reported per class, ICC and Spearman on ratings, calibration error); a gate that cannot be evaluated yet (no gold data) is **not a pass**. A release is *blocked*, *candidate* (all gates pass) or *validated* (all gates pass and a person signed it off); its package holds the manifest, counts (distinct papers, per community, 1/k), time perimeter, full-text status, signal availability, memberships with status, the construct registry, the decided relations, the audit report and a model card.

**Phase 8: read-only retrieval for discovery agents.** Only a validated release can be read. `research0 rag-search --release-id <id> --query "..."`: community filter first (an `agent_profile` fixes the community and policy), then full-text rank and vector distance fused by reciprocal rank, an optional reranker, and a diversity rule (at most N chunks per paper). Hits carry chunk, pages and offsets; every query is logged in `retrieval_run`/`retrieval_hit`, the only writes allowed. `research0 rag-index` builds an HNSW or IVFFlat index for one embedding model, to benchmark against the exact scan on the real corpus. Agents connect as the PostgreSQL role `research_agent` (SELECT everywhere, INSERT only on the retrieval log; no login until you run `ALTER ROLE research_agent LOGIN PASSWORD '...'`) through `AGENT_DATABASE_URL`. The discovery forum below builds on it; the blind expert panels that rate the questions are outside this system.

## Discovery forum (a service of its own)

The forum is where specialist agents work on a release (normally a validated one) and where you can follow them: the conversation between agents, or each agent's own reasoning. It is a separate service with its own process, port, host name and settings, not a page of the dashboard: `research0 forum-serve --run-id <id> [--port 8770]`, then open `http://127.0.0.1:8770/`. It shares only the database with the validation dashboard, which knows nothing of it. Settings: `FORUM_BIND`, `FORUM_ALLOWED_HOSTS` (the public host name) and `FORUM_READ_ONLY=1`, which lets reviewers watch every episode but not create, start or stop one.

An *episode* is one run of one condition of the proposal (v4, 13 Oct 2026): **0** unorganized baseline (full-text passages only), **A** generalist (construct database and full texts of CTO and TIM), **B** isolated specialist (one specialist, CTO or TIM, alone on its own community's view with the whole budget; running both in one episode with half the budget each was removed on 14 Oct 2026 - the episodes made that way can be read, not run - and a batch runs B once for each specialist), **C** open specialist encounter (the opener, chosen in the form or picked by the seed, proposes a theme from evidence following the trigger and the other specialist tests or reframes it from its own corpus; they compare two or three puzzles and the opener writes one joint question; the earlier openings - each from its construct database, a theme by retrieval, random passages - remain as auxiliary controls outside the design). D, E and F were removed from the design: their old episodes can be read, not run. Every episode follows one of nine **ideation triggers** (new phenomenon, anomaly, challenged assumption, paradox, missing mechanism, boundary reversal, cross-level or temporal process, theoretical translation, construct problem), chosen in the form or drawn from the seed alone (the same seed gives the same trigger in every condition): it says how to look for a puzzle, never what to study. Only the **construct problem** hands over relationships between constructs that people decided - to A, B and C as dossiers that begin with the reviewers' reasons and the contrary evidence, to 0 as the raw passages of the same papers; the same seed picks the same pairs in every condition. The agents walk one chain: evidence, puzzle or challenged assumption, at least two rival interpretations, mechanism, a short theoretical question, predictions, and only then the empirical design (unit, context, measures, timing, design), which stays out of the question. From the form you choose the release (one that is not validated can be used too: the episode, its list entry, its Settings and every question it produces are then tagged UNVALIDATED, with a red banner, and its output is exploratory, not a finding), the condition, the trigger, and for each agent three models (private *reflection*, public *discussion*, final *synthesis*) with reasoning effort, temperature and token limits; the protocol (rounds, private reflection on or off, focus, seed, who opens and writes the question); the budgets (tokens, model calls, seconds, searches, new passages and dossiers, which are the information budget; keep them identical across conditions, and the page lists the episodes whose budgets match); retrieval (embedding model, passages per search, per paper, characters per passage, construct relationships handed over, the opening of C); and internet access (off by default). Everything is frozen with the episode and shown in its Settings tab. Questions are archived with their trigger, seed, opener, opening theme, discarded puzzles and conceptual signature (migration 024: apply it before running this version on a database).

Page layout (reorganized on 14 Oct 2026 for the research team): **Overview** (what the experiment tests, the three steps of the work, the four conditions, the recent batches), the sidebar with **batches as groups** (a repetitions × arms grid; open one to see its episodes) and the single episodes apart, and **Results** (the blind evaluation). An episode has five tabs: **Conversation** (the outcome - question and blind score - on top, then the public contributions with the trigger, the opening and any construct-problem dossiers, citation chips that open the passage, and parallel notebooks for the conditions where agents never meet), **Research question** (its chain of argument, predictions, plan to verify, analyses marked obtained / inferred / proposed, and sources checked against what the agents really retrieved), **Proposal** (the 4-page document written from each question), **Behind the scenes** (private reasoning, the event log, and the code runs when the episode could run code) and **Settings**. Everything that makes an episode exploratory (unvalidated release, Batch v1 views, provisional or signals-only relationships) is listed in one notice on top. A citation to evidence nobody retrieved is shown as such, never silently kept. Every retrieval is logged in `retrieval_run` under a per-episode agent profile.

```powershell
research0 forum-run --release-id <id> --config episode.json --by "Your name"      # headless; add --create-only to only register it
```

**Batch (14 Oct 2026).** In the form (*New experiment*), *Matched batch* - the default, step 1 of 4; the other choice is *Single episode* - creates five *arms*, **0, A, B with the CTO specialist, B with the TIM specialist, C**, with the same trigger, seed, budgets, models, retrieval and proposal settings, and runs them one after the other. *Repetitions* repeats that N times (1 to 50), so a batch is 5×N episodes, run in the order repetition 1: 0, A, B (CTO), B (TIM), C; repetition 2: the same five, and so on. The five episodes of a repetition always share the seed. Across repetitions the seed moves on by one by default (seed, seed+1, ...: a random trigger, the opener of C and the construct relationships are drawn again); *the same seed every repetition* repeats the starting material and leaves only the model's variability. There is one block of agent settings for every agent. The lead option applies only to C. Each B specialist works alone with the whole budget. Every configuration is checked before anything is written, and the batch is created whole or not at all. The confirmation shows the ceiling for the whole batch (5×N times the caps of one episode). Only one episode runs at a time: *Stop batch* stops the running episode and the rest of the batch. The episodes not reached stay *Not started*, and *Run the rest* runs them later (also after a server restart). After 3 failed episodes in a row the batch halts by itself, because the cause is then likely common (a key, a quota, a model id). The batch view (`#/b/<id>`, from the *batch r/N* badge) shows a grid of repetitions × arms with status, tokens, questions, proposals and the mean of the three votes of its proposals (novelty, methodology, impact), plus the totals per arm. The grouping lives in each episode's frozen configuration (`config.batch`: id, repetition, position, arm), so it needs no migration, and the episodes of a batch share `budgets_sha256`, which keeps them comparable in Results. From the CLI the configuration is the same JSON without `condition` and with one `agent` block:

```powershell
research0 forum-batch --release-id <id> --config batch.json --repetitions 5 [--seed-mode vary|same] --by "Your name" --acknowledge-send-to-openai   # add --create-only to only register it
# batch.json: {"provider": "openai", "seed": 20261014, "trigger": "random", "agent": {"discussion_model": "gpt-6-luna", "reasoning_effort": "medium"},
#              "protocol": {"rounds": 4, "lead": "random"}, "budgets": {"max_total_tokens": 300000, "max_llm_calls": 60}}
```

Corpus for the agents: `research0 embed-corpus --ingestion-run-id <id> --extraction-run-id <id> [--dry-run]` embeds the chunks of the PDFs (`chunk_embedding`) and the records of the construct database (`record_embedding`, migration 021: constructs, measures, claims, attributes) into PostgreSQL + pgvector with `openai:text-embedding-3-small`. `--dry-run` counts texts and real tokens and calls nothing; a real run needs `--acknowledge-send-to-openai` (the texts go to OpenAI and cost money), commits after every batch and can be rerun to continue. Agents search the records with the `search_constructs` tool. The CTO/TIM views of the forum can use the Batch v1 labels (imported with `preassign-import-v1`) as suggestions: they are binary, never confirmed, and the episode says so.

Provider: **openai** only. It sends the prompts, with passages of the papers, to OpenAI and costs money, so the page asks for confirmation before an episode starts (creating one sends nothing; `forum-run` needs `--acknowledge-send-to-openai`). The free offline simulation (`scripted`, and `research0 forum-demo`) was **removed on 13 Oct 2026**: the page, the API and the CLI no longer offer it, the page shows no trace of it, and the simulated episodes that existed were deleted from the database. Should an old one ever be restored, the server would refuse to run, rate or rank it. The automatic tests still use an offline stand-in for the model ([tests/forum_fake.py](../tests/forum_fake.py)), which the application cannot select. Models are listed in `config/forum_models.json`, whose prices are `null` until you enter the published ones; without a price the page shows tokens, not dollars. **Timeliness and novelty (all conditions).** Every agent is told today's date and that the corpus ends in 2025, and is asked (rule 8 of the forum, the full-text step and the last step of every chain) to reason in time: note the year of each study, trace the topic up to the most recent studies, say what has changed recently that makes the question pressing, and check that no recent study has already answered it. The final question has a required `timeliness` block (*why now*, *most recent evidence*, *already answered?*, *how far the check reached*), shown in the **Questions** tab, and the proposal's introduction must argue the same. With internet **open**, every page a web search returns gets an id **W1, W2, ...** citable like `[E3]`, listed among the sources and in the proposal's references as "not checked by the system"; with internet off, the agents must say that their check covers the corpus only. Warnings flag a question with no such argument, one whose most recent cited study is from 2022 or earlier, one checked against the corpus only, and one that had the web but cites no web source for novelty. These checks say how far the claim was argued, not whether the question is really new: that stays with the experts.

Budgets are checked between model calls (a call can run over by at most its own output limit) and a share is held back so the final question can always be written. The OpenAI path has been tested only against a stand-in client, not against the live API.

**Code for analyses (migration 022, off by default).** With the episode setting *Analysis tools = Python on the server's machine* the agents get a `run_python` tool: a complete program, run in a fresh Python process ([forum_sandbox.py](../src/research0/forum_sandbox.py)) on CSV tables built for that agent (`papers`, `constructs`, `measures`, `claims`, `diagnoses`, all limited to its community view, and `passages` with what it retrieved; the baseline condition gets only `passages`, because it has no database by design). Each run gets an id **A1, A2, ...** that can be cited like `[E3]` or `[P1]`; the code, the SHA-256 of every file it read, what it printed, the files it produced (png, csv, txt, json), the environment and the time are stored (`forum_analysis`, `forum_analysis_file`) and shown in the **Analyses** tab. A question can call an analysis *obtained* only if it cites a run that finished without error; otherwise the question carries a warning. Limits per episode: runs, seconds and MB per run (part of the comparability checksum).

It is a **safeguard against mistakes, not a security boundary**: the process runs as the server's own user, with a scrubbed environment (no `DATABASE_URL`, no API keys), a wall-clock limit, a memory limit (a Windows job object, or `RLIMIT_AS`), a cap on what it prints, no second process, and a prelude that blocks sockets and subprocesses and lets `open()` read only `data/` and the Python installation and write only its own folder. Code that goes around the prelude (`ctypes`, a C extension) can read what your account can read; the realistic risk is a prompt injected through the text of a paper, and the only way out is what the code prints, which is stored and shown back to the model (and, with the OpenAI provider, sent to OpenAI). That is why the server refuses unless `FORUM_ALLOW_LOCAL_CODE=1` is set, the files in `deploy/` never set it, and the episode form says so. Libraries available to the agents are whatever is installed (numpy, scipy, scikit-learn; pandas, matplotlib, statsmodels if you add them).

**Research proposal (migration 022, on by default).** When the question is written, the agent that wrote it writes a proposal of at most 4 pages made only of an *abstract, an Introduction and a Theoretical background*, each in depth ([forum_proposal.py](../src/research0/forum_proposal.py); since 18 Oct 2026 there are no Methods or Expected outcomes in the text). Once the text is final, its author writes a **methodology card** and an **impact card** ([forum_cards.py](../src/research0/forum_cards.py)) with exactly the structure of the cards of the published papers (same fields, `text`/`status`/`passages` and `info_gaps`, same check that every passage is an exact piece of the proposal's text); they are stored with the proposal, outside the page limit, and shown under its references. The impact card reports what the proposal claims it would change; the methodology card is the design its author chose to test the propositions, a plan and not a result. With one agent (conditions 0, A) or isolated specialists (B, each writes their own) it is one call; in a discussion (C-F) the lead drafts, the other specialist writes one critique, and the lead revises. The text is checked, rendered to PDF and its pages **counted**; if it runs over, its author is asked to shorten it (twice at most) and, if it still does, it is kept and marked `over_limit` with a warning, never cut by hand. The reference list is not written by the model: it is built from the ids the text cites, looked up in what the episode retrieved, was given or ran; an id that does not exist is flagged. A paper is listed as *authors (year). Title. Journal.* with the cited pages and DOI, all from the catalog record (Scopus); more than three authors become the first one and *et al.*, and a field the catalog lacks is left out. A proposal made on an unvalidated release, or on premises no person reviewed, starts with an EXPLORATORY notice. Part of the token and call budget is held back for it (about 99,000 tokens and 5 calls per question in a discussion, 57,000 and 3 for a single agent, with the default 9,000-token limit per call; two of those calls and 36,000 tokens are for the two cards), so rounds get less budget than before. Open it in the **Proposals** tab or as a PDF (`/api/forum/proposal.pdf?id=...`, rendered from the stored text). Turn it off per episode in the form.

**Evaluation of the proposals and Results (migration 030, 18 Oct 2026).** The blind evaluator (migration 023), the Baseline Agent Evaluation (028) and, on the forum page, the publishability estimate are gone; their rows stay in the database as history. Each proposal of the current format (abstract, introduction, theoretical background and the two cards) gets three votes from 1 to 10 ([forum_scoring.py](../src/research0/forum_scoring.py)), each given by one judge (`gpt-6-luna`, medium effort, configuration [config/forum_scoring.json](../config/forum_scoring.json)); a fixed 20% of the proposals, drawn from the proposal id (`FORUM_SCORING_DOUBLE_SHARE`), also gets a second judge and the vote is the mean of the two. Each criterion is judged against its OWN nearest published papers: **Novelty** - the title and abstract against the 10,050 abstracts of the CTO/TIM corpus (`cor_a5621a4d95ac`); distance is good, and the judge votes even when every neighbour is far. **Methodology** - the proposal's methodology card against the methodology cards of 2,611 papers (`fsm_…`), retrieved on the card only; similarity is good (a design with published precedents is feasible and defensible). **Impact** - the impact card against the impact cards of 2,614 papers (`fsi_…`), retrieved on the card only; the proposal's potential (it has no results) is compared with the impact those papers really had: Scopus 'Cited by' as a percentile among the papers of the same publication year, and the ABS grade. The code computes the similarity indices (closest paper, mean of the k=5 closest, and the percentile of the closest one against how close the corpus papers are to each other) and, for novelty and methodology, an INDICATIVE vote (novelty: 0.50 -> 8, one point per 0.06; methodology: from the percentile). The indicative vote is guidance, not a rule: each judge must say whether the similarity (or the distance) is a matter of substance or only of wording, setting or style, quote both texts exactly in at least two comparisons, and explain any departure. An answer that breaks a check is asked for again with the reasons. It runs automatically at the end of an OpenAI episode (outside the episode's budget; `FORUM_AUTO_EVALUATE=0` turns it off), from the **Results** page (*Evaluate pending*) or with `research0 forum-score --run-id <id> [--proposal-id <id>] --acknowledge-send-to-openai`; four judge calls per proposal (eight for the double sample): the three votes plus a second, hybrid novelty judge (the closest abstracts by embedding plus those a BM25 lexical search finds, compared on question, mechanism and predictions), shown beside the first novelty judge and never part of the mean (`lexical_k` in `config/forum_scoring.json`, 0 turns it off). To add the hybrid vote to proposals already evaluated: `research0 forum-score --run-id <id> --hybrid-missing --acknowledge-send-to-openai` (a new row each; the other judges' answers are in the register, so only the hybrid judge is paid for). The three corpora are built once with `research0 forum-score-corpus --run-id <id> --kind novelty|method|impact --acknowledge-send-to-openai` (novelty reads the Scopus exports in `inputs/scopus` and the CTO/TIM assignments of the run; a build with the same inputs finds the corpus already built). They live in `reports/anvur_live/corpora/`, outside git. The code: [paper_cards.py](../src/research0/paper_cards.py) (the two cards, their check and the sections they are made from), [scoring_corpora.py](../src/research0/scoring_corpora.py) (the corpora), [scoring_judges.py](../src/research0/scoring_judges.py) (configuration, cache of the calls, limits), [forum_scoring.py](../src/research0/forum_scoring.py) (the three votes). The **Results** page shows the leaderboard (the three votes, their mean and the closest-paper similarity of each criterion) and the statistics by condition: votes, every similarity index, yield per episode, means by trigger. Proposals written before 18 Oct 2026 (four sections, no cards) are not shown. Paths can be changed with `FORUM_SCORING_CONFIG`, `FORUM_SCORING_WORKSPACE`, `FORUM_SCORING_NOVELTY_CORPUS`, `FORUM_SCORING_METHOD_CORPUS`, `FORUM_SCORING_IMPACT_CORPUS`. Two judges of the same model family are not a panel, and with few proposals per condition the differences between means are not evidence.

## Evaluation of published papers (migration 031, 20 Oct 2026)

The novelty judges of the forum can be run on papers published after the corpus perimeter (the 2026 papers in `inputs/evaluation_2026/`, local and outside git: see its README). [paper_evaluation.py](../src/research0/paper_evaluation.py) gives each paper the same evaluation a proposal gets (title and abstract in, the first novelty judge and the hybrid judge out; no web search, no tool). The judge reads only `evaluator_input/abstracts.jsonl` (id, title, abstract); the journal and the ABS grade are in `display_metadata/papers.csv` and only appear on the results page. The model is the one of [config/evaluation_pre2026.json](../config/evaluation_pre2026.json) (`gpt-5.2-2025-12-11`, knowledge cutoff 31 August 2025, declared per judge: a run is refused without a cutoff that precedes the earliest paper of the sample). Start a run from the dashboard page *Evaluation* (side bar, step 6; local server only, `OPENAI_API_KEY` set, the box confirming that the texts go to OpenAI ticked); it is refused on a read-only deployment. Methodology and impact (runs on three criteria) need the two cards of each paper; the page has the steps that make them for the sample: full texts (only those the project already has: the 2026 PDFs of the ANVUR package of 17 Oct 2026, copied from git commit 87c4aaf to `inputs/evaluation_2026/fulltext_archive/`, and files put by hand in `inputs/evaluation_2026/fulltext_manual/` named after the paper id or the DOI with `/` as `_`; nothing is downloaded; by default the sample is drawn among the papers that have one), local parsing, and the cards, made as the corpus cards were (same sections, instructions, profile versions and model, `config/evaluation_cards.json`; a field naming the paper's journal is removed). The cards go by default as an OpenAI Batch (*Submit the cards as a Batch*: half the price, answers within 24 hours; the request files are in `<EVALUATION_WORKSPACE>/card_batches/<job>/`); press *Check the Batches* later to collect and check the answers. A card in a Batch not yet collected is not sent again, a refused card goes into the next Batch (up to three refusals), and *Make them now* still makes them synchronously at full price. Apply `migrations/031_paper_evaluation.sql` and `migrations/032_paper_evaluation_cards.sql` first. Paths: `EVALUATION_DIR`, `EVALUATION_CONFIG`, `EVALUATION_WORKSPACE`; the abstract corpus is the one of the forum (`FORUM_SCORING_NOVELTY_CORPUS`). The page shows the two votes per paper, the mean by ABS grade and the agreement of the judges; a relation between grade and vote describes the sample and validates nothing.

The [proposal evaluation addendum](evaluation/PROPOSAL_EVALUATION_ADDENDUM.md) describe the
new offline integrity checks and a prepared, three-arm novelty ablation.
The private Results preview and generated packets are local under ignored
`reports/`; reproduction commands use the evaluator's copied export without connecting to AWS.
The [Vite/React evaluation atlas](../web/README.md) is a separate, local view of
the same dated export. It shows all 38 proposals, their evidence and integrity
flags, and requires no AWS connection. It is not connected to production.
The [external evaluator handoff](evaluation/EVALUATOR_HANDOFF.md) lists the six
concrete follow-up cases and the next reproducible scorer checks.

## Retired evaluators

Three earlier ways of judging a proposal were tried and retired when the forum moved to the three votes above (19 Oct 2026): the blind evaluator (migration 023), the publishability estimate (migration 027, `pub-*` commands) and the ANVUR evaluator with its material package (`anvur-*` commands, the Baseline Agent Evaluation of migration 028). Their code is in the git history (commit `87c4aaf` and the archive noted in the internal project notes); their migrations and tables stay, because they hold the rows of what was run. The two paper cards and the corpora the three votes use were born in the ANVUR evaluator and now live in `paper_cards.py` and `scoring_corpora.py`; their stored version strings keep the old `anvur-` prefix.

## Mock world and demo (synthetic data, removed automatically)

[src/research0/mock_world.py](../src/research0/mock_world.py) builds a synthetic catalog in a throw-away PostgreSQL schema `mock_<hex>` (the real schema is never written): four invented topics with known membership, generated PDFs, a planted jingle, jangle and partial overlap, and scripted stand-ins for the model and for the people. The tests use it and drop it when they end.

```powershell
research0 demo            # all phases 2-8 end to end on the mock world, prints what each step produced, then deletes the mock world
research0 demo --keep     # keep the schema to look at it
research0 mock-clean      # drop any leftover mock_* schema (for example after an interrupted test run)
```

To remove the mock machinery from the project entirely, delete `src/research0/mock_world.py`, `src/research0/demo.py` and the tests that import them (`pytest` shows which); nothing else depends on them. The demo's release comes out *blocked*, which is the honest outcome without gold extraction and ratings; its final retrieval step forces the gates to pass, and says so, only to exercise Phase 8.

## Limits

Coverage does not establish scientific validity or completeness of Scopus. Partial exports can lower measured coverage. Duplicate detection is a review signal rather than proof of identity. CTO/TIM decisions are model-generated suggestions until checked against the community domain statements and a human-reviewed pilot. PDFs enter only through the open-access download or a person's upload, and a paper without one stays unavailable. Embeddings and clusters are derived aids, never decisions. Phases 2-8 have been tested only on synthetic data.

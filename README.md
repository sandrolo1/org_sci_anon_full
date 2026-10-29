# Discovery Forum: research platform (anonymised review copy)

This repository is the anonymised code and documentation package accompanying a Stage 1 proposal. It contains:

- a **pipeline** that turns a bibliographic export into a canonical catalog, extracts constructs and relations from papers with
  human checkpoints at every decision step, and produces a validated knowledge release;
- a **discovery forum**, a separate web service in which language-model agents work on such a release under controlled conditions
  (0, A, B, C) and are rated blind;
- a read-only **dashboard** for the human validators, and an offline **reviewer artifact** (`web/reviewer`) that needs no database or key.

Everything runs on your own machine. The authors' data are not included; see [What is not included](#what-is-not-included).

| Folder | Content |
|---|---|
| `src/research0/` | the Python package and the `research0` command line |
| `migrations/` | PostgreSQL schema (pgvector), applied by `research0 init` |
| `config/` | scopes, judge and model configuration, scoring configuration |
| `tests/` | the test suite (uses a throw-away database schema) |
| `web/` | Vite/React apps; `web/reviewer` is the offline reviewer artifact |
| `docs/PIPELINE_GUIDE.md` | step-by-step guide to every phase and command |
| `docs/evaluation/`, `docs/architecture/` | evaluation protocols and the architecture diagram |
| `stage1/` | the anonymous Stage 1 proposal and technical supplement, and the next-stage protocol |
| `templates/`, `examples/` | empty input templates and a generator of fictional fixtures |

## Requirements

- Python 3.11 or newer
- Docker with Docker Compose (PostgreSQL with pgvector and an S3-compatible store run as containers)
- Node.js 20.19+ or 22.12+ (only for the web apps)
- Optional: an OpenAI API key (only for commands that call a model; these cost money and always print a confirmation line first)

## 1. Offline reviewer artifact (no Docker, no keys)

```sh
cd web
npm ci
npm run reviewer:check     # evidence gates on the invented case and the pilot aggregate
npm run reviewer:dev       # open the address printed by Vite
```

The artifact uses four invented papers and a verified aggregate of the 30 real pilot episodes. It makes no network calls.

## 2. The platform on your machine

```sh
cp .env.example .env                  # PowerShell: Copy-Item .env.example .env
docker compose up -d                  # PostgreSQL+pgvector on :55432, S3-compatible store on :8333
python -m venv .venv
source .venv/bin/activate             # PowerShell: .\.venv\Scripts\Activate.ps1
pip install -e ".[test]" -c requirements-tested.txt     # pins the versions the suite was run with
research0 init                        # applies the migrations
pytest -q                             # the suite (run from this folder: it reads .env); it creates and drops its own schema
research0 demo                        # all phases end to end on a synthetic world, then deletes it
```

**Windows note.** Inside a virtual environment, the optional local code sandbox of the forum (`FORUM_ALLOW_LOCAL_CODE`, off by default) cannot start,
because the venv's `python.exe` is a launcher and the sandbox limits its job to one process. Thirteen tests in `tests/test_forum_code_proposal.py` then fail.
They pass with a Python that is not a venv; to skip them use `pytest -q --ignore=tests/test_forum_code_proposal.py`. Observed on Windows only.

`research0 demo --keep` keeps the synthetic schema so you can inspect it; `research0 mock-clean` removes any leftover one.
No API key is needed for any of the above: embeddings default to an offline lexical method (`EMBEDDING_MODEL=hash-rp-v1`) and the
demo uses scripted stand-ins for the model and the people.

## 3. Your own `.env`

`.env` is read from the current directory at startup (variables already set in the terminal take precedence). It is never
committed. `.env.example` lists every setting; the ones you will normally touch are:

| Variable | Needed for |
|---|---|
| `OPENAI_API_KEY` | every command that calls a model (assignment, extraction, forum episodes, scoring). Leave empty to stay offline. |
| `UNPAYWALL_EMAIL` | open-access PDF lookup (Unpaywall asks for a real contact address). |
| `OPENALEX_API_KEY`, `SEMANTIC_SCHOLAR_API_KEY` | optional, further PDF sources. |
| `EMBEDDING_MODEL` | `hash-rp-v1` (offline, default) or `openai:<model>` (sends text to OpenAI). |
| `DATABASE_URL`, `S3_*`, `AWS_*` | already point at the Docker containers; the values in `.env.example` are local-only demo credentials. |
| `FORUM_*`, `DASHBOARD_*` | bind address, allowed hosts and read-only mode of the two web services. |

To use your own corpus, put your files under `inputs/` (see `templates/` and `docs/PIPELINE_GUIDE.md`) and follow the phases in order.
Starting the services needs a catalog or release id from your own database:

```sh
research0 dashboard-serve --run-id <catalog_run_id> --port 8765     # validation dashboard
research0 forum-serve --run-id <release_id> --port 8770             # discovery forum (its own service)
```

Commands that would send text to OpenAI never do so on their own: they print an explicit submit or acknowledge line you have to run.

## What is not included

- Scopus exports, abstracts and full texts: licensed or unpublished, so the database starts empty and `inputs/` is a skeleton.
- Model messages, credentials and the raw logs of the 30 real pilot episodes. Only their verified aggregate is in `web/reviewer/public/pilot.json`.
- Deployment scripts for a hosted instance. Everything here is meant to run locally.

## Scope of the claims

The prototype is exploratory. The interactive questions in the reviewer artifact are scripted illustrations. The 30-episode pilot shows
feasibility and mixed arm rankings, not a positive effect; the evaluator audit establishes traceability, not score accuracy. The
validated construct release, independent quality validation and the blinded matched 0/A comparison are planned work; see `stage1/`.

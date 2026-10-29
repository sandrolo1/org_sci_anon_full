# Novelty evaluation: automated prototype

## Current state

The former `proposal-blind-v2` evaluator read a proposal's text without
checking prior papers; its `novelty_net` score is historical. The evaluator's new
`forum_scoring.py` compares a proposal abstract with five semantic neighbors
from a 10,050-paper CTO/TIM corpus and checks exact passages in the judge's
answer. It also scores methodology and impact using separate paper cards.
This improves evidence traceability. Its novelty corpus is a model-assigned
subset of the eligible catalog, its retrieval has no lexical branch, and the
judge is required to give a numeric vote even when evidence is thin. Neither
evaluator has established prior-answer recall.
The agent's own novelty check in `forum_protocol.py` is useful for generating
ideas, but is not an independent evaluation.

The short proposal calls for a retrieval check across both communities and
equivalent construct labels. The evaluator's implementation compares proposal
abstracts with nearby Scopus abstracts. For the
current prototype, the user wants no human annotations or reviews. This plan
therefore makes only automated, evidence-linked claims and records uncertainty.
Expert validation can be a later study; it is not a dependency for building
or running the prototype.

## Unit and evidence

Evaluate each **research question** with its puzzle, mechanism, and
distinguishing predictions. The forum already stores these as structured
fields. `novelty_baseline.question_card` copies them verbatim into a search
card, so the first baseline needs no generated abstract or human fidelity
check. A missing field causes an abstention. Keep the original question and
card hashes. A generated abstract may be tested later as a separate retrieval
variant, with an automatic entailment check against the stored question.

Freeze a manifest for each audit: catalog run, included paper IDs, query and
source-list versions, export batches, date of last literature search, abstract
coverage, text recipe, embedding model, cluster run, alias-map version,
retrieval parameters, and code version. Scopus title, abstract, DOI, year, and
source come from the canonical catalog and representative export record.
Missing abstracts remain visible in the coverage count.

## Candidate retrieval

1. Search **all eligible papers**, not only the proposal's assigned cluster.
   Rank the question card against paper abstracts (or the same documented
   title/abstract/keyword recipe on both sides) with cosine similarity. Show
   the nearest papers with DOI, year, title, source, and score.
2. Rank cluster centroids separately to show the proposal's topical context.
   The centroid is a browsing aid; a distant centroid must not hide an
   individual near match. `novelty_baseline.compare_abstracts` implements the
   deterministic comparison from supplied vectors, with explicit missing-data
   counts and no novelty verdict.
3. Run a lexical search on the question, mechanism, predictions, and mapped
   equivalent construct labels. The offline BM25 function accepts explicit
   query variants. For the automated prototype, use only acronyms or synonyms
   explicitly stated by paper authors with a stored supporting quote;
   model-suggested variants remain separate, uncertain leads. Fuse vector
   and lexical lists by reciprocal rank, deduplicate canonical papers, and
   preserve which query found each hit. Title-only matches remain visible but
   cannot support an abstract-level claim.
4. Check newer work beyond the frozen Scopus window separately and record its
   search date and provenance. If this is unavailable, report the time limit
   prominently. A 2015–2025 corpus cannot establish novelty in 2026.

## Automated evidence check

An evaluator receives the question card and retrieved paper evidence, with
the forum condition hidden. It must choose: **apparent prior answer**,
**substantial partial overlap**, **related but distinct mechanism or
prediction**, or **insufficient evidence**. Every nontrivial claim must cite
a paper ID and an exact passage from its abstract or available full text.
Programmatic checks verify that the passage exists in the stored source and
that the DOI/paper ID resolves. A second independent evaluator can repeat the
classification. Disagreement, missing abstracts, missing full text, or a
question beyond the literature cutoff yields **uncertain**, not a high
novelty score. Similarity ranks only which papers are inspected.

The output is an evidence dossier: prior-answer candidates, passages,
coverage, judge disagreement, and abstentions. It does not claim that no
earlier answer exists, and its labels are not human-validated novelty scores.

## Automated controls before the October 23 batches

Build fixed positive controls from duplicate/version records and paraphrased
paper abstracts; add label-swapped variants to test whether an established
answer is retrieved under another name. Measure recall@k on those known
sources and report misses. Test stability under query paraphrases and paper
order changes, and measure agreement between independent evaluators. Compare
nearest-vector, centroid-only, and combined vector-plus-lexical retrieval.
Freeze the rule, `k`, prompts, model IDs, and corpus snapshot before applying
them to forum output. These controls measure retrieval and consistency. They
do **not** estimate accuracy on genuinely new research questions without
independent ground truth.

## Immediate implementation boundary

The comparison, lexical search, and rank fusion functions are in place. A
read-only catalog function builds a packet from the eligible papers, current
embeddings, and cluster assignments, including title-only lexical hits that
have no embedding. On 19 October the external evaluator supplied a 1,466-paper extraction
subset and 30 forum questions. `research0.online_snapshot` converted those
exports; `research0.novelty_offline` generated 30 lexical candidate dossiers
with 20 hits each. All remain `retrieval_only`. This is a real-data smoke test
of the offline pipeline, not a run on the full catalog or a validation of
retrieval recall. The new Git code has abstract-corpus and card-corpus readers,
but their current vectors are not in this checkout. There is no configured
`DATABASE_URL` here. The next integration step is a whole-catalog retrieval
comparison with automatic controls and explicit abstention for insufficient
evidence. Neither the forum score nor the offline dossier should be presented
as a verified novelty finding.

### Offline export path, ready to run

With a read-only `DATABASE_URL`, the owner can export a consistent snapshot
without granting AWS access:

```sh
python -m research0.novelty_export --run-id RUN_ID --literature-cutoff-year 2025 --catalog-output reports/novelty/catalog.json --episode-id EPISODE_ID --questions-output reports/novelty/questions.json
```

`--episode-id` and `--questions-output` may be omitted before the forum batches
exist. The cutoff is an explicit claim about the literature search, not
inferred from the newest paper in the database. The exporter includes excluded
rows and their status so the retrieval report can show coverage and omissions.

For proposals in the new format, add `--scored-proposals-output
reports/novelty/scored_proposals.json` to the same command. This exports the
original title and abstract and the latest real three-metric score document
from the same read-only transaction. Run once per episode, then compare the
forum's semantic neighbors with lexical retrieval across every eligible paper
in the frozen catalog:

```sh
python -m research0.novelty_retrieval_audit \
  --catalog reports/novelty/catalog.json \
  --scored-proposals reports/novelty/scored_proposals.json \
  --output reports/novelty/retrieval_comparison.json
```

The audit maps semantic neighbors by canonical ID or unambiguous DOI, records
unmapped neighbors, corpus and score versions, input hashes, coverage, and
overlap among the first five hits. It excludes synthetic scores. A disagreement
is a paper to inspect, not proof that either search is correct. The old 30
proposals in the October 19 ZIP lack an abstract and new score, so this
comparison needs the newly deployed proposals and score export.

`python -m research0.novelty_offline --catalog catalog.json --questions
questions.json --output dossier.json` runs a lexical candidate search without
AWS, a database, or API calls. It records input SHA-256 hashes, corpus
coverage, exact structured question cards, title-only hits, and abstentions.
An optional `--claims claims.json` checks that model-proposed passage quotes
occur exactly in an eligible paper's abstract; it does not decide whether the
quote supports the model's interpretation. This is a usable first-stage
evidence packet, not a scored evaluation or semantic novelty check.

The catalog JSON is an object with `catalog_run_id`, `exported_at`,
`literature_cutoff_year`, and `papers`. Every paper has `id`, `title`, `year`,
`abstract`, and explicit `eligible`; `doi` and `source` are strongly
recommended. The question JSON is a list of `{id, content}` records where
`content` is the forum's structured question object. Optional
`author_stated_aliases` entries need `text`, `paper_id`, and an exact `quote`
from that paper's abstract. The tool checks the pointer, though it cannot
prove semantic equivalence. The optional claims list contains `question_id`,
`paper_id`, `label` (`apparent_prior_answer`, `partial_overlap`, or
`related_distinct`), and `quote`. Reports should remain private if the export
contains licensed abstracts.

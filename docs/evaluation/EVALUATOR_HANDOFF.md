# Evaluation follow-up for the external evaluator

This note uses the frozen 20 October 2026 export (SHA-256
`90d7fc7b4828b7df1dde5b0ea8fcbba8dc33e4b3c802d79b246070074841fdc2`).
The source contains 38 proposals and a 10,050-paper 2015–2025 CTO/TIM
novelty corpus. The findings are diagnostics of stored model output, not
article-level validation.

## Actions possible now

1. **Rerun three invalid hybrid novelty checks.** The stored hybrid judge
   answers failed passage validation, so they have no valid hybrid vote. Keep
   the failed attempt and its exact error, then rerun with the same frozen
   configuration. A second failed answer should remain `insufficient_evidence`
   rather than receive a fabricated or default score.
2. **Investigate three large score shifts.** Their hybrid votes differ from
   the primary novelty votes by at least two points. All five primary
   semantic papers were retained in every scored hybrid packet. In one
   three-point shift, the most directly related paper was already in both
   packets. The difference therefore cannot simply be attributed to a
   previously missed paper. Compare both rationales, candidate order and
   prompt/output format; then rerun with one frozen rubric. The hybrid judge
   made structured comparisons with two, one and three newly added papers
   in these three shifted cases, respectively. This records use of the new
   evidence; it does not identify which change caused the higher vote.
   All 35 scored hybrid judgments compared at least one added paper, including
   the 14 whose vote did not change. Mentioning a new paper is therefore not
   enough to explain why some votes shifted.
3. **Run the prepared controlled comparison.** The private package contains
   114 arm-hidden packets: 38 proposals each evaluated with semantic five,
   semantic five plus lexical five, and semantic five plus five year/length
   matched nonretrieved papers. Keep the judge prompt, model and scoring rules
   identical across arms. Compare evidence use, abstention, score shifts and
   repeat stability. The matched papers are a packet-size control, not known
   irrelevant papers. This test can establish sensitivity, not accuracy. The
   [frozen run protocol](NOVELTY_ABLATION_PROTOCOL.md) specifies the private
   result format and an automatic analysis command. Its layout now keeps the
   five shared papers in identical positions across the two ten-paper arms,
   removing paper order as an explanation for that primary contrast. **This
   comparison has not run.** The authors chose no API route for the private text,
   and a five-packet local judge feasibility check
   produced no usable score with the installed Qwen3 4B model. A CLI that
   sends the same private packets externally is not an acceptable workaround.

The local [evaluation workbench](../../web/README.md) now opens directly on the six
follow-up cases, shows both novelty rationales and their paper sets, and lets
one download a private CSV with proposal IDs, scores, failure reasons and
next actions. Regenerate that file with `python3 web/scripts/prepare_data.py`;
it appears at `web/public/data/followup_queue.csv` and is excluded from Git.
The quote audit traced 886 comparison passage pairs to exported texts, but
traceability does not prove the comparisons were interpreted correctly.

## What would unblock a stronger contribution

- The deployed hybrid scorer and its configuration are now in Git (21 October
  update). Adapt its rubric once so every arm of the three-arm comparison uses
  identical instructions and can abstain on insufficient evidence; record the
  resulting prompt/configuration hashes. The packet and analysis code is
  ready, but it has made no judge calls.
- Share the definition and export of the full *eligible* prior-art catalog,
  or explicitly justify the current CTO/TIM subset and 2025 cutoff.
- For the 2026-paper check, the expanded export has 125 papers, including 75
  carried over from the first 80-paper export. The
  expanded audit finds an impact-led
  four-star contrast and weaker novelty separation. It verifies all 5,938
  required nonblank judge quotations against the saved texts, while leaving
  the meaning of those comparisons unvalidated. Four recorded online dates
  and the stated model cutoff still need reconciliation. Venue tier is a weak
  external comparison, not a truth label for any individual article.

## New retrieval control

The known-source benchmark now runs
the deployed BM25 index against all 10,050 papers, using 1,000 deterministic
source papers. A direct 20-word excerpt retrieves its source in the top 20 in
1,000/1,000 cases. After deleting the rarest half of distinct terms, this is
979/1,000; after deleting three quarters, 323/1,000. This is an offline,
repeatable plumbing and word-loss stress test, not a paraphrase test or
novelty accuracy estimate. A next step the external evaluator can run on the same frozen
sources is to compare the semantic and hybrid retrievers on these erased
queries, then test independently adjudicated paraphrases and distinct
mechanisms. The offline benchmark command makes no model calls and shares no
private text; the model-dependent comparison remains unrun.

The [matched 0/A next-stage protocol](../../stage1/NEXT_STAGE_PROTOCOL.md)
turns the mixed 30-episode pilot into a specific research design: frozen
problem packets, balanced order, identical underlying passages, a separate
12-pair feasibility phase, blind ratings, and complete failure accounting.
Its schedule generator is tested but is not connected to the live five-arm
runner, and no new generation experiment has been run. The prototype can
remain fully automated now; the independent ratings are part of a later
confirmatory study.

The most useful immediate change is the invalid-result handling: a failed
passage check should be visible as no verdict, with a repeatable retry path.
The next research test is whether lexical retrieval changes a fixed judge's
reasoning beyond the change caused by adding five other papers.

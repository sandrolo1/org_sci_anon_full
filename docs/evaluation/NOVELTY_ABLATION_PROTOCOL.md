# Frozen novelty comparison: run and interpretation protocol

This is an exploratory, fully automated sensitivity test of the novelty
evaluator on the evaluator's 20 October 2026 export. Its 114 packets and a
snapshot-specific judge configuration are frozen locally. **No judge has run
on these packets yet.** The packet bundle contains unpublished
proposals and licensed abstracts, so its inputs and row-level results stay in
ignored local storage or an approved private channel.

**Execution decision, 23 October 2026:** The authors chose no API use for this
experiment. The API runner below remains dormant as a reproducibility design;
its preparation and fake-client tests involved no private model requests.
An installed Ollama model subsequently ran locally on five packets, but
produced no usable vote; see the local feasibility report.
Do not substitute a CLI or other external route for the rejected private-text
transmission. This small local run does not complete the frozen comparison.

## Question and design

For each of 38 proposals, compare three arm-hidden evidence packets under
one identical judge prompt and scoring rule:

| Arm | Papers shown to the judge |
| --- | --- |
| `semantic_five` | Five existing semantic neighbours |
| `semantic_plus_lexical` | The same five plus five lexical candidates from the deployed export |
| `semantic_plus_placebo` | The same five plus five papers outside both retrieved sets, paired to the lexical papers on publication year and approximate abstract length |

The two ten-paper packets place all five shared semantic papers in exactly
the same slots. Each lexical paper and its matched control occupy the same
remaining slot. The five-paper packet preserves the shared papers' relative
order, but their absolute positions differ. Journal, journal tier, DOI and
citations are absent from judge-facing packets. The control papers are
**not** established negatives; their purpose is to test whether results
change merely because the judge receives more text.

The primary contrast is lexical versus matched control at the same packet
length. Semantic five versus lexical is a useful secondary contrast but may
also reflect the extra five papers and their positions. Neither contrast
establishes that a novelty vote is scientifically correct. The supplied
corpus is a 2015–2025 CTO/TIM subset, so conclusions cannot assert a search
of all eligible prior art.

## Run protocol

1. Freeze the deployed judge's exact model ID, system/developer/user prompt,
   scoring schema, temperature and other decoding settings, tool access,
   retrieval catalog and code revision. Hash the final prompt bytes and a
   canonical saved configuration file separately with SHA-256. Use the
   **same configuration** for every arm and repeat. Do not
   tune it after seeing contrast results.
2. Regenerate the bundle from the source ZIP and verify its hashes. Give
   the judge only one line from `packets.jsonl` at a time. Keep
   `private_key.json` inaccessible to the judge and do not expose arm labels,
   proposal IDs, journal identities or scores. Shuffle execution order. If
   the judge can use outside search or tools, either disable them uniformly
   or record and hold them fixed; otherwise the evidence packets are no
   longer a controlled comparison.
3. Require a structured comparison of the research question, mechanism and
   discriminating predictions, with exact proposal and paper passages for
   every claimed overlap or difference. Allow `insufficient_evidence` with
   a reason and no vote. Preserve model failures as `failed` with a reason;
   never substitute a score. Store one result for every packet and repeat.
   A second repeat with the same frozen configuration is useful for judging
   stability, but a single complete repeat is analyzable.
4. Analyze only after all 114 packets in each repeat have a result. The
   analyzer checks coverage, shared configuration, paper references and
   passages. Invalid responses remain visible in status counts and receive
   no vote. It reports paired score differences per proposal, abstentions,
   failures and whether a paired quote from an added paper supports the explanation.

The prepared runner in `src/research0/novelty_ablation_run.py` freezes
`gpt-5.2-2025-12-11`, one prompt, a strict JSON schema, no tools, no response
storage, and `reasoning.effort=none`. It records a result for every attempted
packet, including failures and abstentions, plus a private raw-answer and
token-usage ledger. It can resume without repeating completed packet/repeat
pairs. The model snapshot supports structured output according to the
[official OpenAI model page](https://developers.openai.com/api/docs/models/gpt-5.2).
The runner is **prepared only**. Sending these packets would transmit
unpublished proposal text and licensed paper abstracts to the OpenAI API; do
not launch it without specific authorization for that payload and destination.
The exact planned destination is the OpenAI Responses API
(`https://api.openai.com/v1/responses`). The 114 requests each contain one
proposal's text, the frozen rubric, and five or ten source-paper titles,
years and abstracts (950 paper entries across all packets, with repeats).
They omit names, journals, DOI, internal paper IDs and the private arm key.
The packet file is about 1.8 MB, or roughly 342,000 input tokens under the
local `cl100k_base` estimate; actual billing may differ. Outputs, raw answers
and usage stay in ignored `reports/` locally. `store=false` is requested, but
the team should still confirm that this API data path is acceptable for the
unpublished and licensed material before executing it.

## Reproduce locally

Run from the repository root. These paths are ignored by Git.

```bash
PYTHONPATH=src python3 -m research0.novelty_ablation \
  --export inputs/evaluator_export_2026-10-20/raw/proposals_evaluations_export_2026-10-20.zip \
  --output reports/evaluator_export_2026-10-20/novelty_ablation

PYTHONPATH=src python3 -m research0.novelty_ablation_analysis \
  --bundle reports/evaluator_export_2026-10-20/novelty_ablation --check-bundle
```

The prepared bundle has 114 packets and a source ZIP hash of
`90d7fc7b4828b7df1dde5b0ea8fcbba8dc33e4b3c802d79b246070074841fdc2`.
The current bundle uses layout `fixed-semantic-slots-v2`; do not mix results
from an earlier layout. Its manifest records the hashes of both private files.

The no-network preparation command verifies the private bundle and writes
the exact prompt and configuration under ignored local storage:

```bash
PYTHONPATH=src python3 -m research0.novelty_ablation_run \
  --bundle reports/evaluator_export_2026-10-20/novelty_ablation \
  --output reports/evaluator_export_2026-10-20/novelty_ablation/run_gpt52_v1 \
  --prepare-only
```

The API execution path is documented for reproducibility, but is not the
selected route for this project run. If the team later changes that decision,
the same command without `--prepare-only` would submit one frozen repeat. It
requires `OPENAI_API_KEY` and an explicit
`--allow-private-api` launch flag; without that flag it refuses to send any
packet. `--max-new` supports a bounded
pilot and a later invocation resumes the remaining packets. The result file
will be `run_gpt52_v1/results.jsonl`. This command has **not** been run here.

After the judge has produced private `results.jsonl`, analyze it with:

```bash
PYTHONPATH=src python3 -m research0.novelty_ablation_analysis \
  --bundle reports/evaluator_export_2026-10-20/novelty_ablation \
  --results reports/evaluator_export_2026-10-20/novelty_ablation/results.jsonl \
  --output reports/evaluator_export_2026-10-20/novelty_ablation/analysis
```

One result line has this shape; the passages below are schematic and must
match passages in the supplied proposal text and paper abstract in a real
run (the checker ignores case and punctuation when matching):

```json
{"packet_id":"np_<ID FROM PACKET>","packets_sha256":"<SHA-256 FROM MANIFEST>","repeat":1,"model_id":"<EXACT MODEL ID>","prompt_sha256":"<64 LOWERCASE HEX CHARACTERS>","config_sha256":"<64 LOWERCASE HEX CHARACTERS>","rubric_version":"novelty-three-dimensions-ablation-v1","status":"scored","vote":6,"comparisons":[{"neighbour_id":"V1","question":{"assessment":"overlap","explanation":"Same empirical question.","proposal_passage":"<EXACT PROPOSAL PASSAGE>","neighbour_passage":"<EXACT ABSTRACT PASSAGE>"},"mechanism":{"assessment":"difference","explanation":"The causal path differs.","proposal_passage":"<EXACT PROPOSAL PASSAGE>","neighbour_passage":"<EXACT ABSTRACT PASSAGE>"},"predictions":{"assessment":"unclear","explanation":"The abstract does not report a comparable prediction.","proposal_passage":null,"neighbour_passage":null}}]}
```

Each dimension's `assessment` is one of `overlap`, `difference`, `unclear` or
`not_addressed`. A scored answer needs at least one comparison and one valid
proposal/paper passage pair. For `insufficient_evidence` or `failed`, set
`vote` to `null`, include a nonempty `reason`, and use an empty comparisons
array. The analyzer rejects duplicate or missing packet/repeat rows, results
from another packet bundle, and mixed model IDs, prompt hashes, configuration
hashes or rubric versions. It writes
`analysis/summary.json` and `analysis/private_cases.csv`; the latter contains
proposal IDs and must remain private.

## Reporting rule

Report the arm status counts before any score contrast. For each contrast,
report the number of proposals with paired scored repeats, the mean and
median vote difference, and the counts of positive, negative and unchanged
proposal-level differences. Report how often each ten-paper arm supported an
added-paper comparison with paired passages. If many runs abstain or fail, report that as the
finding. Describe the result as **evidence-packet sensitivity**, never
prior-art recall, validity or accuracy. A separate known-answer/paraphrase
benchmark and coverage check are needed to evaluate those claims.

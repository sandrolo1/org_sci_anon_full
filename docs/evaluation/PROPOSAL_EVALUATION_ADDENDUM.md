# Evidence grounded evaluation of artificial scientific proposals

## Recommendation for the Organization Science proposal

Add evaluation of the *evaluator* as a distinct part of the discovery study.
The central question is whether an automated system can identify a proposal's
specific contribution against traceable prior work while recognizing when
its evidence is too thin to support a verdict. This makes the project more
than a demonstration of agents generating ideas. It tests a bottleneck that
both artificial scientists and journals face: plausible claims of novelty
that may be rediscoveries, weak mechanisms, or plans that cannot test their
own predictions.

The prototype can run without human attention. Its output should remain an
exploratory screen until it has been checked independently. The human
validation already described in the v4 proposal remains the basis for any
eventual claim that scores measure scientific quality.

## What the current pilot tells us

The 20 October 2026 export contains 38 proposals scored on novelty,
methodology, and impact, searched against 10,050 CTO/TIM abstracts published
in 2015–2025. It is a pilot of the *system's behavior*, not a test of its
accuracy. An offline integrity audit matched 886 cited proposal/paper passage
pairs to the recorded proposal texts, paper abstracts, or paper cards. Three
hybrid novelty runs failed their quotation check. Of 35 completed hybrid
checks, 21 changed the novelty vote relative to the primary five-neighbor
score, including three shifts of at least two points. All five original
semantic neighbors remained in each scored hybrid packet, so the changes
are associated with additional evidence **and** a different comparison
format. They cannot yet be credited to lexical retrieval alone.

The agents cited 141 distinct paper DOI/proposal pairs in the 35 completed
hybrid cases. Their five semantic neighbors included 35 of these pairs;
their ten-paper hybrid packets included 43. This is a useful coverage check,
but a paper cited for background is not necessarily the closest prior art,
and agent-authored bibliographies are not independent ground truth. The
proposal methodology cards also label design as inferred in 31 of 38 cases,
data/sample and procedure as inferred in all 38, and measures as inferred in
37. Methodology scores should therefore describe the plausibility of a
*proposed test*, conditional on these unresolved design choices.

## A falsifiable evaluation design

1. **Freeze the search boundary.** Record the catalog version, included
   years, community filter, paper IDs, text recipe, embedding model, and
   prompts. Search the full eligible catalog for the declared question, not
   only one convenient cluster. Combine semantic and lexical candidates,
   deduplicate by canonical ID/DOI, and retain ranks from both methods. The
   current 10,050-paper set and 2025 cutoff must be displayed as coverage
   limits. A wider catalog or a later literature update is a new version.

2. **Judge claims against passages.** For novelty, compare the proposed
   question, causal mechanism, and discriminating predictions separately
   with directly quoted passages from candidate papers. A similar topic is
   neither proof of duplication nor proof of originality. Record agreement,
   difference, and what the paper does not state for each dimension. A quote
   must map to the exact paper and proposal text before a vote is accepted.

3. **Use criterion-specific evidence.** Methodology asks whether the design
   can distinguish the proposed mechanism from rivals, whether measures and
   units align, and which design choices are merely inferred. Impact has two
   reported components: potential theoretical/practical contribution, and
   realized uptake of similar prior papers. Citations and journal grades are
   contextual indicators of prior work, not a direct measure of the new
   proposal's worth. These components should not be collapsed invisibly.

4. **Allow a genuine no-verdict outcome.** Missing proposal detail, absent
   relevant passages, invalid quotations, unresolved paper identity,
   conflicting evidence, or a literature cutoff material to the claim should
   produce `insufficient_evidence` with a specific reason and coverage
   statement. This is an automated outcome, not a request for a person to
   review every case. A corpus-limited score can still be displayed when it
   is clearly scoped to the frozen corpus.

5. **Test the evaluator rather than tuning to its own scores.** Use repeated,
   arm-hidden runs of the same proposals under a single frozen rubric. The
   prepared ablation has 114 private packets: for each of 38 proposals, the
   semantic five, those five plus five lexical candidates, and those five
   plus five nonretrieved papers matched on year and approximate abstract
   length. The primary contrast is lexical versus control, both with ten
   papers; the five-paper arm shows the effect of adding text at all. Compare
   reference coverage, quoted support, abstention, score shifts, and
   run-to-run stability. The control papers are not known negatives, so this
   experiment measures sensitivity, not novelty accuracy.

The evaluator's 2026-paper experiment is a complementary out-of-time stress test.
Freeze the 2015–2025 literature, evaluation rules, and model before seeing
results. Hide journal names, grades, citation counts, DOI, and explicit venue
cues from the judge; record article type, field, publication date, and any
earlier preprint. Compare score distributions by journal tier only *after*
scoring, within field and article type, with uncertainty intervals. A
high-tier paper can be methodologically strong without being especially new,
and a lower-tier paper can be valuable. Venue is a noisy external proxy, not
an individual-paper truth label (see the [San Francisco Declaration on
Research Assessment](https://sfdora.org/read/)). Do not adjust the rubric
merely to make its scores reproduce the journal hierarchy.

## Paste-ready proposal text

> We will evaluate the discovery system's judgments as well as the questions
> it generates. For each proposal, a versioned retrieval process will search
> the declared eligible literature with both semantic and lexical methods.
> An evaluator will compare the proposal's question, mechanism, and
> discriminating predictions with cited passages from candidate papers and
> will record the search boundary, unresolved alternatives, and an
> `insufficient_evidence` outcome when a defensible judgment is not possible.
> Methodological assessment will ask whether the proposed design can test
> the mechanism rather than whether similar published methods merely exist.
> We will report a proposal's possible contribution separately from citation
> and venue indicators describing prior papers. To identify sensitivity to
> evidence selection, we will hold the proposals and evaluation rubric fixed
> while varying the retrieval packet, including a matched extra-paper
> control. We will then assess stability and evidence traceability before
> using independent expert judgments to test whether the scores correspond
> to research quality. A blinded study of 2026 papers across journal tiers
> will provide an additional, deliberately weak external check, not a
> substitute for article-level validation.

## What to request for the next empirical step

- The deployed hybrid scorer and its exact prompt/configuration version in
  Git, so the controlled ablation can run with one identical judge rubric.
- The full eligible catalog or a documented reason to restrict novelty
  searches to the current 10,050-paper subset.
- The evaluator's 2026-paper sample with title, abstract, DOI, publication and
  preprint dates, article type, field, journal tier, and the frozen selection
  rule. Keep the venue fields in a separate key that the evaluator cannot see.

The proposed application to journal screening should be framed as a research
question: can evidence-traceable screening detect rediscovery, unsupported
contribution claims, and untestable designs while preserving unconventional
but promising ideas? The current pilot establishes the infrastructure to
study that question; it does not yet show a reliable editorial filter.

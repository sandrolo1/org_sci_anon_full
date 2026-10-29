export function candidateRelations(papers) {
  const relations = [];
  for (let i = 0; i < papers.length; i += 1) {
    for (let j = i + 1; j < papers.length; j += 1) {
      const a = papers[i], b = papers[j];
      const sameLabel = a.construct.label.toLowerCase() === b.construct.label.toLowerCase();
      const sameMeaning = a.construct.meaning_key === b.construct.meaning_key;
      if (sameLabel && !sameMeaning) {
        relations.push({ type: 'Jingle candidate', ids: [a.id, b.id],
          reason: `Same label; different referents (${a.construct.referent} versus ${b.construct.referent}) and levels (${a.construct.level} versus ${b.construct.level}).` });
      }
      if (!sameLabel && sameMeaning) {
        relations.push({ type: 'Jangle candidate', ids: [a.id, b.id],
          reason: 'Different labels; the fixture assigns the same referent, level and meaning profile.' });
      }
    }
  }
  return relations;
}

export function auditCase(fixture) {
  const papers = new Map(fixture.papers.map(paper => [paper.id, paper]));
  const claimChecks = fixture.claims.map(claim => ({ ...claim,
    matched: Boolean(papers.get(claim.paper_id)?.passage.includes(claim.quote)),
  }));
  const uniqueIds = papers.size === fixture.papers.length;
  const validRelations = candidateRelations(fixture.papers).every(relation => relation.ids.every(id => papers.has(id)));
  const abstains = fixture.novelty_assessment.status === 'insufficient_evidence' &&
    !('vote' in fixture.novelty_assessment);
  return {
    same_paper_set: uniqueIds && Boolean(fixture.outputs['0'] && fixture.outputs.A),
    exact_claim_quotes: claimChecks.every(check => check.matched),
    candidate_relations_traceable: validRelations,
    novelty_abstains: abstains,
    claim_checks: claimChecks,
    passed: uniqueIds && claimChecks.every(check => check.matched) && validRelations && abstains,
  };
}

export function pilotFacts(pilot) {
  const baseline = pilot.batches.map(batch => batch.means['0']);
  const generalist = pilot.batches.map(batch => batch.means.A);
  const matched = pilot.batches.flatMap(batch => batch.matched_0_A);
  return {
    completedEpisodes: pilot.episodes,
    batches: pilot.batches.length,
    baselineToGeneralist: pilot.batches.map((batch, index) => Number((generalist[index] - baseline[index]).toFixed(3))),
    rankingReverses: generalist[0] > baseline[0] && generalist[1] < baseline[1],
    matchedPairs: matched.length,
    higher: matched.filter(pair => pair.difference_A_minus_0 > 0).length,
    lower: matched.filter(pair => pair.difference_A_minus_0 < 0).length,
    equal: matched.filter(pair => pair.difference_A_minus_0 === 0).length,
  };
}

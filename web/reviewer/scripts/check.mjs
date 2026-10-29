import assert from 'node:assert/strict';
import fs from 'node:fs';
import { auditCase, candidateRelations, pilotFacts } from '../src/logic.js';

const fixture = JSON.parse(fs.readFileSync(new URL('../public/case.json', import.meta.url), 'utf8'));
const pilot = JSON.parse(fs.readFileSync(new URL('../public/pilot.json', import.meta.url), 'utf8'));
const audit = auditCase(fixture);
assert.equal(audit.passed, true, 'The synthetic case must pass all evidence gates');
assert.equal(audit.claim_checks.length, 4);
assert.deepEqual(candidateRelations(fixture.papers).map(item => item.type).sort(),
  ['Jangle candidate', 'Jingle candidate']);
assert.deepEqual(fixture.papers.map(paper => paper.id), ['S1', 'S2', 'S3', 'S4']);
assert.equal(pilotFacts(pilot).completedEpisodes, 30);
assert.equal(pilotFacts(pilot).rankingReverses, true);
assert.deepEqual([pilotFacts(pilot).matchedPairs, pilotFacts(pilot).higher,
  pilotFacts(pilot).lower, pilotFacts(pilot).equal], [6, 2, 2, 2]);
assert.ok(pilot.batches.every(batch => batch.repetitions_per_arm === 3));
console.log('Reviewer artifact: fixture, evidence gates, conceptual relations and pilot summary verified.');

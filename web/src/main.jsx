import React, { useEffect, useMemo, useState } from 'react';
import { createRoot } from 'react-dom/client';
import './styles.css';

const CONDITIONS = { '0': 'Baseline', A: 'Generalist', B: 'Specialist', C: 'Encounter' };
const CRITERIA = [
  ['novelty', 'Novelty'],
  ['methodology', 'Methodology'],
  ['impact', 'Impact'],
];
const fmt = value => value == null || !Number.isFinite(Number(value)) ? '—' : Number(value).toFixed(1);
const fmt2 = value => value == null || !Number.isFinite(Number(value)) ? '—' : Number(value).toFixed(2);
const signed2 = value => value == null || !Number.isFinite(Number(value)) ? '—' : `${Number(value) >= 0 ? '+' : ''}${fmt2(value)}`;
const average = values => values.length ? values.reduce((sum, value) => sum + value, 0) / values.length : null;
const mean = (items, key) => average(items.map(item => item.votes?.[key]).filter(value => value != null && Number.isFinite(Number(value))).map(Number));
const conditionClass = condition => `condition-${condition || '0'}`;
const dateFromSource = source => source?.match(/\d{4}-\d{2}-\d{2}/)?.[0] || null;
const benchmarkIssues = (item, futureDates) => ({
  incomplete: item.status !== 'ok' || BENCHMARK_FIELDS.some(([key]) => item[key] == null),
  futureDate: futureDates.has(item.id),
});

function Mark() {
  return <div className="mark" aria-hidden="true"><span /><span /><span /><span /></div>;
}

function Badge({ children, tone = '' }) {
  return <span className={`badge ${tone}`}>{children}</span>;
}

function SourceDetails({ title, date, files }) {
  return <details className="source-details"><summary>{title} · {date || 'date unavailable'} · inspect source</summary>
    <dl>{files.map(([label, value]) => <div key={label}><dt>{label}</dt><dd>{value || 'Unavailable'}</dd></div>)}</dl>
    <p>These are frozen local exports. This page does not refresh from the forum or AWS.</p>
  </details>;
}

function ScorePlot({ items, selected, onSelect }) {
  const width = 360, height = 190, pad = 30;
  const xy = item => {
    const jitter = (item.id.charCodeAt(0) % 7 - 3) * 1.9;
    return [pad + (Number(item.votes.novelty) - 1) / 9 * (width - 2 * pad) + jitter,
      height - pad - (Number(item.votes.methodology) - 1) / 9 * (height - 2 * pad) - jitter];
  };
  return <div className="plot" aria-label="Novelty versus methodology scores">
    <div className="plot-title"><span>Score landscape</span><small>Each dot is one proposal</small></div>
    <svg viewBox={`0 0 ${width} ${height}`} role="img" aria-label="Scatter plot of novelty and methodology votes">
      {[3, 5, 7, 9].map(tick => <g key={tick}>
        <line x1={pad} x2={width - pad} y1={height - pad - (tick - 1) / 9 * (height - 2 * pad)} y2={height - pad - (tick - 1) / 9 * (height - 2 * pad)} className="plot-grid" />
        <text x="5" y={height - pad - (tick - 1) / 9 * (height - 2 * pad) + 4}>{tick}</text>
        <text x={pad + (tick - 1) / 9 * (width - 2 * pad) - 3} y={height - 7}>{tick}</text>
      </g>)}
      <line x1={pad} x2={pad} y1={pad} y2={height - pad} className="plot-axis" />
      <line x1={pad} x2={width - pad} y1={height - pad} y2={height - pad} className="plot-axis" />
      {items.map(item => { const [cx, cy] = xy(item); return <circle key={item.id} cx={cx} cy={cy} r={item.id === selected ? 7 : 4.5}
        className={`plot-dot ${conditionClass(item.condition)} ${item.id === selected ? 'selected' : ''}`}
        onClick={() => onSelect(item.id)} tabIndex="0" role="button" aria-label={`Open ${item.title}`}
        onKeyDown={event => { if (event.key === 'Enter' || event.key === ' ') onSelect(item.id); }}>
        <title>{item.title} · novelty {fmt(item.votes.novelty)} · methodology {fmt(item.votes.methodology)}</title>
      </circle>; })}
    </svg>
    <div className="plot-axis-labels"><span>Methodology ↑</span><span>Novelty →</span></div>
  </div>;
}

function ScoreCard({ label, value, featured = false }) {
  return <div className={`score-card ${featured ? 'featured' : ''}`}><span>{label}</span><strong>{fmt(value)}</strong><small>/ 10</small></div>;
}

function Candidate({ paper }) {
  const provenance = Array.isArray(paper.found_by) ? paper.found_by : [];
  return <div className="candidate">
    <div className="candidate-top"><span className="candidate-label">{paper.label || 'Paper'}</span><strong>{paper.title || 'Untitled paper'}</strong></div>
    <div className="candidate-meta">
      {provenance.map(source => <Badge key={source} tone={source === 'lexical' ? 'blue' : 'green'}>{source}</Badge>)}
      <span>{paper.journal || 'Journal unrecorded'}{paper.year ? ` · ${paper.year}` : ''}</span>
      {paper.ajg2024 && <span>AJG {paper.ajg2024}</span>}
      {paper.similarity != null && <span>similarity {fmt(paper.similarity)}</span>}
      {paper.citation_percentile != null && <span>citations {Math.round(paper.citation_percentile * 100)}th percentile</span>}
    </div>
  </div>;
}

function Comparison({ item, paper }) {
  const dimensions = item.question ? [['Question', item.question], ['Mechanism', item.mechanism], ['Predictions', item.predictions]] : [];
  return <div className="comparison">
    <div className="comparison-head"><strong>{item.neighbour_id || 'Paper'}{paper?.title ? ` · ${paper.title}` : ''}</strong>{item.assessment && <Badge>{item.assessment.replaceAll('_', ' ')}</Badge>}</div>
    {item.explanation && <p>{item.explanation}</p>}
    {!dimensions.length && (item.proposal_passage || item.neighbour_passage) && <div className="quote-pair">
      {item.proposal_passage && <blockquote><span>Proposal</span>“{item.proposal_passage}”</blockquote>}
      {item.neighbour_passage && <blockquote><span>Prior paper</span>“{item.neighbour_passage}”</blockquote>}
    </div>}
    {dimensions.map(([name, detail]) => detail && <div className="dimension" key={name}>
      <div className="dimension-head"><span>{name}</span><b>{detail.assessment?.replaceAll('_', ' ') || ''}</b></div>
      {detail.explanation && <p>{detail.explanation}</p>}
      {(detail.proposal_passage || detail.neighbour_passage) && <div className="quote-pair">
        {detail.proposal_passage && <blockquote><span>Proposal</span>“{detail.proposal_passage}”</blockquote>}
        {detail.neighbour_passage && <blockquote><span>Prior paper</span>“{detail.neighbour_passage}”</blockquote>}
      </div>}
    </div>)}
  </div>;
}

function Judge({ judges, papers = [] }) {
  const entries = Object.entries(judges || {});
  const paperByLabel = Object.fromEntries(papers.map(paper => [paper.label, paper]));
  if (!entries.length) return <p className="muted">No judge rationale is stored.</p>;
  return entries.map(([role, judge]) => judge && <section className="judge" key={role}>
    <div className="judge-head"><span>{role.replace('_', ' ')}</span><strong>{fmt(judge.vote)} / 10</strong></div>
    <p>{judge.rationale || 'No rationale stored.'}</p>
    {!!judge.comparisons?.length && <details className="comparisons"><summary>Inspect cited comparisons <span>{judge.comparisons.length}</span></summary>
      <div className="comparison-list">{judge.comparisons.map((comparison, index) => <Comparison key={`${comparison.neighbour_id}-${index}`} item={comparison} paper={paperByLabel[comparison.neighbour_id]} />)}</div>
    </details>}
  </section>);
}

function Card({ title, record }) {
  const card = record?.card;
  if (!card) return <p className="muted">No {title.toLowerCase()} is stored.</p>;
  const fields = Object.entries(card).filter(([, value]) => value && typeof value === 'object' && !Array.isArray(value));
  return <section className="card-section"><div className="section-heading"><h3>{title}</h3><span>Agent-authored</span></div>
    <p className="section-note">“Inferred” means the agent chose or interpreted this detail from the proposal. It is a plan, not an observed study.</p>
    <div className="field-grid">{fields.map(([name, value]) => <div className="field" key={name}>
      <div><strong>{name.replaceAll('_', ' ')}</strong><Badge tone={value.status === 'inferred' ? 'amber' : value.status === 'missing' ? 'rose' : 'green'}>{value.status}</Badge></div>
      <p>{value.text || 'Not stated'}</p>
    </div>)}</div>
    {!!card.info_gaps?.length && <details className="gaps"><summary>Open design questions ({card.info_gaps.length})</summary><ul>{card.info_gaps.map((gap, index) => <li key={index}>{gap}</li>)}</ul></details>}
  </section>;
}

function Followup({ item }) {
  const followup = item.followup;
  if (!followup || followup.kind === 'none') return null;
  return <section className={`followup followup-${followup.kind}`} aria-label="Follow-up action">
    <div><span className="eyebrow">Evidence follow-up</span><h3>{followup.label}</h3></div>
    <p>{followup.action}</p><small>{followup.detail}</small>
  </section>;
}

function NoveltyReview({ item }) {
  const primary = item.criteria?.novelty || {};
  const hybrid = primary.hybrid || {};
  const primaryPapers = primary.neighbours || [];
  const primaryIds = new Set(primaryPapers.map(paper => paper.id));
  const hybridPapers = hybrid.candidates || [];
  const added = hybridPapers.filter(paper => !primaryIds.has(paper.id));
  const delta = item.followup?.delta;
  return <div className="criterion-page novelty-review">
    <div className="criterion-intro"><span className="eyebrow">Retrieval sensitivity</span><h3>Novelty comparison</h3><p>Two model judgments on the same proposal with different evidence packets. A change in score does not show which judgment is more accurate.</p></div>
    <div className="vote-comparison">
      <div><span>Primary · 5 semantic papers</span><strong>{fmt(primary.vote)}</strong><small>Stored novelty vote</small></div>
      <div><span>Hybrid · 5 semantic + 5 keyword</span><strong>{hybrid.status === 'scored' ? fmt(hybrid.vote) : 'Failed'}</strong><small>{hybrid.status === 'scored' ? `${delta > 0 ? '+' : ''}${fmt(delta)} versus primary` : 'No valid vote retained'}</small></div>
    </div>
    {hybrid.status === 'failed' && <div className="notice rose"><strong>Validation failure.</strong> {hybrid.reason || 'No valid hybrid judgment was retained.'}</div>}
    {hybrid.status === 'scored' && <p className="retrieval-note">All {item.retrieval_diagnostics?.primary_retained ?? primaryPapers.length} primary papers remained in the hybrid packet. Its judge made structured comparisons with {item.retrieval_diagnostics?.added_compared ?? '—'} of the {added.length} added papers. This shows evidence use, not what caused the score change.</p>}
    <div className="two-up judgment-columns">
      <div><div className="section-heading"><h4>Primary reasoning</h4><span>{primaryPapers.length} papers</span></div><Judge judges={primary.judges} papers={primaryPapers} /></div>
      <div><div className="section-heading"><h4>Hybrid reasoning</h4><span>{hybrid.status === 'scored' ? `${hybridPapers.length} papers` : 'Invalid response'}</span></div>
        {hybrid.status === 'scored' ? <Judge judges={hybrid.judges} papers={hybridPapers} /> : <p className="muted">The hybrid judgment did not pass its quotation check. The primary judgment remains available; the failed hybrid attempt should be rerun rather than treated as a low score.</p>}</div>
    </div>
    <div className="retrieval-sets"><div className="section-heading"><h4>Evidence packets</h4><span>{added.length} additional papers in hybrid</span></div>
      <details open><summary>Primary semantic five</summary><div className="candidate-list">{primaryPapers.map((paper, index) => <Candidate key={`${paper.id}-${index}`} paper={paper} />)}</div></details>
      <details><summary>Additional keyword candidates ({added.length})</summary><div className="candidate-list">{added.length ? added.map((paper, index) => <Candidate key={`${paper.id}-${index}`} paper={paper} />) : <p className="muted">No additional candidate list was retained with this result.</p>}</div></details>
    </div>
  </div>;
}

function Criterion({ item, name, label }) {
  const criterion = item.criteria?.[name] || {};
  const methodCard = item.cards?.method?.card || {};
  const inferred = ['design', 'measures', 'data_sample', 'procedure'].filter(field => methodCard[field]?.status === 'inferred');
  return <div className="criterion-page">
    <div className="criterion-intro"><span className="eyebrow">Evidence review</span><h3>{label}</h3><p>{name === 'methodology' ? 'Can the proposed design test its mechanism against alternatives?' : 'What might this proposal contribute? Citations below describe prior papers, not the proposal’s future impact.'}</p></div>
    <div className="criterion-score"><span>Stored model vote</span><strong>{fmt(criterion.vote)}</strong><small>Exploratory · 1 to 10</small></div>
    {name === 'methodology' && <div className="method-note"><strong>{inferred.length} of 4 core plan fields inferred</strong><span>{inferred.length ? inferred.join(', ').replaceAll('_', ' ') : 'The card marks all four core fields as extracted.'} · These are proposed design choices, not an executed method.</span></div>}
    <div className="two-up"><div><div className="section-heading"><h4>Judge's reasoning</h4><span>{criterion.judges_answered || Object.keys(criterion.judges || {}).length} judge(s)</span></div><Judge judges={criterion.judges} papers={criterion.neighbours || []} /></div>
      <div><div className="section-heading"><h4>Retrieved papers</h4><span>{criterion.neighbours?.length || 0} candidates</span></div><div className="candidate-list">{(criterion.neighbours || []).map((paper, index) => <Candidate key={`${paper.id}-${index}`} paper={paper} />)}</div></div></div>
    {name === 'methodology' && <Card title="Proposal methodology card" record={item.cards?.method} />}
    {name === 'impact' && <Card title="Proposal impact card" record={item.cards?.impact} />}
  </div>;
}

function Detail({ item, integrity }) {
  const [tab, setTab] = useState('overview');
  useEffect(() => setTab('overview'), [item?.id]);
  if (!item) return <article className="detail empty-state"><h2>Choose a proposal</h2><p>Select a result to inspect its scores and evidence.</p></article>;
  const tabs = [['overview', 'Overview'], ['novelty', 'Novelty comparison'], ['methodology', 'Methodology'], ['impact', 'Impact']];
  return <article className="detail">
    <div className="detail-top"><div className="detail-kicker"><span className={`condition-mark ${conditionClass(item.condition)}`}>{item.condition}</span>{CONDITIONS[item.condition] || item.condition} <span className="dot-separator">•</span> {item.trigger || 'Research question'}</div><span className="detail-date">{item.scored_on?.slice(0, 10) || 'Date unavailable'}</span></div>
    <h2>{item.title}</h2><div className="detail-meta"><Badge>{item.episode || 'Forum episode'}</Badge><Badge>{item.schema || 'Evaluation rubric'}</Badge>{item.followup?.kind !== 'none' && <Badge tone="amber">{item.followup?.label}</Badge>}</div>
    {!!item.warnings?.length && <div className="snapshot-warning">{item.warnings.join(' ')}</div>}
    <Followup item={item} />
    <nav className="tabs" aria-label="Proposal details">{tabs.map(([key, label]) => <button key={key} type="button" className={tab === key ? 'active' : ''} onClick={() => setTab(key)}>{label}</button>)}</nav>
    {tab === 'overview' ? <div className="summary-page">
      <div className="score-grid"><ScoreCard label="Mean of three" value={item.mean} featured />{CRITERIA.map(([key, label]) => <ScoreCard key={key} label={label} value={item.votes?.[key]} />)}</div>
      <div className="overview-next"><div><span className="eyebrow">First check</span><h3>Compare the novelty evidence</h3><p>Read both judges’ reasoning alongside the papers each could see. The hybrid result is a sensitivity check, not a replacement score.</p></div><button type="button" onClick={() => setTab('novelty')}>Open comparison →</button></div>
      <details className="abstract-details"><summary>Read proposal abstract</summary><div className="abstract-block"><p>{item.abstract || 'No abstract in export.'}</p></div></details>
      <div className="summary-lower"><div><span className="eyebrow">Interpretation</span><h3>Exploratory model judgments</h3><p>The three scores cover different questions. They have not been validated against independent judgments of research quality.</p></div><div className="coverage"><strong>{integrity?.corpus_papers?.toLocaleString() ?? '—'}</strong><span>papers in the exported novelty corpus</span><small>2015–2025 · CTO/TIM subset · 2026 papers absent</small></div></div>
    </div> : tab === 'novelty' ? <NoveltyReview item={item} /> : <Criterion item={item} name={tab} label={CRITERIA.find(([key]) => key === tab)?.[1]} />}
  </article>;
}

const BENCHMARK_FIELDS = [
  ['novelty_first', 'First novelty'], ['novelty_hybrid', 'Hybrid novelty'],
  ['methodology', 'Methodology'], ['impact', 'Impact'], ['mean_of_three', 'Three-score mean'],
];
const RETRIEVAL_CONTROLS = [
  ['full_abstract', 'Full abstract', 'The source abstract, unchanged'],
  ['title_only', 'Title only', 'The source title, unchanged'],
  ['lead_40_words', 'First 40 words', 'Extracted from the source abstract'],
  ['middle_20_words', 'Middle 20 words', 'Extracted from the source abstract'],
  ['lead_20_words', 'First 20 words', 'Extracted from the source abstract'],
  ['lead_20_drop_rarest_quarter', 'Remove rarest quarter', 'Delete one quarter of unique lexical clues'],
  ['lead_20_drop_rarest_half', 'Remove rarest half', 'Delete half of unique lexical clues'],
  ['lead_20_drop_rarest_three_quarters', 'Remove rarest three quarters', 'Severe adversarial deletion of lexical clues'],
];
const recovered = (row, field) => row ? `${row[field].toLocaleString()} / ${row.n.toLocaleString()}` : '—';
const recoveredPercent = (row, field) => row && row.n ? 100 * row[field] / row.n : 0;

function RetrievalView({ data, error }) {
  if (error || !data) return <main><div className="retrieval-unavailable"><h1>Retrieval diagnostic unavailable</h1><p>{error || 'Prepare the local aggregate report first.'}</p><code>cd web && npm run prepare:data</code><p>This view needs the offline aggregate report; it does not run model calls.</p></div></main>;
  const { metrics, coverage, source, sample_size: sampleSize } = data;
  const stress = RETRIEVAL_CONTROLS.slice(4);
  return <main className="retrieval-main">
    <section className="workbench-top"><div><span className="eyebrow">Offline control · BM25 lexical retrieval</span><h1>Known-source retrieval</h1><p>Can the index recover an article when a query is mechanically taken from that same article?</p></div><span className="snapshot-badge benchmark-snapshot">Aggregate only</span></section>
    <div className="retrieval-definition"><strong>What is measured</strong><p>The source is already among {coverage.catalog_rows.toLocaleString()} indexed papers. For each of {sampleSize.toLocaleString()} sampled articles, the check asks whether its own article appears in the top 1, 5, or 20 BM25 results. This is a retrieval plumbing test within the catalog.</p></div>
    {data.vector_self_check && <div className="retrieval-vector"><div><span className="eyebrow">Separate index-alignment check</span><strong>{recovered(data.vector_self_check, 'source_at_1')} at rank 1</strong></div><p>Each paper’s own stored vector retrieves itself. This checks vector/source alignment only. It does not embed an altered query or test whether semantic search rescues word deletion.</p></div>}
    <details className="source-details retrieval-source"><summary>Source and reproducibility · inspect</summary><dl>
      <div><dt>Aggregate report</dt><dd>{source.report_name}</dd></div><div><dt>Report SHA-256</dt><dd>{source.report_sha256}</dd></div>
      <div><dt>Source archive SHA-256</dt><dd>{source.archive_sha256}</dd></div><div><dt>Local archive check</dt><dd>{source.local_archive_verified ? 'Checksum verified' : 'Archive unavailable here; checksum not rechecked'}</dd></div>
      <div><dt>Method</dt><dd>{data.method}</dd></div><div><dt>Sample seed</dt><dd>{data.sample_seed}</dd></div>
      <div><dt>Eligible abstracts</dt><dd>{coverage.eligible_unique_abstracts.toLocaleString()}</dd></div><div><dt>Excluded short, missing or duplicate</dt><dd>{coverage.excluded_short_missing_or_duplicate.toLocaleString()}</dd></div>
    </dl><p>The generated browser file contains aggregate counts only. Rerun the offline benchmark to refresh the source report.</p></details>
    <section className="retrieval-panel"><div className="benchmark-section-head"><div><span className="eyebrow">Word-deletion stress test</span><h2>Source recovered in top 20</h2></div><p>The first 20 words are extractive. Later rows mechanically remove the rarest terms; this is not paraphrasing.</p></div>
      <div className="retrieval-bars">{stress.map(([key, label, note]) => { const row = metrics[key]; const percent = recoveredPercent(row, 'source_at_20'); return <div className="retrieval-bar-row" key={key}><div><strong>{label}</strong><span>{note}</span></div><div className="retrieval-bar-track" role="img" aria-label={`${label}: ${recovered(row, 'source_at_20')} recovered in top 20`}><span style={{ width: `${percent}%` }} /></div><b>{recovered(row, 'source_at_20')} <small>({fmt(percent)}%)</small></b></div>; })}</div>
      <p className="benchmark-caveat">Removing half of the rare terms gives {recovered(metrics.lead_20_drop_rarest_half, 'source_at_20')} at 20; removing three quarters gives {recovered(metrics.lead_20_drop_rarest_three_quarters, 'source_at_20')}. This diagnoses dependence on distinctive words in an adversarial extractive control.</p>
    </section>
    <section className="retrieval-panel"><div className="benchmark-section-head"><div><span className="eyebrow">All controls</span><h2>Where the known source ranks</h2></div><p>Counts, not model judgment scores. “Positive” means BM25 assigned the source a score above zero.</p></div>
      <div className="benchmark-table-wrap"><table className="benchmark-table retrieval-table"><thead><tr><th>Query control</th><th>Positive</th><th>Top 1</th><th>Top 5</th><th>Top 20</th><th>Median rank when scored</th></tr></thead><tbody>{RETRIEVAL_CONTROLS.map(([key, label]) => { const row = metrics[key]; return <tr key={key}><th>{label}</th><td>{recovered(row, 'positive_source_score')}</td><td>{recovered(row, 'source_at_1')}</td><td>{recovered(row, 'source_at_5')}</td><td>{recovered(row, 'source_at_20')}</td><td>{row?.median_source_rank_when_scored ?? '—'}</td></tr>; })}</tbody></table></div>
    </section>
    <section className="retrieval-limits"><div><span className="eyebrow">Interpretation boundary</span><h2>What this does not establish</h2></div><ul><li>Every query was taken from the article being sought. The test does not measure paraphrase retrieval or unseen questions.</li><li>The target article is in the index by construction. This does not establish full prior-art coverage.</li><li>Finding a source does not show that a judge correctly compares its question, mechanism, or predictions with a proposal.</li><li>These controls test the lexical BM25 component. They do not validate hybrid ranking or substantive novelty scores.</li></ul></section>
  </main>;
}

function BenchmarkView({ data, error }) {
  const [grade, setGrade] = useState('All');
  const [review, setReview] = useState('all');
  const [query, setQuery] = useState('');
  const [sort, setSort] = useState('mean_of_three');
  const [selectedId, setSelectedId] = useState(null);
  if (error || !data) return <main><div className="load-state"><h1>2026 benchmark unavailable</h1><p>{error || 'Preparing the private benchmark data.'}</p><code>cd web && npm run prepare:data</code></div></main>;
  const { summary, items, source } = data;
  const datedLater = new Set(summary.future_online_dates);
  const issueCount = items.filter(item => {
    const issues = benchmarkIssues(item, datedLater);
    return issues.incomplete || issues.futureDate;
  }).length;
  const displayed = items.filter(item => (grade === 'All' || item.grade === grade) &&
    (review === 'all' || (() => {
      const issues = benchmarkIssues(item, datedLater);
      return review === 'attention' ? issues.incomplete || issues.futureDate : issues[review];
    })()) &&
    (!query || `${item.title} ${item.journal} ${item.doi}`.toLowerCase().includes(query.toLowerCase().trim())))
    .sort((a, b) => (b[sort] == null ? -Infinity : Number(b[sort])) - (a[sort] == null ? -Infinity : Number(a[sort])) || a.title.localeCompare(b.title));
  const selected = displayed.find(item => item.id === selectedId) || displayed[0];
  const contrasts = summary.four_star_vs_other;
  const selectedIssues = selected ? benchmarkIssues(selected, datedLater) : null;
  const missingScores = selected ? BENCHMARK_FIELDS.filter(([key]) => selected[key] == null).map(([, label]) => label) : [];
  const unavailableCounts = Object.fromEntries(BENCHMARK_FIELDS.map(([key]) => [key, items.filter(item => item[key] == null).length]));
  return <main className="benchmark-main">
    <section className="workbench-top"><div><span className="eyebrow">{summary.as_of} · Frozen export</span><h1>2026 paper benchmark</h1><p>Explore {summary.papers} paper records ({summary.completed} with the three primary scores) and check what the journal-tier comparison actually measures.</p></div><span className="snapshot-badge benchmark-snapshot">Private local snapshot</span></section>
    <SourceDetails title="Benchmark source" date={source?.exported_at} files={[["CSV", source?.csv_name], ["CSV SHA-256", source?.csv_sha256], ["JSON", source?.json_name], ["JSON SHA-256", source?.json_sha256], ["Run ID", summary.run_id], ["Model", summary.model]]} />
    <section className="benchmark-facts" aria-label="Benchmark facts">
      <div><strong>{summary.papers}</strong><span>paper records</span><small>{summary.completed} core complete · {items.filter(item => benchmarkIssues(item, datedLater).incomplete).length} missing a displayed score · {summary.sampling.pool} eligible</small></div>
      <div><strong>{summary.four_star_papers}</strong><span>four-star papers</span><small>across {summary.four_star_journals} journals</small></div>
      <div><strong>{signed2(contrasts.impact.difference)}</strong><span>impact gap</span><small>four-star versus other papers</small></div>
      <div><strong>{signed2(contrasts.methodology.difference)}</strong><span>methodology gap</span><small>small in this sample</small></div>
    </section>
    <section className="benchmark-panel"><div className="benchmark-section-head"><div><span className="eyebrow">Criterion check</span><h2>The difference is concentrated in impact</h2></div><p>Mean scores, 0–10. The final column resamples journals within each group; it is exploratory.</p></div>
      <div className="benchmark-table-wrap"><table className="benchmark-table"><thead><tr><th>Criterion</th><th>4★ ({summary.four_star_papers})</th><th>Other ({summary.other_papers})</th><th>Gap</th><th>95% journal bootstrap interval</th></tr></thead><tbody>{BENCHMARK_FIELDS.map(([key, label]) => { const row = contrasts[key]; const fourN = items.filter(item => item.grade === '4*' && item[key] != null).length; const otherN = items.filter(item => item.grade !== '4*' && item[key] != null).length; return <tr key={key} className={key === 'impact' ? 'strongest' : ''}><th>{label}</th><td>{fmt2(row.four_star_mean)} <small>n={fourN}</small></td><td>{fmt2(row.other_mean)} <small>n={otherN}</small></td><td>{signed2(row.difference)}</td><td>{Array.isArray(row.journal_cluster_bootstrap_95pct) ? row.journal_cluster_bootstrap_95pct.map(fmt2).join(' to ') : 'Unavailable'}</td></tr>; })}</tbody></table></div>
      <p className="benchmark-caveat">The three-score mean uses first novelty, methodology and impact. Hybrid novelty is separate. Missing scores: {BENCHMARK_FIELDS.filter(([key]) => unavailableCounts[key]).map(([key, label]) => `${label} ${unavailableCounts[key]}`).join(' · ') || 'none'}. Each row shows its actual denominator. Journal grade is a group comparison, not a correct-answer label for any paper. The five grade levels do not show a steady upward pattern on every score.</p>
      {contrasts.impact.leave_one_journal_out && <p className="benchmark-caveat">Leaving out each of {contrasts.impact.leave_one_journal_out.journals_tested} sampled journals in turn gives an impact gap from {signed2(contrasts.impact.leave_one_journal_out.minimum_gap)} to {signed2(contrasts.impact.leave_one_journal_out.maximum_gap)}. {contrasts.impact.leave_one_journal_out.all_gaps_positive ? 'The point gap stays positive each time.' : 'The point gap does not stay positive in every omission.'} This checks single-journal influence, not scoring accuracy.</p>}
    </section>
    <section className="benchmark-browser"><div className="benchmark-browser-head"><div><span className="eyebrow">Paper register</span><h2>Inspect the sample <span>{displayed.length} of {items.length}</span></h2></div><div className="benchmark-controls"><label className="search"><span aria-hidden="true">⌕</span><input value={query} onChange={event => setQuery(event.target.value)} placeholder="Search title, journal or DOI" aria-label="Search benchmark papers" /></label><label>Grade <select value={grade} onChange={event => setGrade(event.target.value)}><option>All</option>{['4*', '4', '3', '2', '1'].map(value => <option key={value}>{value}</option>)}</select></label><label>Sort <select value={sort} onChange={event => setSort(event.target.value)}>{BENCHMARK_FIELDS.map(([key, label]) => <option key={key} value={key}>{label}</option>)}</select></label></div></div>
      <div className="benchmark-review-filters" role="group" aria-label="Review paper records">{[['all', 'All records', items.length], ['attention', 'Needs verification', issueCount], ['incomplete', 'Incomplete score', items.filter(item => benchmarkIssues(item, datedLater).incomplete).length], ['futureDate', 'Date to verify', datedLater.size]].map(([key, label, count]) => <button type="button" key={key} className={review === key ? 'active' : ''} aria-pressed={review === key} onClick={() => setReview(key)}>{label} <span>{count}</span></button>)}</div>
      <div className="benchmark-grid"><div className="benchmark-list">{displayed.length ? displayed.map(item => { const issues = benchmarkIssues(item, datedLater); return <button type="button" key={item.id} className={selected?.id === item.id ? 'active' : ''} aria-current={selected?.id === item.id ? 'true' : undefined} onClick={() => setSelectedId(item.id)}><span className="benchmark-list-head"><b>AJG {item.grade}</b><span>{fmt(item.mean_of_three)} mean</span></span><strong>{item.title}</strong><small>{item.journal}</small>{issues.incomplete && <em>Incomplete score</em>}{issues.futureDate && <em>Date to verify</em>}</button>; }) : <p className="no-results">No papers match these filters. Try All records or clear the search.</p>}</div>
      <article className="benchmark-paper">{selected ? <><span className="eyebrow">Paper record · {selected.id}</span><h3>{selected.title}</h3><p className="benchmark-paper-meta">{selected.journal} · AJG {selected.grade} · recorded online date {selected.online_date}</p>{selectedIssues.futureDate && <p className="benchmark-date-flag">This recorded date is after {summary.as_of}. Confirm whether it is the first online date or a future issue date.</p>}{selectedIssues.incomplete && <p className="benchmark-date-flag">Unavailable score: {missingScores.join(', ') || 'one or more required evaluations'}. Treat the dash below as missing, not zero. The table above uses only available scores for each criterion.</p>}
        <div className="benchmark-score-grid">{BENCHMARK_FIELDS.map(([key, label]) => <div key={key}><span>{label}</span><strong>{fmt(selected[key])}</strong></div>)}</div><p className="benchmark-paper-note">Closest abstract similarity {selected.closest_similarity == null ? 'unavailable' : Number(selected.closest_similarity).toFixed(2)}. The model’s scores describe this paper under the current rubric; the journal tier does not verify their accuracy.</p><p className="benchmark-paper-note">{selected.judged_twice ? `Double judged · gap ${fmt(selected.gap_between_copies)}` : 'Single judge set for this paper'}. This export contains score records, not the underlying per-paper judge rationales or cited passages.</p>{selected.doi && <a className="benchmark-doi" href={`https://doi.org/${encodeURIComponent(selected.doi)}`} target="_blank" rel="noreferrer">Open DOI ↗</a>}</> : <p>Choose a paper. No records match the current filters.</p>}</article></div>
    </section>
    <section className="benchmark-limits"><div><span className="eyebrow">Interpretation</span><h2>What to check before sharing</h2></div><ul>{summary.previous_export && <li>{summary.previous_export.reused} of these papers also appear in the earlier {summary.previous_export.papers}-paper export; {summary.previous_export.new} are newly included. This is an expanded sample, not an independent replication.</li>}<li>{summary.source_versions ? `${summary.source_versions.versions.submitted} of ${summary.source_versions.papers} PDFs are submitted manuscripts; evaluated text can differ from the final article.` : 'Many PDF sources are manuscripts; evaluated text can differ from the final article.'}</li><li>{summary.future_online_dates.length} recorded online dates fall after the export date and need first-publication checks.</li><li>The export records a {summary.knowledge_cutoff} model cutoff; the evaluator's email says November 2025.</li><li>{summary.usage.failed_calls} failed model attempts were logged before {summary.completed} fully scored rows. Review the attempt ledger and causes.</li></ul></section>
  </main>;
}

function App() {
  const [view, setView] = useState('proposals');
  const [benchmark, setBenchmark] = useState(null);
  const [benchmarkError, setBenchmarkError] = useState('');
  const [retrieval, setRetrieval] = useState(null);
  const [retrievalError, setRetrievalError] = useState('');
  const [data, setData] = useState(null);
  const [error, setError] = useState('');
  const [query, setQuery] = useState('');
  const [condition, setCondition] = useState('All');
  const [focus, setFocus] = useState('followup');
  const [sort, setSort] = useState('priority');
  const [selectedId, setSelectedId] = useState(null);
  useEffect(() => { fetch('/data/evaluations.json', { cache: 'no-store' }).then(response => {
    if (!response.ok) throw new Error('The local evaluation data has not been prepared.');
    return response.json();
  }).then(payload => { setData(payload); setSelectedId(payload.items?.[0]?.id || null); }).catch(err => setError(err.message)); }, []);
  useEffect(() => { fetch('/data/benchmark.json', { cache: 'no-store' }).then(response => {
    if (!response.ok) throw new Error('Place the private 2026 CSV and JSON in inputs/evaluation_2026/exports, then prepare the app data.');
    return response.json();
  }).then(setBenchmark).catch(err => setBenchmarkError(err.message)); }, []);
  useEffect(() => { fetch('/data/retrieval.json', { cache: 'no-store' }).then(response => {
    if (!response.ok) throw new Error('Generate the offline aggregate report, then run npm run prepare:data.');
    return response.json();
  }).then(setRetrieval).catch(err => setRetrievalError(err.message)); }, []);
  const items = data?.items || [];
  const filtered = useMemo(() => items.filter(item =>
    (condition === 'All' || item.condition === condition) &&
    (focus === 'all' || (focus === 'followup' && item.followup?.kind !== 'none') ||
      (focus === 'failed' && item.followup?.kind === 'rerun') ||
      (focus === 'shifted' && item.integrity_flags?.includes('hybrid:shift_at_least_two'))) &&
    (!query || `${item.title} ${item.abstract}`.toLowerCase().includes(query.toLowerCase().trim()))
  ).sort((a, b) => {
    if (sort === 'novelty') return (b.votes?.novelty ?? -1) - (a.votes?.novelty ?? -1) || (a.title || '').localeCompare(b.title || '');
    if (sort === 'shift') return Math.abs(b.followup?.delta ?? 0) - Math.abs(a.followup?.delta ?? 0) || (a.title || '').localeCompare(b.title || '');
    const rank = item => item.followup?.kind === 'rerun' ? 0 : item.followup?.kind === 'inspect' ? 1 : 2;
    return rank(a) - rank(b) || Math.abs(b.followup?.delta ?? 0) - Math.abs(a.followup?.delta ?? 0) || (a.title || '').localeCompare(b.title || '');
  }), [items, condition, focus, query, sort]);
  const selected = filtered.find(item => item.id === selectedId) || filtered[0] || null;
  useEffect(() => { if (filtered.length && !filtered.some(item => item.id === selectedId)) setSelectedId(filtered[0].id); }, [filtered, selectedId]);
  const counts = data?.integrity?.counts || {};
  const proposalDate = dateFromSource(data?.source) || data?.items?.[0]?.scored_on?.slice(0, 10);
  const quotePairs = ['novelty', 'methodology', 'impact', 'hybrid'].reduce((sum, key) => sum + (counts[`${key}_quote_pairs`] || 0), 0);
  const focusOptions = [
    { key: 'followup', label: 'Follow-up', count: items.filter(item => item.followup?.kind !== 'none').length, note: 'Cases with mechanical flags' },
    { key: 'failed', label: 'Failed checks', count: counts.hybrid_failed || 0, note: 'No valid hybrid vote' },
    { key: 'shifted', label: 'Large shifts', count: counts['flag_hybrid:shift_at_least_two'] || 0, note: 'Novelty change ≥2' },
    { key: 'all', label: 'All proposals', count: items.length, note: 'Browse the export' },
  ];
  if (view === 'proposals' && error) return <div className="load-state"><Mark /><h1>Evaluation data is not ready</h1><p>{error}</p><code>cd web && npm run prepare:data</code></div>;
  if (view === 'proposals' && !data) return <div className="load-state"><Mark /><h1>Loading the evaluation atlas</h1></div>;
  return <div className="app">
    <header className="site-header"><div className="header-inner"><div className="identity"><Mark /><div><strong>Discovery Forum</strong><span>Evaluation workbench</span></div></div><div className="header-right"><span>{view === 'benchmark' ? `${benchmark?.summary?.as_of || 'Date unavailable'} benchmark` : view === 'retrieval' ? 'Offline retrieval control' : `${proposalDate || 'Date unavailable'} export`}</span><span className="snapshot-badge"><i /> Private local snapshot</span></div></div></header>
    <nav className="workbench-nav" aria-label="Workbench views"><button type="button" className={view === 'proposals' ? 'active' : ''} aria-current={view === 'proposals' ? 'page' : undefined} onClick={() => setView('proposals')}>Proposal evaluations</button><button type="button" className={view === 'benchmark' ? 'active' : ''} aria-current={view === 'benchmark' ? 'page' : undefined} onClick={() => setView('benchmark')}>2026 paper benchmark <span>{benchmark?.summary?.papers ?? '—'}</span></button><button type="button" className={view === 'retrieval' ? 'active' : ''} aria-current={view === 'retrieval' ? 'page' : undefined} onClick={() => setView('retrieval')}>Retrieval checks <span>{retrieval?.sample_size ?? '—'}</span></button></nav>
    {view === 'benchmark' ? <BenchmarkView data={benchmark} error={benchmarkError} /> : view === 'retrieval' ? <RetrievalView data={retrieval} error={retrievalError} /> : <>
    <main>
      <section className="workbench-top"><div><span className="eyebrow">{proposalDate || 'Undated'} · Evaluation export</span><h1>Proposal evaluations</h1><p>Inspect model judgments, compare retrieval evidence, and resolve failed or unstable novelty checks.</p></div><a className="download-link" href="/data/followup_queue.csv" download>Download follow-up CSV ↓</a></section>
      <SourceDetails title="Proposal source" date={proposalDate} files={[["Archive", data.source], ["Archive SHA-256", data.source_sha256], ["Proposals", String(items.length)], ["Scored on", [...new Set(items.map(item => item.scored_on?.slice(0, 10)).filter(Boolean))].join(', ')]]} />
      <section className="issue-overview" aria-label="Choose a case queue">{focusOptions.map(option => <button key={option.key} type="button" className={`issue-card ${focus === option.key ? 'active' : ''}`} aria-pressed={focus === option.key} onClick={() => { setFocus(option.key); setCondition('All'); setQuery(''); setSort('priority'); }}><strong>{option.count}</strong><span>{option.label}</span><small>{option.note}</small></button>)}</section>
      <section className="context-bar"><span><strong>{quotePairs}</strong> passage pairs matched to exported texts</span><span><strong>{data.integrity?.corpus_papers?.toLocaleString() ?? '—'}</strong> papers in 2015–2025 CTO/TIM subset</span><span>Primary mean votes: <strong>N {fmt(mean(items, 'novelty'))}</strong> · <strong>M {fmt(mean(items, 'methodology'))}</strong> · <strong>I {fmt(mean(items, 'impact'))}</strong></span></section>
      <p className="interpretive-note">This is a diagnostic view of exploratory model outputs. Exact quote matches show traceability; score differences and journal indicators do not establish scientific accuracy.</p>
      <section className="workspace"><aside className="rail"><div className="rail-head"><div><span className="eyebrow">Case queue</span><h2>{focusOptions.find(option => option.key === focus)?.label} <span>{filtered.length}</span></h2></div><span className="rail-count">{items.length} total</span></div>
        <label className="search"><span aria-hidden="true">⌕</span><input value={query} onChange={event => setQuery(event.target.value)} placeholder="Search proposals" aria-label="Search proposals" /></label>
        <div className="rail-controls"><span>Condition</span><div className="filters" aria-label="Filter proposals by condition">{['All', '0', 'A', 'B', 'C'].map(key => <button type="button" key={key} className={condition === key ? 'active' : ''} onClick={() => setCondition(key)}>{key === 'All' ? 'All' : key}</button>)}</div></div>
        <label className="sort-control">Sort <select value={sort} onChange={event => setSort(event.target.value)}><option value="priority">Follow-up priority</option><option value="shift">Largest novelty change</option><option value="novelty">Highest novelty vote</option></select></label>
        <div className="proposal-list">{filtered.length ? filtered.map(item => <button type="button" key={item.id} className={`proposal-row ${selected?.id === item.id ? 'selected' : ''}`} onClick={() => setSelectedId(item.id)}>
          <span className={`row-mark ${conditionClass(item.condition)}`}>{item.condition}</span><span className="row-body"><strong>{item.title}</strong><small>{CONDITIONS[item.condition] || item.condition} <i>·</i> {item.trigger || 'Research question'}</small><span className="row-scores">Novelty {fmt(item.votes?.novelty)} → {item.followup?.kind === 'rerun' ? 'failed' : fmt(item.hybrid_vote)}{item.followup?.kind !== 'none' && <em>{item.followup?.label}</em>}</span></span>
        </button>) : <div className="no-results">No proposals match this view.<button type="button" onClick={() => { setFocus('all'); setCondition('All'); setQuery(''); }}>Show all proposals</button></div>}</div>
      </aside><Detail item={selected} integrity={data.integrity} /></section>
      <details className="dataset-context"><summary>Explore the score landscape and dataset limits</summary><div><ScorePlot items={items} selected={selected?.id} onSelect={id => { setFocus('all'); setCondition('All'); setQuery(''); setSelectedId(id); }} /><p>The source corpus includes {data.integrity?.corpus_papers?.toLocaleString() ?? '—'} papers, covers 2015–2025, and is limited to the CTO/TIM subset. The 2026-paper study is available in the separate benchmark view. Among 35 completed hybrid checks, cited-paper instances in the candidate pool rose from {counts.paired_reference_dois_in_semantic_five || 0} to {counts.paired_reference_dois_in_hybrid_ten || 0} of {counts.paired_reference_dois_in_corpus || 0}. These are agent-cited references, not prior-art recall.</p></div></details>
    </main>
    </>}
    <footer><span>Discovery Forum · Private research workspace</span><span>{view === 'benchmark' ? '2026 benchmark · Journal tier is an exploratory comparison' : view === 'retrieval' ? 'Known-source lexical control · Not a novelty validation' : `Export SHA-256 ${data?.source_sha256 || '—'} · Model scores are exploratory`}</span></footer>
  </div>;
}

createRoot(document.getElementById('root')).render(<App />);

"""The corpora a forum proposal is compared with (forum_scoring.py):

  * the ABSTRACT corpus (novelty): the CTO/TIM papers of the Scopus exports with their abstracts, `cor_<id>/` (corpus.json, entries.jsonl, texts.jsonl, vectors.npy); searched by
    embedding (`neighbours`) and, for the hybrid novelty judge, also by BM25 on the same texts (`candidates`, scoring_lexical.py: built in memory, nothing more is stored);
  * the CARD corpora (methodology, impact): the valid cards of the papers of a catalog run, each embedded whole, `fsm_<id>/` and `fsi_<id>/` (forum_method_corpus.json or
    forum_impact_corpus.json, entries.jsonl, cards.jsonl, vectors_card.npy, mask_card.npy).

They live in reports/anvur_live/corpora (outside git: they hold abstracts and embeddings) and are derived data: each can be rebuilt with `research0 forum-score-corpus`. A build is
idempotent: the same inputs give the same id, and an existing corpus is loaded instead of being rebuilt. Building one sends its texts to the embedder.
"""
import csv
import re
import sys
from pathlib import Path

import numpy as np

from . import core, paper_cards, scoring_io, scoring_lexical

ABSTRACT_CORPUS_VERSION = "anvur-rag-v1"      # part of the id of the abstract corpus already built (cor_a5621a4d95ac): keep it, or the same inputs get a new id
ABSTRACT_CARD = "Abstract (testo originale del paper): "
NO_ABSTRACT = ("", "[no abstract available]")
EMBED_BATCH = 32
CARD_CORPORA = {"method": ("fsm_", "forum_method_corpus.json"), "impact": ("fsi_", "forum_impact_corpus.json")}      # kind -> (directory prefix, manifest file)


class CorpusError(ValueError):
    pass


def load_grades(path):
    """{normalized journal title: {"ajg2024": ..., "issn": ...}} from the project's grades file (comment lines start with #)."""
    out = {}
    with open(path, encoding="utf-8", newline="") as stream:
        lines = [line for line in stream if not line.startswith("#")]
    for row in csv.DictReader(lines):
        out[scoring_io.title_key(row["journal"])] = {"ajg2024": row["ajg2024"], "issn": row["issn"]}
    return out


# --- the abstract corpus (novelty) ------------------------------------------------------------------------------------------------------------

class AbstractCorpus:
    def __init__(self, directory):
        self.dir = Path(directory)
        self.manifest = scoring_io.read_json(self.dir / "corpus.json")
        self.entries = scoring_io.read_jsonl(self.dir / "entries.jsonl")
        self.vectors = np.load(self.dir / "vectors.npy")
        self.texts = {e["id"]: e["text"] for e in scoring_io.read_jsonl(self.dir / "texts.jsonl")}
        self._bm25 = None

    def neighbours(self, vector, *, k):
        """The k most similar entries, best first: [{"entry", "sim"}]."""
        sims = self.vectors @ np.asarray(vector, dtype=np.float32)
        return [{"entry": self.entries[int(i)], "sim": float(sims[i])} for i in np.argsort(-sims)[:k]]

    def lexical(self):
        """The BM25 index over title and abstract of every entry, built the first time it is asked for."""
        if self._bm25 is None:
            self._bm25 = scoring_lexical.Bm25([f"{e['title']}. {self.abstract(e['id'])}" for e in self.entries])
        return self._bm25

    def candidates(self, vector, query, *, k, lexical_k):
        """The papers a hybrid judge reads: the k closest by embedding (the same as `neighbours`, same order), then up to `lexical_k` more that BM25 ranks highest on `query` and the
        embedding did not bring (a paper both searches find is listed once, among the embedding's). Each carries its cosine similarity, its rank in the embedding's ordering of the whole
        corpus and in BM25's (None when it shares no term with the query), its BM25 score and which searches brought it. -> [{"entry", "sim", "semantic_rank", "lexical_rank", "bm25", "found_by"}]"""
        sims = self.vectors @ np.asarray(vector, dtype=np.float32)
        chosen = [int(i) for i in np.argsort(-sims)[:k]]
        scores = self.lexical().scores(query) if lexical_k > 0 else np.zeros(len(self.entries), dtype=np.float32)
        extras, reached = [], 0
        if lexical_k > 0:
            for rank, i in enumerate(np.argsort(-scores, kind="stable"), start=1):
                if scores[i] <= 0 or len(extras) >= lexical_k:
                    break
                reached = rank
                if int(i) not in chosen:
                    extras.append(int(i))
        out = []
        for i in chosen + extras:
            lexical_rank = int((scores > scores[i]).sum()) + 1 if scores[i] > 0 else None
            out.append({"entry": self.entries[i], "sim": float(sims[i]), "semantic_rank": int((sims > sims[i]).sum()) + 1, "lexical_rank": lexical_rank, "bm25": round(float(scores[i]), 3),
                        "found_by": (["semantic"] if i in chosen else []) + (["lexical"] if lexical_rank is not None and lexical_rank <= reached else [])})
        return out

    def abstract(self, entry_id):
        text = self.texts[entry_id]
        return text[len(ABSTRACT_CARD):] if text.startswith(ABSTRACT_CARD) else text


def load_assignments(catalog_csv, assignments_csv):
    """{EID: {"canonical_id", "community"}} for the works the Batch v1 assigned to CTO and/or TIM. These labels are MODEL SUGGESTIONS (community-binary-v1), not decisions validated by people."""
    labels = {}
    with open(assignments_csv, encoding="utf-8-sig", newline="") as stream:
        for r in csv.DictReader(stream):
            cto, tim = r["cto"] == "1", r["tim"] == "1"
            if cto or tim:
                labels[r["record_id"]] = "CTO+TIM" if cto and tim else ("CTO" if cto else "TIM")
    out = {}
    with open(catalog_csv, encoding="utf-8-sig", newline="") as stream:
        for r in csv.DictReader(stream):
            if r["canonical_id"] in labels:
                for eid in re.split(r"[;,|\s]+", r.get("all_eids") or r.get("eid") or ""):
                    if eid:
                        out[eid] = {"canonical_id": r["canonical_id"], "community": labels[r["canonical_id"]]}
    return out


def read_scopus_rows(csv_paths, min_words=40, assigned=None):
    """Records of Scopus exports that have a usable abstract, one per work (DOI, else title+year; one per canonical paper when `assigned` is given, which also keeps ONLY the works assigned
    to CTO and/or TIM), in a stable order. 'Cited by' is deliberately not read: it is later than the paper and is not impact."""
    csv.field_size_limit(min(sys.maxsize, 2 ** 31 - 1))
    seen, out = set(), []
    for path in sorted(Path(p) for p in csv_paths):
        with open(path, encoding="utf-8-sig", newline="") as stream:
            for row in csv.DictReader(stream):
                abstract = " ".join((row.get("Abstract") or "").split())
                if abstract.casefold() in NO_ABSTRACT or len(abstract.split()) < min_words or not (row.get("Title") or "").strip():
                    continue
                label = None
                if assigned is not None:
                    label = assigned.get(row.get("EID"))
                    if label is None:
                        continue
                doi = core.norm_doi(row.get("DOI"))
                work = label["canonical_id"] if label else (doi or f"{scoring_io.title_key(row['Title'])}|{row.get('Year')}")
                if work in seen:
                    continue
                seen.add(work)
                year = int(row["Year"]) if (row.get("Year") or "").isdigit() else None
                issn = (row.get("ISSN") or "").strip()
                out.append({"id": row.get("EID") or "sc_" + scoring_io.sha256_text(work)[:12], "title": " ".join(row["Title"].split()), "journal": (row.get("Source title") or "").strip(),
                            "year": year, "issn": (issn[:4] + "-" + issn[4:]) if len(issn) == 8 else issn, "doi": doi, "family": work, "abstract": abstract,
                            "community": label["community"] if label else "", "canonical_id": label["canonical_id"] if label else ""})
    return sorted(out, key=lambda r: r["id"])


def build_abstract_corpus(csv_paths, out_root, grades_csv, embedder, *, min_words=40, chunk=1024, batch=64, log=None, assignments=None):
    """The corpus of the NOVELTY criterion: the works of Scopus exports with their abstracts (the abstract is quoted verbatim: no model writes it). With `assignments` = (catalog.csv,
    assignments.csv) only the works the Batch v1 assigned to CTO and/or TIM are kept. The embeddings are saved in parts, so an interrupted build resumes where it stopped. Idempotent:
    the same exports, grades, embedder and filter give the same id and directory. -> AbstractCorpus."""
    csv_paths = sorted(Path(p) for p in csv_paths)
    grades = load_grades(grades_csv)
    assigned = load_assignments(*assignments) if assignments else None                    # assignments = (catalog.csv, assignments.csv)
    corpus_id = "cor_" + scoring_io.sha256_obj(["scopus", [scoring_io.sha256_file(p) for p in csv_paths], embedder.model_id, scoring_io.sha256_file(grades_csv), ABSTRACT_CORPUS_VERSION, min_words,
                                              [scoring_io.sha256_file(p) for p in assignments] if assignments else None])[:12]
    out_dir = Path(out_root) / corpus_id
    if (out_dir / "corpus.json").exists():
        return AbstractCorpus(out_dir)
    rows = read_scopus_rows(csv_paths, min_words, assigned)
    if not rows:
        raise CorpusError("no Scopus record with a usable abstract")
    plan = {"corpus_id": corpus_id, "records": len(rows), "ids_sha256": scoring_io.sha256_obj([r["id"] for r in rows]), "chunk": chunk}
    (out_dir / "parts").mkdir(parents=True, exist_ok=True)
    plan_path = out_dir / "plan.json"
    if plan_path.exists() and scoring_io.read_json(plan_path) != plan:
        raise CorpusError(f"{out_dir} holds a different, unfinished build: remove it and start again")
    if not plan_path.exists():
        scoring_io.write_json_once(plan_path, plan)
    texts = [f"{r['title']}\n{r['abstract']}" for r in rows]
    for start in range(0, len(texts), chunk):
        part = out_dir / "parts" / f"part_{start // chunk:05d}.npy"
        if part.exists():
            continue
        block = np.vstack([np.asarray(embedder.embed(texts[i:i + batch]), dtype=np.float32) for i in range(start, min(start + chunk, len(texts)), batch)])
        np.save(part, block)
        if log:
            log(f"embedded {min(start + chunk, len(texts))} of {len(texts)}")
    vectors = np.vstack([np.load(out_dir / "parts" / f"part_{i:05d}.npy") for i in range((len(texts) + chunk - 1) // chunk)])
    if vectors.shape[0] != len(rows):
        raise CorpusError("the saved parts do not add up to the records: remove the build folder and start again")
    entries = []
    for r in rows:
        grade = grades.get(scoring_io.title_key(r["journal"]), {}).get("ajg2024", "")
        entries.append({"id": r["id"], "input_id": None, "title": r["title"], "journal": r["journal"], "journal_key": scoring_io.title_key(r["journal"]), "year": r["year"],
                        "publication_date": f"{r['year']}-12-31" if r["year"] else "", "family_id": "fam_" + scoring_io.sha256_text(r["family"])[:12], "quartile": "", "ajg2024": grade,
                        "doi": r["doi"], "text_sha256": scoring_io.sha256_text(r["abstract"]), "card_kind": "abstract", "source": "scopus", "community": r["community"],
                        "canonical_id": r["canonical_id"]})
    journals = np.array([hash(e["journal_key"]) for e in entries])
    rng = np.random.default_rng(20261017)
    sample = rng.choice(len(entries), size=min(3000, len(entries)), replace=False)
    nearest = []
    for s in range(0, len(sample), 250):
        idx = sample[s:s + 250]
        sims = vectors[idx] @ vectors.T
        sims[journals[idx][:, None] == journals[None, :]] = -1.0                       # the closest paper of ANOTHER journal
        nearest.extend(float(x) for x in sims.max(axis=1))
    threshold = float(np.percentile(nearest, 25))
    np.save(out_dir / "vectors.npy", vectors)
    for e, r in zip(entries, rows):
        scoring_io.append_jsonl(out_dir / "entries.jsonl", e)
        scoring_io.append_jsonl(out_dir / "texts.jsonl", {"id": e["id"], "text": ABSTRACT_CARD + r["abstract"]})
    for part in (out_dir / "parts").glob("part_*.npy"):
        part.unlink()
    (out_dir / "parts").rmdir()
    scoring_io.write_json_once(out_dir / "corpus.json", {
        "schema": "anvur-corpus-v1", "corpus_id": corpus_id, "source": "scopus exports", "files": [p.name for p in csv_paths], "embedder": embedder.model_id, "dimensions": int(vectors.shape[1]),
        "entries": len(entries), "journals": len({e["journal_key"] for e in entries}), "with_ajg2024": sum(1 for e in entries if e["ajg2024"]), "min_similarity": threshold,
        "min_similarity_rule": "25th percentile of the best other-journal similarity of 3,000 sampled entries (not calibrated against any judgement)", "grades_sha256": scoring_io.sha256_file(grades_csv),
        "rag_version": ABSTRACT_CORPUS_VERSION, "years": sorted({e["year"] for e in entries if e["year"]}), "card": "the abstract, verbatim",
        "ajg2024_papers": {g: sum(1 for e in entries if e["ajg2024"] == g) for g in sorted({e["ajg2024"] for e in entries})},
        "communities": {c: sum(1 for e in entries if e["community"] == c) for c in sorted({e["community"] for e in entries})},
        "community_note": ("only the works the Batch v1 assigned to CTO and/or TIM; those labels are model suggestions (community-binary-v1), not validated by people" if assignments
                           else "no community filter"),
        "note": "journals of the project's CTO/TIM perimeter only, so almost all neighbours are high-grade (ABS 3-4*); publication date = end of the paper's year"})
    return AbstractCorpus(out_dir)


# --- the card corpora (methodology, impact) ---------------------------------------------------------------------------------------------------

class CardCorpus:
    """The valid cards of one kind, each embedded whole (`paper_cards.card_text`). A card that states nothing has no vector (mask false) and is never a neighbour."""

    def __init__(self, directory, kind):
        self.dir, self.kind = Path(directory), kind
        self.manifest = scoring_io.read_json(self.dir / CARD_CORPORA[kind][1])
        self.entries = scoring_io.read_jsonl(self.dir / "entries.jsonl")
        self.cards = {r["id"]: r["card"] for r in scoring_io.read_jsonl(self.dir / "cards.jsonl")}
        self.vectors = np.load(self.dir / "vectors_card.npy")
        self.mask = np.load(self.dir / "mask_card.npy")

    def neighbours(self, vector, *, k):
        """The k most similar cards, best first: [{"entry", "sim"}]. Each similarity is recomputed as its own dot product, as when the scores of 18 Oct 2026 were made: the
        matrix product rounds differently in the last float32 digit, and the rounded similarity is part of the judge's prompt (and so of the cache key)."""
        query = np.asarray(vector, dtype=np.float32)
        sims = np.where(self.mask, self.vectors @ query, -np.inf)
        picked = [int(i) for i in np.argsort(-sims)[:k] if np.isfinite(sims[i])]
        out = [{"entry": self.entries[i], "sim": float(self.vectors[i] @ query)} for i in picked]
        return sorted(out, key=lambda n: -n["sim"])


def is_card_corpus(directory, kind):
    return (Path(directory) / CARD_CORPORA[kind][1]).is_file()


def newest_card_corpus(root, kind):
    """The most recently built card corpus of a kind under `root` (a path that does not exist when none is built)."""
    prefix, manifest = CARD_CORPORA[kind]
    found = sorted((d for d in Path(root).glob(prefix + "*") if (d / manifest).is_file()), key=lambda d: (d / manifest).stat().st_mtime)
    return found[-1] if found else Path(root) / f"{prefix}not_built"


def _embed(embedder, texts):
    return np.vstack([np.asarray(embedder.embed(texts[i:i + EMBED_BATCH]), dtype=np.float32) for i in range(0, len(texts), EMBED_BATCH)])


def build_card_corpus(entries, cards, kind, embedder, out_root, *, source):
    """A card corpus from `entries` (dicts with at least id, title, journal, journal_key, year, ajg2024) and their cards ({id: card}); entries without a card are left out and counted.
    `source` says where the cards come from and is part of the id. Idempotent. -> CardCorpus."""
    profile, (prefix, manifest) = paper_cards.PROFILES[kind], CARD_CORPORA[kind]
    kept = sorted((e for e in entries if e["id"] in cards), key=lambda e: e["id"])
    # the id formula of the corpora built on 18 Oct 2026 (fsm_474150332a9e, fsi_0c5b08677bbe): keep it, so that the same cards find the same directory
    corpus_id = prefix + scoring_io.sha256_obj([[e["id"] for e in kept], [scoring_io.sha256_obj(cards[e["id"]]) for e in kept], embedder.model_id, profile.version, source])[:12]
    out_dir = Path(out_root) / corpus_id
    if (out_dir / manifest).exists():
        return CardCorpus(out_dir, kind)
    if out_dir.exists():
        raise CorpusError(f"{out_dir} exists without a manifest (an interrupted build): remove it and build again")
    if not kept:
        raise CorpusError(f"no entry has a {kind} card")
    texts = [paper_cards.card_text(cards[e["id"]], profile) for e in kept]
    mask = np.array([bool(t) for t in texts], dtype=bool)
    got = _embed(embedder, [t for t in texts if t])
    vectors = np.zeros((len(kept), got.shape[1]), dtype=np.float32)
    vectors[mask] = got
    out_dir.mkdir(parents=True)
    np.save(out_dir / "vectors_card.npy", vectors)
    np.save(out_dir / "mask_card.npy", mask)
    for e in kept:
        scoring_io.append_jsonl(out_dir / "entries.jsonl", e)
        scoring_io.append_jsonl(out_dir / "cards.jsonl", {"id": e["id"], "card": cards[e["id"]]})
    scoring_io.write_json_once(out_dir / manifest, {
        "schema": f"forum-{kind}-card-corpus-v1", "corpus_id": corpus_id, "embedder": embedder.model_id, "entries": len(kept), "without_card": len(entries) - len(kept),
        "with_text": int(mask.sum()), "journals": len({e["journal_key"] for e in kept}), "source": source, "version": profile.version})
    return CardCorpus(out_dir, kind)

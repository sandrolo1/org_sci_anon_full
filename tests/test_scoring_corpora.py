"""The corpora a forum proposal is compared with (scoring_corpora.py): the abstract corpus built from Scopus exports filtered to the works assigned to CTO and/or TIM, and the card
corpora. Nothing here is a real paper; the embedder is the offline hash embedder."""
import csv
import json
import os
import time

import numpy as np
import pytest

from research0 import embeddings, paper_cards, scoring_corpora as sc, scoring_io

SCOPUS_COLUMNS = ["Authors", "Title", "Year", "Source title", "Cited by", "DOI", "Abstract", "ISSN", "Document Type", "EID"]


def scopus_csv(path, n=40, journals=5, extra=()):
    rows = []
    for k in range(n):
        rows.append({"Authors": "A B", "Title": f"Study {k} of coordination in teams", "Year": str(2015 + k % 11), "Source title": f"Journal Alpha {k % journals}", "Cited by": str(k * 7),
                     "DOI": f"10.1000/sc.{k}", "ISSN": f"1234{k % journals:04d}", "Document Type": "Article", "EID": f"2-s2.0-{1000 + k}",
                     "Abstract": " ".join(["Organizations coordinate work with digital tools and managers study the outcomes carefully"] * 3) + f" Case {k} adds a distinct mechanism number {k * k}."})
    rows += list(extra)
    with open(path, "w", encoding="utf-8-sig", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=SCOPUS_COLUMNS)
        w.writeheader()
        w.writerows(rows)
    return path


def grades_csv(path, journals=5):
    path.write_text("# provenance\njournal,issn,matched_by,abs_title,abs_field,ajg2024\n" + "".join(f"Journal Alpha {k},1234-{k:04d},issn,x,ORG,{['3', '4', '4*'][k % 3]}\n" for k in range(journals)),
                    encoding="utf-8")
    return path


def assignment_files(tmp_path, labels):
    """labels: {canonical_id: (cto, tim)}; the catalog links each canonical paper to the EID of the matching Scopus row (paper k <-> EID 2-s2.0-(1000+k))."""
    catalog, assigned = tmp_path / "catalog.csv", tmp_path / "assignments.csv"
    with open(catalog, "w", encoding="utf-8", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["canonical_id", "doi", "eid", "all_eids"])
        for k in range(40):
            w.writerow([f"can-{k}", f"10.1000/sc.{k}", f"2-s2.0-{1000 + k}", f"2-s2.0-{1000 + k};2-s2.0-{5000 + k}"])
    with open(assigned, "w", encoding="utf-8", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["record_id", "cto", "tim", "cto_rationale", "tim_rationale", "custom_id", "model", "scope_version", "prompt_version"])
        for cid, (cto, tim) in labels.items():
            w.writerow([cid, int(cto), int(tim), "r", "r", "c", "m", "s", "p"])
    return catalog, assigned


# --- the abstract corpus ----------------------------------------------------------------------------------------------------------------------

def test_scopus_rows_are_deduplicated_filtered_and_do_not_carry_citations(tmp_path):
    dup = {"Authors": "X", "Title": "Another title", "Year": "2020", "Source title": "Journal Alpha 0", "Cited by": "9", "DOI": "https://doi.org/10.1000/SC.3", "Abstract": "x " * 80,
           "ISSN": "12340000", "Document Type": "Article", "EID": "2-s2.0-9999"}
    rows = sc.read_scopus_rows([scopus_csv(tmp_path / "a.csv", n=10, extra=[dup, {**dup, "DOI": "10.1000/e", "Abstract": "[No abstract available]", "EID": "e1"},
                                                                           {**dup, "DOI": "10.1000/s", "Abstract": "too short", "EID": "e2"},
                                                                           {**dup, "DOI": "", "Title": "No DOI study", "EID": "e3"}])])
    assert len(rows) == 11 and [r["id"] for r in rows] == sorted(r["id"] for r in rows)
    assert not any(r["id"] in ("e1", "e2", "2-s2.0-9999") for r in rows) and any(r["id"] == "e3" for r in rows)
    assert all("cited" not in " ".join(str(v) for v in r).lower() for r in rows) and rows[0]["issn"].count("-") == 1


def test_only_the_works_assigned_to_cto_and_or_tim_stay_and_carry_their_community(tmp_path):
    csv_path = scopus_csv(tmp_path / "a.csv", n=40)
    catalog, assigned = assignment_files(tmp_path, {"can-0": (True, False), "can-1": (False, True), "can-2": (True, True), "can-3": (False, False), "can-4": (True, False)})
    a = sc.load_assignments(catalog, assigned)
    assert a and {v["community"] for v in a.values()} == {"CTO", "TIM", "CTO+TIM"} and "2-s2.0-1003" not in a and a["2-s2.0-5000"]["canonical_id"] == "can-0"
    rows = sc.read_scopus_rows([csv_path], 40, a)
    assert sorted(r["canonical_id"] for r in rows) == ["can-0", "can-1", "can-2", "can-4"] and {r["community"] for r in rows} == {"CTO", "TIM", "CTO+TIM"}
    assert len(sc.read_scopus_rows([csv_path], 40, {})) == 0


def test_a_paper_exported_twice_under_two_eids_counts_once(tmp_path):
    first = scopus_csv(tmp_path / "a.csv", n=3)
    twin = {"Authors": "A B", "Title": "Study 0 of coordination in teams", "Year": "2015", "Source title": "Journal Alpha 0", "Cited by": "0", "DOI": "10.1000/sc.0",
            "Abstract": "Organizations coordinate work with digital tools. " * 10, "ISSN": "12340000", "Document Type": "Article", "EID": "2-s2.0-5000"}
    second = scopus_csv(tmp_path / "b.csv", n=0, extra=[twin])
    catalog, assigned = assignment_files(tmp_path, {"can-0": (True, False)})
    rows = sc.read_scopus_rows([first, second], 40, sc.load_assignments(catalog, assigned))
    assert len(rows) == 1 and rows[0]["canonical_id"] == "can-0"


def test_the_corpus_keeps_only_assigned_works_with_abstracts_dates_grades_and_communities(tmp_path):
    csv_path, grades = scopus_csv(tmp_path / "a.csv"), grades_csv(tmp_path / "g.csv")
    catalog, assigned = assignment_files(tmp_path, {f"can-{k}": (k % 2 == 0, k % 3 == 0) for k in range(40)})
    corpus = sc.build_abstract_corpus([csv_path], tmp_path / "c", grades, embeddings.HashEmbedder(), assignments=(catalog, assigned))
    labelled = [k for k in range(40) if k % 2 == 0 or k % 3 == 0]
    assert corpus.manifest["entries"] == len(labelled) and sum(corpus.manifest["communities"].values()) == len(labelled)
    assert "model suggestions" in corpus.manifest["community_note"] and corpus.manifest["with_ajg2024"] == len(labelled)
    e = corpus.entries[0]
    assert e["card_kind"] == "abstract" and e["publication_date"] == f"{e['year']}-12-31" and e["community"] in ("CTO", "TIM", "CTO+TIM") and e["canonical_id"].startswith("can-")
    assert corpus.texts[e["id"]].startswith(sc.ABSTRACT_CARD) and not corpus.abstract(e["id"]).startswith(sc.ABSTRACT_CARD)
    assert "Cited" not in json.dumps(corpus.entries) and not (corpus.dir / "parts").exists()
    again = sc.build_abstract_corpus([csv_path], tmp_path / "c", grades, embeddings.HashEmbedder(), assignments=(catalog, assigned))
    assert again.dir == corpus.dir                                                                     # idempotent: the same inputs find the same corpus
    unfiltered = sc.build_abstract_corpus([csv_path], tmp_path / "c", grades, embeddings.HashEmbedder())
    assert unfiltered.manifest["entries"] == 40 and unfiltered.manifest["corpus_id"] != corpus.manifest["corpus_id"] and unfiltered.manifest["community_note"] == "no community filter"


def test_the_neighbours_are_the_most_similar_abstracts_best_first(tmp_path):
    corpus = sc.build_abstract_corpus([scopus_csv(tmp_path / "a.csv")], tmp_path / "c", grades_csv(tmp_path / "g.csv"), embeddings.HashEmbedder())
    query = corpus.vectors[7]
    found = corpus.neighbours(query, k=3)
    best3 = sorted((float(s) for s in corpus.vectors @ query), reverse=True)[:3]
    assert len(found) == 3 and [round(n["sim"], 5) for n in found] == [round(s, 5) for s in best3]


def test_an_interrupted_embedding_resumes_without_redoing_finished_parts(tmp_path):
    class Dies(embeddings.HashEmbedder):
        calls = 0

        def embed(self, texts):
            Dies.calls += 1
            if Dies.calls == 3:
                raise RuntimeError("the network dropped")
            return super().embed(texts)
    csv_path, grades = scopus_csv(tmp_path / "a.csv", n=40), grades_csv(tmp_path / "g.csv")
    with pytest.raises(RuntimeError):
        sc.build_abstract_corpus([csv_path], tmp_path / "c", grades, Dies(), chunk=10, batch=10)
    build_dir = next((tmp_path / "c").iterdir())
    assert sorted(p.name for p in (build_dir / "parts").glob("part_*.npy")) == ["part_00000.npy", "part_00001.npy"] and not (build_dir / "corpus.json").exists()
    Dies.calls = -100
    seen = []
    corpus = sc.build_abstract_corpus([csv_path], tmp_path / "c", grades, Dies(), chunk=10, batch=10, log=seen.append)
    assert corpus.manifest["entries"] == 40 and seen[0].startswith("embedded 30") and Dies.calls == -100 + 2            # only the two remaining parts were embedded
    whole = sc.build_abstract_corpus([scopus_csv(tmp_path / "b.csv", n=40)], tmp_path / "c2", grades, embeddings.HashEmbedder(), chunk=10, batch=10)
    assert np.allclose(whole.vectors, corpus.vectors, atol=1e-6)


def test_an_unfinished_build_of_something_else_is_refused(tmp_path):
    grades, csv_path = grades_csv(tmp_path / "g.csv"), scopus_csv(tmp_path / "a.csv", n=20)

    class Dies(embeddings.HashEmbedder):
        def embed(self, texts):
            raise RuntimeError("stop")
    with pytest.raises(RuntimeError):
        sc.build_abstract_corpus([csv_path], tmp_path / "c", grades, Dies(), chunk=10, batch=10)
    build_dir = next((tmp_path / "c").iterdir())
    plan = scoring_io.read_json(build_dir / "plan.json")
    (build_dir / "plan.json").write_text(json.dumps({**plan, "ids_sha256": "0" * 64}), encoding="utf-8")
    with pytest.raises(sc.CorpusError, match="different, unfinished build"):
        sc.build_abstract_corpus([csv_path], tmp_path / "c", grades, embeddings.HashEmbedder(), chunk=10, batch=10)


# --- the card corpora -------------------------------------------------------------------------------------------------------------------------

TOPICS = ("survey of employees in two waves with multilevel models", "case study of one platform firm with interviews", "panel of patents with fixed effects",
          "experiment with managers in a laboratory", "archival data on mergers with difference in differences")


def method_card(text):
    stated = {"text": text, "status": "extracted", "passages": ["a passage of the paper"]}
    return {**{f: stated for f in paper_cards.METHOD.fields[:4]}, **{f: {"text": None, "status": "missing", "passages": []} for f in paper_cards.METHOD.fields[4:]}, "info_gaps": []}


def entries(n):
    return [{"id": f"P{i}", "title": f"Paper {i}", "journal": f"Journal {i % 2}", "journal_key": f"journal {i % 2}", "year": 2020 + i, "ajg2024": "4"} for i in range(n)]


def test_a_card_corpus_is_built_once_and_found_again_by_its_inputs(tmp_path):
    cards = {f"P{i}": method_card(t) for i, t in enumerate(TOPICS)}
    source = {"run_id": "r1", "kind": "method"}
    corpus = sc.build_card_corpus(entries(6), cards, "method", embeddings.HashEmbedder(), tmp_path, source=source)          # P5 has no card
    assert corpus.dir.name.startswith("fsm_") and corpus.manifest["entries"] == 5 and corpus.manifest["without_card"] == 1 and corpus.manifest["version"] == "anvur-method-v2"
    assert sc.build_card_corpus(entries(6), cards, "method", embeddings.HashEmbedder(), tmp_path, source=source).dir == corpus.dir
    assert sc.build_card_corpus(entries(6), cards, "method", embeddings.HashEmbedder(), tmp_path, source={**source, "run_id": "r2"}).dir != corpus.dir
    assert sc.is_card_corpus(corpus.dir, "method") and not sc.is_card_corpus(corpus.dir, "impact")


def test_a_card_that_states_nothing_is_never_a_neighbour(tmp_path):
    empty = {**{f: {"text": None, "status": "missing", "passages": []} for f in paper_cards.METHOD.fields}, "info_gaps": []}
    cards = {"P0": method_card(TOPICS[0]), "P1": empty, "P2": method_card(TOPICS[2])}
    corpus = sc.build_card_corpus(entries(3), cards, "method", embeddings.HashEmbedder(), tmp_path, source={})
    assert corpus.mask.tolist() == [True, False, True] and corpus.manifest["with_text"] == 2
    found = corpus.neighbours(corpus.vectors[0], k=5)
    assert [n["entry"]["id"] for n in found] == ["P0", "P2"] and found[0]["sim"] >= found[1]["sim"]


def test_an_interrupted_card_corpus_build_is_refused_until_removed(tmp_path):
    cards = {f"P{i}": method_card(t) for i, t in enumerate(TOPICS)}
    corpus = sc.build_card_corpus(entries(5), cards, "method", embeddings.HashEmbedder(), tmp_path, source={})
    (corpus.dir / sc.CARD_CORPORA["method"][1]).unlink()
    with pytest.raises(sc.CorpusError, match="interrupted build"):
        sc.build_card_corpus(entries(5), cards, "method", embeddings.HashEmbedder(), tmp_path, source={})


def test_the_newest_card_corpus_of_a_kind_is_the_default(tmp_path):
    assert sc.newest_card_corpus(tmp_path, "impact").name == "fsi_not_built"
    first = sc.build_card_corpus(entries(5), {f"P{i}": method_card(t) for i, t in enumerate(TOPICS)}, "method", embeddings.HashEmbedder(), tmp_path, source={"n": 1})
    second = sc.build_card_corpus(entries(5), {f"P{i}": method_card(t) for i, t in enumerate(TOPICS)}, "method", embeddings.HashEmbedder(), tmp_path, source={"n": 2})
    older = time.time() - 3600
    os.utime(first.dir / sc.CARD_CORPORA["method"][1], (older, older))
    assert sc.newest_card_corpus(tmp_path, "method") == second.dir

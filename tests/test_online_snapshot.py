import csv
import io
import json
import tempfile
import unittest
import zipfile
from pathlib import Path

from research0.online_snapshot import convert
from research0.novelty_offline import build_dossier


def csv_bytes(headers, rows):
    stream = io.StringIO(newline="")
    writer = csv.DictWriter(stream, fieldnames=headers)
    writer.writeheader()
    writer.writerows(rows)
    return stream.getvalue().encode("utf-8")


class OnlineSnapshotTests(unittest.TestCase):
    def make_export(self, folder, *, proposal_question_id="q1"):
        papers = folder / "papers_v4_2026-10-14.csv"
        papers.write_bytes(csv_bytes(
            ["catalog_run_id", "extraction_run_id", "paper_id", "title", "doi",
             "year", "abstract", "eligible", "n_constructs_v4"],
            [{"catalog_run_id": "catalog-1", "extraction_run_id": "extract-1",
              "paper_id": "p1", "title": "Distributed decision rights", "doi": "10.1/example",
              "year": "2025", "abstract": "Distributed decision rights change coordination in teams.",
              "eligible": "True", "n_constructs_v4": "2"},
             {"catalog_run_id": "catalog-1", "extraction_run_id": "extract-1",
              "paper_id": "p2", "title": "Authority in teams", "doi": "", "year": "2024",
              "abstract": "[No abstract available]", "eligible": "True", "n_constructs_v4": "0"}]))
        archive = folder / "forum_online_2026-10-19.zip"
        question = {"rq": "How do teams share decision rights?", "puzzle": "Teams differ.",
                    "mechanism": "Distributed authority changes coordination.",
                    "predictions": [{"statement": "Coordination improves."}]}
        tables = {
            "forum_episode": (["episode_id", "release_id", "condition", "config", "status"],
                              [{"episode_id": "e1", "release_id": "r1", "condition": "A",
                                "config": json.dumps({"batch": {"id": "b1", "arm": "A"}}),
                                "status": "completed"}]),
            "question": (["question_id", "episode_id", "content"],
                         [{"question_id": "q1", "episode_id": "e1", "content": json.dumps(question)}]),
            "forum_proposal": (["proposal_id", "question_id", "episode_id", "status"],
                               [{"proposal_id": "pr1", "question_id": proposal_question_id,
                                 "episode_id": "e1", "status": "ok"}]),
            "forum_proposal_evaluation": (["evaluation_id", "proposal_id", "status",
                                           "rubric_version", "mean_score"],
                                          [{"evaluation_id": "ev1", "proposal_id": "pr1",
                                            "status": "ok", "rubric_version": "proposal-blind-v2",
                                            "mean_score": "7.0"}]),
        }
        with zipfile.ZipFile(archive, "w") as output:
            for name, (headers, rows) in tables.items():
                output.writestr(f"forum_export/{name}.csv", csv_bytes(headers, rows))
        return papers, archive

    def test_real_export_shape_feeds_retrieval_with_explicit_coverage(self):
        with tempfile.TemporaryDirectory() as directory:
            papers, archive = self.make_export(Path(directory))
            catalog, questions, summary = convert(papers, archive)
            dossier = build_dossier(catalog, questions)
            self.assertEqual(summary["papers_with_abstract"], 1)
            self.assertEqual(summary["evaluation_rubrics"], {"proposal-blind-v2": 1})
            self.assertEqual(summary["batches"]["b1"]["blind_mean_score"], 7.0)
            self.assertEqual(dossier["coverage"]["title_only"], 1)
            self.assertEqual(dossier["questions"][0]["status"], "retrieval_only")
            self.assertEqual(dossier["questions"][0]["candidates"][0]["id"], "p1")

    def test_rejects_broken_forum_relationship(self):
        with tempfile.TemporaryDirectory() as directory:
            papers, archive = self.make_export(Path(directory), proposal_question_id="missing")
            with self.assertRaisesRegex(ValueError, "unknown question or episode"):
                convert(papers, archive)


if __name__ == "__main__":
    unittest.main()

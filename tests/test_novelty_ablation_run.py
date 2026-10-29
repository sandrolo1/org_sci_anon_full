"""Scientific control: the judge runner must retain abstentions and failures."""

import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

from research0.novelty_ablation_analysis import _bundle, analyze
from research0.novelty_ablation_run import _configuration, _freeze, _prior_results, _run_one, run
from test_novelty_ablation_analysis import fixture


PACKET = {"packet_id": "np_test", "rubric": "Use evidence.",
          "proposal_text": "Workers contest algorithmic scheduling decisions.",
          "papers": [{"ref": "V1", "title": "Voice", "year": 2025,
                      "abstract": "Workers contest algorithmic scheduling decisions during rollout."}]}
MANIFEST = {"packets_sha256": "a" * 64}


class FakeClient:
    def __init__(self, answer=None, error=None):
        self.answer, self.error = answer, error
        self.responses = self

    def create(self, **kwargs):
        assert kwargs["model"] == "gpt-5.2-2025-12-11"
        assert kwargs["store"] is False and kwargs["tools"] == []
        assert "packet_id" not in kwargs["input"]
        if self.error:
            raise self.error
        return SimpleNamespace(status="completed", output_text=json.dumps(self.answer), id="resp_test",
                               usage=SimpleNamespace(input_tokens=100, output_tokens=50))


def _answer(status="scored"):
    if status == "insufficient_evidence":
        return {"status": status, "vote": None, "reason": "The abstract gives no testable prediction.",
                "comparisons": []}
    dimension = {"assessment": "overlap", "explanation": "Both concern contesting assignments.",
                 "proposal_passage": "Workers contest algorithmic scheduling decisions",
                 "neighbour_passage": "Workers contest algorithmic scheduling decisions"}
    return {"status": "scored", "vote": 5, "reason": None,
            "comparisons": [{"neighbour_id": "V1", "question": dimension,
                             "mechanism": {"assessment": "unclear", "explanation": "Not given.",
                                           "proposal_passage": None, "neighbour_passage": None},
                             "predictions": {"assessment": "unclear", "explanation": "Not given.",
                                             "proposal_passage": None, "neighbour_passage": None}}]}


class RunnerTests(unittest.TestCase):
    def test_scored_abstained_invalid_and_api_failure_have_distinct_records(self):
        config = _configuration(MANIFEST)
        cases = [(_answer(), "scored"), (_answer("insufficient_evidence"), "insufficient_evidence"),
                 ({**_answer(), "vote": 12}, "failed")]
        for answer, expected in cases:
            row, usage, raw = _run_one(PACKET, config, "b" * 64, 1,
                                       lambda answer=answer: FakeClient(answer=answer))
            self.assertEqual((row["status"], row["packet_id"]), (expected, PACKET["packet_id"]))
            self.assertEqual((usage["input_tokens"], raw), (100, answer))
            if expected != "scored":
                self.assertIsNone(row["vote"])
        row, usage, raw = _run_one(PACKET, config, "b" * 64, 1,
                                   lambda: FakeClient(error=RuntimeError("private material")))
        self.assertEqual((row["status"], row["reason"]), ("failed", "api_error:RuntimeError"))
        self.assertNotIn("private material", json.dumps(row))
        self.assertIsNone(raw)
        self.assertEqual(usage["response_status"], "api_error")

    def test_frozen_configuration_rejects_changed_run_and_duplicate_results(self):
        with tempfile.TemporaryDirectory() as directory:
            tmp_path = Path(directory)
            config = _configuration(MANIFEST)
            digest = _freeze(tmp_path, config)
            self.assertEqual(_freeze(tmp_path, config), digest)
            row = {"packet_id": "np_test", "repeat": 1,
                   "packets_sha256": config["packets_sha256"], "model_id": config["model"],
                   "prompt_sha256": config["prompt_sha256"], "config_sha256": digest,
                   "rubric_version": config["rubric_version"]}
            results = tmp_path / "results.jsonl"
            results.write_text(json.dumps(row) + "\n")
            self.assertEqual(len(_prior_results(results, {"np_test": PACKET}, config, digest)), 1)
            results.write_text(json.dumps(row) + "\n" + json.dumps(row) + "\n")
            with self.assertRaisesRegex(ValueError, "duplicate"):
                _prior_results(results, {"np_test": PACKET}, config, digest)
            with self.assertRaisesRegex(ValueError, "frozen configuration"):
                _freeze(tmp_path, {**config, "model": "different"})

    def test_complete_fake_run_resumes_without_repeat_and_can_be_analyzed(self):
        with tempfile.TemporaryDirectory() as directory:
            tmp_path = Path(directory)
            bundle, _ = fixture(tmp_path)
            _, packets, _ = _bundle(bundle)

            class DynamicClient:
                def __init__(self):
                    self.responses = self

                def create(self, **kwargs):
                    data = json.loads(kwargs["input"])
                    dimension = {"assessment": "overlap", "explanation": "Both discuss the stated phenomenon.",
                                 "proposal_passage": data["proposal_text"],
                                 "neighbour_passage": data["papers"][0]["abstract"]}
                    answer = {"status": "scored", "vote": 5, "reason": None,
                              "comparisons": [{"neighbour_id": data["papers"][0]["ref"],
                                               "question": dimension, "mechanism": dimension,
                                               "predictions": dimension}]}
                    return SimpleNamespace(status="completed", output_text=json.dumps(answer), id="resp_test",
                                           usage=SimpleNamespace(input_tokens=100, output_tokens=50))

            output = tmp_path / "run"
            first = run(bundle, output, workers=2, client_factory=DynamicClient)
            self.assertEqual((first["new"], first["scored"], first["remaining"]), (3, 3, 0))
            second = run(bundle, output, workers=2, client_factory=DynamicClient)
            self.assertEqual((second["new"], second["already_done"]), (0, 3))
            self.assertEqual(len((output / "results.jsonl").read_text().splitlines()), 3)
            report = analyze(bundle, output / "results.jsonl", tmp_path / "analysis")
            self.assertEqual(report["arm_status"], {arm: {"scored": 1} for arm in
                                                    ("semantic_five", "semantic_plus_lexical", "semantic_plus_placebo")})
            selected_output = tmp_path / "selected_run"
            selected = run(bundle, selected_output, workers=1, client_factory=DynamicClient,
                           packet_ids={next(iter(packets))})
            self.assertEqual((selected["new"], selected["remaining"]), (1, 0))
            resumed = run(bundle, selected_output, workers=1, client_factory=DynamicClient)
            self.assertEqual((resumed["new"], resumed["already_done"]), (2, 1))

    def test_private_api_requires_an_explicit_launch_flag(self):
        with tempfile.TemporaryDirectory() as directory:
            tmp_path = Path(directory)
            bundle, _ = fixture(tmp_path)
            with self.assertRaisesRegex(RuntimeError, "--allow-private-api"):
                run(bundle, tmp_path / "private_run")

    def test_incomplete_text_is_kept_in_private_raw_ledger(self):
        class PartialClient:
            responses = None

            def __init__(self):
                self.responses = self

            def create(self, **kwargs):
                return SimpleNamespace(status="incomplete", output_text="partial private answer",
                                       id="local:partial", usage=SimpleNamespace(
                                           input_tokens=80, output_tokens=20))

        row, usage, raw = _run_one(PACKET, _configuration(MANIFEST), "b" * 64, 1,
                                   PartialClient)
        self.assertEqual((row["status"], row["reason"]), ("failed", "response_incomplete_or_empty"))
        self.assertEqual(raw, {"raw_text": "partial private answer"})
        self.assertEqual(usage["output_tokens"], 20)


if __name__ == "__main__":
    unittest.main()

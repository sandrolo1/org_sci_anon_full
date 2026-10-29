"""The private local judge uses an installed localhost model only."""

import json
import unittest
from unittest.mock import patch

from research0.novelty_ablation_local import LocalClient, _installed_digest, local_configuration
from research0.novelty_ablation_run import PROMPT, SCHEMA


class LocalJudgeTests(unittest.TestCase):
    def test_refuses_a_model_that_is_not_already_installed(self):
        with patch("research0.novelty_ablation_local._local_request",
                   return_value=({"models": []}, b"{}")):
            with self.assertRaisesRegex(RuntimeError, "refusing to pull"):
                _installed_digest()

    def test_request_is_local_structured_and_pinned(self):
        response = {"model": "qwen3:4b", "done": True, "done_reason": "stop",
                    "message": {"content": '{"status":"insufficient_evidence"}'},
                    "prompt_eval_count": 12, "eval_count": 8}
        captured = {}

        def fake_request(method, path, body=None, timeout=240):
            captured.update(method=method, path=path, body=body)
            return response, json.dumps(response).encode()

        with patch("research0.novelty_ablation_local._installed_digest", return_value="a" * 64):
            config = local_configuration({"packets_sha256": "b" * 64})
        self.assertEqual((config["model"], config["transport"], config["local_model_sha256"]),
                         ("ollama:qwen3:4b", "http://127.0.0.1:11434", "a" * 64))
        with patch("research0.novelty_ablation_local._local_request", side_effect=fake_request):
            answer = LocalClient().responses.create(
                model=config["model"], instructions=PROMPT, input='{"fixture":true}',
                text={"format": {"schema": SCHEMA}}, tools=[], store=False)
        self.assertEqual((captured["method"], captured["path"]), ("POST", "/api/chat"))
        self.assertEqual(captured["body"]["model"], "qwen3:4b")
        self.assertEqual(captured["body"]["format"], SCHEMA)
        self.assertFalse(captured["body"]["think"])
        self.assertEqual((answer.status, answer.usage.input_tokens), ("completed", 12))


if __name__ == "__main__":
    unittest.main()

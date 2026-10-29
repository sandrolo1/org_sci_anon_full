"""Run a small private novelty-judge feasibility check on local Ollama only.

HTTP is pinned to 127.0.0.1:11434. There is no provider fallback, proxy,
remote model pull, or external API key. Results stay in the caller's directory.
This is a judge-format check with a small local model, not a novelty result.
"""

import argparse
import hashlib
import http.client
import json
from pathlib import Path
from types import SimpleNamespace

from .novelty_ablation_analysis import _bundle
from .novelty_ablation_run import PROMPT, SCHEMA, _canonical, _configuration, _sha, run


HOST = "127.0.0.1"
PORT = 11434
MODEL = "qwen3:4b"
NUM_CTX = 12288
NUM_PREDICT = 2200


def _local_request(method, path, body=None, timeout=240):
    connection = http.client.HTTPConnection(HOST, PORT, timeout=timeout)
    try:
        encoded = json.dumps(body, ensure_ascii=False).encode("utf-8") if body is not None else None
        connection.request(method, path, body=encoded,
                           headers={"Content-Type": "application/json"} if encoded is not None else {})
        response = connection.getresponse()
        raw = response.read()
        if response.status != 200:
            raise RuntimeError(f"local Ollama returned HTTP {response.status}")
        return json.loads(raw), raw
    finally:
        connection.close()


def _installed_digest():
    tags, _ = _local_request("GET", "/api/tags", timeout=10)
    matches = [row for row in tags.get("models", []) if row.get("name") == MODEL]
    if len(matches) != 1 or not isinstance(matches[0].get("digest"), str):
        raise RuntimeError(f"{MODEL} is not already installed locally; refusing to pull a model")
    digest = matches[0]["digest"]
    if len(digest) != 64 or any(c not in "0123456789abcdef" for c in digest):
        raise RuntimeError("local model digest is malformed")
    return digest


class LocalClient:
    def __init__(self):
        self.responses = self

    def create(self, **kwargs):
        if (kwargs.get("model") != f"ollama:{MODEL}" or kwargs.get("store") is not False or
                kwargs.get("tools") != [] or kwargs.get("instructions") != PROMPT or
                kwargs.get("text", {}).get("format", {}).get("schema") != SCHEMA):
            raise ValueError("unexpected local judge request")
        body = {"model": MODEL,
                "messages": [{"role": "system", "content": PROMPT},
                             {"role": "user", "content": kwargs["input"]}],
                "format": SCHEMA, "stream": False, "think": False,
                "options": {"temperature": 0, "num_ctx": NUM_CTX,
                            "num_predict": NUM_PREDICT}}
        result, raw = _local_request("POST", "/api/chat", body)
        if result.get("model") != MODEL:
            raise RuntimeError("local response used the wrong model")
        return SimpleNamespace(
            status="completed" if result.get("done") and result.get("done_reason") == "stop" else "incomplete",
            output_text=(result.get("message") or {}).get("content"),
            id="local:" + hashlib.sha256(raw).hexdigest(),
            usage=SimpleNamespace(input_tokens=result.get("prompt_eval_count"),
                                  output_tokens=result.get("eval_count")))


def local_configuration(manifest):
    config = _configuration(manifest)
    config.update(model=f"ollama:{MODEL}", local_model_sha256=_installed_digest(),
                  transport=f"http://{HOST}:{PORT}", max_output_tokens=NUM_PREDICT,
                  context_tokens=NUM_CTX, temperature=0, runner_version="novelty-ablation-local-v1")
    return config


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bundle", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--max-new", type=int, default=3,
                        help="maximum new packets for this local feasibility pass (default 3)")
    parser.add_argument("--repeat", type=int, default=1)
    parser.add_argument("--matched-first-proposal", action="store_true",
                        help="complete all three frozen arms for the first packet's proposal")
    args = parser.parse_args(argv)
    manifest, packets, private_key = _bundle(args.bundle)
    config = local_configuration(manifest)
    selected = None
    if args.matched_first_proposal:
        first = next(iter(packets))
        proposal_id = private_key[first]["proposal_id"]
        selected = {packet_id for packet_id in packets
                    if private_key[packet_id]["proposal_id"] == proposal_id}
        if len(selected) != 3:
            raise ValueError("the selected proposal does not have all three frozen arms")
    print(json.dumps(run(args.bundle, args.output, repeat=args.repeat,
                         max_new=args.max_new, workers=1,
                         client_factory=LocalClient, config_override=config,
                         packet_ids=selected)))


if __name__ == "__main__":
    main()

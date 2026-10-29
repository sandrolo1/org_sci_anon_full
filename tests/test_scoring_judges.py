"""The model calls of the proposal evaluation (scoring_judges.py): configuration, cache, retries, limits checked before a call, refusals of the provider and network faults.
The provider is a scripted stand-in, never a model."""
import copy
import json

import pytest

from research0 import scoring_io, scoring_judges as sj
from research0.forum_providers import BudgetExceeded, Cancelled, TurnResult

SCHEMA = {"type": "object", "additionalProperties": False, "required": ["vote"], "properties": {"vote": {"type": "integer", "minimum": 1, "maximum": 10}}}


def config(provider="synthetic", run_live=False, **budget):
    judge = lambda role: {"role": role, "provider": provider, "model": {"id": "m-1"}, "params": {"max_output_tokens": 1000, "reasoning_effort": None, "temperature": None}}
    return {"schema": sj.CONFIG_SCHEMA, "name": "test", "run_live": run_live, "embedding": "hash-rp-v1", "k": 5, "concurrency": 1, "judges": [judge(r) for r in sj.JUDGE_ROLES],
            "budget": {"max_live_calls": 0, "max_tokens": 0, "max_seconds": 0, **budget}, "retries": {"max_attempts": 2, "max_total_failures": 3}}


class Scripted:
    """Answers with the next item of `answers`: a dict (sent as JSON), a string (sent as it is), an exception (raised) or "CUT" (an answer cut off at the output limit)."""
    name, synthetic, cache_identity = "synthetic", True, "scripted-judges-1"

    def __init__(self, *answers):
        self.answers, self.requests = list(answers), []

    def run(self, request):
        self.requests.append(request)
        a = self.answers.pop(0) if len(self.answers) > 1 else self.answers[0]
        if isinstance(a, Exception):
            raise a
        if a == "CUT":
            return TurnResult(text="{", finish="incomplete", model=request.model, input_tokens=10, output_tokens=1000, latency_ms=1)
        text = a if isinstance(a, str) else json.dumps(a)
        return TurnResult(text=text, model=request.model, resolved_model="scripted", input_tokens=100, output_tokens=20, latency_ms=5)


def judges(tmp_path, provider, cfg=None, session="s1", **kw):
    return sj.Judges(tmp_path, cfg or config(), {r: provider for r in sj.JUDGE_ROLES}, session_id=session, **kw)


def ask(j, role="judge_1", prompt="P", validate=lambda o: [] if 1 <= o.get("vote", 0) <= 10 else ["vote out of range"]):
    return j.ask(role, task="t", system="S", prompt=prompt, schema_name="t", schema=SCHEMA, validate=validate, key_parts=["k"])


# --- configuration ------------------------------------------------------------------------------------------------------------------------

def test_the_project_configuration_is_valid_and_live():
    cfg = sj.load_config("config/forum_scoring.json")
    assert cfg["run_live"] and [j["role"] for j in cfg["judges"]] == list(sj.JUDGE_ROLES) and cfg["k"] == 5 and all(cfg["budget"][k] > 0 for k in sj.BUDGET_KEYS)


@pytest.mark.parametrize("change, message", [
    (lambda c: c.update(schema="anvur-config-v1"), "config.schema"),
    (lambda c: c.update(run_live=True), "test stand-in"),
    (lambda c: c["judges"].reverse(), "judge_1 and judge_2"),
    (lambda c: c["judges"][0]["params"].update(max_output_tokens=100), "max_output_tokens"),
    (lambda c: c.update(k=0), "config.k"),
    (lambda c: c["budget"].update(max_tokens=-1), "max_tokens"),
])
def test_a_wrong_configuration_is_refused(change, message):
    cfg = config()
    change(cfg)
    with pytest.raises(sj.ConfigError, match=message):
        sj.validate_config(cfg)


def test_a_live_run_needs_openai_and_positive_limits():
    with pytest.raises(sj.ConfigError, match="positive max_live_calls"):
        sj.validate_config(config("openai", run_live=True, max_tokens=10, max_seconds=10))
    assert sj.validate_config(config("openai", run_live=True, max_live_calls=1, max_tokens=10, max_seconds=10))


def test_openai_without_run_live_and_a_stand_in_for_a_real_provider_are_refused(tmp_path):
    with pytest.raises(sj.JudgeError, match="run_live"):
        judges(tmp_path, Scripted({"vote": 5}), config("openai"))
    real_looking = Scripted({"vote": 5})
    real_looking.synthetic = False
    with pytest.raises(sj.JudgeError, match="do not match"):
        judges(tmp_path, real_looking)


# --- cache and retries ----------------------------------------------------------------------------------------------------------------------

def test_a_valid_answer_is_cached_across_sessions_and_a_different_prompt_or_key_part_is_a_new_call(tmp_path):
    p = Scripted({"vote": 6})
    j = judges(tmp_path, p)
    assert ask(j) == ({"vote": 6}, False) and ask(j) == ({"vote": 6}, True) and len(p.requests) == 1
    ask(j, prompt="Q")
    assert len(p.requests) == 2
    # the role is not in the key by itself: a caller that wants two independent judges puts the role in key_parts, as forum_scoring does
    assert ask(j, role="judge_2") == ({"vote": 6}, True)
    j.ask("judge_2", task="t", system="S", prompt="P", schema_name="t", schema=SCHEMA, validate=lambda o: [], key_parts=["k", "judge_2"])
    assert len(p.requests) == 3
    assert ask(judges(tmp_path, Scripted({"vote": 1}), session="s2")) == ({"vote": 6}, True)


def test_the_request_carries_the_stripped_schema_and_no_tools(tmp_path):
    p = Scripted({"vote": 6})
    ask(judges(tmp_path, p))
    r = p.requests[0]
    assert r.json_schema == {"name": "t", "schema": scoring_io.provider_schema(SCHEMA)} and "minimum" not in json.dumps(r.json_schema) and r.tools == [] and not r.web_search


def test_an_invalid_answer_is_retried_and_then_refused_and_every_attempt_is_registered(tmp_path):
    p = Scripted("not json", {"vote": 11}, {"vote": 4})
    j = judges(tmp_path, p)
    with pytest.raises(sj.JudgeError, match="vote out of range"):
        ask(j)
    rows = scoring_io.read_jsonl(tmp_path / "calls.jsonl")
    assert [r["status"] for r in rows] == ["failed", "failed"] and rows[0]["error"] == "the answer is not valid JSON" and rows[1]["raw_text"] == '{"vote": 11}'
    assert ask(j) == ({"vote": 4}, False)                                       # a later call may still answer: one more attempt is left of the lifetime cap of 3
    with pytest.raises(sj.JudgeError, match="no valid answer"):
        ask(judges(tmp_path, Scripted("x"), session="s3"), prompt="other")     # two failed attempts
    with pytest.raises(sj.JudgeError, match="left without an answer"):
        ask(judges(tmp_path, Scripted("x"), session="s4"), prompt="other")     # a third one reaches the lifetime cap, whatever the session


def test_a_cut_off_answer_and_an_api_error_are_failed_attempts_not_crashes(tmp_path):
    j = judges(tmp_path, Scripted("CUT", RuntimeError("boom")))
    with pytest.raises(sj.JudgeError):
        ask(j)
    errors = [r["error"] for r in scoring_io.read_jsonl(tmp_path / "calls.jsonl")]
    assert errors[0].startswith("the answer was cut off") and errors[1] == "RuntimeError: boom"


# --- limits ---------------------------------------------------------------------------------------------------------------------------------

def openai_like(*answers):
    p = Scripted(*answers)
    p.synthetic, p.cache_identity = False, None
    return p


def test_a_real_provider_with_a_zero_limit_cannot_call(tmp_path):
    p = openai_like({"vote": 5})
    j = judges(tmp_path, p, config("openai", run_live=True, max_live_calls=5, max_tokens=100000, max_seconds=100))
    j.budget.limits = {**j.budget.limits, "max_seconds": 0}
    with pytest.raises(BudgetExceeded, match="limit 0"):
        ask(j)
    assert p.requests == []


def test_the_limits_are_checked_before_the_call_and_failures_count(tmp_path):
    p = openai_like("garbage", {"vote": 5})
    j = judges(tmp_path, p, config("openai", run_live=True, max_live_calls=2, max_tokens=100000, max_seconds=100))
    ask(j)                                                                       # one failed attempt and one valid answer: two calls
    with pytest.raises(BudgetExceeded, match="max_live_calls"):
        ask(j, prompt="another")
    assert len(p.requests) == 2 and j.budget.used()["calls"] == 2 and j.budget.used()["failed_calls"] == 1
    small = judges(tmp_path, openai_like({"vote": 5}), config("openai", run_live=True, max_live_calls=9, max_tokens=500, max_seconds=100), session="s-small")
    with pytest.raises(BudgetExceeded, match="max_tokens"):
        ask(small, prompt="not cached yet")                                      # the whole output allowance (1000) is reserved before the call


def test_the_synthetic_stand_in_is_unlimited_with_zero_limits(tmp_path):
    j = judges(tmp_path, Scripted({"vote": 5}))
    assert all(ask(j, prompt=str(i))[0] == {"vote": 5} for i in range(5))


def test_the_gate_can_stop_a_session(tmp_path):
    def gate():
        raise Cancelled("stopped")
    with pytest.raises(Cancelled):
        ask(judges(tmp_path, Scripted({"vote": 5}), gate=gate))


# --- the provider -------------------------------------------------------------------------------------------------------------------------

def test_a_refusal_of_the_provider_stops_at_once_and_is_not_a_failed_attempt(tmp_path):
    p = Scripted(RuntimeError("Error code: 429 insufficient_quota"), {"vote": 7})
    j = judges(tmp_path, p)
    with pytest.raises(sj.ProviderBlocked):
        ask(j)
    assert j.ledger.failures(j.ledger._load()[0]["cache_key"]) == 0 and j.budget.used()["calls"] == 0
    assert ask(j) == ({"vote": 7}, False)                                       # resumes unchanged once the cause is fixed
    assert sj.is_provider_block("AuthenticationError: Incorrect API key provided")


def test_network_faults_are_waited_out_and_a_network_that_stays_down_stops_the_session(tmp_path):
    p = Scripted(RuntimeError("APIConnectionError: Connection error."), {"vote": 3})
    j = judges(tmp_path, p)
    j.transport_backoff = (0, 0)
    assert ask(j) == ({"vote": 3}, False) and j.budget.used()["calls"] == 1 and j.budget.used()["failed_calls"] == 0
    down = judges(tmp_path, Scripted(RuntimeError("APITimeoutError: timed out")), session="s-down")
    down.transport_backoff = (0, 0)
    with pytest.raises(sj.ProviderBlocked, match="unreachable"):
        ask(down, prompt="other")


def test_a_rate_limit_that_outlasts_the_provider_is_waited_out_like_a_network_fault_but_a_spent_quota_still_blocks(tmp_path):
    j = judges(tmp_path, Scripted(RuntimeError("RateLimitError: Error code: 429 - Rate limit reached"), {"vote": 4}))
    j.transport_backoff = (0, 0)
    assert ask(j) == ({"vote": 4}, False) and j.budget.used()["calls"] == 1 and j.budget.used()["failed_calls"] == 0     # not the model's answer: not counted
    blocked = judges(tmp_path, Scripted(RuntimeError("RateLimitError: Error code: 429 - insufficient_quota")), session="s-quota")
    blocked.transport_backoff = (0, 0)
    with pytest.raises(sj.ProviderBlocked, match="provider blocked"):
        ask(blocked, prompt="other")


def test_the_cache_key_is_the_one_of_the_calls_already_registered(tmp_path):
    """The key formula is shared with the calls made since 18 Oct 2026; changing it would ask (and pay for) every cached answer again."""
    block = config()["judges"][0]
    expected = scoring_io.sha256_obj(["t", block["provider"], block["model"]["id"], block["params"], ["k"], scoring_io.sha256_text("S"), scoring_io.sha256_text("P"), "scripted-judges-1"])
    ask(judges(tmp_path, Scripted({"vote": 5})))
    assert scoring_io.read_jsonl(tmp_path / "calls.jsonl")[0]["cache_key"] == expected
    assert copy.deepcopy(block) == config()["judges"][0]

"""The model calls of the evaluation of the forum proposals (forum_scoring.py): its configuration, a cache, an append-only register of every call, and the limits checked BEFORE a call.

Configuration (`config/forum_scoring.json`, schema `forum-scoring-config-v1`): two judges (`judge_1`, `judge_2`) with provider, model and parameters, the embedder of the corpora, how
many neighbours each judge reads (`k`), how many more the lexical search adds for the hybrid novelty judge (`lexical_k`, optional, 0 = none), how many calls may run at once, the limits of one scoring session and the retries. Nothing is guessed: a live run needs `run_live: true` and
the exact model id. The provider `synthetic` is a stand-in for tests and can never run live.

Cache and register: `<workspace>/calls.jsonl`, one line per attempt (valid or not): cache key, session, task, raw text, the validated answer, tokens, latency, error. A call whose
valid answer is already in the register is not made again; a deliberate repetition needs a different key.

Limits (`budget`): calls, tokens, model time, per session (one scoring run). They are checked before every call against what it could still consume (estimated input plus its whole
output allowance); failed attempts count. For a real provider a limit of 0 means no remote work at all; for the synthetic stand-in it means unlimited. A refusal of the provider (no
credits, invalid key) stops the session like a limit; a network fault is waited out.
"""
import threading
import time
import uuid
from pathlib import Path

from . import scoring_io
from .forum_providers import BudgetExceeded, Cancelled, TurnRequest

CONFIG_SCHEMA = "forum-scoring-config-v1"
JUDGE_ROLES = ("judge_1", "judge_2")
PROVIDERS = ("openai", "synthetic")
BUDGET_KEYS = ("max_live_calls", "max_tokens", "max_seconds")
MAX_RAW = 20000
PROVIDER_BLOCK_MARKERS = ("insufficient_quota", "no credits remaining", "invalid_api_key", "Incorrect API key", "AuthenticationError", "PermissionDeniedError")
TRANSPORT_MARKERS = ("APIConnectionError", "APITimeoutError", "Connection error", "ReadTimeout", "ConnectTimeout", "InternalServerError", "ServiceUnavailableError",
                     "RateLimitError")    # a 429 that reaches here has outlasted the provider's own waits; insufficient_quota is a PROVIDER_BLOCK_MARKER, checked first


class JudgeError(ValueError):
    pass


class ConfigError(ValueError):
    pass


class ProviderBlocked(BudgetExceeded):
    """The provider refuses to serve us (no credits, invalid key). A session stops here like it does at a limit, and resumes unchanged once the cause is fixed."""


# --- configuration ------------------------------------------------------------------------------------------------------------------------

def validate_config(cfg):
    """-> the configuration, checked, or ConfigError."""
    if not isinstance(cfg, dict) or cfg.get("schema") != CONFIG_SCHEMA:
        raise ConfigError(f"config.schema must be {CONFIG_SCHEMA!r}")
    if not (isinstance(cfg.get("name"), str) and cfg["name"].strip()):
        raise ConfigError("config.name is required")
    live = cfg.get("run_live")
    if not isinstance(live, bool):
        raise ConfigError("config.run_live must be true or false")
    judges = cfg.get("judges")
    if not isinstance(judges, list) or [j.get("role") for j in judges] != list(JUDGE_ROLES):
        raise ConfigError("config.judges must be exactly judge_1 and judge_2, in this order")
    for j in judges:
        if j.get("provider") not in PROVIDERS:
            raise ConfigError(f"{j['role']}: provider must be one of {PROVIDERS}")
        if j["provider"] == "synthetic" and live:
            raise ConfigError("the synthetic provider is a test stand-in: it can never be used in a live run")
        if not (j.get("model") or {}).get("id"):
            raise ConfigError(f"{j['role']}: model.id is required")
        params = j.get("params") or {}
        if params.get("reasoning_effort") not in (None, "minimal", "low", "medium", "high"):
            raise ConfigError(f"{j['role']}: params.reasoning_effort must be minimal, low, medium, high or null")
        if params.get("temperature") is not None and not 0 <= params["temperature"] <= 2:
            raise ConfigError(f"{j['role']}: params.temperature must be between 0 and 2 or null")
        if not isinstance(params.get("max_output_tokens"), int) or params["max_output_tokens"] < 500:
            raise ConfigError(f"{j['role']}: params.max_output_tokens must be a whole number >= 500")
    if live and any(j["provider"] != "openai" for j in judges):
        raise ConfigError("a live run needs provider openai for every judge")
    if not (isinstance(cfg.get("embedding"), str) and (cfg["embedding"] == "hash-rp-v1" or cfg["embedding"].startswith("openai:"))):
        raise ConfigError("config.embedding must be hash-rp-v1 or openai:<model>")
    if not isinstance(cfg.get("k"), int) or not 1 <= cfg["k"] <= 10:
        raise ConfigError("config.k must be a whole number from 1 to 10")
    lexical = cfg.get("lexical_k", 0)
    if isinstance(lexical, bool) or not isinstance(lexical, int) or not 0 <= lexical <= 10:
        raise ConfigError("config.lexical_k must be a whole number from 0 to 10 (how many papers only the lexical search brings to the hybrid novelty judge; 0 = no hybrid judge)")
    if not isinstance(cfg.get("concurrency"), int) or not 1 <= cfg["concurrency"] <= 16:
        raise ConfigError("config.concurrency must be a whole number from 1 to 16")
    budget = cfg.get("budget") or {}
    for key in BUDGET_KEYS:
        if isinstance(budget.get(key), bool) or not isinstance(budget.get(key), (int, float)) or budget[key] < 0:
            raise ConfigError(f"config.budget.{key} must be a non-negative number (0 = no remote work allowed)")
    if live and not all(budget[k] for k in BUDGET_KEYS):
        raise ConfigError("a live run needs a positive " + ", ".join(k for k in BUDGET_KEYS if not budget[k]))
    retries = cfg.get("retries") or {}
    for key in ("max_attempts", "max_total_failures"):
        if not isinstance(retries.get(key), int) or retries[key] < 1:
            raise ConfigError(f"config.retries.{key} must be a whole number >= 1")
    return cfg


def load_config(path):
    return validate_config(scoring_io.read_json(path))


def fingerprint(cfg):
    return scoring_io.sha256_obj(cfg)


# --- register and limits --------------------------------------------------------------------------------------------------------------------

def is_provider_block(error):
    return any(marker in (error or "") for marker in PROVIDER_BLOCK_MARKERS)


def is_transport_fault(error):
    return any(marker in (error or "") for marker in TRANSPORT_MARKERS)


def _not_a_model_answer(row):
    """A refusal of the provider or a network fault: the model never answered, so the attempt does not count against the call or the limits."""
    return is_provider_block(row.get("error")) or is_transport_fault(row.get("error"))


def estimate_tokens(*texts):
    return sum(len(t) for t in texts) // 3 + 1                    # deliberately high: characters / 3


class CallLedger:
    """calls.jsonl with an in-memory copy; appended to, never rewritten."""

    def __init__(self, path):
        self.path = Path(path)
        self._rows = None
        self._lock = threading.Lock()

    def _load(self):
        if self._rows is None:
            self._rows = scoring_io.read_jsonl(self.path)
        return self._rows

    def append(self, row):
        with self._lock:                                  # several worker threads may finish together: one line at a time, in the file and in memory
            scoring_io.append_jsonl(self.path, row)
            self._load().append(row)

    def find_ok(self, key):
        for row in reversed(self._load()):
            if row["cache_key"] == key and row["status"] == "ok":
                return row
        return None

    def failures(self, key):
        return sum(1 for r in self._load() if r["cache_key"] == key and r["status"] == "failed" and not _not_a_model_answer(r))

    def session(self, session_id):
        return [r for r in self._load() if r["session_id"] == session_id]


class Budget:
    """The limits of one session, rebuilt from the register so that a resumed session honours what was already spent."""

    def __init__(self, limits, ledger, session_id):
        self.limits, self.ledger, self.session_id = limits, ledger, session_id

    def used(self):
        rows = [r for r in self.ledger.session(self.session_id) if not _not_a_model_answer(r)]
        return {"calls": len(rows), "failed_calls": sum(1 for r in rows if r["status"] == "failed"), "tokens": sum(r["input_tokens"] + r["output_tokens"] for r in rows),
                "seconds": round(sum(r["latency_ms"] or 0 for r in rows) / 1000, 3)}

    def check(self, *, estimated_input, reserve_output, remote, inflight_calls=0, inflight_tokens=0):
        """Raise BudgetExceeded if the call about to be made could pass a limit. Calls already started and not yet recorded (other workers) count with their whole allowance."""
        used = self.used()
        for key in BUDGET_KEYS:
            if remote and not self.limits.get(key):
                raise BudgetExceeded(f"{key}: remote work is not allowed by this configuration (limit 0)")
        limit = self.limits.get("max_live_calls")
        if limit and used["calls"] + inflight_calls + 1 > limit:
            raise BudgetExceeded(f"max_live_calls: call {used['calls'] + inflight_calls + 1} would pass the limit {limit}")
        limit = self.limits.get("max_tokens")
        if limit and used["tokens"] + inflight_tokens + estimated_input + reserve_output > limit:
            raise BudgetExceeded(f"max_tokens: {used['tokens'] + inflight_tokens + estimated_input + reserve_output} could pass the limit {limit}")
        limit = self.limits.get("max_seconds")
        if limit and used["seconds"] >= limit:
            raise BudgetExceeded(f"max_seconds: {used['seconds']} s of model time already used, limit {limit}")


# --- the judges -----------------------------------------------------------------------------------------------------------------------------

class Judges:
    """The two judges of a configuration, with the register and the limits of one session. `providers` = {judge_role: provider}."""

    def __init__(self, workspace, cfg, providers, *, session_id, gate=None):
        self.cfg = validate_config(cfg)
        self.providers = providers
        if set(providers) != set(JUDGE_ROLES):
            raise JudgeError("providers must cover judge_1 and judge_2")
        for j in self.cfg["judges"]:
            fake = bool(getattr(providers[j["role"]], "synthetic", False))
            if j["provider"] == "openai" and not self.cfg["run_live"]:
                raise JudgeError(f"{j['role']}: provider openai needs run_live true in the configuration")
            if (j["provider"] == "synthetic") != fake:
                raise JudgeError(f"{j['role']}: the configured provider ({j['provider']}) and the provider object given do not match")
        self.ledger = CallLedger(Path(workspace) / "calls.jsonl")
        self.session_id = session_id
        self.budget = Budget(self.cfg["budget"], self.ledger, session_id)
        self.gate = gate or (lambda: None)
        self.config_sha = fingerprint(self.cfg)
        self.transport_backoff = (10, 30, 60, 120, 240)    # seconds to wait after successive network faults on one call; after the last one the session stops
        self._lock = threading.Lock()
        self._inflight = {"calls": 0, "tokens": 0}

    def judge(self, role):
        return next(j for j in self.cfg["judges"] if j["role"] == role)

    def _reserve(self, block, estimated_input):
        """Check the limits and RESERVE this call's worst case until it is recorded."""
        reserve = estimated_input + block["params"]["max_output_tokens"]
        with self._lock:
            self.budget.check(estimated_input=estimated_input, reserve_output=block["params"]["max_output_tokens"], remote=block["provider"] != "synthetic",
                              inflight_calls=self._inflight["calls"], inflight_tokens=self._inflight["tokens"])
            self._inflight["calls"] += 1
            self._inflight["tokens"] += reserve
        return reserve

    def _release(self, reserve):
        with self._lock:
            self._inflight["calls"] -= 1
            self._inflight["tokens"] -= reserve

    def ask(self, role, *, task, system, prompt, schema_name, schema, validate, key_parts):
        """One cached, budgeted, validated call of judge `role`. `validate(obj)` -> list of problems. -> (obj, cached). An invalid answer is asked again within retries.max_attempts;
        after that JudgeError. Budget, Stop and a refusal of the provider propagate."""
        block, provider = self.judge(role), self.providers[role]
        # the key of the calls made since 18 Oct 2026: keep the formula, or the answers already in the register are asked (and paid for) again
        key = scoring_io.sha256_obj([task, block["provider"], block["model"].get("id"), block["params"], key_parts, scoring_io.sha256_text(system), scoring_io.sha256_text(prompt),
                                     getattr(provider, "cache_identity", None)])
        hit = self.ledger.find_ok(key)
        if hit:
            return hit["response"], True
        retries, failures, attempts, streak, last = self.cfg["retries"], self.ledger.failures(key), 0, 0, None
        while attempts < retries["max_attempts"]:
            if failures >= retries["max_total_failures"]:
                raise JudgeError(f"{task}: left without an answer after {failures} failed attempts: {last}")
            reserved = self._reserve(block, estimate_tokens(system, prompt))
            try:
                self.gate()
                params, turn, error, obj = block["params"], None, None, None
                request = TurnRequest(model=block["model"]["id"], system=system, prompt=prompt, task=task, tools=[], web_search=False, max_tool_rounds=0,
                                      reasoning_effort=params.get("reasoning_effort"), temperature=params.get("temperature"), max_output_tokens=params["max_output_tokens"],
                                      json_schema={"name": schema_name, "schema": scoring_io.provider_schema(schema)}, hints={"role": task})
                try:
                    turn = provider.run(request)
                    if turn.finish == "incomplete":
                        error = "the answer was cut off at max_output_tokens"
                    else:
                        obj, why = scoring_io.parse_json(turn.text)
                        error = why or ("; ".join(validate(obj)[:4]) or None)
                except (BudgetExceeded, Cancelled):
                    raise
                except Exception as exc:                                  # an API error is an attempt that failed and is counted, not a crash
                    error = f"{type(exc).__name__}: {str(exc)[:200]}"
                self.ledger.append({"call_id": str(uuid.uuid4()), "session_id": self.session_id, "created_at": time.strftime("%Y-%m-%dT%H:%M:%S+00:00", time.gmtime()),
                                    "config_sha256": self.config_sha, "task": task, "cache_key": key, "judge_role": role, "status": "failed" if error else "ok", "error": error,
                                    "raw_text": (getattr(turn, "text", "") or "")[:MAX_RAW], "response": None if error else obj, "provider": block["provider"],
                                    "model": block["model"]["id"], "model_resolved": getattr(turn, "resolved_model", None) or None, "synthetic": bool(getattr(provider, "synthetic", False)),
                                    "input_tokens": getattr(turn, "input_tokens", 0), "output_tokens": getattr(turn, "output_tokens", 0),
                                    "reasoning_tokens": getattr(turn, "reasoning_tokens", 0), "latency_ms": getattr(turn, "latency_ms", 0) or 0})
            finally:
                self._release(reserved)
            if error and is_provider_block(error):
                raise ProviderBlocked("provider blocked: " + error[:160])
            if error and is_transport_fault(error):
                streak += 1                                            # a network fault is not the model's answer: wait, and try the same attempt again
                if streak > len(self.transport_backoff):
                    raise ProviderBlocked("provider unreachable: " + error[:160])
                time.sleep(self.transport_backoff[streak - 1])
                continue
            streak = 0
            if not error:
                return obj, False
            attempts += 1
            failures += 1
            last = error
        raise JudgeError(f"{task}: no valid answer: {last}")

"""Language-model providers for the discovery forum.

A provider runs ONE turn: it sends the request to a model, lets the model call the tools the orchestrator offers (looping until the model answers in
text or the tool rounds run out) and returns the text with the tokens used. The orchestrator owns everything else (what the agent sees, the budgets,
what is stored). Before every model call the provider calls `request.gate()`, which raises when a budget is spent or a person pressed Stop, and after
every call it reports the usage through `request.on_usage`, so a turn can never run past a cap by more than the call already in flight.

`OpenAIProvider` uses the Responses API and SENDS the prompts to OpenAI (costs money). It is selected only by an explicit acknowledgement in the interface.
It is the only provider: the free, offline simulation (`scripted`) was removed on 13 Oct 2026 at the user's request and survives only as a test fixture
(tests/forum_fake.py), which the application cannot select.
"""
import json
import time
from dataclasses import dataclass, field


class BudgetExceeded(Exception):
    """A cap of the episode is spent (tokens, calls, time, retrievals, passages)."""


class Cancelled(Exception):
    """A person pressed Stop."""


@dataclass
class ToolSpec:
    name: str
    description: str
    parameters: dict


@dataclass
class TurnRequest:
    model: str
    system: str
    prompt: str
    task: str                                  # reflect | speak | synthesize | evaluate | proposal_* (the tests' stand-in needs it; a real model ignores it)
    tools: list = field(default_factory=list)
    run_tool: object = None                    # (name, arguments dict) -> str, the text handed back to the model
    gate: object = lambda: None
    on_usage: object = lambda usage: None
    max_output_tokens: int = 2000
    reasoning_effort: str = None               # low | medium | high | None
    temperature: float = None
    max_tool_rounds: int = 4
    json_schema: dict = None                   # {"name": ..., "schema": ...} for a structured answer
    web_search: bool = False
    hints: dict = field(default_factory=dict)


@dataclass
class TurnResult:
    text: str = ""
    reasoning_summary: str = ""
    finish: str = "completed"                  # completed | incomplete (the output limit was reached)
    model: str = ""
    resolved_model: str = ""                   # the model name the API says it served (a snapshot id), when the provider reports one
    latency_ms: int = 0
    llm_calls: int = 0
    input_tokens: int = 0
    output_tokens: int = 0
    reasoning_tokens: int = 0
    sources: list = field(default_factory=list)   # web sources the provider's own search tool returned


def _usage(input_tokens=0, output_tokens=0, reasoning_tokens=0):
    return {"input_tokens": int(input_tokens or 0), "output_tokens": int(output_tokens or 0), "reasoning_tokens": int(reasoning_tokens or 0)}


RATE_LIMIT_WAITS = (15, 30, 60, 120)    # seconds to wait after successive 429s on one call, on top of the SDK's own ~8 s of retries; after the last one the error propagates
MAX_RETRY_AFTER = 120                   # a longer Retry-After from the server is not trusted: the schedule above applies


def _is_rate_limit(exc):
    """A 429 that waiting can cure. `insufficient_quota` also comes as a 429 but no wait fixes it: it must reach the caller at once."""
    return type(exc).__name__ == "RateLimitError" and "insufficient_quota" not in str(exc) and getattr(exc, "code", None) != "insufficient_quota"


def _retry_after(exc):
    try:
        seconds = float(exc.response.headers.get("retry-after"))
    except (AttributeError, TypeError, ValueError):
        return None
    return seconds if 0 < seconds <= MAX_RETRY_AFTER else None


class OpenAIProvider:
    name = "openai"

    def __init__(self, client=None, sleep=time.sleep):
        if client is None:
            import os
            if not os.environ.get("OPENAI_API_KEY"):
                raise ValueError("OPENAI_API_KEY is required for the openai provider")
            from openai import OpenAI
            client = OpenAI(max_retries=4, timeout=300)
        self.client = client
        self.sleep = sleep

    def _create(self, request, kwargs):
        """responses.create, waiting out a rate limit instead of losing the turn (and the tokens it already spent). The wait is in one-second steps through `request.gate()`,
        so a Stop or a spent time limit ends it at once."""
        waits = iter(RATE_LIMIT_WAITS)
        while True:
            try:
                return self.client.responses.create(**kwargs)
            except Exception as exc:
                wait = next(waits, None)
                if wait is None or not _is_rate_limit(exc):
                    raise
                for _ in range(max(1, round(_retry_after(exc) or wait))):
                    request.gate()
                    self.sleep(1)

    def run(self, request):
        started = time.monotonic()
        result = TurnResult(model=request.model)
        items = [{"role": "user", "content": request.prompt}]
        tools = [{"type": "function", "name": t.name, "description": t.description, "parameters": t.parameters} for t in request.tools]
        if request.web_search:
            tools.append({"type": "web_search"})
        summaries = []
        for round_number in range(request.max_tool_rounds + 1):
            request.gate()
            kwargs = {"model": request.model, "instructions": request.system, "input": items, "max_output_tokens": request.max_output_tokens, "store": False}
            if tools and round_number < request.max_tool_rounds:       # the last round has no tools, so the turn ends in text
                kwargs["tools"] = tools
            if request.reasoning_effort:
                kwargs["reasoning"] = {"effort": request.reasoning_effort, "summary": "auto"}
                kwargs["include"] = ["reasoning.encrypted_content"]   # lets a stateless call carry its reasoning into the next tool round
            if request.temperature is not None:
                kwargs["temperature"] = request.temperature
            if request.json_schema:
                kwargs["text"] = {"format": {"type": "json_schema", "name": request.json_schema["name"], "strict": True, "schema": request.json_schema["schema"]}}
            response = self._create(request, kwargs)
            result.llm_calls += 1
            result.resolved_model = str(getattr(response, "model", "") or result.resolved_model)
            details =getattr(getattr(response, "usage", None), "output_tokens_details", None)
            usage = _usage(getattr(response.usage, "input_tokens", 0), getattr(response.usage, "output_tokens", 0), getattr(details, "reasoning_tokens", 0))
            result.input_tokens += usage["input_tokens"]
            result.output_tokens += usage["output_tokens"]
            result.reasoning_tokens += usage["reasoning_tokens"]
            request.on_usage(usage)
            output = list(getattr(response, "output", []) or [])
            for item in output:
                if item.type == "reasoning":
                    summaries += [s.text for s in (item.summary or []) if getattr(s, "text", "")]
                if item.type == "message":
                    for part in item.content or []:
                        for note in getattr(part, "annotations", None) or []:
                            if getattr(note, "type", "") == "url_citation":
                                result.sources.append({"url": note.url, "title": getattr(note, "title", "")})
            calls = [item for item in output if item.type == "function_call"]
            if getattr(response, "status", "completed") == "incomplete":
                result.finish = "incomplete"
            if not calls:
                result.text = (getattr(response, "output_text", "") or "").strip()
                break
            items += [item.model_dump(exclude_none=True) for item in output]
            for call in calls:
                try:
                    arguments = json.loads(call.arguments or "{}")
                    answer = request.run_tool(call.name, arguments)
                except (ValueError, TypeError) as exc:
                    answer = json.dumps({"error": f"bad tool call: {exc}"})
                items.append({"type": "function_call_output", "call_id": call.call_id, "output": answer})
        result.reasoning_summary = "\n\n".join(summaries)
        result.latency_ms = int((time.monotonic() - started) * 1000)
        return result


# The providers an episode can be run with. The scripted simulation was removed on 13 Oct 2026 (the user's request); the automatic tests add their offline stand-in
# here (tests/forum_fake.py), nothing in the application does.
FACTORIES = {"openai": OpenAIProvider}


def get_provider(name):
    if name not in FACTORIES:
        raise ValueError("The scripted simulation was removed: a simulated episode can be read but no longer run or rated" if name == "scripted" else f"Unknown provider {name!r}")
    return FACTORIES[name]()

"""
Reverse-engineer a model's preferred tool names and parameters for:
  - read_file
  - write_file
  - update_file
  - execute_bash
  - ask_user_question

Strategy:
  Round 0 (format)  -- Send a minimal tool schema and detect which output format
                       the model uses: OpenAI structured tool_calls, [TOOL_CALLS],
                       <tool_call> XML, <toolcall> XML, or inline JSON in content.
  Round 1 (elicit)  -- Ask the model to freely describe which function it would
                       call, with no tool schema provided.  Parse its answer to
                       extract candidate names / parameter keys.
  Round 2 (probe)   -- Build a minimal OpenAI-style tool schema from those
                       candidates and call the model again with tools enabled.
                       Record the actual tool_calls the model emits, OR detect
                       content-embedded JSON if the model doesn't emit structured
                       tool_calls (a known Ollama behaviour).
  Round 3 (dispatch)-- Build a tool_dispatch table: model tool_name ->
                       {python_function, param_map}.  For any tool that cannot
                       be matched to a known canonical op, ask the LLM to
                       synthesise a Python implementation and store it in
                       generated_source.

The final inferred schema is printed as JSON.  A behavioural note is included
that records whether the model emits structured tool_calls or falls back to
inline JSON in the message content.
"""

import argparse
import json
import os
import random
import re
import subprocess
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path
from textwrap import indent

import lark
import openai

# -- configuration ------------------------------------------------------------

ENDPOINT = "https://openrouter.ai/api/v1"
MODEL    = "qwen2.5-coder:7b"

API_TYPE_LABELS = {
    "openai-completions": "OpenAI Completions",
    "openai-responses":   "OpenAI Responses",
    "anthropic-messages": "Anthropic Messages",
}

_PROBE_DIR: Path | None = None
_KEY_NAME: str | None = None
_PROVIDER: str | None = None
_OBSERVED_PROVIDER: str | None = None
# Upstream URL last reported by a `script:` wrapper (see _script_upstream).
_SCRIPT_UPSTREAM: str | None = None


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Probe a model's preferred tool names/parameters.")
    p.add_argument("--endpoint", default=None, help="OpenAI-compatible base URL")
    p.add_argument("--model",    default=None, help="Model ID")
    p.add_argument("--key-name", default="OPENROUTER_API_KEY", dest="key_name",
                   help="Env-var name / keyring slot holding the API key (default: OPENROUTER_API_KEY).")
    p.add_argument("--output",   default=None, help="Output JSON file (default: reports/<server>/<model>/capabilities_<model>.json)")
    p.add_argument("--quick-summary", action="store_true", dest="quick_summary",
                   help="Read reports/<server>/<model>/capabilities_*.json files and list models with native "
                        "structured tool_call support along with their main tool parameters.")
    p.add_argument("--quote-test", action=argparse.BooleanOptionalAction, dest="quote_test",
                   default=True,
                   help="Run an extra round that probes whether the model correctly escapes "
                        "double-quotes inside JSON argument values. On by default; pass "
                        "--no-quote-test to skip.")
    p.add_argument("--efficiency-test", action=argparse.BooleanOptionalAction, dest="efficiency_test",
                   default=True,
                   help="Run an extra round that probes whether the model prefers "
                        "filtered/targeted calls (grep, sed -n, head, offset/limit reads) "
                        "over pulling entire large files/outputs into context. On by "
                        "default; pass --no-efficiency-test to skip.")
    p.add_argument("--askq-test", action=argparse.BooleanOptionalAction, dest="askq_test",
                   default=True,
                   help="Run an extra round (8 phrasing variants, full schema, one "
                        "sample each) that probes how strongly task wording drives "
                        "the model to call its own ask_user_question tool. On by "
                        "default; pass --no-askq-test to skip.")
    p.add_argument("--gram-knowledge-test", action=argparse.BooleanOptionalAction,
                   dest="gram_knowledge_test", default=True,
                   help="Run an extra round (no tool schema) that probes whether "
                        "the model naturally knows OpenAI's apply_patch envelope "
                        "grammar, parsed against the real Lark grammar. On by "
                        "default; pass --no-gram-knowledge-test to skip.")
    p.add_argument("--gram-transport-test", action=argparse.BooleanOptionalAction,
                   dest="gram_transport_test", default=True,
                   help="Run an extra round that sends a real OpenAI custom/freeform "
                        "tool (type:'custom', format:{type:'grammar', syntax:'lark'}) "
                        "and checks whether the endpoint honours it end to end, "
                        "instead of falling back to classic function calling. On by "
                        "default; pass --no-gram-transport-test to skip.")
    p.add_argument("--rjson-test", action=argparse.BooleanOptionalAction, dest="rjson_test",
                   default=True,
                   help="Run an extra round that sends a strict "
                        "response_format:{type:'json_schema'} request (no tool schema) "
                        "and checks whether the endpoint honours schema-constrained "
                        "structured output end to end. On by default; pass "
                        "--no-rjson-test to skip.")
    p.add_argument("--stream-test", action=argparse.BooleanOptionalAction, dest="stream_test",
                   default=True,
                   help="Run an extra round that sends stream:true and checks whether "
                        "the endpoint delivers real incremental SSE chunks rather than "
                        "rejecting the parameter or buffering the whole reply into one "
                        "chunk. On by default; pass --no-stream-test to skip.")
    p.add_argument("--reasoning-test", action=argparse.BooleanOptionalAction, dest="reasoning_test",
                   default=True,
                   help="Run an extra round that checks whether the endpoint surfaces "
                        "reasoning tokens (reasoning_content/reasoning/thinking field, "
                        "or usage.reasoning_tokens) and whether it accepts either "
                        "reasoning-effort syntax (native reasoning_effort param, or "
                        "extra_body={'reasoning': {'effort': ...}}) without erroring. "
                        "On by default; pass --no-reasoning-test to skip.")
    p.add_argument("--agentknit-test", action=argparse.BooleanOptionalAction,
                   dest="agentknit_test", default=True,
                   help="Run an extra round that offers agentknit's actual shipped "
                        "default tool schema (read_file/write_file/str_replace/"
                        "exec_shell, imported live from agentknit._core) and dispatches "
                        "each call through agentknit's real dispatch() against a scratch "
                        "directory, verifying both tool selection and the on-disk effect. "
                        "Requires agentknit to be importable. On by default; pass "
                        "--no-agentknit-test to skip.")
    p.add_argument("--cache-ttl-test", action=argparse.BooleanOptionalAction,
                   dest="cache_ttl_test", default=False,
                   help="Run an extra round that empirically measures the endpoint's "
                        "prompt-cache TTL by re-sending a large fixed prompt at "
                        "increasing delays and finding where the cached-token count "
                        "collapses. Off by default (it takes several minutes); pass "
                        "--cache-ttl-test to enable. Use --cache-ttl-only to run just "
                        "this round against an existing report, skipping all other "
                        "rounds.")
    p.add_argument("--cache-ttl-only", action="store_true", dest="cache_ttl_only",
                   help="Run only the CACH prompt-cache TTL round: load the existing "
                        "capabilities_<model>.json, add/refresh its cache_ttl_test "
                        "section, re-render the markdown, and exit. Implies "
                        "--cache-ttl-test. Requires a previous full probe run.")
    p.add_argument("--context-test", action=argparse.BooleanOptionalAction,
                   dest="context_test", default=True,
                   help="Recover the model's context-window size: try the endpoint's "
                        "/models metadata first, then send one deliberately oversized "
                        "prompt and parse the limit out of the rejection message (a "
                        "rejected request produces no tokens). If the oversized prompt "
                        "is accepted the value is recorded as a lower bound and nothing "
                        "larger is sent. On by default; pass --no-context-test to skip. "
                        "Use --context-only to run just this round against an existing "
                        "report.")
    p.add_argument("--context-only", action="store_true", dest="context_only",
                   help="Run only the CTX context-window recovery: load the existing "
                        "capabilities_<model>.json, add/refresh its context_window "
                        "section, re-render the markdown, and exit. Implies "
                        "--context-test. Requires a previous full probe run.")
    p.add_argument("--context-tokens", type=int, default=None, dest="context_tokens",
                   help="Record this context-window size verbatim instead of probing for "
                        "it, for endpoints that neither publish limits nor name them in "
                        "their errors. Pair with --context-evidence to say where the "
                        "number came from.")
    p.add_argument("--context-evidence", default=None, dest="context_evidence",
                   help="Free-text provenance for --context-tokens, stored in the report.")
    p.add_argument("--context-source", default="explicit", dest="context_source",
                   choices=("explicit", "api_metadata", "api_error_message",
                            "empirical_bisection"),
                   help="How the --context-tokens value was obtained, recorded verbatim in "
                        "the report. Default: explicit.")
    p.add_argument("--cors-test", action=argparse.BooleanOptionalAction,
                   dest="cors_test", default=True,
                   help="Run an extra round that replays the CORS preflight a browser "
                        "sends before a cross-origin POST /chat/completions carrying "
                        "Authorization and application/json (OPTIONS with Origin and "
                        "Access-Control-Request-* headers), plus a plain Origin-tagged "
                        "GET /models, and records the endpoint's access-control-* answer "
                        "(wildcard '*', reflected origin, or none). No credentials "
                        "needed. On by default; pass --no-cors-test to skip. Use "
                        "--cors-only to run just this round against an existing report.")
    p.add_argument("--cors-only", action="store_true", dest="cors_only",
                   help="Run only the CORS round: load the existing "
                        "capabilities_<model>.json, add/refresh its cors_test section, "
                        "re-render the markdown, and exit. Implies --cors-test. "
                        "Requires a previous full probe run.")
    p.add_argument("--tdef-test", action=argparse.BooleanOptionalAction,
                   dest="tdef_test", default=True,
                   help="Run an extra round that checks whether the endpoint honours "
                        "deferred tool loading (`defer_loading` plus a tool-search tool) "
                        "on each protocol surface it serves -- chat completions, OpenAI "
                        "Responses and Anthropic Messages. Because compatibility layers "
                        "silently drop unknown fields, the verdict comes from comparing "
                        "reported input tokens with the schemas inline vs. deferred, not "
                        "from the HTTP status. On by default; pass --no-tdef-test to "
                        "skip. Use --tdef-only to run just this round against an "
                        "existing report.")
    p.add_argument("--tdef-only", action="store_true", dest="tdef_only",
                   help="Run only the TDEF round: load the existing "
                        "capabilities_<model>.json, add/refresh its tdef_test section, "
                        "re-render the markdown, and exit. Implies --tdef-test. "
                        "Requires a previous full probe run.")
    p.add_argument("--tdef-responses-base", default=None, dest="tdef_responses_base",
                   help="Base URL of the endpoint's OpenAI Responses surface for TDEF "
                        "(default: --endpoint, i.e. <endpoint>/responses).")
    p.add_argument("--tdef-anthropic-base", default=None, dest="tdef_anthropic_base",
                   help="Base URL of the endpoint's Anthropic Messages surface for TDEF "
                        "(default: guessed as <host>/.../anthropic, which is where every "
                        "vendor probed so far puts it).")
    p.add_argument("--tdef-anthropic-model", default=None, dest="tdef_anthropic_model",
                   help="Model ID to send to the Anthropic Messages surface, for vendors "
                        "that map claude-* names onto their own models (default: --model).")
    p.add_argument("--provider", default=None, dest="provider",
                   help="Pin an aggregator's routing to one upstream provider, by its "
                        "OpenRouter slug (e.g. 'deepinfra/fp8', 'fireworks'). The same "
                        "model behind the same aggregator is served by different "
                        "software with different configuration per provider, and the "
                        "capabilities differ accordingly, so a pinned run is filed under "
                        "reports/<server>/<model>/<provider>/ and fallbacks are "
                        "disabled -- a request that provider cannot serve fails instead "
                        "of being silently rerouted. List the choices with "
                        "--list-providers.")
    p.add_argument("--list-providers", action="store_true", dest="list_providers",
                   help="Print the upstream providers OpenRouter can route --model to, "
                        "as slugs usable with --provider, and exit.")
    p.add_argument("--api-type", default="openai-completions",
                   choices=["openai-completions", "openai-responses", "anthropic-messages"],
                   help="The *actual* backend transport behind the endpoint/script, for "
                        "the report only -- probe_inference.py always speaks Chat "
                        "Completions on the wire regardless of this flag. Set it when "
                        "the target translates under the hood (e.g. a script that "
                        "accepts Chat Completions JSON but forwards to a Responses API "
                        "or Anthropic Messages backend). Default: openai-completions.")
    p.add_argument("--script", default=None,
                   help="Treat this local script as the inference server instead of "
                        "hitting --endpoint over HTTP: it must read one OpenAI-style "
                        "chat/completions JSON payload from stdin and print one JSON "
                        "response to stdout (e.g. ~/bin/*-completions.py). --model is "
                        "still used to label/log this run; --endpoint/--key-name are ignored.")
    p.add_argument("--server", default=None,
                   help="Name of the server answering this run, used as the "
                        "reports/<server>/<model>/ directory (default: the "
                        "endpoint's host, or, for --script, the upstream the "
                        "wrapper reports in 'x_upstream_endpoint').")
    p.add_argument("--render-md", action="store_true", dest="render_md_only",
                   help="Skip probing entirely; just (re)render the Markdown report "
                        "from the existing capabilities_<model>.json (or --output) on disk.")
    return p.parse_args()


def get_api_key(key_name: str = "OPENROUTER_API_KEY") -> str:
    key = os.environ.get(key_name)
    if key:
        return key
    try:
        result = subprocess.run(
            ["keyring", "get", "login2", key_name],
            capture_output=True, text=True, check=True,
        )
        return result.stdout.strip()
    except Exception as e:
        sys.exit(f"Cannot obtain API key for '{key_name}': {e}")


def _llmprobe_commit() -> str | None:
    """Return the commit of the checkout running this probe, if available."""
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=Path(__file__).resolve().parent,
            capture_output=True, text=True, check=True, timeout=2,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    return result.stdout.strip() or None


# -- helpers ------------------------------------------------------------------

# -- provider pinning ----------------------------------------------------------
#
# An aggregator like OpenRouter is not one endpoint: the same model id is served
# by a dozen upstream providers running different inference software (vLLM,
# SGLang, TensorRT, a vendor's own stack) under different configuration. They do
# not agree on capabilities -- observed directly while developing TDEF, where
# two consecutive unpinned runs of the same model on the same URL answered
# `accepted-but-ignored` once and `rejected` (an SGLang validation error) the
# next, because the request landed on different providers.
#
# So an unpinned aggregator report measures "whatever served that request",
# which is not a property of anything reproducible. `--provider` pins routing to
# one upstream and turns fallbacks off, and pinned reports are filed per
# provider. The observed provider is recorded alongside the requested one,
# because a pin that silently fails to apply would otherwise look like a result.


def _provider_routing(provider: str) -> dict:
    """OpenRouter's routing block: this provider only, never a fallback."""
    return {"order": [provider], "allow_fallbacks": False}


def _observed_provider(resp: object) -> str | None:
    """The provider an aggregator says actually served a response, if it says."""
    for source in (getattr(resp, "model_extra", None), resp):
        if isinstance(source, dict) and isinstance(source.get("provider"), str):
            return source["provider"]
    provider = getattr(resp, "provider", None)
    return provider if isinstance(provider, str) else None


def _record_observed_provider(resp: object) -> None:
    """Remember who actually served a response, so a failed pin is visible."""
    global _OBSERVED_PROVIDER
    provider = _observed_provider(resp)
    if provider:
        _OBSERVED_PROVIDER = provider


def openrouter_providers(model: str) -> list[dict]:
    """Providers OpenRouter can route `model` to, newest metadata first.

    Returns `{"slug", "name", "context_length", "supports_tools"}` per endpoint;
    `slug` is what --provider takes.
    """
    url = ("https://openrouter.ai/api/v1/models/"
           + urllib.parse.quote(model) + "/endpoints")
    with urllib.request.urlopen(url, timeout=60) as resp:
        data = json.load(resp).get("data") or {}
    providers = []
    for endpoint in data.get("endpoints") or []:
        supported = endpoint.get("supported_parameters") or []
        providers.append({
            "slug": endpoint.get("tag"),
            "name": endpoint.get("provider_name"),
            "context_length": endpoint.get("context_length"),
            "supports_tools": "tools" in supported,
        })
    return providers


class _PinnedCompletions:
    """Wraps `.chat.completions`, adding the provider block to every request."""

    def __init__(self, inner, provider: str):
        self._inner = inner
        self._provider = provider

    def create(self, **kwargs):
        extra_body = dict(kwargs.get("extra_body") or {})
        extra_body.setdefault("provider", _provider_routing(self._provider))
        kwargs["extra_body"] = extra_body
        return self._inner.create(**kwargs)


class _PinnedChat:
    def __init__(self, inner, provider: str):
        self.completions = _PinnedCompletions(inner.completions, provider)


class PinnedClient:
    """A client that pins every chat call to one upstream provider.

    Everything other than `.chat` is delegated untouched, so `/models` metadata
    lookups and the like keep working exactly as before.
    """

    def __init__(self, client, provider: str):
        self._client = client
        self.chat = _PinnedChat(client.chat, provider)

    def __getattr__(self, name):
        return getattr(self._client, name)


def make_client(api_key: str) -> "openai.OpenAI | PinnedClient":
    if not api_key.strip():
        # Some free endpoints (e.g. opencode.ai/zen) require no auth at all
        # and 401 on any non-empty bearer token. httpx also rejects a
        # whitespace-only header value outright, so blank it via
        # default_headers rather than passing it through api_key.
        client = openai.OpenAI(api_key="unused", base_url=ENDPOINT,
                               default_headers={"Authorization": ""})
    else:
        client = openai.OpenAI(api_key=api_key, base_url=ENDPOINT)
    return PinnedClient(client, _PROVIDER) if _PROVIDER else client


# -- local-script "inference server" adapter -----------------------------------
#
# Some models are only reachable through a local wrapper script rather than a
# plain HTTP endpoint (auth handled internally, non-standard transport, etc.
# see e.g. ~/bin/*-completions.py). Such a script reads one OpenAI-style
# chat/completions JSON payload from stdin and prints one JSON response to
# stdout. ScriptClient duck-types just enough of the openai.OpenAI client
# (`.chat.completions.create(**kwargs)`) for chat() to treat that script as
# the inference server, with no knowledge of what's inside it.

class _ScriptChatCompletions:
    def __init__(self, script_path: str):
        self.script_path = script_path

    def create(self, **kwargs) -> "openai.types.chat.ChatCompletion":
        global _SCRIPT_UPSTREAM
        timeout = kwargs.pop("timeout", 300)
        proc = subprocess.run(
            [sys.executable, self.script_path],
            input=json.dumps(kwargs),
            capture_output=True, text=True, timeout=timeout,
        )
        if proc.returncode != 0:
            raise RuntimeError(f"{self.script_path} exited {proc.returncode}: "
                               f"{proc.stderr.strip()[:1000]}")
        data = json.loads(proc.stdout)
        # A wrapper hides which server actually answered, so it may name it in
        # "x_upstream_endpoint" (see _script_upstream). Strip it before
        # validation: it is bookkeeping for the report, not part of the reply.
        upstream = data.pop("x_upstream_endpoint", None)
        if isinstance(upstream, str) and upstream.strip():
            _SCRIPT_UPSTREAM = upstream.strip()
        # Some scripts return a near-OpenAI-compatible payload missing the
        # bookkeeping fields (e.g. Copilot's API omits "object"/"created").
        # Backfill rather than fail validation over fields nothing here reads.
        data.setdefault("object", "chat.completion")
        data.setdefault("created", 0)
        return openai.types.chat.ChatCompletion.model_validate(data)


class _ScriptChat:
    def __init__(self, script_path: str):
        self.completions = _ScriptChatCompletions(script_path)


class ScriptClient:
    def __init__(self, script_path: str):
        self.chat = _ScriptChat(script_path)


def _safe_server(endpoint: str) -> str:
    """Filesystem-safe slug for the server named by `endpoint`.

    Endpoints collapse to their host (`https://api.kimi.com/coding/v1` ->
    `api.kimi.com`), so every model served by the same provider shares one
    directory. A `script:` target is a local wrapper, not a server: its real
    upstream comes from _script_upstream() and is passed here instead.
    """
    endpoint = (endpoint or "").strip()
    if not endpoint or endpoint.startswith("script:"):
        return "unknown-server"
    host = urllib.parse.urlsplit(endpoint).netloc or endpoint
    return re.sub(r"[^A-Za-z0-9._-]+", "_", host).strip("_") or "unknown-server"


def _script_upstream(script_path: str) -> str | None:
    """Ask a wrapper script which server it actually talks to.

    A wrapper reaches the real inference server on our behalf, so only it knows
    the upstream URL — which for e.g. Copilot is handed out at auth time and
    cannot be read off the script. The script may report it in an
    "x_upstream_endpoint" field on any response; this sends the smallest
    possible request to obtain one. Returns None if the script does not
    cooperate, in which case the run is filed under "unknown-server".
    """
    global _SCRIPT_UPSTREAM
    _SCRIPT_UPSTREAM = None
    try:
        # No max_tokens: some upstreams 400 on it, and a rejected request
        # never gets far enough to name the server.
        _ScriptChatCompletions(script_path).create(
            model=MODEL,
            messages=[{"role": "user", "content": "hi"}],
            timeout=120,
        )
    except Exception:
        pass  # a wrapper that cannot answer this cannot name its server either
    return _SCRIPT_UPSTREAM


def _known_server_for(endpoint: str, safe_model: str) -> str | None:
    """Server slug of an existing report for this model probed via `endpoint`.

    Lets a re-run of a script target land in the directory the previous run
    resolved, without paying for another upstream-discovery call.
    """
    for path in sorted(Path("reports").glob(f"*/{safe_model}/capabilities_{safe_model}.json")):
        try:
            with open(path) as f:
                data = json.load(f)
        except (OSError, ValueError):
            continue
        if data.get("endpoint") == endpoint:
            return path.parent.parent.name
    return None


def _safe_provider(provider: str) -> str:
    """Filesystem-safe slug for a provider tag (`deepinfra/fp8` -> `deepinfra_fp8`)."""
    return re.sub(r"[^A-Za-z0-9._-]+", "_", provider).strip("_") or "unknown-provider"


def _report_dir(endpoint: str, safe_model: str, server: str | None = None,
                provider: str | None = None) -> Path:
    """reports/<server>/<model>/ — one directory per model per server.

    A run pinned to one upstream provider of an aggregator nests one level
    deeper, reports/<server>/<model>/<provider>/, because those are genuinely
    different servers behind one URL. Unpinned runs keep the old two-level
    path, so existing reports stay where they are.
    """
    base = Path("reports") / (server or _safe_server(endpoint)) / safe_model
    return base / _safe_provider(provider) if provider else base


def _init_probe_dir(safe_model: str) -> None:
    global _PROBE_DIR
    _PROBE_DIR = Path("probes") / safe_model
    _PROBE_DIR.mkdir(parents=True, exist_ok=True)


def _load_previous_report(out_path: str) -> dict:
    """Load a previous run's report, if any, so results can be preserved."""
    path = Path(out_path)
    if not path.exists():
        return {}
    try:
        with open(path) as f:
            previous = json.load(f)
        return previous if isinstance(previous, dict) else {}
    except (json.JSONDecodeError, OSError) as e:
        print(f"WARNING: ignoring unreadable previous report {out_path}: {e}")
        return {}


_RESULT_KEYS = (
    "status", "error",
    "format_detection", "elicited_names", "inferred_tool_schema",
    "behaviour", "tool_dispatch", "dispatch_conflicts",
    "quote_test", "token_efficiency_test", "askq_test",
    "gram_knowledge_test", "gram_transport_test", "rjson_test",
    "stream_test", "reasoning_test", "agentknit_test", "cache_ttl_test",
    "context_window", "cors_test", "tdef_test",
)


def _is_rate_limit_error(exc: Exception) -> bool:
    """True for quota/rate-limit failures (429) whose results should be kept."""
    status = getattr(exc, "status_code", None)
    if status == 429:
        return True
    text = str(exc)
    return "429" in text or "rate limit" in text.lower() or "limit exhausted" in text.lower()


def _restore_previous_on_429(previous: dict, output: dict) -> None:
    """Copy every usable previous-run result into `output`.

    Used when this run dies on a 429 quota/rate-limit error: keys for which
    the current run already holds a usable result keep it; the rest are
    restored from the previous report so a quota-exhausted rerun never wipes
    a good earlier answer.
    """
    for key in _RESULT_KEYS:
        current = output.get(key)
        fresh = bool(current) and not (isinstance(current, dict) and set(current.keys()) == {"error"})
        if fresh:
            continue
        prev = previous.get(key)
        if not prev:
            continue
        if isinstance(prev, dict) and set(prev.keys()) == {"error"}:
            continue
        output[key] = prev


def _keep_previous_result(exc: Exception, previous: dict, key: str) -> bool:
    """Decide whether to keep the previous run's result for `key` on failure.

    True when this run failed on a 429 quota/rate-limit error and the
    previous report holds a usable (non-empty, non-error) result for `key`.
    """
    if not _is_rate_limit_error(exc):
        return False
    prev = previous.get(key)
    if not prev:
        return False
    if isinstance(prev, dict) and set(prev.keys()) == {"error"}:
        return False
    return True


def _save_probe(label: str, messages: list[dict],
                resp: openai.types.chat.ChatCompletion,
                tools: list[dict] | None = None) -> None:
    if _PROBE_DIR is None:
        return
    safe_label = re.sub(r"[^a-zA-Z0-9_\-]", "_", label)
    data: dict = {"label": label, "messages": messages, "response": resp.model_dump()}
    if tools is not None:
        data["tools"] = tools
    with open(_PROBE_DIR / f"{safe_label}.json", "w") as f:
        json.dump(data, f, indent=2)


def _probe_temperature() -> int:
    """Return a temperature accepted by the active endpoint.

    Kimi's Coding API currently permits only its default temperature (1),
    whereas the probe normally uses 0 for repeatable elicitation.
    """
    return 1 if "api.kimi.com/coding/" in ENDPOINT else 0


def chat(client: openai.OpenAI, messages: list[dict], tools: list[dict] | None = None,
         tool_choice="auto") -> openai.types.chat.ChatCompletion:
    kwargs: dict = dict(model=MODEL, messages=messages,
                        temperature=_probe_temperature(), timeout=300)
    if tools:
        kwargs["tools"] = tools
        kwargs["tool_choice"] = tool_choice
    try:
        resp = client.chat.completions.create(**kwargs)
        _record_observed_provider(resp)
        return resp
    except openai.APITimeoutError as e:
        sys.exit(f"ERROR: LLM call timed out after 300 seconds. Model={MODEL}, messages={json.dumps(messages, indent=2, default=str)[:500]}... Exception: {e}")


def extract_json_block(text: str) -> dict | list | None:
    """Pull the first JSON object or array out of free text."""
    for pattern in (r"```json\s*([\s\S]+?)\s*```", r"```\s*([\s\S]+?)\s*```"):
        m = re.search(pattern, text)
        if m:
            try:
                return json.loads(m.group(1))
            except json.JSONDecodeError:
                pass
    for pattern in (r"\[[\s\S]+\]", r"\{[\s\S]+\}"):
        m = re.search(pattern, text)
        if m:
            try:
                return json.loads(m.group(0))
            except json.JSONDecodeError:
                pass
    return None


def section(title: str):
    print(f"\n{'─'*60}")
    print(f"  {title}")
    print(f"{'─'*60}")


# -- unified call extraction --------------------------------------------------

class ToolCallResult:
    """Holds whatever the model returned -- structured tool_call or inline JSON."""

    def __init__(self, function_name: str, arguments: dict, structured: bool):
        self.function_name = function_name
        self.arguments = arguments
        self.structured = structured

    def to_dict(self) -> dict:
        return {"function_name": self.function_name, "arguments": self.arguments}

    def __repr__(self):
        mode = "structured tool_call" if self.structured else "inline JSON in content"
        return f"ToolCallResult({self.function_name}, mode={mode}, args={self.arguments})"


def _extract_xml_tool_call(text: str) -> tuple[str, dict] | None:
    """Extract (name, arguments) from <tool_call> or <toolcall> XML in content.

    Falls back to regex extraction when the JSON payload is truncated/malformed.
    """
    m = re.search(r"<tool_?call[^>]*>\s*([\s\S]+?)(?:\s*</tool_?call>|$)", text, re.IGNORECASE)
    if not m:
        return None
    body = m.group(1).strip()

    parsed = extract_json_block(body)
    if isinstance(parsed, dict):
        fn   = parsed.get("name") or parsed.get("function_name")
        args = parsed.get("arguments") or parsed.get("parameters") or {}
        if fn:
            return fn, args if isinstance(args, dict) else {}

    # Optimistic: name is almost always intact even when the rest is truncated.
    name_m = re.search(r'"name"\s*:\s*"([^"]+)"', body)
    if not name_m:
        return None
    fn = name_m.group(1)

    args: dict = {}
    args_m = re.search(r'"arguments"\s*:\s*(\{[\s\S]*)', body)
    if args_m:
        candidate = extract_json_block(args_m.group(1))
        if isinstance(candidate, dict):
            args = candidate
    return fn, args


def extract_call_from_response(resp: openai.types.chat.ChatCompletion) -> ToolCallResult | None:
    """Extract a tool call from the response regardless of delivery mechanism."""
    msg = resp.choices[0].message
    if msg.tool_calls:
        tc = msg.tool_calls[0]
        try:
            arguments = json.loads(tc.function.arguments)
        except (json.JSONDecodeError, ValueError):
            # Some models (e.g. kimi-k2.6) emit truncated/malformed JSON in
            # tool_call arguments.  Fall back to partial extraction so the
            # caller receives a ToolCallResult instead of a crash.
            arguments = extract_json_block(tc.function.arguments) or {}
            if not isinstance(arguments, dict):
                arguments = {}
        return ToolCallResult(
            function_name=tc.function.name,
            arguments=arguments,
            structured=True,
        )
    if msg.content:
        xml_result = _extract_xml_tool_call(msg.content)
        if xml_result:
            fn, args = xml_result
            return ToolCallResult(function_name=fn, arguments=args, structured=False)
        parsed = extract_json_block(msg.content)
        if isinstance(parsed, dict):
            fn = parsed.get("name") or parsed.get("function_name")
            args = parsed.get("arguments") or parsed.get("parameters") or {}
            if fn and isinstance(args, dict):
                return ToolCallResult(function_name=fn, arguments=args, structured=False)
    return None


# -- Round 0: tool-call format detection -------------------------------------

_FORMAT_PROBE_TOOL = [{
    "type": "function",
    "function": {
        "name": "read_file",
        "description": "Read a file from disk.",
        "parameters": {
            "type": "object",
            "properties": {"path": {"type": "string", "description": "file path"}},
            "required": ["path"],
        },
    },
}]

_FORMAT_PATTERNS: list[tuple[str, str]] = [
    ("TOOL_CALLS_bracket", r"\[TOOL_CALLS\]"),
    ("xml_tool_call",      r"<tool_call\b"),
    ("xml_toolcall",       r"<toolcall\b"),
]


def _detect_call_format(msg) -> tuple[str, str]:
    """Return the transport/style detected in one completion message."""
    raw_content = msg.content or ""
    if msg.tool_calls:
        return "structured_tool_calls", raw_content
    for fmt, pattern in _FORMAT_PATTERNS:
        if re.search(pattern, raw_content, re.IGNORECASE):
            return fmt, raw_content
    if extract_json_block(raw_content):
        return "inline_json", raw_content
    return "unknown", raw_content


def format_detection_round(client: openai.OpenAI) -> dict:
    """Round 0 -- detect the model's preferred tool-call output format."""
    section("Round 0 -- Tool-call format detection")
    messages = [
        {"role": "system", "content": "You are a helpful assistant with tool access."},
        {"role": "user",   "content": "Read the file /etc/hostname."},
    ]
    resp = chat(client, messages, tools=_FORMAT_PROBE_TOOL)
    _save_probe("round0_format_detection", messages, resp, tools=_FORMAT_PROBE_TOOL)

    msg = resp.choices[0].message
    detected, raw_content = _detect_call_format(msg)

    # Do not offer a function schema here: this deliberately checks whether
    # the model can serialize an XML-style call in message content when the
    # task explicitly requests it, rather than merely noticing XML by chance.
    xml_messages = [
        {
            "role": "system",
            "content": "You are testing tool-call serialization. Follow the requested output format exactly.",
        },
        {
            "role": "user",
            "content": (
                "Return ONLY this XML-style call to read /etc/hostname; no prose, Markdown, "
                "or JSON outside the XML envelope:\n"
                '<tool_call>{"name":"read_file","arguments":{"path":"/etc/hostname"}}</tool_call>'
            ),
        },
    ]
    try:
        xml_resp = chat(client, xml_messages)
        _save_probe("round0_xml_tool_call", xml_messages, xml_resp)
        xml_msg = xml_resp.choices[0].message
        xml_detected, xml_content = _detect_call_format(xml_msg)
        xml_call = _extract_xml_tool_call(xml_content) if not xml_msg.tool_calls else None
        xml_name, xml_args = xml_call if xml_call else (None, None)
        xml_passed = bool(
            xml_detected in {"xml_tool_call", "xml_toolcall"}
            and xml_name == "read_file"
            and isinstance(xml_args, dict)
            and xml_args.get("path") == "/etc/hostname"
        )
        xml_result = {
            "pass": xml_passed,
            "detected_format": xml_detected,
            "structured_tool_calls": bool(xml_msg.tool_calls),
            "function_name": xml_name,
            "arguments": xml_args,
            "raw_content_snippet": xml_content[:300] or None,
            "error": None,
        }
    except Exception as e:
        xml_result = {
            "pass": False,
            "detected_format": None,
            "structured_tool_calls": None,
            "function_name": None,
            "arguments": None,
            "raw_content_snippet": None,
            "error": str(e),
        }

    result = {
        "detected_format":            detected,
        "has_structured_tool_calls":  bool(msg.tool_calls),
        "raw_content_snippet":        raw_content[:300] or None,
        "xml_tool_call_test":          xml_result,
    }
    print(f"  detected_format           : {detected}")
    print(f"  has_structured_tool_calls : {result['has_structured_tool_calls']}")
    if raw_content:
        print(f"  raw content snippet       : {raw_content[:200]!r}")
    if xml_result["error"]:
        print(f"  XML tool-call test        : ERROR -- {xml_result['error']}")
    else:
        print(f"  XML tool-call test        : {'PASS' if xml_result['pass'] else 'FAIL'} "
              f"(detected `{xml_result['detected_format']}`)")
    return result


# -- Round 1: elicit free-form descriptions -----------------------------------

ELICIT_TASKS = {
    "read_file": (
        "You need to read the contents of the file /etc/hostname. "
        "What function would you call and what arguments would you pass? "
        "Respond with a JSON object like: "
        '{"function_name": "...", "arguments": {"arg1": "val1", ...}}'
    ),
    "write_file": (
        "You need to write the text 'hello world' to the file /tmp/test.txt. "
        "What function would you call and what arguments would you pass? "
        "Respond with a JSON object like: "
        '{"function_name": "...", "arguments": {"arg1": "val1", ...}}'
    ),
    "update_file": (
        "The file /tmp/test.py already exists and contains Python code. "
        "You need to make a targeted edit: replace the exact string 'x = 1' with 'x = 42', "
        "without rewriting the whole file. "
        "What function would you call and what arguments would you pass? "
        "Respond with a JSON object like: "
        '{"function_name": "...", "arguments": {"arg1": "val1", ...}}'
    ),
    "execute_bash": (
        "You need to run the shell command `ls -la /tmp`. "
        "What function would you call and what arguments would you pass? "
        "Respond with a JSON object like: "
        '{"function_name": "...", "arguments": {"arg1": "val1", ...}}'
    ),
    "ask_user_question": (
        "You need to ask the user a clarifying question: "
        "'Should I overwrite the existing file, or create a backup first?' "
        "with options 'Overwrite' and 'Backup'. "
        "What function would you call and what arguments would you pass? "
        "Respond with a JSON object like: "
        '{"function_name": "...", "arguments": {"arg1": "val1", ...}}'
    ),
    "list_directory": (
        "You need to list all files and subdirectories inside /tmp. "
        "What function would you call and what arguments would you pass? "
        "Respond with a JSON object like: "
        '{"function_name": "...", "arguments": {"arg1": "val1", ...}}'
    ),
    "search_files": (
        "You need to find every line containing the string 'def main' "
        "in any file under /tmp/myproject (search recursively). "
        "What function would you call and what arguments would you pass? "
        "Respond with a JSON object like: "
        '{"function_name": "...", "arguments": {"arg1": "val1", ...}}'
    ),
    "glob": (
        "You need to find all Python source files (matching *.py) "
        "anywhere under /tmp/myproject, recursively. "
        "What function would you call and what arguments would you pass? "
        "Respond with a JSON object like: "
        '{"function_name": "...", "arguments": {"arg1": "val1", ...}}'
    ),
}

FALLBACK_ELICITED = {
    "read_file":         {"function_name": "read_file",          "arguments": {"file_path": ""}},
    "write_file":        {"function_name": "write_file",         "arguments": {"file_path": "", "content": ""}},
    "update_file":       {"function_name": "str_replace",        "arguments": {"file_path": "", "old_str": "", "new_str": ""}},
    "execute_bash":      {"function_name": "run_shell_command",  "arguments": {"command": ""}},
    "ask_user_question": {"function_name": "ask_user_question",  "arguments": {"question": ""}},
    "list_directory":    {"function_name": "list_directory",     "arguments": {"path": ""}},
    "search_files":      {"function_name": "search_files",       "arguments": {"path": "", "pattern": ""}},
    "glob":              {"function_name": "glob",               "arguments": {"pattern": ""}},
}


def elicit_round(client: openai.OpenAI) -> dict[str, dict]:
    section("Round 1 -- Free-form elicitation (no tool schema)")
    results: dict[str, dict] = {}
    for op, task in ELICIT_TASKS.items():
        messages = [
            {
                "role": "system",
                "content": (
                    "You are a helpful assistant with access to tools. "
                    "When asked what function to call, respond ONLY with a JSON object."
                ),
            },
            {"role": "user", "content": task},
        ]
        resp = chat(client, messages)
        _save_probe(f"round1_elicit_{op}", messages, resp)
        text = resp.choices[0].message.content or ""
        parsed = extract_json_block(text)
        print(f"\n[{op}] raw response:\n{indent(text.strip(), '  ')}")
        if isinstance(parsed, dict) and ("function_name" in parsed or "name" in parsed):
            fn = parsed.get("function_name") or parsed.get("name")
            args = parsed.get("arguments", {})
            results[op] = {"function_name": fn, "arguments": args}
            print(f"[{op}] parsed: {json.dumps(results[op], indent=2)}")
        else:
            results[op] = FALLBACK_ELICITED[op]
            print(f"[{op}] could not parse -- using fallback: {results[op]}")
    return results


# -- Round 2: probe with tool schema ------------------------------------------

def args_to_schema_properties(args: dict) -> dict:
    props: dict[str, dict] = {}
    for key, val in args.items():
        if isinstance(val, bool):
            typ = "boolean"
        elif isinstance(val, int):
            typ = "integer"
        elif isinstance(val, float):
            typ = "number"
        elif isinstance(val, list):
            typ = "array"
        else:
            typ = "string"
        prop: dict = {"type": typ, "description": key.replace("_", " ")}
        # A bare-token string value (e.g. command="str_replace", no spaces or
        # punctuation) is far more likely to be a mode/action selector than
        # free-form content -- surface it as a single-value enum so the
        # model isn't handed a required field with no hint of what's valid.
        # Free-text fields (paths, file content, shell commands) contain
        # spaces or punctuation and fall through unchanged.
        if typ == "string" and isinstance(val, str) and re.fullmatch(r"[a-z][a-z0-9_]*", val):
            prop["enum"] = [val]
            prop["description"] = f"{prop['description']} (must be {val!r})"
        props[key] = prop
    return props


def _deduplicate_elicited(elicited: dict[str, dict]) -> dict[str, dict]:
    """Noop any canonical op whose elicited function_name duplicates an earlier op's name.

    When a model maps two distinct canonical ops (e.g. write_file and update_file)
    to the same tool name, the second one cannot be meaningfully distinguished at
    runtime.  Mark it with function_name=None so it is excluded from the schema
    and dispatch table.
    """
    seen: dict[str, str] = {}
    result: dict[str, dict] = {}
    for op, info in elicited.items():
        fn = info.get("function_name")
        if fn and fn in seen:
            print(f"  [dedup] '{op}' elicited name '{fn}' already claimed by "
                  f"'{seen[fn]}' -- marking as noop")
            result[op] = {**info, "function_name": None}
        else:
            if fn:
                seen[fn] = op
            result[op] = info
    return result


def _sanitize_tool_name(name: str) -> str:
    """Ensure a tool name matches the OpenAI pattern ^[a-zA-Z0-9_-]{1,64}$."""
    sanitized = re.sub(r"[^a-zA-Z0-9_\-]", "_", name)
    return sanitized[:64] or "tool"


def build_tool_schema(elicited: dict[str, dict]) -> list[dict]:
    tools = []
    for op, info in elicited.items():
        fn_name = info.get("function_name")
        if not fn_name:   # None => nooped due to name collision
            continue
        fn_name = _sanitize_tool_name(fn_name)
        args    = info.get("arguments", {})
        props   = args_to_schema_properties(args)
        tools.append({
            "type": "function",
            "function": {
                "name": fn_name,
                "description": f"Perform the '{op}' operation.",
                "parameters": {
                    "type": "object",
                    "properties": props,
                    "required": list(props.keys()),
                },
            },
        })
    return tools


PROBE_TASKS = {
    "read_file":         "Please read the file /etc/hostname and tell me its contents.",
    "write_file":        "Please write 'hello world\\n' to the file /tmp/test.txt.",
    "update_file":       "In the file /tmp/test.py, replace the exact string 'x = 1' with 'x = 42'. Do not rewrite the whole file.",
    "execute_bash":      "Please run `ls -la /tmp` and show me the output.",
    "ask_user_question": "Before you start, ask the user whether they want to overwrite /tmp/test.txt or create a backup first.",
    "list_directory":    "List the files and directories inside /tmp.",
    "search_files":      "Search for the string 'hello' in all files under /tmp.",
    "glob":              "Find all .py files anywhere under /tmp.",
}


def probe_round(client: openai.OpenAI, tools: list[dict],
                label: str = "Round 2") -> dict[str, ToolCallResult | None]:
    section(f"{label} -- Probing with tool schema")
    print("\nSchema offered to model:")
    print(json.dumps(tools, indent=2))

    calls: dict[str, ToolCallResult | None] = {}
    for op, task in PROBE_TASKS.items():
        messages = [
            {"role": "system", "content": "You are a helpful assistant with tool access."},
            {"role": "user",   "content": task},
        ]
        resp = chat(client, messages, tools=tools)
        _save_probe(f"{re.sub(r'[^a-z0-9]+', '_', label.lower()).strip('_')}_{op}",
                    messages, resp, tools=tools)
        result = extract_call_from_response(resp)
        calls[op] = result
        if result:
            mode = "structured tool_call" if result.structured else "inline JSON in content"
            print(f"\n[{op}] ({mode}): {json.dumps(result.to_dict(), indent=2)}")
        else:
            raw = resp.choices[0].message.content
            print(f"\n[{op}] no call detected. content: {raw!r}")
    return calls


# -- behavioural summary ------------------------------------------------------

def behavioural_summary(probe_calls: dict[str, ToolCallResult | None]) -> dict:
    structured = sum(1 for r in probe_calls.values() if r and r.structured)
    inline     = sum(1 for r in probe_calls.values() if r and not r.structured)
    missing    = sum(1 for r in probe_calls.values() if r is None)
    if structured == len(probe_calls):
        mode = "structured_tool_calls"
    elif inline > 0 and structured == 0:
        mode = "inline_json_in_content"
    else:
        mode = "mixed"
    return {
        "call_delivery_mode": mode,
        "structured_tool_calls": structured,
        "inline_json_in_content": inline,
        "no_call_detected": missing,
        "note": (
            "This model outputs tool invocations as JSON inside the message "
            "content field rather than the structured tool_calls API field. "
            "Callers must parse the content field to extract function calls."
            if mode == "inline_json_in_content" else
            "Model correctly uses the structured tool_calls API field."
        ),
    }


def tool_selection_test(
    probe_calls: dict[str, ToolCallResult | None], elicited_names: dict[str, str]
) -> dict:
    """Score whether each full-schema probe call selected its expected tool.

    This deliberately measures selection, not whether a model repeats a name it
    invented during elicitation: every Round-2 task sees the complete inferred
    tool list, and its result is compared to the tool assigned to that task.
    """
    results: dict[str, dict] = {}
    for op, expected_name in elicited_names.items():
        if not expected_name:
            continue  # No distinct tool was available for this operation.
        call = probe_calls.get(op)
        called_name = call.function_name if call else None
        passed = called_name == expected_name
        if call is None:
            error = "no tool call detected"
        elif not passed:
            error = f"called {called_name!r} instead of {expected_name!r}"
        else:
            error = None
        results[op] = {
            "pass": passed,
            "expected_function_name": expected_name,
            "function_name": called_name,
            "error": error,
        }

    passed = sum(1 for result in results.values() if result["pass"])
    return {
        "tsel_results": results,
        "tsel_passed": passed,
        "tsel_total": len(results),
    }


# -- tool dispatch table -------------------------------------------------------
#
# _CANONICAL_OPS maps each canonical operation name to:
#   python_function : name of the callable in agent_probe.TOOL_LIBRARY
#   kwarg_roles     : {kwarg_name: semantic_role}
#
# _PARAM_ROLES is the ordered list of (role, hint_substrings) used to classify
# a model's parameter name into a semantic role.  More-specific patterns first.

_CANONICAL_OPS: dict[str, dict] = {
    "read_file": {
        "python_function": "t_read",
        "kwarg_roles": {"path": "path"},
    },
    "write_file": {
        "python_function": "t_write",
        "kwarg_roles": {"path": "path", "content": "content"},
    },
    "update_file": {
        "python_function": "t_update",
        "kwarg_roles": {"path": "path", "old": "old", "new": "new"},
    },
    "execute_bash": {
        "python_function": "t_run",
        "kwarg_roles": {"command": "command"},
    },
    "ask_user_question": {
        "python_function": "t_ask_user",
        "kwarg_roles": {"question": "question"},
    },
    "list_directory": {
        "python_function": "t_list_dir",
        "kwarg_roles": {"path": "path"},
    },
    "search_files": {
        "python_function": "t_search",
        "kwarg_roles": {"path": "path", "pattern": "query"},
    },
    "glob": {
        "python_function": "t_glob",
        "kwarg_roles": {"pattern": "glob_pattern"},
    },
}

_PARAM_ROLES: list[tuple[str, tuple]] = [
    ("old",          ("old_str", "old_string", "old_text", "search", "find", "before", "original")),
    ("new",          ("new_str", "new_string", "new_text", "replac", "after", "replacement")),
    ("question",     ("question", "questions", "prompt")),
    ("query",        ("query", "grep", "regex", "keyword", "term", "search_string", "search_term", "pattern")),
    ("glob_pattern", ("glob", "wildcard")),
    ("content",      ("content", "text", "data", "body")),
    ("command",      ("command", "cmd", "shell_command", "bash")),
    ("path",         ("path", "file", "filename", "file_path", "filepath", "directory", "dir")),
]


def _classify_param(name: str) -> str | None:
    """Return the semantic role for a parameter name, or None if unrecognised."""
    n = name.lower()
    for role, hints in _PARAM_ROLES:
        if any(h in n for h in hints):
            return role
    return None


def _match_op(tool_name: str, param_names: list[str],
               elicited_names: dict[str, str]) -> str | None:
    """Return the canonical op name for a tool, or None if unrecognised.

    Priority:
      1. elicited_names reverse lookup (probe already told us the op).
      2. Parameter-role fingerprint (set of roles present in the tool).
    """
    # 1. Direct lookup: elicited_names maps tool_name -> canonical op.
    if tool_name in elicited_names:
        return elicited_names[tool_name]

    # 1b. If the tool name is itself a canonical op name, use it directly.
    if tool_name in _CANONICAL_OPS:
        return tool_name

    # 2. Role fingerprint -- more-specific patterns first.
    roles = {_classify_param(p) for p in param_names} - {None}
    if "old" in roles and "new" in roles:
        return "update_file"
    if "question" in roles:
        return "ask_user_question"
    if "query" in roles:
        return "search_files"
    if "glob_pattern" in roles:
        return "glob"
    if "content" in roles and "path" in roles:
        return "write_file"
    if "command" in roles:
        return "execute_bash"
    if "path" in roles:
        return "read_file"   # list_directory is indistinguishable here; elicited_names wins
    return None


def _build_param_map(param_names: list[str], kwarg_roles: dict[str, str]) -> dict[str, str]:
    """Map each model param name -> Python kwarg name.

    kwarg_roles: {kwarg_name: role}  (from _CANONICAL_OPS)
    Returns:     {model_param_name: kwarg_name}

    Any param whose role is not in kwarg_roles is passed through unchanged
    (identity mapping), so the function still receives it even if we don't
    know what to do with it.
    """
    role_to_kwarg = {role: kwarg for kwarg, role in kwarg_roles.items()}
    param_map: dict[str, str] = {}
    for p in param_names:
        role = _classify_param(p)
        kwarg = role_to_kwarg.get(role) if role else None
        if kwarg:
            # Only include params that map to a known kwarg for this op;
            # extras (e.g. the 'command' enum on str_replace_editor) are dropped.
            param_map[p] = kwarg
    return param_map


def _append_to_tool_library(fn_name: str, source: str) -> None:
    """Append a generated function to tool_library.py and register it in TOOL_LIBRARY."""
    lib_path = Path(__file__).resolve().parent / "tool_library.py"
    addition = f"\n\n# --- generated: {fn_name} ---\n{source}\n\nTOOL_LIBRARY[{fn_name!r}] = {fn_name}\n"
    with lib_path.open("a") as f:
        f.write(addition)
    print(f"  [codegen] appended {fn_name} to {lib_path.name}")


def _synthesise_function(
    client: openai.OpenAI,
    tool_name: str,
    tool_description: str,
    param_names: list[str],
) -> tuple[str, str]:
    """Ask the LLM to write a Python implementation for an unrecognised tool.

    Returns (python_function_name, source_code).

    Contract for the generated function:
      - Named  t_<sanitised_tool_name>
      - Accepts the model's exact parameter names as keyword arguments (str defaults)
      - Returns tuple[str, dict]: (human-readable result, log dict with 'result' key)
      - Uses only stdlib (pathlib, subprocess, json, os are pre-imported in scope)
      - Handles exceptions; returns "ERROR: ..." on failure
    """
    fn_name = "t_" + re.sub(r"[^a-z0-9_]", "_", tool_name.lower()).strip("_")
    params_sig = ", ".join(f"{p}: str = ''" for p in param_names)

    lines = [
        f"Write a Python function called `{fn_name}` that implements the tool described below.",
        "",
        f"Tool name       : {tool_name}",
        f"Tool description: {tool_description}",
        f"Parameters      : {param_names}",
        "",
        "Requirements:",
        f"  - Function signature: def {fn_name}({params_sig}) -> tuple[str, dict]:",
        "  - Return a tuple: (human-readable result string, dict with at least a 'result' key)",
        "  - Use only the Python standard library (pathlib, subprocess, json, os, etc.)",
        "  - Handle exceptions and return an 'ERROR: ...' string on failure",
        "  - Do NOT include import statements -- assume Path, subprocess, json, os are in scope",
        "",
        "Respond with ONLY the function source code, no prose, no markdown fences.",
    ]
    prompt = "\n".join(lines)

    section(f"Code generation -- synthesising {fn_name} for unrecognised tool '{tool_name}'")
    messages = [
        {"role": "system", "content": "You are an expert Python programmer. "
                                       "Respond with only raw Python source code."},
        {"role": "user", "content": prompt},
    ]
    resp = chat(client, messages)
    _save_probe(f"codegen_{fn_name}", messages, resp)
    raw = resp.choices[0].message.content or ""

    # Strip markdown fences if the model added them despite instructions.
    source = re.sub(r"^```(?:python)?\s*", "", raw.strip(), flags=re.MULTILINE)
    source = re.sub(r"\s*```$", "", source.strip(), flags=re.MULTILINE)
    source = source.strip()

    print(f"\nGenerated source for {fn_name}:\n{indent(source, '  ')}")
    return fn_name, source


def build_tool_dispatch(
    elicited: dict[str, dict],
    final_probes: dict[str, "ToolCallResult | None"],
    client: openai.OpenAI,
) -> tuple[dict[str, dict], dict[str, str]]:
    """Build the tool_dispatch table stored in the probe JSON.

    For each tool observed in final_probes:
      - Match it to a canonical op (via elicited_names or param-role fingerprint).
      - Build a param_map: model param name -> Python kwarg name.
      - If no canonical op matches, ask the LLM to synthesise a Python function
        and store its source in generated_source.

    Returns (dispatch, conflicts):

      dispatch (keyed by model tool name):
        {
          "str_replace_editor": {
            "python_function": "t_update",
            "param_map": {"path": "path", "old_str": "old", "new_str": "new"}
          },
          "some_unknown_tool": {
            "python_function": "t_some_unknown_tool",
            "param_map": {"x": "x"},
            "generated_source": "def t_some_unknown_tool(x: str = '') -> tuple[str, dict]: ..."
          }
        }

      conflicts (keyed by op): the op's own probe call resolved to a tool
      name some *other* op already claimed a dispatch entry for, e.g.
        {"update_file": "read_file"}
      means update_file's Round-2 probe call was actually `read_file(...)`
      instead of its own elicited tool (e.g. str_replace_editor) -- so that
      tool never gets a dispatch entry, not because the schema is broken,
      but because the model substituted a different tool for this op.
    """
    section("Round 4 -- Building tool dispatch table")

    # Reverse map: tool_name -> canonical op name (from elicited_names).
    elicited_names: dict[str, str] = {
        v["function_name"]: op
        for op, v in elicited.items()
        if v.get("function_name")
    }

    dispatch: dict[str, dict] = {}
    conflicts: dict[str, str] = {}

    for op, result in final_probes.items():
        if result is None:
            print(f"  [{op}] no probe result -- skipping")
            continue

        tool_name   = result.function_name
        param_names = list(result.arguments.keys())

        if tool_name in dispatch:
            # This op's probe call resolved to a tool name another op
            # already claimed a dispatch entry for -- record the collision
            # instead of silently dropping it, so the report can explain
            # *why* this op's own tool never got dispatched.
            conflicts[op] = tool_name
            print(f"  [{op}] probe call resolved to '{tool_name}', already claimed "
                  f"by another op -- recorded as a conflict, not dispatched")
            continue

        canonical_op = _match_op(tool_name, param_names, elicited_names)

        if canonical_op and canonical_op in _CANONICAL_OPS:
            canon     = _CANONICAL_OPS[canonical_op]
            param_map = _build_param_map(param_names, canon["kwarg_roles"])
            dispatch[tool_name] = {
                "python_function": canon["python_function"],
                "param_map":       param_map,
            }
            print(f"  [{tool_name}] -> {canon['python_function']}  param_map={param_map}")
        else:
            # Unrecognised tool -- synthesise a Python implementation.
            print(f"  [{tool_name}] unrecognised -- requesting code generation")
            fn_name, source = _synthesise_function(
                client,
                tool_name=tool_name,
                tool_description=f"Tool '{tool_name}' with parameters {param_names}",
                param_names=param_names,
            )
            _append_to_tool_library(fn_name, source)
            dispatch[tool_name] = {
                "python_function": fn_name,
                "param_map":       {p: p for p in param_names},  # identity
            }
            print(f"  [{tool_name}] -> {fn_name} (generated)")

    return dispatch, conflicts


# -- quick summary from local JSON files --------------------------------------

def _tool_param_signature(tool: dict) -> str:
    """Return 'name(p1*, p2)' where '*' marks required params."""
    fn = tool.get("function") or tool
    name = fn.get("name", "?")
    params = fn.get("parameters") or {}
    props = params.get("properties")
    if not isinstance(props, dict):
        props = {k: v for k, v in params.items()
                 if isinstance(v, dict) and "type" in v}
    required = set(params.get("required") or [])
    parts = []
    for pname, pinfo in props.items():
        marker = "*" if pname in required or (isinstance(pinfo, dict) and pinfo.get("required")) else ""
        ptype = pinfo.get("type", "?") if isinstance(pinfo, dict) else "?"
        parts.append(f"{pname}{marker}:{ptype}")
    return f"{name}({', '.join(parts)})"


def _summary_label(data: dict, fallback: str) -> str:
    """`model` in listings, or `model @ provider` for a provider-pinned run."""
    model = data.get("model", fallback)
    provider = data.get("provider")
    return f"{model} @ {provider}" if provider else model


def quick_summary() -> None:
    import glob
    # Three depths: the legacy reports/<model>/, reports/<server>/<model>/, and
    # reports/<server>/<model>/<provider>/ for runs pinned to one upstream.
    paths = sorted(set(glob.glob("reports/*/*/*/capabilities_*.json"))
                   | set(glob.glob("reports/*/*/capabilities_*.json"))
                   | set(glob.glob("reports/*/capabilities_*.json")))
    if not paths:
        print("No schema json files found under reports/<server>/<model>/.")
        return

    structured_list: list[dict] = []
    other: list[tuple[str, str]] = []
    for path in paths:
        try:
            with open(path) as f:
                data = json.load(f)
        except Exception as e:
            other.append((path, f"unreadable: {e}"))
            continue
        model = _summary_label(data, path)
        behaviour = data.get("behaviour") or {}
        mode = behaviour.get("call_delivery_mode")
        status = data.get("status", "ok")
        if mode == "structured_tool_calls":
            structured_list.append(data)
        else:
            other.append((model, mode or status or "unknown"))

    print(f"Models with native structured tool_calls support  ({len(structured_list)}/{len(paths)}):\n")
    for data in structured_list:
        model    = _summary_label(data, "?")
        endpoint = data.get("endpoint", "?")
        print(f"  * {model}   [{endpoint}]")
        for tool in data.get("inferred_tool_schema") or []:
            print(f"      - {_tool_param_signature(tool)}")
        print()

    if other:
        print(f"Models without native structured tool_calls  ({len(other)}):")
        for model, mode in other:
            print(f"  * {model}  ->  {mode}")

    print("\n('*' marks required parameters)")


# -- quote-escaping test ------------------------------------------------------

QUOTE_TEST_TASKS = {
    "write_file": (
        r'Write the following text exactly to /tmp/quote_test.txt: '
        r'She said "hello" and he replied "goodbye, world".'
    ),
    "execute_bash": (
        r'Run this exact shell command: echo "hello \"world\""'
    ),
    "update_file": (
        r'In /tmp/test.py, replace the string x = "old value" with x = "new value". '
        r'Do not rewrite the whole file.'
    ),
}

QUOTE_TEST_EXPECTED = {
    "write_file":   '"',
    "execute_bash": '"',
    "update_file":  '"',
}


def quote_test_round(
    client: openai.OpenAI, tools: list[dict], elicited_names: dict[str, str]
) -> dict:
    section("Quote-escaping test -- arguments must contain literal double-quotes")
    print("\nEach task requires a double-quote character inside a JSON string value.")
    print("PASS = model emits valid JSON with the quote present in the parsed value.")
    print("FAIL = JSON parse error, or the quote is silently dropped/mangled.")
    print("Only the tool relevant to each task is advertised (not the full schema), so")
    print("this isolates quote-escaping fidelity from tool-selection behaviour -- a model")
    print("that skips the intended tool for an unrelated one fails GREP/dispatch checks,")
    print("not QUOTE. See CAPABILITIES.md.\n")

    results: dict[str, dict] = {}
    for op, task in QUOTE_TEST_TASKS.items():
        tool_name  = elicited_names.get(op)
        task_tools = [t for t in tools if (t.get("function") or t).get("name") == tool_name]
        isolated   = bool(task_tools)
        if not task_tools:
            task_tools = tools  # fallback: no matching tool found, offer everything
        messages = [
            {"role": "system", "content": "You are a helpful assistant with tool access."},
            {"role": "user",   "content": task},
        ]
        resp = chat(client, messages, tools=task_tools)
        _save_probe(f"quote_test_{op}", messages, resp, tools=task_tools)
        entry: dict = {"task": task, "pass": False, "error": None,
                       "structured": None, "parsed_args": None, "isolated": isolated}
        result = extract_call_from_response(resp)
        if result is None:
            raw = resp.choices[0].message.content
            entry["error"] = "no tool call detected"
            entry["raw_content"] = raw
            print(f"[{op}] FAIL -- no call detected. content: {raw!r}")
            results[op] = entry
            continue
        entry["structured"]    = result.structured
        entry["function_name"] = result.function_name
        entry["parsed_args"]   = result.arguments
        expected_char = QUOTE_TEST_EXPECTED[op]
        found = any(isinstance(v, str) and expected_char in v
                    for v in result.arguments.values())
        if found:
            entry["pass"] = True
            mode = "structured" if result.structured else "inline JSON"
            print(f"[{op}] PASS  ({mode})  args={json.dumps(result.arguments)}")
        else:
            entry["error"] = "double-quote not found in any argument value"
            print(f"[{op}] FAIL -- quote missing from args: {json.dumps(result.arguments)}")
        results[op] = entry

    passed = sum(1 for r in results.values() if r["pass"])
    total  = len(results)
    print(f"\nQuote-test summary: {passed}/{total} passed")
    return {"quote_test_results": results, "quote_test_passed": passed, "quote_test_total": total}


# -- token-efficiency test ------------------------------------------------------
#
# Each task describes a large file/output where the token-cheap move is a
# targeted, filtered call (grep, sed -n, head, wc -l, a dedicated search
# tool, or a read with an offset/limit) instead of pulling the whole
# file/output back into context. PASS = the model chose the cheap call.

TOKEN_EFFICIENCY_TASKS = {
    "large_log_grep": (
        "The file /var/log/app.log is 500,000 lines long. Find all lines "
        "containing the exact string 'FATAL ERROR' and show them to me."
    ),
    "count_occurrences": (
        "The file /var/data/access.log has several million lines. Tell me "
        "how many lines contain the IP address 203.0.113.42. I only need the count."
    ),
    "specific_line": (
        "The file /opt/build/output.txt is over a million lines long. "
        "What is on line 48213 of that file?"
    ),
    "check_string_exists": (
        "The file /var/log/build.log is huge (hundreds of thousands of lines). "
        "Does the word 'DeprecationWarning' appear anywhere in it? Just answer yes or no."
    ),
    "function_definition_search": (
        "The file /repo/src/model.py is 20,000 lines long. Find the line "
        "number where the function `def train_model` is defined."
    ),
    "process_output_filter": (
        "List all currently running processes, but I only care about the "
        "ones related to 'python'. Show me just those."
    ),
}

_EFFICIENT_BASH_MARKERS = ("grep", "awk", "sed -n", "sed '", 'sed "', "head ",
                           "tail ", "wc -l", "wc -c", "cut ")
_WASTEFUL_BASH_MARKERS  = ("cat ", "type ", "more ", "less ")


def _classify_bash_command(command: str) -> tuple[bool, str]:
    cmd = command.lower()
    if any(m in cmd for m in _EFFICIENT_BASH_MARKERS):
        return True, "command includes a filtering tool (grep/awk/sed/head/tail/wc/cut)"
    if any(m in cmd for m in _WASTEFUL_BASH_MARKERS):
        return False, "command dumps the file unfiltered (cat/more/less) instead of filtering"
    return False, "command has no recognisable filtering -- likely reads everything"


def _classify_read_args(args: dict) -> tuple[bool, str]:
    partial_hints = ("offset", "limit", "start", "end", "line", "head", "max", "range")
    if any(any(h in k.lower() for h in partial_hints) for k in args):
        return True, "read call includes a range/offset/limit argument"
    return False, "read call has no offset/limit -- requests the whole file"


def token_efficiency_test_round(
    client: openai.OpenAI, tools: list[dict], tool_dispatch: dict
) -> dict:
    section("Token-efficiency test -- prefer filtering over full dumps")
    print("\nEach task involves a large file/output. PASS = model chooses a filtered/")
    print("targeted call (grep, sed -n, head, a dedicated search tool, or a read with")
    print("offset/limit). FAIL = model requests the entire file/output unfiltered.\n")

    results: dict[str, dict] = {}
    for op, task in TOKEN_EFFICIENCY_TASKS.items():
        messages = [
            {"role": "system", "content": "You are a helpful assistant with tool access. "
                                           "Be mindful of token costs: avoid reading or "
                                           "printing more data than necessary to answer "
                                           "the question."},
            {"role": "user", "content": task},
        ]
        resp = chat(client, messages, tools=tools)
        _save_probe(f"token_efficiency_{op}", messages, resp, tools=tools)
        entry: dict = {"task": task, "pass": False, "reason": None,
                       "function_name": None, "python_function": None,
                       "parsed_args": None}
        result = extract_call_from_response(resp)
        if result is None:
            raw = resp.choices[0].message.content
            entry["reason"] = "no tool call detected"
            entry["raw_content"] = raw
            print(f"[{op}] FAIL -- no call detected. content: {raw!r}")
            results[op] = entry
            continue

        entry["function_name"] = result.function_name
        entry["parsed_args"]   = result.arguments
        dispatch_entry  = tool_dispatch.get(result.function_name) or {}
        python_function = dispatch_entry.get("python_function")
        entry["python_function"] = python_function

        if python_function == "t_run":
            passed, reason = _classify_bash_command(str(result.arguments.get("command", "")))
        elif python_function == "t_search":
            passed, reason = True, "dedicated search tool used -- inherently filtered"
        elif python_function == "t_read":
            passed, reason = _classify_read_args(result.arguments)
        elif python_function in ("t_list_dir", "t_glob"):
            passed, reason = True, f"{python_function} does not dump file contents"
        else:
            passed, reason = False, f"unrecognised python_function={python_function!r}"

        entry["pass"]   = passed
        entry["reason"] = reason
        status = "PASS" if passed else "FAIL"
        print(f"[{op}] {status} -- {result.function_name}({json.dumps(result.arguments)}) :: {reason}")
        results[op] = entry

    passed = sum(1 for r in results.values() if r["pass"])
    total  = len(results)
    print(f"\nToken-efficiency summary: {passed}/{total} passed")
    return {"token_efficiency_results": results, "token_efficiency_passed": passed,
            "token_efficiency_total": total}


# -- ask_user_question phrasing test (ASKQ) -------------------------------------
#
# Explores how strongly a task's wording drives the model to actually invoke
# its own ask_user_question tool, versus silently guessing or investigating
# instead. All 8 variants run once each against the full tool schema (all
# inferred tools available, matching real usage) -- no retries, so this
# measures phrasing sensitivity, not sampling noise. A borderline phrasing
# can still flip pass/fail run to run; see CAPABILITIES.md.

_ASKQ_SYSTEM_DEFAULT = "You are a helpful assistant with tool access."
_ASKQ_SYSTEM_NUDGE = (
    "You are a helpful assistant with tool access. When a task requires user "
    "input, preference, or confirmation before proceeding, you MUST call "
    "ask_user_question rather than guessing, investigating, or acting unilaterally."
)

ASKQ_VARIANTS: dict[str, tuple[str, str]] = {
    "original_probe": (
        _ASKQ_SYSTEM_DEFAULT,
        "Before you start, ask the user whether they want to overwrite "
        "/tmp/test.txt or create a backup first."),
    "imperative_must": (
        _ASKQ_SYSTEM_DEFAULT,
        "You must ask the user this exact question before doing anything else: "
        "'Should I overwrite /tmp/test.txt or create a backup first?' Do not "
        "perform any other action."),
    "explicit_tool_name": (
        _ASKQ_SYSTEM_DEFAULT,
        "Call the ask_user_question tool now with the question: 'Overwrite "
        "/tmp/test.txt or create a backup first?'"),
    "first_person_unsure": (
        _ASKQ_SYSTEM_DEFAULT,
        "I'm not sure whether you want me to overwrite /tmp/test.txt or make "
        "a backup first. Which do you prefer?"),
    "ambiguous_no_ask_word": (
        _ASKQ_SYSTEM_DEFAULT,
        "There's ambiguity about whether /tmp/test.txt should be overwritten "
        "or backed up before I proceed. Resolve it."),
    "destructive_warning": (
        _ASKQ_SYSTEM_DEFAULT,
        "This action is destructive and irreversible. Before proceeding you "
        "MUST get explicit user confirmation on whether to overwrite "
        "/tmp/test.txt or back it up first. Ask now."),
    "options_style": (
        _ASKQ_SYSTEM_DEFAULT,
        "Present the user with two options -- 'Overwrite /tmp/test.txt' or "
        "'Create a backup first' -- and wait for their choice before doing "
        "anything."),
    "system_nudge": (
        _ASKQ_SYSTEM_NUDGE,
        "Before you start, ask the user whether they want to overwrite "
        "/tmp/test.txt or create a backup first."),
}


def _likert_label(passed: int, total: int) -> str:
    """Map a passed/total fraction to a 5-point Likert frequency label."""
    if total == 0:
        return "N/A"
    ratio = passed / total
    if ratio == 0:
        return "Never"
    if ratio <= 0.25:
        return "Rarely"
    if ratio <= 0.625:
        return "Sometimes"
    if ratio < 1.0:
        return "Often"
    return "Always"


def ask_user_question_test_round(
    client: openai.OpenAI, tools: list[dict], ask_tool_name: str | None
) -> dict:
    section("ASKQ test -- does phrasing drive the model to call ask_user_question?")
    print("\nAll 8 variants run once each against the full tool schema (all inferred")
    print("tools available). PASS = model calls its own ask_user_question tool.\n")

    if not ask_tool_name:
        print("No ask_user_question tool was elicited for this model -- skipping.")
        return {"error": "no ask_user_question tool elicited"}

    results: dict[str, dict] = {}
    for variant, (system, task) in ASKQ_VARIANTS.items():
        messages = [
            {"role": "system", "content": system},
            {"role": "user", "content": task},
        ]
        resp = chat(client, messages, tools=tools)
        _save_probe(f"askq_{variant}", messages, resp, tools=tools)
        entry: dict = {"task": task, "system": system, "pass": False,
                       "function_name": None, "parsed_args": None}
        result = extract_call_from_response(resp)
        if result is None:
            raw = resp.choices[0].message.content
            entry["raw_content"] = raw
            print(f"[{variant}] SKIPPED -- no tool call. content: {(raw or '')[:150]!r}")
            results[variant] = entry
            continue
        entry["function_name"] = result.function_name
        entry["parsed_args"]   = result.arguments
        entry["pass"] = result.function_name == ask_tool_name
        status = "ASKED" if entry["pass"] else "SKIPPED"
        print(f"[{variant}] {status} -- {result.function_name}({json.dumps(result.arguments)})")
        results[variant] = entry

    passed = sum(1 for r in results.values() if r["pass"])
    total  = len(results)
    print(f"\nASKQ summary: {passed}/{total} ({_likert_label(passed, total)})")
    return {"askq_results": results, "askq_passed": passed, "askq_total": total}


# -- apply_patch grammar-knowledge test (APPLY_PATCH) ------------------
#
# Tests whether a model *naturally* knows OpenAI's apply_patch envelope
# syntax (the format used by the real "custom"/freeform apply_patch tool --
# see ~/bin/copilot-notes.md) from pretraining, independent of whether the
# probing endpoint exposes that tool at all. No tool schema is offered; the
# model is asked in free text to produce ONLY the raw patch, and the result
# is parsed against the real grammar (adapted from openai/codex's
# apply_patch.lark) rather than checked with a loose regex.
#
# The grammar below is restructured from the upstream Lark source: Python's
# `lark` Earley engine rejects zero-width regex terminals (e.g. `/(.*)/ `),
# which the upstream grammar uses for line content. Each terminal here folds
# its trailing newline into the same regex so it can never match zero
# characters, while accepting the exact same language.
# Upstream: https://github.com/openai/codex/blob/main/codex-rs/core/src/tools/handlers/apply_patch.lark

_APPLY_PATCH_LARK_GRAMMAR = r"""
start: begin_patch hunk+ end_patch
begin_patch: "*** Begin Patch" NL
end_patch: "*** End Patch" NL?

hunk: add_hunk | delete_hunk | update_hunk
add_hunk: "*** Add File: " FILENAME NL add_line+
delete_hunk: "*** Delete File: " FILENAME NL
update_hunk: "*** Update File: " FILENAME NL change_move? change?

FILENAME: /[^\n]+/
add_line: ADD_LINE -> line
ADD_LINE: /\+[^\n]*\n/

change_move: "*** Move to: " FILENAME NL
change: (change_context | change_line)+ eof_line?
change_context: CONTEXT_LINE
CONTEXT_LINE: /@@[^\n]*\n/
change_line: CHANGE_LINE
CHANGE_LINE: /[+\- ][^\n]*\n/
eof_line: "*** End of File" NL

NL: /\n/
"""

_apply_patch_parser: lark.Lark | None = None


def _get_apply_patch_parser() -> lark.Lark:
    global _apply_patch_parser
    if _apply_patch_parser is None:
        _apply_patch_parser = lark.Lark(
            _APPLY_PATCH_LARK_GRAMMAR, start="start", parser="earley", lexer="dynamic_complete"
        )
    return _apply_patch_parser


GRAM_KNOWLEDGE_TASKS = {
    "update_file": (
        "You need to edit the file /tmp/test.py: replace the exact string "
        "'x = 1' with 'x = 42'. Express this change as a patch using the "
        "exact format OpenAI's apply_patch tool expects (the same patch "
        "envelope format used by Codex CLI). Respond with ONLY the raw "
        "patch text -- no prose, no JSON, no markdown code fences."
    ),
    "add_file": (
        "You need to create a new file /tmp/hello.txt containing exactly: "
        "Hello, world! Express this as a patch using the exact format "
        "OpenAI's apply_patch tool expects (the same patch envelope format "
        "used by Codex CLI). Respond with ONLY the raw patch text -- no "
        "prose, no JSON, no markdown code fences."
    ),
}


def _strip_fences(text: str) -> str:
    text = re.sub(r"^```(?:\w+)?\s*\n?", "", text.strip())
    text = re.sub(r"\n?```\s*$", "", text)
    return text.strip("\n") + "\n"


def gram_knowledge_test_round(client: openai.OpenAI) -> dict:
    section("APPLY_PATCH test -- does the model naturally know apply_patch grammar?")
    print("\nNo tool schema is offered -- the model is asked in free text to produce")
    print("ONLY a raw apply_patch-format patch, then it's parsed against the real")
    print("grammar (not a loose regex). PASS = syntactically valid patch.\n")

    parser  = _get_apply_patch_parser()
    results: dict[str, dict] = {}
    for op, task in GRAM_KNOWLEDGE_TASKS.items():
        messages = [
            {"role": "system", "content": "You are a helpful coding assistant."},
            {"role": "user", "content": task},
        ]
        resp = chat(client, messages)
        _save_probe(f"gram_knowledge_{op}", messages, resp)
        raw  = resp.choices[0].message.content or ""
        text = _strip_fences(raw)
        entry: dict = {"task": task, "pass": False, "error": None, "raw_content": raw}
        try:
            parser.parse(text)
            entry["pass"] = True
            print(f"[{op}] PASS\n{indent(text.strip(), '  ')}")
        except lark.exceptions.LarkError as e:
            entry["error"] = str(e)[:500]
            print(f"[{op}] FAIL -- {entry['error'][:200]}")
            print(f"  raw content: {raw[:300]!r}")
        results[op] = entry

    passed = sum(1 for r in results.values() if r["pass"])
    total  = len(results)
    print(f"\nAPPLY_PATCH summary: {passed}/{total} passed")
    return {"gram_knowledge_results": results, "gram_knowledge_passed": passed,
            "gram_knowledge_total": total}


# -- constrained-decoding / custom-tool test (GRAMT) -------------------
#
# APPLY_PATCH tests whether the model *knows* the apply_patch grammar from
# pretraining, with no tool schema at all. This tests something different:
# whether the *endpoint* actually implements OpenAI's real freeform/custom-tool
# transport (`type: "custom"`, `format: {type: "grammar", syntax: "lark", ...}`)
# end to end -- i.e. genuine grammar-constrained decoding, not classic
# JSON-schema function calling with a tool that happens to be named
# apply_patch. See CAPABILITIES.md and ~/bin/copilot-notes.md for the
# background (confirmed directly: gpt-5-mini's endpoint 400s on this request
# shape entirely; gpt-5.6-luna's Responses API returns a genuine
# custom_tool_call once its wrapper script's translator is fixed to pass
# `type: "custom"` tools through instead of silently dropping them).
#
# Sent verbatim as OpenAI/Codex define it -- unlike the APPLY_PATCH grammar
# above, this is not rewritten for Python-lark's zero-width-terminal
# restriction, since it's the *endpoint's* grammar engine that has to accept
# it, not ours.
# Source: https://github.com/openai/codex/blob/main/codex-rs/core/src/tools/handlers/apply_patch.lark

_APPLY_PATCH_UPSTREAM_LARK_GRAMMAR = r"""
start: begin_patch hunk+ end_patch
begin_patch: "*** Begin Patch" LF
end_patch: "*** End Patch" LF?

hunk: add_hunk | delete_hunk | update_hunk
add_hunk: "*** Add File: " filename LF add_line+
delete_hunk: "*** Delete File: " filename LF
update_hunk: "*** Update File: " filename LF change_move? change?

filename: /(.+)/
add_line: "+" /(.*)/ LF -> line

change_move: "*** Move to: " filename LF
change: (change_context | change_line)+ eof_line?
change_context: ("@@" | "@@ " /(.+)/) LF
change_line: ("+" | "-" | " ") /(.*)/ LF
eof_line: "*** End of File" LF

%import common.LF
"""

GRAM_TRANSPORT_TASKS = {
    "apply_patch": {
        "task": (
            "In the file /tmp/test.py, replace the exact string 'x = 1' with "
            "'x = 42'. Do not rewrite the whole file. Use the apply_patch tool."
        ),
        "name": "apply_patch",
        "description": "Use the apply_patch tool to edit files, expressed as a V4A diff.",
        "grammar": _APPLY_PATCH_UPSTREAM_LARK_GRAMMAR,
        "local_validator": _get_apply_patch_parser,
    },
}


def _build_custom_tool(name: str, description: str, grammar: str) -> dict:
    return {
        "type": "custom",
        "custom": {
            "name": name,
            "description": description,
            "format": {"type": "grammar", "grammar": {"syntax": "lark", "definition": grammar}},
        },
    }


# -- RJSON test ---------------------------------------------------------------
#
# Sends a strict response_format:{type:'json_schema'} request with no tool
# schema, and checks whether the endpoint honours schema-constrained
# structured output end to end -- i.e. genuine constrained decoding on the
# response body, not just the model being good at writing JSON.  This is an
# endpoint/provider feature, deliberately separate from the model-behaviour
# capabilities (TCALL & co).  Mirrors the GRAMT test, which does the
# same for grammar-constrained custom tools.

_RJSON_SCHEMA = {
    "type": "object",
    "properties": {"value": {"type": "string"}},
    "required": ["value"],
    "additionalProperties": False,
}


def response_format_test_round(client: openai.OpenAI) -> dict:
    section("RJSON test -- does the endpoint support strict json_schema responses?")
    print("\nSends response_format:{type:'json_schema', strict:true} with no tool schema --")
    print("PASS = the request is accepted and the reply parses as JSON conforming to")
    print("the schema. FAIL = the endpoint rejects the request, or the content does")
    print("not conform.\n")

    entry: dict = {"pass": False, "supported": False, "schema_conformant": None, "error": None}
    messages = [
        {"role": "system", "content": "You are a helpful coding assistant."},
        {"role": "user",   "content": "Reply with the word 'hello'."},
    ]
    try:
        resp = client.chat.completions.create(
            model=MODEL,
            messages=messages,
            temperature=_probe_temperature(),
            timeout=300,
            response_format={
                "type": "json_schema",
                "json_schema": {"name": "answer", "strict": True, "schema": _RJSON_SCHEMA},
            },
        )
    except Exception as e:
        entry["error"] = f"request failed: {str(e)[:500]}"
        print(f"[json_schema] FAIL -- {entry['error']}")
        return {"rjson_results": {"json_schema": entry},
                "rjson_passed": 0, "rjson_total": 1}

    _save_probe("rjson_json_schema", messages, resp)
    entry["supported"] = True
    content = resp.choices[0].message.content or ""
    try:
        parsed = json.loads(content)
        conformant = isinstance(parsed, dict) and isinstance(parsed.get("value"), str)
    except json.JSONDecodeError:
        parsed, conformant = None, False
    entry["schema_conformant"] = conformant
    entry["parsed_value"]      = parsed.get("value") if conformant else None
    # Head + tail so a thinking-token prefix doesn't hide the actual JSON.
    entry["raw_content"] = content[:100] + ("…" + content[-100:] if len(content) > 200 else content[100:])
    entry["pass"] = bool(conformant)
    verdict = "PASS" if conformant else "FAIL"
    print(f"[json_schema] {verdict} -- supported, conformant={conformant}, "
          f"content={content[:120]!r}")

    passed = 1 if conformant else 0
    print(f"\nRJSON summary: {passed}/1 passed")
    return {"rjson_results": {"json_schema": entry},
            "rjson_passed": passed, "rjson_total": 1}


def gram_transport_test_round(client: openai.OpenAI) -> dict:
    section("GRAMT test -- does the endpoint support real constrained-decoding custom tools?")
    print("\nSends a type:'custom' tool with format:{type:'grammar', syntax:'lark', ...} --")
    print("the actual OpenAI freeform-tool transport, not classic JSON-schema function")
    print("calling with a tool that happens to be named the same thing. PASS = a genuine")
    print("'custom' tool_call comes back with grammar-valid input.\n")

    results: dict[str, dict] = {}
    for op, spec in GRAM_TRANSPORT_TASKS.items():
        tool = _build_custom_tool(spec["name"], spec["description"], spec["grammar"])
        messages = [
            {"role": "system", "content": "You are a helpful coding assistant with tool access."},
            {"role": "user", "content": spec["task"]},
        ]
        entry: dict = {"task": spec["task"], "pass": False, "error": None,
                       "tool_call_type": None, "raw_input": None}
        try:
            resp = chat(client, messages, tools=[tool])
        except Exception as e:
            entry["error"] = f"request failed: {str(e)[:500]}"
            print(f"[{op}] FAIL -- {entry['error']}")
            results[op] = entry
            continue

        _save_probe(f"gram_transport_{op}", messages, resp, tools=[tool])
        msg        = resp.choices[0].message
        tool_calls = getattr(msg, "tool_calls", None) or []
        if not tool_calls:
            entry["error"] = f"no tool call -- content: {(msg.content or '')[:200]!r}"
            print(f"[{op}] FAIL -- {entry['error']}")
            results[op] = entry
            continue

        tc      = tool_calls[0]
        tc_type = getattr(tc, "type", None)
        entry["tool_call_type"] = tc_type
        if tc_type == "custom":
            raw_input = tc.custom.input
            entry["raw_input"] = raw_input
            try:
                spec["local_validator"]().parse(raw_input)
                entry["pass"] = True
                print(f"[{op}] PASS -- genuine custom_tool_call, grammar-valid input:\n"
                     f"{indent(raw_input.strip(), '  ')}")
            except lark.exceptions.LarkError as e:
                entry["error"] = f"custom tool_call but grammar-invalid: {str(e)[:300]}"
                print(f"[{op}] FAIL -- {entry['error']}")
        else:
            entry["error"] = (f"got tool_call type={tc_type!r} instead of 'custom' -- endpoint "
                              "likely downgrades or silently rejects freeform tools")
            print(f"[{op}] FAIL -- {entry['error']}")
        results[op] = entry

    passed = sum(1 for r in results.values() if r["pass"])
    total  = len(results)
    print(f"\nGRAMT summary: {passed}/{total} passed")
    return {"gram_transport_results": results, "gram_transport_passed": passed,
            "gram_transport_total": total}


# -- STRM test ----------------------------------------------------------------
#
# Sends stream:true and checks whether the endpoint actually delivers the
# response incrementally over SSE (multiple ChatCompletionChunk events, a
# real gap between the first and last chunk) rather than either rejecting
# the parameter outright or accepting it but buffering the whole reply into
# one chunk. This is an endpoint/transport feature, independent of any
# model-behaviour capability.

_STREAM_TASK = "Count from 1 to 8, one number per line."


def stream_test_round(client: openai.OpenAI) -> dict:
    section("STRM test -- does the endpoint deliver real incremental SSE chunks?")
    print("\nSends stream:true and iterates the response as Server-Sent Events. PASS =")
    print("the request is accepted, more than one chunk arrives, content reconstructed")
    print("from delta.content is non-empty, and a finish_reason chunk is seen.\n")

    messages = [
        {"role": "system", "content": "You are a helpful coding assistant."},
        {"role": "user",   "content": _STREAM_TASK},
    ]
    entry: dict = {
        "pass": False, "supported": False, "error": None,
        "chunk_count": 0, "reconstructed_content": "", "finish_reason": None,
        "time_to_first_chunk_seconds": None, "total_seconds": None,
    }
    t0 = time.monotonic()
    try:
        stream = client.chat.completions.create(
            model=MODEL, messages=messages, temperature=_probe_temperature(), timeout=300, stream=True,
        )
        chunks: list = []
        content = ""
        finish_reason = None
        first_chunk_t = None
        for chunk in stream:
            chunks.append(chunk)
            if first_chunk_t is None:
                first_chunk_t = time.monotonic() - t0
            if not chunk.choices:
                continue
            delta = chunk.choices[0].delta
            if delta and delta.content:
                content += delta.content
            if chunk.choices[0].finish_reason:
                finish_reason = chunk.choices[0].finish_reason
    except Exception as e:
        entry["error"] = f"request failed: {str(e)[:500]}"
        print(f"[stream] FAIL -- {entry['error']}")
        return {"stream_results": {"basic": entry}, "stream_passed": 0, "stream_total": 1}

    total_t = time.monotonic() - t0
    entry["supported"]                     = True
    entry["chunk_count"]                   = len(chunks)
    entry["reconstructed_content"]         = content
    entry["finish_reason"]                 = finish_reason
    entry["time_to_first_chunk_seconds"]   = round(first_chunk_t, 3) if first_chunk_t is not None else None
    entry["total_seconds"]                 = round(total_t, 3)
    if _PROBE_DIR is not None:
        safe_label = "stream_basic"
        with open(_PROBE_DIR / f"{safe_label}.json", "w") as f:
            json.dump({"label": safe_label, "messages": messages,
                      "chunk_count": len(chunks), "finish_reason": finish_reason,
                      "reconstructed_content": content,
                      "chunks": [c.model_dump() for c in chunks]}, f, indent=2)

    entry["pass"] = len(chunks) > 1 and bool(content.strip()) and finish_reason is not None
    verdict = "PASS" if entry["pass"] else "FAIL"
    if not entry["pass"] and not entry["error"]:
        entry["error"] = (f"accepted stream:true but delivered {len(chunks)} chunk(s) "
                          f"(finish_reason={finish_reason!r}) -- looks buffered, not real SSE")
    print(f"[stream] {verdict} -- {len(chunks)} chunks, "
          f"first chunk at {entry['time_to_first_chunk_seconds']}s, "
          f"total {entry['total_seconds']}s, finish_reason={finish_reason!r}")

    passed = 1 if entry["pass"] else 0
    print(f"\nSTRM summary: {passed}/1 passed")
    return {"stream_results": {"basic": entry}, "stream_passed": passed, "stream_total": 1}


# -- REASN test ------------------------------------------------------------------
#
# Two things, both endpoint/provider features rather than raw model
# knowledge:
#   1. does the endpoint surface reasoning tokens at all -- a
#      `reasoning_content`/`reasoning`/`thinking` field on the message, or a
#      non-zero `usage.completion_tokens_details.reasoning_tokens` -- for a
#      task that plausibly benefits from it, with no special params sent.
#   2. does the endpoint accept either of the two real-world syntaxes for
#      tuning reasoning effort without erroring: OpenAI's native top-level
#      `reasoning_effort` (Chat Completions param on o-series/gpt-5), or the
#      OpenRouter-style `extra_body={"reasoning": {"effort": ...}}` passthrough
#      many other providers proxy. Accepting the parameter without a 400 is
#      the bar -- this does not attempt to prove the effort setting changed
#      model behaviour, only that the wire syntax is honoured rather than
#      rejected.

_REASON_TASK = ("Solve step by step: a train leaves city A at 60mph, another leaves "
                 "city B (300 miles away) at 40mph toward each other. How long until "
                 "they meet, and how far from A?")


def _extract_reasoning(resp) -> dict:
    msg  = resp.choices[0].message
    dump = msg.model_dump()
    for field in ("reasoning_content", "reasoning", "thinking", "thought"):
        val = dump.get(field)
        if isinstance(val, str) and val.strip():
            return {"found": True, "field": field, "chars": len(val)}
    usage = getattr(resp, "usage", None)
    details = getattr(usage, "completion_tokens_details", None) if usage else None
    reasoning_tokens = getattr(details, "reasoning_tokens", None) if details else None
    if reasoning_tokens:
        return {"found": True, "field": "usage.completion_tokens_details.reasoning_tokens",
                "chars": None, "reasoning_tokens": reasoning_tokens}
    return {"found": False, "field": None, "chars": 0}


def reasoning_test_round(client: openai.OpenAI) -> dict:
    section("REASN test -- reasoning tokens exposed, and can effort be set?")
    print("\nCheck 1: with no special params, does the response carry a reasoning trace")
    print("(reasoning_content/reasoning/thinking field, or usage.reasoning_tokens)?")
    print("Check 2: does the endpoint accept reasoning_effort (native OpenAI param) or")
    print("extra_body={'reasoning': {'effort': ...}} (OpenRouter-style) without a 400?\n")

    results: dict[str, dict] = {}
    messages = [
        {"role": "system", "content": "You are a helpful reasoning assistant."},
        {"role": "user",   "content": _REASON_TASK},
    ]

    baseline_entry: dict = {"pass": False, "error": None}
    try:
        resp = client.chat.completions.create(model=MODEL, messages=messages, timeout=300)
        _save_probe("reasoning_baseline", messages, resp)
        info = _extract_reasoning(resp)
        baseline_entry.update(info)
        baseline_entry["pass"] = info["found"]
        print(f"[baseline] {'PASS' if info['found'] else 'FAIL'} -- "
              f"field={info['field']!r}")
    except Exception as e:
        baseline_entry["error"] = f"request failed: {str(e)[:500]}"
        print(f"[baseline] FAIL -- {baseline_entry['error']}")
    results["reasoning_tokens_present"] = baseline_entry

    def _try_effort(label: str, kwargs: dict) -> dict:
        entry: dict = {"pass": False, "error": None}
        try:
            resp = client.chat.completions.create(
                model=MODEL, messages=messages, timeout=300, **kwargs,
            )
            _save_probe(f"reasoning_effort_{label}", messages, resp)
            info = _extract_reasoning(resp)
            entry.update(info)
            entry["pass"] = True  # accepted without error -- the bar for this check
            print(f"[effort_{label}] PASS -- accepted, reasoning_found={info['found']}")
        except Exception as e:
            entry["error"] = f"request failed: {str(e)[:500]}"
            print(f"[effort_{label}] FAIL -- {entry['error']}")
        return entry

    results["effort_control_native"]     = _try_effort("native", {"reasoning_effort": "high"})
    results["effort_control_extra_body"] = _try_effort(
        "extra_body", {"extra_body": {"reasoning": {"effort": "high"}}})

    passed = sum(1 for r in results.values() if r["pass"])
    total  = len(results)
    print(f"\nREASN summary: {passed}/{total} passed")
    return {"reason_results": results, "reason_passed": passed, "reason_total": total}


# -- agentknit default-tool compatibility test (AKDEF) -------------------------
#
# Unlike TSEL (which scores selection among the model's own elicited tool
# names), AKDEF exercises agentknit's actual shipped default tool schema --
# imported live from agentknit._core, not reconstructed here -- end to end:
# each task's call is run through agentknit's real dispatch() against a
# scratch directory, so PASS means the model's call would actually work
# inside agentknit today. This doubles as a new-model compatibility check
# and a regression test to rerun after fine-tuning a model against
# agentknit's tool contract.

def _akdef_verify_effect(tool_name: str, result_str: str,
                          write_path: str, update_path: str) -> bool:
    """Check the observable effect of a dispatched AKDEF call."""
    if tool_name == "read_file":
        return "hello agentknit" in result_str
    if tool_name == "write_file":
        return os.path.exists(write_path) and Path(write_path).read_text() == "hello world\n"
    if tool_name == "str_replace":
        return os.path.exists(update_path) and "x = 42" in Path(update_path).read_text()
    if tool_name == "exec_shell":
        try:
            return json.loads(result_str).get("returncode") == 0
        except (json.JSONDecodeError, AttributeError):
            return False
    return False


def agentknit_default_tools_test_round(client: openai.OpenAI) -> dict:
    section("AKDEF -- agentknit default-tool compatibility")
    try:
        from agentknit._core import _DEFAULT_TOOL_DISPATCH, _DEFAULT_TOOL_SCHEMA, dispatch
    except ImportError as e:
        print(f"\nERROR: agentknit not importable: {e}")
        return {"error": f"agentknit not importable: {e}"}

    import shutil
    import tempfile
    tmpdir = tempfile.mkdtemp(prefix="llmprobe_akdef_")
    read_path   = os.path.join(tmpdir, "hello.txt")
    write_path  = os.path.join(tmpdir, "write_target.txt")
    update_path = os.path.join(tmpdir, "update_target.py")
    Path(read_path).write_text("hello agentknit\n")
    Path(update_path).write_text("x = 1\n")

    tasks = {
        "read_file":   f"Please read the file {read_path} and tell me its contents.",
        "write_file":  f"Please write 'hello world\\n' to the file {write_path}.",
        "str_replace": f"In the file {update_path}, replace the exact string 'x = 1' "
                        "with 'x = 42'.",
        "exec_shell":  f"Please run `ls -la {tmpdir}` and show me the output.",
    }

    results: dict[str, dict] = {}
    try:
        for tool_name, task in tasks.items():
            messages = [
                {"role": "system", "content": "You are a helpful assistant with tool access."},
                {"role": "user",   "content": task},
            ]
            resp = chat(client, messages, tools=_DEFAULT_TOOL_SCHEMA)
            _save_probe(f"akdef_{tool_name}", messages, resp, tools=_DEFAULT_TOOL_SCHEMA)
            call = extract_call_from_response(resp)
            entry: dict = {
                "expected_tool":  tool_name,
                "called_tool":    call.function_name if call else None,
                "arguments":      call.arguments if call else None,
                "tool_selected":  bool(call and call.function_name == tool_name),
                "dispatch_ok":    False,
                "effect_verified": False,
                "error": None,
            }
            if not call:
                entry["error"] = "no tool call detected"
            elif not entry["tool_selected"]:
                entry["error"] = f"called {call.function_name!r} instead of {tool_name!r}"
            else:
                try:
                    result_str, _ = dispatch(tool_name, call.arguments, _DEFAULT_TOOL_DISPATCH)
                except Exception as e:
                    entry["error"] = f"dispatch raised: {e}"
                    result_str = None
                if result_str is not None:
                    entry["dispatch_result"] = result_str[:500]
                    if result_str.startswith("ERROR"):
                        entry["error"] = result_str
                    else:
                        entry["dispatch_ok"] = True
                        entry["effect_verified"] = _akdef_verify_effect(
                            tool_name, result_str, write_path, update_path)
                        if not entry["effect_verified"]:
                            entry["error"] = "dispatch succeeded but the on-disk effect was wrong"
            entry["pass"] = entry["dispatch_ok"] and entry["effect_verified"]
            print(f"[{tool_name}] {'PASS' if entry['pass'] else 'FAIL'} -- "
                  f"called={entry['called_tool']!r} {entry['error'] or ''}")
            results[tool_name] = entry
    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)

    passed = sum(1 for r in results.values() if r["pass"])
    total  = len(results)
    print(f"\nAKDEF summary: {passed}/{total} passed")
    return {"akdef_results": results, "akdef_passed": passed, "akdef_total": total}


# -- CACH test -----------------------------------------------------------------
#
# Empirically measures the endpoint's prompt-cache TTL: prime the cache with a
# large fixed prefix, then re-send the exact same prefix (with only a distinct
# trailing question) after increasing delays and watch the provider-reported
# cached-prompt-token count (`usage.prompt_cache_hit_tokens`,
# `usage.cached_tokens` / `usage.prompt_tokens_details.cached_tokens`). While a
# delay still shows a warm cache, the TTL is longer; the first delay at which
# the cached-token count collapses to ~0 brackets the TTL. The probe stops
# early at the first cold sample, so it takes roughly one prime plus the delay
# of the first cold rung -- usually a few minutes, which is why it is off by
# default (`--cache-ttl-test` to enable).
#
# The result is informational rather than pass/fail: a TTL has no right or
# wrong value. It only fails when the endpoint never reports cached tokens at
# all (no observable prompt cache) or never answers.

_CACH_TOKEN_TARGET   = 2500   # rough token budget for the fixed prefix
_CACH_DELAYS_SECONDS = (30, 120, 300, 600)
_CACH_WARM_FRACTION  = 0.5    # cached >= fraction of prompt tokens -> warm
_CACH_WARM_MIN       = 500    # absolute floor: ignore trivial sub-prefix matches


class _CachChatError(Exception):
    """The endpoint refused the bare cache-TTL probe call itself."""


def _cach_build_prefix() -> str:
    """Build a large, fixed prompt prefix (~_CACH_TOKEN_TARGET tokens)."""
    words = ("alpha beta gamma delta epsilon zeta eta theta iota kappa lambda "
             "mu nu xi omicron pi rho sigma tau upsilon phi chi psi omega "
             "cache prompt token latency provider endpoint sample probe").split()
    rng = random.Random(42)  # deterministic across runs/delays
    lines = ["The following reference text is fixed. Read it once and keep it "
             "in mind; the actual question comes after it.", ""]
    i = 0
    while len("\n".join(lines)) // 4 < _CACH_TOKEN_TARGET:
        sent = " ".join(rng.choice(words) for _ in range(50))
        lines.append(f"Paragraph {i}: {sent}.")
        i += 1
    return "\n".join(lines)


def _cach_provider(resp) -> str | None:
    """Return the upstream provider name reported for this response, if any.

    OpenRouter load-balances every request across several upstreams (SiliconFlow,
    Novita, GMICloud, Morph, ...) and echoes the one it picked in a top-level
    ``provider`` field.  Each upstream has its own prefix cache, so a re-send
    served by a *different* upstream can never be a cache hit: without this
    field, a provider switch is indistinguishable from an expired cache entry.
    """
    dump = resp.model_dump() if hasattr(resp, "model_dump") else {}
    val = dump.get("provider")
    return val if isinstance(val, str) and val else None


def _cach_cached_tokens(resp) -> int | None:
    """Extract the provider-reported cached-prompt-token count, if any."""
    usage = getattr(resp, "usage", None)
    if usage is None:
        return None
    dump = usage.model_dump() if hasattr(usage, "model_dump") else {}
    for field in ("prompt_cache_hit_tokens", "cached_tokens",
                  "cache_read_input_tokens", "prompt_tokens_cached"):
        val = dump.get(field)
        if isinstance(val, int) and not isinstance(val, bool):
            return val
    details = dump.get("prompt_tokens_details") or {}
    val = details.get("cached_tokens")
    if isinstance(val, int) and not isinstance(val, bool):
        return val
    return None


def cache_ttl_test_round(client: openai.OpenAI) -> dict:
    section("CACH test -- empirically measuring the endpoint's prompt-cache TTL")
    print("\nPrimes the cache with a large fixed prefix, then re-sends the same prefix")
    print("after increasing delays, stopping at the first cold (cache-miss) sample.")
    print("The TTL is bracketed between the last warm and the first cold delay.\n")

    prefix = _cach_build_prefix()

    def _one_call(tag: str, pin_provider: str | None = None) -> dict:
        messages = [
            {"role": "user",
             "content": f"{prefix}\n\nQuestion ({tag}): in one word, what colour is the sky?"},
        ]
        # This test needs a bare tools-free request -- going through chat()
        # would send tools=None verbatim, which some endpoints (e.g. Kimi's
        # Coding API) 400 on.
        kwargs: dict = dict(model=MODEL, messages=messages,
                            temperature=_probe_temperature(), timeout=300)
        if pin_provider:
            # Load-balanced endpoints pick an upstream per request; pin it so
            # every re-send hits the upstream that wrote the cache entry.
            kwargs["extra_body"] = {"provider": {"order": [pin_provider],
                                                 "allow_fallbacks": False}}
        t0 = time.monotonic()
        try:
            resp = client.chat.completions.create(**kwargs)
        except Exception as e:
            raise _CachChatError(str(e)) from e
        usage = getattr(resp, "usage", None)
        sample = {
            "prompt_tokens":     getattr(usage, "prompt_tokens", None) if usage else None,
            "completion_tokens": getattr(usage, "completion_tokens", None) if usage else None,
            "cached_tokens":     _cach_cached_tokens(resp),
            "latency_seconds":   round(time.monotonic() - t0, 3),
            "provider":          _cach_provider(resp),
        }
        return {"messages": messages, "resp": resp, "sample": sample}

    samples: dict[str, dict] = {}

    try:
        prime = _one_call("prime")
    except _CachChatError as e:
        print(f"[prime] endpoint refused the bare probe call -- {e}")
        return {"error": f"prime request failed: {str(e)[:500]}"}
    samples["prime"] = prime["sample"]
    if _PROBE_DIR is not None:
        _save_probe("cache_ttl_prime", prime["messages"], prime["resp"])
    print(f"[prime] prompt_tokens={prime['sample']['prompt_tokens']} "
          f"cached={prime['sample']['cached_tokens']} "
          f"provider={prime['sample'].get('provider') or '?'}")

    # A load-balanced endpoint (OpenRouter) serves each request from a different
    # upstream by default, which makes a cold re-send ambiguous: it can mean
    # "cache expired" or just "another upstream".  Pin the provider that answered
    # the prime call so all re-sends share its cache namespace.
    pinned_provider = prime["sample"].get("provider")
    if pinned_provider:
        print(f"[pin] pinning provider to {pinned_provider!r} for all re-sends")
    result_provider: dict = {"provider": pinned_provider, "provider_pinned": pinned_provider}

    # Immediate (0-delay) re-send: proves the prefix is cacheable at all on this
    # provider, before any expiry can be blamed.  Without it, a cold first
    # delayed sample cannot be told apart from "no cache reuse ever".
    try:
        confirm = _one_call("confirm_0s", pin_provider=pinned_provider)
    except _CachChatError as e:
        print(f"[confirm_0s] endpoint refused the bare probe call -- {e}")
        return {"error": f"confirm request failed: {str(e)[:500]}", "samples": samples,
                **result_provider}
    samples["confirm_0s"] = confirm["sample"]
    if _PROBE_DIR is not None:
        _save_probe("cache_ttl_confirm_0s", confirm["messages"], confirm["resp"])
    confirm_prompt = confirm["sample"]["prompt_tokens"] or 0
    confirm_cached = confirm["sample"]["cached_tokens"]
    confirm_warm = (confirm_cached is not None
                    and confirm_cached >= max(_CACH_WARM_MIN,
                                              _CACH_WARM_FRACTION * confirm_prompt))
    print(f"[confirm_0s] prompt_tokens={confirm_prompt} cached={confirm_cached} "
          f"provider={confirm['sample'].get('provider') or '?'} "
          f"-> {'WARM' if confirm_warm else 'COLD'}")
    if pinned_provider and confirm["sample"].get("provider") not in (None, pinned_provider):
        print(f"[warn] provider changed despite the pin "
              f"({pinned_provider!r} -> {confirm['sample'].get('provider')!r}): "
              f"cache reuse across upstreams is not comparable")
    if not confirm_warm:
        print("\nCACH summary: no cache reuse even on an immediate re-send -- the prefix "
              "is not served from cache on this provider, so no TTL can be measured")
        return {"samples": samples, "delays_seconds": list(_CACH_DELAYS_SECONDS),
                "cache_observed": False,
                "ttl_min_seconds": None, "ttl_max_seconds": None,
                "ttl_min_minutes": None, "ttl_max_minutes": None,
                "error": "no cache reuse on an immediate re-send; no TTL measurable",
                **result_provider}

    ttl_min = None   # last delay still warm (seconds)
    ttl_max = None   # first delay observed cold (seconds)
    for delay in _CACH_DELAYS_SECONDS:
        print(f"[delay {delay}s] waiting before re-sending the same prefix...")
        time.sleep(delay)
        try:
            run = _one_call(f"delay_{delay}s", pin_provider=pinned_provider)
        except _CachChatError as e:
            print(f"[delay {delay}s] endpoint refused the bare probe call -- {e}")
            return {"error": f"request at delay {delay}s failed: {str(e)[:500]}",
                    "samples": samples, **result_provider}
        sample = run["sample"]
        samples[f"delay_{delay}s"] = sample
        if _PROBE_DIR is not None:
            _save_probe(f"cache_ttl_delay_{delay}s", run["messages"], run["resp"])
        prompt = sample["prompt_tokens"] or 0
        cached = sample["cached_tokens"]
        served_by = sample.get("provider")
        warm = cached is not None and cached >= max(_CACH_WARM_MIN, _CACH_WARM_FRACTION * prompt)
        print(f"[delay {delay}s] prompt_tokens={prompt} cached={cached} "
              f"provider={served_by or '?'} -> {'WARM' if warm else 'COLD'}")
        if pinned_provider and served_by not in (None, pinned_provider):
            print(f"\nCACH summary: the endpoint answered from a different upstream "
                  f"({served_by!r} instead of {pinned_provider!r}) despite the provider pin, "
                  f"so a cache miss here cannot be attributed to expiry -- TTL not measurable")
            return {"samples": samples, "delays_seconds": list(_CACH_DELAYS_SECONDS),
                    "cache_observed": False,
                    "ttl_min_seconds": None, "ttl_max_seconds": None,
                    "ttl_min_minutes": None, "ttl_max_minutes": None,
                    "provider_mismatch": {"expected": pinned_provider, "got": served_by},
                    "error": f"provider switched to {served_by!r} despite pinning to "
                             f"{pinned_provider!r}; TTL not measurable on a load-balanced "
                             f"endpoint without a stable upstream",
                    **result_provider}
        if warm:
            ttl_min = delay
        elif cached is None:
            # the endpoint never reports cached tokens: the cache is not
            # observable from here, so a TTL cannot be measured at all
            print("\nCACH summary: no cached tokens ever reported -- no observable prompt cache")
            return {"samples": samples, "delays_seconds": list(_CACH_DELAYS_SECONDS),
                    "cache_observed": False,
                    "error": "no cached tokens ever reported; prompt cache not observable "
                             "from this endpoint",
                    **result_provider}
        else:
            ttl_max = delay
            break

    if ttl_min is None and ttl_max is None:
        print("\nCACH summary: no cached tokens ever reported -- no observable prompt cache")
        return {"samples": samples, "delays_seconds": list(_CACH_DELAYS_SECONDS),
                "cache_observed": False,
                "error": "no cached tokens ever reported; prompt cache not observable "
                         "from this endpoint",
                **result_provider}

    result: dict = {
        "samples": samples,
        "delays_seconds": list(_CACH_DELAYS_SECONDS),
        "cache_observed": True,
        "ttl_min_seconds": ttl_min,
        "ttl_max_seconds": ttl_max,
        "ttl_min_minutes": round(ttl_min / 60, 2) if ttl_min is not None else None,
        "ttl_max_minutes": round(ttl_max / 60, 2) if ttl_max is not None else None,
        **result_provider,
    }
    if ttl_max is not None:
        result["measured_ttl_seconds"] = (ttl_min or 0) + (ttl_max - (ttl_min or 0)) / 2
        result["measured_ttl_minutes"] = round(result["measured_ttl_seconds"] / 60, 2)
        print(f"\nCACH summary: TTL between {ttl_min or 0}s and {ttl_max}s "
              f"(midpoint estimate {result['measured_ttl_minutes']} min)")
    else:
        result["measured_ttl_seconds"] = None
        result["measured_ttl_minutes"] = None
        print(f"\nCACH summary: still warm after {ttl_min}s -- TTL > {ttl_min}s "
              f"(upper bound not reached)")
    return result


# -- context-window recovery (CTX) ---------------------------------------------
#
# The context window is the number an agent needs to size its prompt budget, so
# every probe tries to recover it. Both sources below are the endpoint speaking
# for itself, tried in order of trust:
#
#   1. endpoint metadata -- GET {base}/models, where providers that publish
#      per-model limits expose them: OpenRouter and Kimi return `context_length`,
#      and the Copilot API returns `capabilities.limits.*` (context window, max
#      prompt, max output). The official DeepSeek and z.ai endpoints publish no
#      limits at all.
#   2. the endpoint's own rejection -- send one deliberately oversized prompt and
#      parse the limit out of the error message ("This model's maximum context
#      length is 1048576 tokens", "prompt token count of N exceeds the limit of
#      M"). A rejected request produces no tokens, so this costs one request and
#      nothing else.
#
# If the endpoint *accepts* the oversized prompt, the value is recorded as a
# lower bound and nothing larger is sent: a prompt that large is billed for real,
# so the probe never escalates against a model with a huge window. When neither
# source names a limit -- z.ai answers an oversized prompt with a bare "Prompt
# exceeds max length" and publishes no metadata -- the entry is recorded as
# unknown unless a value is supplied explicitly with --context-tokens.
#
# `kind` records which limit the number is: a true context window (prompt plus
# completion) as DeepSeek/OpenRouter/Kimi describe it, or the max *prompt* tokens
# the Copilot API enforces, which is the binding number for a request. Mixing the
# two silently would make reports incomparable.

_CTX_OVERSIZED_TOKENS = 1_200_000
_CTX_FALLBACK_TOKENS  = 200_000

# Provider metadata fields carrying a per-model limit, most specific first.
_CTX_METADATA_FIELDS = (
    "max_context_window_tokens", "context_length", "context_window",
    "context_window_tokens", "max_context_length", "max_context_tokens",
    "max_prompt_tokens", "input_token_limit",
)

# Limits worth reporting alongside the context window but never mistaken for it.
_CTX_EXTRA_FIELDS = ("max_output_tokens", "max_prompt_images")

# Endpoint rejection messages that name the limit. Each pattern's single capture
# group is the token count.
_CTX_ERROR_PATTERNS = (
    r"maximum context length is\s+(\d+)",
    r"exceeds the limit of\s+(\d+)",
    r"maximum context length of\s+(\d+)",
    r"max(?:imum)? context (?:length|window)[^0-9]{0,24}(\d{3,})",
    r"(?:prompt|input) (?:token count|tokens)[^0-9]{0,24}(\d{4,})",
)


def _ctx_parse_limit(text: str) -> tuple[int, str] | None:
    """Pull a token limit out of an endpoint's rejection message, if it names one."""
    for pattern in _CTX_ERROR_PATTERNS:
        m = re.search(pattern, text, re.IGNORECASE)
        if m:
            tokens = int(m.group(1))
            if tokens >= 1000:  # ignore capture noise like a stray small integer
                return tokens, m.group(0).strip()
    return None


def _ctx_find_model_entry(payload: dict, model_id: str) -> dict | None:
    """Find this model's entry in a /models response, tolerating id decoration.

    Providers decorate ids differently from the id used on the wire (OpenRouter
    carries a `vendor/` prefix; a probe may be labelled `k3` while the API calls
    it `k3-256k`). Match exact first, then case-insensitively, then on the final
    path segment, and finally on a prefix relationship, so a labelled id still
    resolves.
    """
    entries = payload.get("data")
    if not isinstance(entries, list):
        return None
    ids = [(e.get("id") or "", e) for e in entries if isinstance(e, dict)]
    for candidate in (model_id, model_id.split("/")[-1]):
        for entry_id, entry in ids:
            if entry_id == candidate:
                return entry
    for candidate in (model_id, model_id.split("/")[-1]):
        for entry_id, entry in ids:
            if entry_id.lower() == candidate.lower():
                return entry
    for candidate in (model_id, model_id.split("/")[-1]):
        for entry_id, entry in ids:
            if entry_id.lower().endswith(candidate.lower()):
                return entry
    return None


def _ctx_metadata_limits(entry: dict) -> dict:
    """Flatten a model entry's advertised limits into {field: int}."""
    limits: dict = {}
    stack = [entry]
    while stack:
        node = stack.pop()
        if not isinstance(node, dict):
            continue
        for key, value in node.items():
            if isinstance(value, dict):
                if key in ("limits", "capabilities", "top_provider"):
                    stack.append(value)
            elif isinstance(value, int) and not isinstance(value, bool):
                if key in _CTX_METADATA_FIELDS or key in _CTX_EXTRA_FIELDS:
                    limits[key] = value
    return limits


def _ctx_from_metadata() -> dict | None:
    """Recover the context window from the endpoint's own /models metadata."""
    if not ENDPOINT.startswith(("http://", "https://")):
        return None  # a local --script wrapper exposes no metadata over HTTP
    payload = _ctx_fetch_models()
    if payload is None:
        return None
    entry = _ctx_find_model_entry(payload, MODEL)
    if entry is None:
        return None
    limits = _ctx_metadata_limits(entry)
    chosen_field = next((f for f in _CTX_METADATA_FIELDS if f in limits), None)
    if chosen_field is None:
        return None
    kind = "max_prompt_tokens" if chosen_field == "max_prompt_tokens" else "context_window"
    detail = {k: v for k, v in limits.items() if k != chosen_field}
    return {
        "tokens":  limits[chosen_field],
        "kind":    kind,
        "source":  "api_metadata",
        "evidence": f"{ENDPOINT.rstrip('/')}/models -> {entry.get('id')}: "
                    f"{chosen_field}={limits[chosen_field]}",
        "metadata": detail,
    }


def _ctx_fetch_models() -> dict | None:
    """GET {endpoint}/models, returning the parsed body or None on any failure."""
    try:
        headers = {}
        if _KEY_NAME:
            try:
                key = get_api_key(_KEY_NAME)
            except SystemExit:
                key = ""
            if key.strip():
                headers["Authorization"] = f"Bearer {key}"
        url = ENDPOINT.rstrip("/") + "/models"
        req = urllib.request.Request(url, headers=headers)
        with urllib.request.urlopen(req, timeout=60) as response:
            data = json.load(response)
        return data if isinstance(data, dict) else None
    except Exception:
        # Metadata is a best-effort source; the error-message path still stands.
        return None


def _ctx_from_error(client) -> dict | None:
    """Recover the context window from the endpoint's own rejection message.

    One oversized prompt is sent. If the endpoint rejects it and names a limit,
    that is the answer. If it rejects without naming a limit, nothing smaller can
    teach us more -- a smaller prompt can only ever be *accepted*, which yields a
    useless lower bound -- so the probe stops rather than spend those tokens. The
    one exception is a failure that never reached the model (a 413 or a transport
    error, e.g. a size-capping local wrapper), where a smaller prompt is worth
    trying because the real limit was never exercised.
    """
    for tokens in (_CTX_OVERSIZED_TOKENS, _CTX_FALLBACK_TOKENS):
        messages = [{"role": "user", "content": "alpha " * tokens}]
        try:
            resp = client.chat.completions.create(
                model=MODEL, messages=messages, max_tokens=16, timeout=900)
        except Exception as e:
            text = str(e)
            parsed = _ctx_parse_limit(text)
            if parsed:
                limit, evidence = parsed
                return {
                    "tokens":   limit,
                    "kind":     "max_prompt_tokens" if "prompt" in text.lower()
                                else "context_window",
                    "source":   "api_error_message",
                    "evidence": evidence,
                    "probed_prompt_tokens": tokens,
                }
            if tokens == _CTX_FALLBACK_TOKENS or not _ctx_is_transport_failure(text):
                return None
            continue  # never reached the model -- try one smaller prompt
        usage = getattr(resp, "usage", None)
        accepted = getattr(usage, "prompt_tokens", None) or tokens
        return {
            "tokens":      accepted,
            "kind":        "context_window",
            "source":      "accepted_oversized_prompt",
            "lower_bound": True,
            "evidence":    f"endpoint accepted a ~{tokens}-token prompt "
                           f"(prompt_tokens={accepted}); the real limit is at least this",
        }
    return None


def _ctx_is_transport_failure(text: str) -> bool:
    """True when a failure looks like the request never reached the model.

    A too-large *body* (413) or a broken connection says nothing about the
    context window, so those are worth retrying smaller; a context rejection
    (which this function is only called on when it named no number) is not.
    """
    lowered = text.lower()
    if any(marker in lowered for marker in ("413", "entity too large", "payload too large",
                                            "request body too large", "connection",
                                            "timed out", "timeout", "broken pipe")):
        return True
    return False


def context_window_round(client) -> dict:
    """Recover the model's context window from metadata, else from a rejection."""

    section("CTX test -- recovering the context window size")
    print("\nTries the endpoint's own /models metadata first, then falls back to")
    print("sending one deliberately oversized prompt and parsing the limit out of")
    print("the rejection message. A rejected request produces no tokens.\n")

    for label, probe_fn in (("metadata", _ctx_from_metadata),
                            ("oversized prompt", lambda: _ctx_from_error(client))):
        print(f"[ctx] trying {label}...")
        result = probe_fn()
        if result:
            kind = result.get("kind", "context_window")
            suffix = " (lower bound)" if result.get("lower_bound") else ""
            print(f"[ctx] {result['tokens']} tokens ({kind}, via {result['source']}){suffix}")
            print(f"[ctx] evidence: {result['evidence']}")
            return result
        print(f"[ctx] {label} gave no limit")

    print("\nCTX summary: no context window published or named by this endpoint -- "
          "record one with --context-tokens")
    return {"tokens": None, "kind": None, "source": None,
            "error": "no context window advertised in /models metadata and no limit "
                     "named in the endpoint's rejection message"}


def _make_target_client():
    """Client for the probe target: a local script wrapper, or the HTTP endpoint."""
    if ENDPOINT.startswith("script:"):
        return ScriptClient(ENDPOINT.split("script:", 1)[1])
    return make_client(get_api_key(_KEY_NAME) if _KEY_NAME else "")


def _resolve_context_window(args, client_factory) -> dict | None:
    """Pick the context-window source: an explicit value, else live recovery.

    `client_factory` is only called when live recovery is needed, so recording a
    known value with --context-tokens requires no credentials at all.
    """
    if args.context_tokens is not None:
        evidence = args.context_evidence or "supplied on the command line"
        return {
            "tokens":   args.context_tokens,
            "kind":     "max_prompt_tokens" if "prompt token" in evidence.lower()
                        else "context_window",
            "source":   args.context_source,
            "evidence": evidence,
        }
    if not args.context_test:
        return None
    return context_window_round(client_factory())


# -- CORS preflight test (CORS) ------------------------------------------------
#
# Whether a plain web page can call the endpoint directly depends on its CORS
# policy: a browser first sends an OPTIONS preflight (because the request
# carries Authorization and Content-Type: application/json) and refuses to
# deliver the response unless the server answers with a matching
# access-control-allow-origin. So the probe replays exactly that handshake:
#
#   1. OPTIONS {base}/chat/completions with Origin, Access-Control-Request-
#      Method: POST, and Access-Control-Request-Headers: authorization,
#      content-type -- the preflight every browser sends for an SDK call.
#   2. GET {base}/models with just Origin set -- some endpoints answer the
#      preflight yet strip the CORS headers from the actual response, which
#      breaks the call one request later; the one that worked is the evidence.
#
# The requests carry no Authorization, so this round needs no credentials.
# `allow_origin` records the endpoint's answer verbatim: `*` (any origin can
# integrate), the echoed request origin (browser-direct works from anywhere,
# but only without cookies), or null (server sent none). `preflight_passed`
# additionally requires allow-methods to cover POST and allow-headers to cover
# both authorization and content-type, since a preflight that omits them fails
# in the browser just like a missing origin.

_CORS_REQUEST_ORIGIN = "https://llmprobe.example"
_CORS_REQUEST_HEADERS = "authorization, content-type"


def _cors_header(headers, name: str) -> str | None:
    """Case-insensitively fetch one header from an HTTPResponse / HTTPMessage."""
    return headers.get(name)


def _cors_probe_url(path: str) -> str:
    return ENDPOINT.rstrip("/") + path


def _cors_send(req: urllib.request.Request) -> tuple[int | None, object]:
    """One HTTP request that never raises; status is None on a transport error."""
    try:
        with urllib.request.urlopen(req, timeout=30) as resp:
            return resp.status, resp.headers
    except urllib.error.HTTPError as e:
        # A preflight answered with 4xx/5xx still carries its headers, and that
        # answer is the datum: many endpoints "handle" OPTIONS with a 404/405.
        return e.code, e.headers
    except Exception:
        return None, {}


def _cors_classify(headers) -> dict:
    origin = _cors_header(headers, "access-control-allow-origin")
    methods = _cors_header(headers, "access-control-allow-methods") or ""
    allowed_headers = _cors_header(headers, "access-control-allow-headers") or ""
    credentials = _cors_header(headers, "access-control-allow-credentials") or ""
    expose = _cors_header(headers, "access-control-expose-headers") or ""
    return {
        "allow_origin":      origin,
        "allow_methods":     methods or None,
        "allow_headers":     allowed_headers or None,
        "allow_credentials": (credentials.lower() == "true") or None,
        "expose_headers":    expose or None,
    }


def _cors_allows_probe_origin(origin: str | None) -> bool:
    """Whether an ACAO value permits the probe page's origin."""
    return origin in {"*", _CORS_REQUEST_ORIGIN}


def _cors_wildcard(headers) -> dict:
    """Detail rows for the report: which headers were set and to what."""
    interesting = (
        "access-control-allow-origin", "access-control-allow-methods",
        "access-control-allow-headers", "access-control-allow-credentials",
        "access-control-expose-headers", "vary",
    )
    out: dict[str, str] = {}
    for name in interesting:
        value = headers.get(name) if hasattr(headers, "get") else None
        if value is not None:
            out[name] = value
    return out


def _cors_run() -> dict:
    """Replay the browser handshake for POST /chat/completions + GET /models."""
    preflight = urllib.request.Request(
        _cors_probe_url("/chat/completions"), method="OPTIONS",
        headers={"Origin": _CORS_REQUEST_ORIGIN,
                 "Access-Control-Request-Method": "POST",
                 "Access-Control-Request-Headers": _CORS_REQUEST_HEADERS})
    status, headers = _cors_send(preflight)
    pre = _cors_classify(headers)
    pre["status"] = status
    pre["headers"] = _cors_wildcard(headers)
    pre["pass"] = bool(
        status is not None
        and _cors_allows_probe_origin(pre["allow_origin"])
        and "POST" in pre["allow_methods"].upper()
        and all(h in pre["allow_headers"].lower()
                for h in ("authorization", "content-type")))

    actual = urllib.request.Request(
        _cors_probe_url("/models"), method="GET",
        headers={"Origin": _CORS_REQUEST_ORIGIN})
    status, headers = _cors_send(actual)
    act = _cors_classify(headers)
    act["status"] = status
    act["headers"] = _cors_wildcard(headers)
    act["pass"] = bool(status is not None and _cors_allows_probe_origin(act["allow_origin"]))

    return {"preflight": pre, "actual_response": act}


def cors_test_round() -> dict:
    """Probe the endpoint's CORS policy for browser-direct integration."""
    section("CORS test -- browser-direct integration")
    print("\nReplays the CORS preflight a browser sends before a cross-origin")
    print("POST /chat/completions (Authorization + application/json), then a")
    print("plain Origin-tagged GET /models. No credentials are sent.\n")

    if not ENDPOINT.startswith(("http://", "https://")):
        print("[cors] skipped: a local --script wrapper is not an HTTP endpoint")
        return {"error": "CORS does not apply: the probe target is a local script "
                         "wrapper, not an HTTP endpoint"}

    result = _cors_merge(_cors_run())
    for label, part in (("preflight", result["preflight"]),
                        ("actual response", result["actual_response"])):
        origin = part["allow_origin"]
        verdict = "no access-control-allow-origin" if origin is None else repr(origin)
        print(f"[cors] {label}: HTTP {part['status']}, {verdict} -> "
              f"{'PASS' if part['pass'] else 'FAIL'}")
    print("\nCORS summary: "
          + _cors_summary_line(result))
    return result


def _cors_origin_cell(cors_test: dict) -> str:
    """The `CORS` capabilities-table cell: direct verdict plus observed origin."""
    if cors_test.get("mode") == "blocked":
        return "preflight or response blocks browser-direct access"
    origin = cors_test.get("allow_origin")
    if origin is None:
        return "no access-control-allow-origin"
    if "*" in origin:
        return "`*` (any origin)"
    return f"reflected origin `{origin}`"


def _cors_summary_line(cors_test: dict) -> str:
    """One-line verdict for the console and the report intro."""
    mode = cors_test.get("mode")
    if mode == "wildcard":
        return "wildcard access-control-allow-origin: * -- any web page can call this endpoint"
    if mode == "reflected":
        return (f"access-control-allow-origin echoes the request origin "
                f"({cors_test.get('allow_credentials') and 'with' or 'without'} "
                f"credentials) -- browser-direct works, no wildcard")
    if mode == "blocked":
        return "CORS headers were observed, but preflight or the actual response blocks browser-direct access"
    if mode == "none":
        return "no access-control-allow-origin on any CORS probe -- a browser cannot read responses from this endpoint"
    return "no verdict"


def _cors_reduce(cors: dict) -> dict:
    """Collapse the two probe legs into the summary fields the report shows."""
    pre = cors.get("preflight") or {}
    act = cors.get("actual_response") or {}
    preflight_passed = bool(pre.get("pass"))
    actual_response_passed = bool(act.get("pass"))
    if preflight_passed and actual_response_passed:
        if pre.get("allow_origin") == "*" and act.get("allow_origin") == "*":
            mode = "wildcard"
            allow_origin = "*"
        else:
            mode = "reflected"
            allow_origin = (pre.get("allow_origin") if pre.get("allow_origin") != "*"
                            else act.get("allow_origin"))
    elif pre.get("allow_origin") or act.get("allow_origin"):
        mode = "blocked"
        allow_origin = None
    else:
        mode = "none"
        allow_origin = None
    evidence = (f"OPTIONS {ENDPOINT.rstrip('/')}/chat/completions -> "
                f"access-control-allow-origin: {pre.get('allow_origin') or '(none)'}; "
                f"GET /models -> access-control-allow-origin: "
                f"{act.get('allow_origin') or '(none)'}")
    return {
        "mode":                mode,
        "allow_origin":        allow_origin,
        "allow_credentials":   (pre.get("allow_credentials")
                                or act.get("allow_credentials") or None),
        "preflight_passed":    preflight_passed,
        "actual_response_passed": actual_response_passed,
        "evidence":            evidence,
    }


def _cors_merge(cors_test: dict) -> dict:
    """Return the stored dict with its summary fields filled in (idempotent).

    An empty dict (key present but the round never stored anything) or an
    error-only dict is returned untouched, so the report can still tell
    "not run" and "does not apply" apart from a measured result.
    """
    if not cors_test or "error" in cors_test:
        return cors_test
    merged = dict(cors_test)
    merged.update(_cors_reduce(merged))
    return merged


# -- deferred tool loading test (TDEF) -----------------------------------------
#
# "Deferred tool loading" keeps a tool's parameter schema out of the prompt
# until the model asks for it through a tool-search tool. Two incompatible
# implementations exist upstream, and an OpenAI/Anthropic-compatible endpoint
# could plausibly carry either:
#
#   - OpenAI, Responses API only: a {"type": "tool_search"} entry in `tools`,
#     with deferrable function tools marked "defer_loading": true.
#   - Anthropic, Messages API: a server tool
#     {"type": "tool_search_tool_bm25_20251119", ...}, same "defer_loading"
#     marker, with results coming back as tool_search_tool_result blocks.
#
# The hazard this probe exists for: compatibility layers built in front of a
# non-OpenAI model routinely *drop unknown request fields silently* (DeepSeek
# documents exactly that). So an HTTP 200 on a request carrying defer_loading
# proves nothing at all, and neither does the absence of an error message.
# Three rounds per surface, cheapest first:
#
#   1. Rejection  -- send the deferred tool set. A 4xx naming the field is a
#      clean negative and ends it; 404/405 on the path means the surface is
#      simply absent (n/a). HTTP 200 proves nothing, so continue.
#   2. Token accounting -- the real test. Send the same request three ways:
#      with no tools (baseline), with all tool schemas inline (control), and
#      with most of them deferred. If deferral is honoured, the reported input
#      tokens must fall by most of what the schemas cost; if the counts match
#      the control, the field was swallowed and the schemas went in anyway.
#   3. Reachability -- ask for something only a deferred tool can do. Native
#      support shows a tool-search step before the real call; a direct call to
#      a deferred tool with valid arguments proves the schema was in the prompt
#      all along, corroborating round 2 without trusting usage accounting.
#
# One pitfall is encoded in the tool set itself: never mark every tool
# deferred. Anthropic answers `400 All tools have defer_loading set`, and a
# probe that trips that reads as "rejected" for entirely the wrong reason.

_TDEF_ANTHROPIC_VERSION = "2023-06-01"
_TDEF_SEARCH_TOOL_ANTHROPIC = {"type": "tool_search_tool_bm25_20251119",
                               "name": "tool_search_tool_bm25"}
_TDEF_SEARCH_TOOL_OPENAI = {"type": "tool_search"}
_TDEF_MAX_TOKENS = 64
# The reachability round needs headroom: a reasoning model spends its first
# hundreds of tokens thinking, and a tool call truncated away reads as "no
# call at all", which would silently turn the round into a non-observation.
_TDEF_REACH_MAX_TOKENS = 1024
_TDEF_BASE_PROMPT = "Reply with the single word OK."
_TDEF_REACH_PROMPT = ("You must call a tool, not answer from memory. How many "
                      "units of SKU AB-1234 are on hand in warehouse HAM-3 "
                      "right now?")

# Each deferred tool carries a few hundred tokens of parameter schema, so the
# difference between "deferred" and "inline" is far outside usage-counter noise.
_TDEF_TOOL_SPECS = (
    ("warehouse_inventory_lookup",
     "Look up live on-hand stock for one SKU in one warehouse, including "
     "reserved, damaged and in-transit quantities.", True),
    ("freight_quote_calculator",
     "Quote freight for a shipment between two facilities across every "
     "contracted carrier, lane and service level.", True),
    ("customs_tariff_classifier",
     "Classify a product under the harmonised tariff schedule and return duty "
     "rates, restrictions and required documents per destination.", True),
    ("report_status",
     "Report a short free-text status line back to the operator. Always "
     "available.", False),
)


def _tdef_parameters(name: str) -> dict:
    """A deliberately large JSON-Schema parameter object for one probe tool."""
    properties: dict = {}
    for i in range(12):
        properties[f"{name}_field_{i}"] = {
            "type": "string",
            "description": (
                f"Field {i} of the {name} request envelope. Supply the fully "
                f"qualified identifier as issued by the upstream system of "
                f"record, including its region prefix, its four-digit revision "
                f"suffix and any correlation token the caller was handed by a "
                f"previous call. The value is matched verbatim and is case "
                f"sensitive."),
        }
    properties[f"{name}_options"] = {
        "type": "object",
        "description": (
            "Optional execution settings controlling pagination, currency "
            "conversion, unit normalisation and whether historical revisions "
            "are included in the answer."),
        "properties": {
            "page_size": {"type": "integer", "description": "Rows per page, 1 to 500."},
            "currency": {"type": "string", "description": "ISO 4217 currency for monetary fields."},
            "include_history": {"type": "boolean", "description": "Include superseded revisions."},
        },
    }
    return {"type": "object", "properties": properties,
            "required": [f"{name}_field_0"]}


def _tdef_tools() -> list[dict]:
    """The canonical probe tool set: most deferrable, at least one never is."""
    return [{"name": name, "description": description,
             "parameters": _tdef_parameters(name), "deferrable": deferrable}
            for name, description, deferrable in _TDEF_TOOL_SPECS]


def _tdef_anthropic_base(endpoint: str) -> str:
    """Guess the Anthropic-Messages base URL that sits beside an OpenAI one.

    Every vendor probed so far hangs its Anthropic-compatible surface off
    `/anthropic` under the same host, one level above the OpenAI version
    segment: `https://api.deepseek.com` -> `.../anthropic`,
    `https://api.moonshot.ai/v1` -> `.../anthropic`,
    `https://api.z.ai/api/paas/v4` -> `https://api.z.ai/api/anthropic`.
    Override with --tdef-anthropic-base when a vendor disagrees.
    """
    parts = urllib.parse.urlsplit(endpoint)
    segments = [s for s in parts.path.split("/") if s]
    while segments:
        last = segments[-1]
        if re.fullmatch(r"v\d+(?:\.\d+)?", last) or last in {"paas", "coding"}:
            segments.pop()
            continue
        break
    segments.append("anthropic")
    return urllib.parse.urlunsplit(
        (parts.scheme, parts.netloc, "/" + "/".join(segments), "", ""))


def _tdef_post(url: str, payload: dict, headers: dict) -> tuple[int | None, object]:
    """One JSON POST that never raises; status is None on a transport error.

    The body is returned parsed when it is JSON and as text otherwise, because
    the error body is the evidence: whether a 400 names `defer_loading` is
    exactly what separates "rejected" from "the model disliked the prompt".
    """
    req = urllib.request.Request(
        url, method="POST", data=json.dumps(payload).encode(),
        headers=dict(headers, **{"Content-Type": "application/json"}))
    try:
        with urllib.request.urlopen(req, timeout=120) as resp:
            body = resp.read().decode("utf-8", "replace")
            status = resp.status
    except urllib.error.HTTPError as e:
        body = e.read().decode("utf-8", "replace")
        status = e.code
    except Exception as e:
        return None, f"transport error: {e}"
    try:
        return status, json.loads(body)
    except json.JSONDecodeError:
        return status, body[:2000]


def _tdef_surface_payload(surface: str, model: str, tools: list[dict],
                          prompt: str, mode: str,
                          max_tokens: int = _TDEF_MAX_TOKENS) -> dict:
    """Build one request for `surface` in mode bare / inline / deferred."""
    if mode == "bare":
        entries: list[dict] = []
    elif surface == "anthropic-messages":
        entries = [dict({"name": t["name"], "description": t["description"],
                         "input_schema": t["parameters"]},
                        **({"defer_loading": True}
                           if mode == "deferred" and t["deferrable"] else {}))
                   for t in tools]
        if mode == "deferred":
            entries.append(dict(_TDEF_SEARCH_TOOL_ANTHROPIC))
    elif surface == "openai-responses":
        entries = [dict({"type": "function", "name": t["name"],
                         "description": t["description"],
                         "parameters": t["parameters"]},
                        **({"defer_loading": True}
                           if mode == "deferred" and t["deferrable"] else {}))
                   for t in tools]
        if mode == "deferred":
            entries.append(dict(_TDEF_SEARCH_TOOL_OPENAI))
    else:
        entries = [dict({"type": "function",
                         "function": {"name": t["name"],
                                      "description": t["description"],
                                      "parameters": t["parameters"]}},
                        **({"defer_loading": True}
                           if mode == "deferred" and t["deferrable"] else {}))
                   for t in tools]
        if mode == "deferred":
            entries.append(dict(_TDEF_SEARCH_TOOL_OPENAI))

    if surface == "openai-responses":
        payload: dict = {"model": model, "input": prompt,
                         "max_output_tokens": max_tokens, "stream": False}
    else:
        payload = {"model": model, "max_tokens": max_tokens,
                   "messages": [{"role": "user", "content": prompt}]}
    if entries:
        payload["tools"] = entries
    if _PROVIDER:
        # Same pin as every other round: an aggregator that reroutes mid-probe
        # would compare token counts across two different inference stacks.
        payload["provider"] = _provider_routing(_PROVIDER)
    return payload


def _tdef_input_tokens(body: object) -> int | None:
    """Input/prompt tokens from any of the three response shapes."""
    if not isinstance(body, dict):
        return None
    usage = body.get("usage")
    if not isinstance(usage, dict):
        return None
    for key in ("prompt_tokens", "input_tokens"):
        value = usage.get(key)
        if isinstance(value, int):
            return value
    return None


def _tdef_error_text(body: object) -> str:
    """Flatten an error body to a single searchable string."""
    if isinstance(body, str):
        return body
    if isinstance(body, dict):
        error = body.get("error")
        if isinstance(error, dict):
            return " ".join(str(v) for v in error.values())
        if isinstance(error, str):
            return error
        for key in ("message", "detail", "msg"):
            if isinstance(body.get(key), str):
                return body[key]
        return json.dumps(body)[:2000]
    return ""


def _tdef_routing_failure(text: str) -> bool:
    """True when an aggregator could not route the request at all.

    OpenRouter answers a pin it cannot satisfy with `404 No endpoints found
    for <model>` plus a `routing_funnel` — the same status an unimplemented
    path returns. Reading that as "this surface is not served here" would
    quietly turn a broken pin into a capability finding, so it is checked
    first and reported as an error.
    """
    lowered = text.lower()
    return any(marker in lowered for marker in
               ("no endpoints found", "routing_funnel", "no allowed providers",
                "no provider", "provider not found"))


def _tdef_surface_absent(status: int | None, text: str) -> bool:
    """True when the failure says the surface itself is not there.

    A 404/405 on the path, or a body naming an unknown path/method, means the
    vendor never implemented this protocol here -- which is `n/a`, not a
    verdict about deferred loading.
    """
    if status in (404, 405, 501):
        return True
    lowered = text.lower()
    return any(marker in lowered for marker in
               ("not found", "no such endpoint", "unknown path",
                "method not allowed", "cannot post"))


def _tdef_rejected_component(text: str, search_index: int | None = None,
                             deferred_indices: tuple[int, ...] = ()) -> str | None:
    """Which half of the feature an error body rejects, if it identifies one.

    The two halves fail separately: an endpoint can whitelist tool types (and
    so reject the search tool) while still swallowing `defer_loading` on the
    function tools, or the other way round. Either one alone makes the feature
    unusable, but the report should say which.

    Some endpoints name neither and merely point at the offending array slot
    ("tools[4].type:type is illegal"), so the indices the probe used are the
    only way to read that answer -- hence `search_index` / `deferred_indices`.
    """
    lowered = text.lower()
    field = "defer_loading" in lowered or "defer loading" in lowered
    search = any(marker in lowered for marker in
                 ("tool_search", "tool search", "tool_search_tool"))
    if not search and search_index is not None:
        search = bool(re.search(rf"tools[\[.]{search_index}\b", lowered))
    if not search:
        # The search tool is the one entry in the request with no `function`
        # member, so every inference stack complains about that same missing
        # field in its own dialect: serde's "missing field `function`",
        # pydantic's "Field required", Fireworks' "Input should be...".
        search = bool(
            re.search(r"missing (the )?(required )?field `?function`?", lowered)
            or re.search(r"`?function`?\s+(field\s+)?is required", lowered)
            or ("field required" in lowered and "function.function" in lowered))
    if not field:
        field = any(re.search(rf"tools[\[.]{i}\b", lowered) for i in deferred_indices)
    if field and search:
        return "both"
    if field:
        return "defer_loading"
    if search:
        return "tool_search"
    return None


def _tdef_search_used(body: object) -> bool:
    """True when the response shows the model going through a tool search.

    Every shape spells the step differently -- Anthropic's
    `tool_search_tool_result` content block, a Responses output item, a
    chat-completions tool call -- and vendors namespace their own variants
    (OpenRouter emits `"type": "openrouter:tool_search"`). So rather than
    matching a fixed list, walk the response and accept any `type`/`name`
    that mentions a tool search at all.
    """
    def walk(node: object) -> bool:
        if isinstance(node, dict):
            for key in ("type", "name"):
                value = node.get(key)
                if isinstance(value, str) and "tool_search" in value.lower():
                    return True
            return any(walk(v) for v in node.values())
        if isinstance(node, list):
            return any(walk(v) for v in node)
        return False

    if not isinstance(body, dict):
        return False
    # Only what the model produced counts: several endpoints echo the request's
    # `tools` array back in the response, search tool included.
    return any(walk(body.get(channel))
               for channel in ("output", "content", "choices"))


def _tdef_called_tools(body: object) -> list[str]:
    """Names of the non-search tools the model called, across all shapes."""
    names: list[str] = []
    if not isinstance(body, dict):
        return names
    for choice in body.get("choices") or []:
        message = (choice or {}).get("message") or {}
        for call in message.get("tool_calls") or []:
            name = ((call or {}).get("function") or {}).get("name")
            if name:
                names.append(name)
    for item in body.get("output") or []:
        if isinstance(item, dict) and item.get("type") == "function_call":
            if item.get("name"):
                names.append(item["name"])
    content = body.get("content")
    if isinstance(content, list):
        for block in content:
            if isinstance(block, dict) and block.get("type") == "tool_use":
                if block.get("name"):
                    names.append(block["name"])
    search_names = {"tool_search_tool_bm25", "tool_search_tool_regex", "tool_search"}
    return [n for n in names if n not in search_names]


def _tdef_classify_tokens(bare: int | None, inline: int | None,
                          deferred: int | None) -> tuple[str | None, str]:
    """Decide from input-token counts whether the schemas left the prompt."""
    if inline is None or deferred is None:
        return None, ("the endpoint reported no input-token usage, so the "
                      "prompt size could not be compared")
    drop = inline - deferred
    schema_cost = (inline - bare) if bare is not None else None
    if schema_cost is not None and schema_cost > 0:
        threshold = max(0.5 * schema_cost, 100)
        detail = (f"tool schemas cost {schema_cost} input tokens inline; "
                  f"deferring them changed the prompt by {drop}")
    else:
        threshold = 300
        detail = (f"deferring the schemas changed the prompt by {drop} input "
                  f"tokens (no tool-free baseline available)")
    return ("native" if drop >= threshold else "accepted-but-ignored"), detail


def _tdef_probe_surface(surface: str, url: str, model: str,
                        headers: dict) -> dict:
    """Run the three TDEF rounds against one protocol surface."""
    tools = _tdef_tools()
    result: dict = {"surface": surface, "url": url, "model": model}

    status, body = _tdef_post(
        url, _tdef_surface_payload(surface, model, tools, _TDEF_BASE_PROMPT,
                                   "deferred"), headers)
    text = _tdef_error_text(body)
    result["deferred_request_status"] = status
    served_by = _observed_provider(body)
    if served_by:
        result["served_by"] = served_by
        _record_observed_provider(body)
    if status is None:
        result["verdict"] = "n/a"
        result["evidence"] = text
        return result
    if status >= 400:
        rejected = _tdef_rejected_component(
            text, search_index=len(tools),
            deferred_indices=tuple(i for i, t in enumerate(tools) if t["deferrable"]))
        if _tdef_routing_failure(text):
            result["verdict"] = "error"
            result["evidence"] = (f"HTTP {status}: the request was never routed to a "
                                  f"model — {text[:300]}")
        elif _tdef_surface_absent(status, text):
            result["verdict"] = "n/a"
            result["evidence"] = f"HTTP {status}: this protocol surface is not served here"
        elif rejected:
            result["verdict"] = "rejected"
            result["rejected_component"] = rejected
            result["evidence"] = f"HTTP {status}: {text[:400]}"
        else:
            result["verdict"] = "error"
            result["evidence"] = f"HTTP {status}: {text[:400]}"
        return result

    deferred_tokens = _tdef_input_tokens(body)
    inline_status, inline_body = _tdef_post(
        url, _tdef_surface_payload(surface, model, tools, _TDEF_BASE_PROMPT,
                                   "inline"), headers)
    bare_status, bare_body = _tdef_post(
        url, _tdef_surface_payload(surface, model, tools, _TDEF_BASE_PROMPT,
                                   "bare"), headers)
    inline_tokens = _tdef_input_tokens(inline_body)
    bare_tokens = _tdef_input_tokens(bare_body)
    result["input_tokens"] = {"bare": bare_tokens, "inline": inline_tokens,
                              "deferred": deferred_tokens}
    if bare_status != 200:
        result["baseline_request_status"] = bare_status
    control_failed = inline_status != 200
    if control_failed:
        # Without the control request there is nothing to compare against, and
        # a missing token count must not be read as "the schemas stayed in".
        result["control_request_status"] = inline_status
        verdict, detail = None, (
            f"the control request (schemas inline) failed with HTTP "
            f"{inline_status}, so no comparison was possible: "
            f"{_tdef_error_text(inline_body)[:200]}")
    else:
        verdict, detail = _tdef_classify_tokens(bare_tokens, inline_tokens,
                                                deferred_tokens)
    result["token_accounting"] = detail

    _, reach_body = _tdef_post(
        url, _tdef_surface_payload(surface, model, tools, _TDEF_REACH_PROMPT,
                                   "deferred", _TDEF_REACH_MAX_TOKENS), headers)
    searched = _tdef_search_used(reach_body)
    called = _tdef_called_tools(reach_body)
    deferred_names = {t["name"] for t in tools if t["deferrable"]}
    called_deferred = [n for n in called if n in deferred_names]
    result["reachability"] = (
        "search_then_call" if searched
        else "direct_call" if called_deferred
        else "no_call")
    result["called_tools"] = called

    if verdict is None and searched:
        # A genuine search round-trip stands on its own.
        verdict = "native"
        result["token_accounting"] = detail + "; verdict taken from the reachability round instead"
    elif verdict is None and control_failed:
        # Nothing was measured and nothing was observed: report that, rather
        # than defaulting to a verdict the probe did not earn.
        verdict = "error"
    elif verdict is None:
        # Usage was simply never reported. A direct call to a deferred tool is
        # still proof the schema was in the prompt; anything else is a miss.
        verdict = "accepted-but-ignored" if called_deferred else "error"
        result["token_accounting"] = detail + "; verdict taken from the reachability round instead"
    elif searched and verdict != "native":
        # A real search round-trip outranks the counters, which a
        # compatibility layer may synthesise rather than measure.
        verdict = "native"
        result["token_accounting"] = detail + "; overridden by an observed tool-search round-trip"
    result["verdict"] = verdict
    return result


_TDEF_VERDICT_RANK = {"native": 3, "accepted-but-ignored": 2, "rejected": 1,
                      "error": 0, "n/a": 0}


def _tdef_overall(surfaces: dict) -> str:
    """Best verdict across surfaces -- one honouring surface is enough."""
    verdicts = [s.get("verdict", "n/a") for s in surfaces.values()]
    if not verdicts:
        return "n/a"
    return max(verdicts, key=lambda v: _TDEF_VERDICT_RANK.get(v, 0))


def _tdef_cell(tdef: dict) -> str:
    """The `TDEF` capabilities-table cell: overall verdict plus per surface."""
    surfaces = tdef.get("surfaces") or {}
    if not surfaces:
        return "no surface reachable"
    per = ", ".join(f"{name}: {data.get('verdict', 'n/a')}"
                    for name, data in surfaces.items())
    return f"**{tdef.get('verdict', 'n/a')}** ({per})"


def _tdef_summary_line(tdef: dict) -> str:
    """One-line verdict for the console and the report."""
    verdict = tdef.get("verdict")
    if verdict == "native":
        return ("at least one surface genuinely defers tool schemas -- they leave "
                "the prompt and come back through a tool search")
    if verdict == "accepted-but-ignored":
        return ("no surface defers anything: `defer_loading` is either rejected or "
                "accepted and silently dropped, leaving the schemas in the prompt "
                "and on the bill")
    if verdict == "rejected":
        return ("every reachable surface rejects deferred tool loading outright, "
                "which is at least honest")
    return "no surface exposed a deferred-loading answer"


def tdef_test_round(api_key: str, args: argparse.Namespace) -> dict:
    """Probe every protocol surface for genuine deferred tool loading."""
    section("TDEF test -- deferred tool loading")
    print("\nSends the same request with tool schemas inline and deferred, and")
    print("compares the reported input tokens: a compatibility layer that drops")
    print("`defer_loading` still answers 200 with the schemas in the prompt.\n")

    if not ENDPOINT.startswith(("http://", "https://")):
        print("[tdef] skipped: a local --script wrapper is not an HTTP endpoint")
        return {"error": "TDEF does not apply: the probe target is a local script "
                         "wrapper, not an HTTP endpoint"}

    bearer = {"Authorization": f"Bearer {api_key}"} if api_key.strip() else {}
    anthropic_headers = dict(bearer, **{"x-api-key": api_key,
                                        "anthropic-version": _TDEF_ANTHROPIC_VERSION})
    responses_base = (args.tdef_responses_base or ENDPOINT).rstrip("/")
    anthropic_base = (args.tdef_anthropic_base
                      or _tdef_anthropic_base(ENDPOINT)).rstrip("/")
    targets = (
        ("openai-completions", f"{ENDPOINT.rstrip('/')}/chat/completions", MODEL, bearer),
        ("openai-responses", f"{responses_base}/responses", MODEL, bearer),
        ("anthropic-messages", f"{anthropic_base}/v1/messages",
         args.tdef_anthropic_model or MODEL, anthropic_headers),
    )

    surfaces: dict = {}
    for surface, url, model, headers in targets:
        print(f"[tdef] {surface}: POST {url}")
        surfaces[surface] = _tdef_probe_surface(surface, url, model, headers)
        data = surfaces[surface]
        detail = data.get("token_accounting") or data.get("evidence") or ""
        print(f"[tdef] {surface}: {data['verdict']}"
              + (f" -- {detail[:160]}" if detail else ""))

    result = {"surfaces": surfaces, "verdict": _tdef_overall(surfaces)}
    print("\nTDEF summary: " + _tdef_summary_line(result))
    return result


# -- markdown report -----------------------------------------------------------

def _md_escape(text: str) -> str:
    return text.replace("|", "\\|").replace("\n", " ")


_NO_DATA_NOTE = "no data, please rerun the probing"

_CTX_SOURCE_LABELS = {
    "api_metadata":              "the endpoint's /models metadata",
    "api_error_message":         "the endpoint's rejection message",
    "accepted_oversized_prompt": "an accepted oversized prompt",
    "empirical_bisection":       "empirical bisection",
    "explicit":                  "the command line",
}


_MULTI_PROVIDER_HOSTS = {"openrouter.ai"}


def _md_provider_line(output: dict) -> str:
    """The report's provider bullet, for aggregators that route to several.

    Names both the pinned provider and the one that answered: a pin that did
    not take hold makes every number in the report a measurement of something
    else, and that has to be visible rather than inferred.
    """
    pinned = output.get("provider")
    observed = output.get("observed_provider")
    if not pinned and not observed:
        if _safe_server(output.get("endpoint") or "") in _MULTI_PROVIDER_HOSTS:
            return ("- **Provider:** unknown — this aggregator routes one model id to "
                    "many upstreams running different software, and this run was not "
                    "pinned, so each result below describes whichever one answered. "
                    "Rerun with `--provider` for a reproducible report.")
        return ""
    if pinned and observed and observed.lower() not in pinned.lower():
        return (f"- **Provider:** pinned to `{pinned}`, but `{observed}` answered "
                f"— the pin did not hold, treat these results with suspicion")
    if pinned:
        return f"- **Provider:** `{pinned}` (pinned, fallbacks disabled)"
    return f"- **Provider:** `{observed}` (not pinned — a rerun may land elsewhere)"


def _md_context_window(output: dict) -> str:
    """One-line context-window summary for the report's header bullets."""
    ctx = output.get("context_window")
    if _capability_absent(output, "context_window") or not ctx:
        return f"*({_NO_DATA_NOTE})*"
    tokens = ctx.get("tokens")
    if tokens is None:
        return f"unknown — *{_md_escape(ctx.get('error') or 'endpoint named no limit')}*"
    label = _CTX_SOURCE_LABELS.get(ctx.get("source") or "", ctx.get("source") or "unknown")
    summary = f"{tokens:,} tokens ({ctx.get('kind') or 'context_window'}, from {label})"
    if ctx.get("lower_bound"):
        summary += " — lower bound: the oversized probe was accepted"
    return summary


def _capability_absent(output: dict, key: str) -> bool:
    """True when the report JSON carries no data at all for a capability.

    An *absent* key means the report was written by a probe version that did not
    have this capability yet, so there is genuinely nothing to render for it.
    That is different from a key that is present but ``None``: there the probe
    ran and skipped the test on purpose (``--no-*``), which the report can state
    exactly rather than claiming there is no data.
    """
    return key not in output


def _append_no_data_section(lines: list[str], heading: str) -> None:
    """Emit a capability section whose data is missing from the report JSON."""
    lines.append(heading)
    lines.append("")
    lines.append(f"*({_NO_DATA_NOTE})*")
    lines.append("")


def _capabilities_md_path(out_path: str) -> str:
    """Derive the markdown report path from the JSON output path (same stem, .md)."""
    return str(Path(out_path).with_suffix(".md"))


def render_markdown_report(output: dict) -> str:
    """Render the probe's JSON output dict as a human-readable Markdown report."""
    lines: list[str] = []
    model    = output.get("model", "?")
    endpoint = output.get("endpoint", "?")
    status   = output.get("status", "?")
    commit   = output.get("llmprobe_commit")
    cors_test = _cors_merge(output.get("cors_test") or {})

    lines.append(f"# Model capability probe: {model}")
    lines.append("")
    if commit:
        lines.append(
            "Generated by [llmprobe](https://github.com/superleanai/llmprobe) "
            f"at commit [`{commit}`](https://github.com/superleanai/llmprobe/commit/{commit})."
        )
    else:
        lines.append("Generated by [llmprobe](https://github.com/superleanai/llmprobe).")
    lines.append("")
    lines.append(f"- **Endpoint:** {endpoint}")
    server = output.get("server")
    if server and server != _safe_server(endpoint):
        # Worth printing only when the endpoint does not already name it, i.e.
        # when a wrapper script fronts the real server.
        lines.append(f"- **Server:** {server}")
    provider_line = _md_provider_line(output)
    if provider_line:
        lines.append(provider_line)
    lines.append(f"- **API type:** {output.get('api_type', 'OpenAI Completions')}")
    lines.append(f"- **Context window:** {_md_context_window(output)}")
    if _capability_absent(output, "cors_test"):
        lines.append(f"- **CORS:** *({_NO_DATA_NOTE})*")
    elif cors_test and "error" not in cors_test:
        lines.append(f"- **CORS:** {_cors_origin_cell(cors_test)}")
    elif cors_test and cors_test.get("error"):
        lines.append(f"- **CORS:** *({_md_escape(cors_test['error'])})*")
    else:
        lines.append("- **CORS:** *(not run — rerun without `--no-cors-test`)*")
    if status != "ok":
        lines.append(f"- **Status:** {status}")
    if output.get("error"):
        lines.append(f"- **Error:** {output['error']}")
    lines.append("")

    fmt        = output.get("format_detection") or {}
    behaviour  = output.get("behaviour") or {}
    quote_test = output.get("quote_test")
    tok_test   = output.get("token_efficiency_test")
    askq_test  = output.get("askq_test")
    gram_knowledge_test = output.get("gram_knowledge_test")
    gram_transport_test = output.get("gram_transport_test")
    rjson_test = output.get("rjson_test")
    stream_test = output.get("stream_test")
    reasoning_test = output.get("reasoning_test")
    agentknit_test = output.get("agentknit_test")
    cache_ttl_test = output.get("cache_ttl_test")

    lines.append("## Capabilities summary")
    lines.append("")
    lines.append("See `CAPABILITIES.md` for what each codename measures, its unit, and its range.")
    lines.append("")
    lines.append("| Codename | Value |")
    lines.append("|---|---|")
    if _capability_absent(output, "behaviour"):
        lines.append(f"| `TCALL` | *({_NO_DATA_NOTE})* |")
    elif behaviour:
        structured = behaviour.get("structured_tool_calls", 0)
        total_b    = structured + behaviour.get("inline_json_in_content", 0) + behaviour.get("no_call_detected", 0)
        lines.append(f"| `TCALL` | {structured}/{total_b} |")
    else:
        lines.append("| `TCALL` | *(not run)* |")
    if _capability_absent(output, "quote_test"):
        lines.append(f"| `QUOTE` | *({_NO_DATA_NOTE})* |")
    elif quote_test and "error" not in quote_test:
        lines.append(f"| `QUOTE` | {quote_test.get('quote_test_passed', 0)}/{quote_test.get('quote_test_total', 0)} |")
    else:
        lines.append("| `QUOTE` | *(not run — rerun without `--no-quote-test`)* |")
    if _capability_absent(output, "token_efficiency_test"):
        lines.append(f"| `GREP` | *({_NO_DATA_NOTE})* |")
    elif tok_test and "error" not in tok_test:
        lines.append(f"| `GREP` | {tok_test.get('token_efficiency_passed', 0)}/{tok_test.get('token_efficiency_total', 0)} |")
    else:
        lines.append("| `GREP` | *(not run — rerun without `--no-efficiency-test`)* |")
    if _capability_absent(output, "askq_test"):
        lines.append(f"| `ASKQ` | *({_NO_DATA_NOTE})* |")
    elif askq_test and "error" not in askq_test:
        askq_passed = askq_test.get("askq_passed", 0)
        askq_total  = askq_test.get("askq_total", 0)
        lines.append(f"| `ASKQ` | {askq_passed}/{askq_total} ({_likert_label(askq_passed, askq_total)}) |")
    else:
        lines.append("| `ASKQ` | *(not run — rerun without `--no-askq-test`)* |")
    if _capability_absent(output, "gram_knowledge_test"):
        lines.append(f"| `APPLY_PATCH` | *({_NO_DATA_NOTE})* |")
    elif gram_knowledge_test and "error" not in gram_knowledge_test:
        lines.append(f"| `APPLY_PATCH` | {gram_knowledge_test.get('gram_knowledge_passed', 0)}/{gram_knowledge_test.get('gram_knowledge_total', 0)} |")
    else:
        lines.append("| `APPLY_PATCH` | *(not run — rerun without `--no-gram-knowledge-test`)* |")
    if _capability_absent(output, "gram_transport_test"):
        lines.append(f"| `GRAMT` | *({_NO_DATA_NOTE})* |")
    elif gram_transport_test and "error" not in gram_transport_test:
        lines.append(f"| `GRAMT` | {gram_transport_test.get('gram_transport_passed', 0)}/{gram_transport_test.get('gram_transport_total', 0)} |")
    else:
        lines.append("| `GRAMT` | *(not run — rerun without `--no-gram-transport-test`)* |")
    if _capability_absent(output, "rjson_test"):
        lines.append(f"| `RJSON` | *({_NO_DATA_NOTE})* |")
    elif rjson_test and "error" not in rjson_test:
        lines.append(f"| `RJSON` | {rjson_test.get('rjson_passed', 0)}/{rjson_test.get('rjson_total', 0)} |")
    else:
        lines.append("| `RJSON` | *(not run — rerun without `--no-rjson-test`)* |")
    if _capability_absent(output, "stream_test"):
        lines.append(f"| `STRM` | *({_NO_DATA_NOTE})* |")
    elif stream_test and "error" not in stream_test:
        lines.append(f"| `STRM` | {stream_test.get('stream_passed', 0)}/{stream_test.get('stream_total', 0)} |")
    else:
        lines.append("| `STRM` | *(not run — rerun without `--no-stream-test`)* |")
    if _capability_absent(output, "reasoning_test"):
        lines.append(f"| `REASN` | *({_NO_DATA_NOTE})* |")
    elif reasoning_test and "error" not in reasoning_test:
        lines.append(f"| `REASN` | {reasoning_test.get('reason_passed', 0)}/{reasoning_test.get('reason_total', 0)} |")
    else:
        lines.append("| `REASN` | *(not run — rerun without `--no-reasoning-test`)* |")
    if _capability_absent(output, "agentknit_test"):
        lines.append(f"| `AKDEF` | *({_NO_DATA_NOTE})* |")
    elif agentknit_test and "error" not in agentknit_test:
        lines.append(f"| `AKDEF` | {agentknit_test.get('akdef_passed', 0)}/{agentknit_test.get('akdef_total', 0)} |")
    elif agentknit_test and agentknit_test.get("error"):
        lines.append(f"| `AKDEF` | *(error: {_md_escape(agentknit_test['error'])})* |")
    else:
        lines.append("| `AKDEF` | *(not run — rerun without `--no-agentknit-test`)* |")
    if _capability_absent(output, "cache_ttl_test"):
        lines.append(f"| `CACH` | *({_NO_DATA_NOTE})* |")
    elif cache_ttl_test and (cache_ttl_test.get("samples") or "error" not in cache_ttl_test):
        ttl_min = cache_ttl_test.get("ttl_min_minutes")
        ttl_max = cache_ttl_test.get("ttl_max_minutes")
        if ttl_min is not None and ttl_max is not None:
            lines.append(f"| `CACH` | TTL between {ttl_min} and {ttl_max} min |")
        elif ttl_min is not None:
            lines.append(f"| `CACH` | TTL > {ttl_min} min |")
        elif ttl_max is not None:
            lines.append(f"| `CACH` | TTL < {ttl_max} min |")
        else:
            lines.append("| `CACH` | TTL not measurable (no cache reuse) |")
    elif cache_ttl_test and cache_ttl_test.get("error"):
        lines.append(f"| `CACH` | *(error: {_md_escape(cache_ttl_test['error'])})* |")
    else:
        lines.append("| `CACH` | *(not run — rerun with `--cache-ttl-test`)* |")
    tsel_test = output.get("tsel_test")
    if _capability_absent(output, "tsel_test"):
        lines.append(f"| `TSEL` | *({_NO_DATA_NOTE})* |")
    elif tsel_test and "error" not in tsel_test:
        tsel_passed = tsel_test.get("tsel_passed", 0)
        tsel_total = tsel_test.get("tsel_total", 0)
        lines.append(f"| `TSEL` | {tsel_passed}/{tsel_total} |")
    else:
        lines.append("| `TSEL` | *(not run)* |")
    if _capability_absent(output, "cors_test"):
        lines.append(f"| `CORS` | *({_NO_DATA_NOTE})* |")
    elif cors_test and "error" not in cors_test:
        lines.append(f"| `CORS` | {_cors_origin_cell(cors_test)} |")
    elif cors_test and cors_test.get("error"):
        lines.append(f"| `CORS` | *(error: {_md_escape(cors_test['error'])})* |")
    else:
        lines.append("| `CORS` | *(not run — rerun without `--no-cors-test`)* |")
    tdef_test = output.get("tdef_test")
    if _capability_absent(output, "tdef_test"):
        lines.append(f"| `TDEF` | *({_NO_DATA_NOTE})* |")
    elif tdef_test and "error" not in tdef_test:
        lines.append(f"| `TDEF` | {_md_escape(_tdef_cell(tdef_test))} |")
    elif tdef_test and tdef_test.get("error"):
        lines.append(f"| `TDEF` | *({_md_escape(tdef_test['error'])})* |")
    else:
        lines.append("| `TDEF` | *(not run — rerun without `--no-tdef-test`)* |")
    lines.append("")

    ctx = output.get("context_window")
    lines.append("## Context window")
    lines.append("")
    if _capability_absent(output, "context_window"):
        lines.append(f"*({_NO_DATA_NOTE})*")
        lines.append("")
    elif not ctx or ctx.get("tokens") is None:
        lines.append("No context-window limit was recovered for this endpoint.")
        lines.append("")
        detail = (ctx or {}).get("error") or "the endpoint neither published limits nor named one in its errors"
        lines.append(f"Reason: {_md_escape(detail)}")
        lines.append("")
    else:
        tokens = ctx["tokens"]
        lines.append(f"**{tokens:,} tokens** — recovered from "
                     f"{_CTX_SOURCE_LABELS.get(ctx.get('source') or '', ctx.get('source') or 'an unknown source')}.")
        lines.append("")
        lines.append("| Field | Value |")
        lines.append("|---|---|")
        lines.append(f"| Tokens | {tokens:,} |")
        lines.append(f"| Kind | {ctx.get('kind') or 'context_window'} |")
        lines.append(f"| Source | `{ctx.get('source') or 'unknown'}` |")
        if ctx.get("lower_bound"):
            lines.append("| Confidence | lower bound — the oversized prompt was accepted, "
                         "so the real limit is at least this |")
        lines.append(f"| Evidence | {_md_escape(ctx.get('evidence') or '')} |")
        also = ctx.get("metadata") or {}
        if also:
            shown = ", ".join(f"{k}={v:,}" for k, v in also.items())
            lines.append(f"| Also advertised by the endpoint | {_md_escape(shown)} |")
        lines.append("")

    if _capability_absent(output, "cors_test"):
        _append_no_data_section(lines, "## CORS preflight test (`CORS`)")
    elif cors_test and "error" not in cors_test:
        lines.append("## CORS preflight test (`CORS`)")
        lines.append("")
        lines.append(_cors_summary_line(cors_test) + ".")
        lines.append("")
        lines.append("A browser must clear a CORS preflight before any cross-origin")
        lines.append("POST that carries `Authorization` and `Content-Type: application/json`,")
        lines.append("so the probe replays that exact handshake (OPTIONS with `Origin` and")
        lines.append("`Access-Control-Request-*`), then a plain Origin-tagged GET, and")
        lines.append("records the `access-control-*` answer verbatim. No credentials are")
        lines.append("sent for any of this.")
        lines.append("")
        lines.append("| Field | Value |")
        lines.append("|---|---|")
        lines.append(f"| Allow-origin | {_cors_origin_cell(cors_test)} |")
        lines.append(f"| Preflight | {'PASS' if cors_test.get('preflight_passed') else 'FAIL'} — "
                     f"allow-origin present, POST allowed, authorization + content-type headers allowed |")
        lines.append(f"| Actual response | {'PASS' if cors_test.get('actual_response_passed') else 'FAIL'} — "
                     f"CORS headers also present on a plain Origin-tagged GET |")
        lines.append(f"| Allow-credentials | "
                     f"{'yes' if cors_test.get('allow_credentials') else '—'} |")
        lines.append(f"| Evidence | {_md_escape(cors_test.get('evidence') or '')} |")
        lines.append("")
        legs = (("preflight (OPTIONS /chat/completions)", cors_test.get("preflight")),
                ("actual response (GET /models)", cors_test.get("actual_response")))
        lines.append("| Probe | Status | Allow-origin | Allow-methods | Allow-headers | Result |")
        lines.append("|---|---|---|---|---|---|")
        for label, leg in legs:
            if not isinstance(leg, dict):
                continue
            methods = leg.get("allow_methods")
            headers_ = leg.get("allow_headers")
            origin_ = leg.get("allow_origin")

            def truncate(value: str) -> str:
                return value if len(value) <= 80 else value[:77] + "..."

            lines.append(f"| {label} | {leg.get('status') or '*(transport error)*'} | "
                         f"{f'`{_md_escape(truncate(origin_))}`' if origin_ else '—'} | "
                         f"{f'`{_md_escape(truncate(methods))}`' if methods else '—'} | "
                         f"{f'`{_md_escape(truncate(headers_))}`' if headers_ else '—'} | "
                         f"{'PASS' if leg.get('pass') else 'FAIL'} |")
        lines.append("")
    elif cors_test and cors_test.get("error"):
        lines.append("## CORS preflight test (`CORS`)")
        lines.append("")
        lines.append(f"Error: {cors_test['error']}")
        lines.append("")

    if _capability_absent(output, "tdef_test"):
        _append_no_data_section(lines, "## Deferred tool loading (`TDEF`)")
    elif tdef_test and "error" not in tdef_test:
        lines.append("## Deferred tool loading (`TDEF`)")
        lines.append("")
        lines.append(f"**{tdef_test.get('verdict', 'n/a')}** — "
                     + _tdef_summary_line(tdef_test) + ".")
        lines.append("")
        lines.append("A tool marked `defer_loading` is supposed to keep its parameter")
        lines.append("schema out of the prompt until the model asks for it through a")
        lines.append("tool-search tool. Compatibility layers routinely accept the field")
        lines.append("and drop it, so HTTP 200 proves nothing: the verdict below comes")
        lines.append("from sending the same request with the schemas inline and deferred")
        lines.append("and comparing the endpoint's own input-token count, then asking for")
        lines.append("something only a deferred tool can answer.")
        lines.append("")
        lines.append("| Surface | Verdict | Input tokens (none / inline / deferred) | Reachability | Evidence |")
        lines.append("|---|---|---|---|---|")
        for name, data in (tdef_test.get("surfaces") or {}).items():
            tokens = data.get("input_tokens") or {}
            counts = " / ".join(
                str(tokens.get(k)) if tokens.get(k) is not None else "—"
                for k in ("bare", "inline", "deferred")) if tokens else "—"
            evidence = data.get("token_accounting") or data.get("evidence") or ""
            # Provider error bodies arrive multiply JSON-escaped and can run to
            # hundreds of unreadable characters; the JSON report keeps them whole.
            if len(evidence) > 200:
                evidence = evidence[:197] + "..."
            if data.get("rejected_component"):
                evidence = f"rejects `{data['rejected_component']}` — {evidence}"
            if data.get("served_by"):
                # Surfaces of one aggregator can route to different upstreams.
                evidence = f"served by `{data['served_by']}`; {evidence}"
            lines.append(f"| `{name}` | {data.get('verdict', 'n/a')} | {counts} | "
                         f"{data.get('reachability') or '—'} | "
                         f"{_md_escape(evidence) or '—'} |")
        lines.append("")
        lines.append("`native` means the schemas genuinely left the prompt; "
                     "`accepted-but-ignored` means the field was swallowed and the "
                     "schemas were billed anyway; `rejected` means the endpoint said so; "
                     "`n/a` means the surface is not served here.")
        lines.append("")
    elif tdef_test and tdef_test.get("error"):
        lines.append("## Deferred tool loading (`TDEF`)")
        lines.append("")
        lines.append(f"Error: {tdef_test['error']}")
        lines.append("")

    lines.append("## Format detection & call delivery (`TCALL`)")
    lines.append("")
    if _capability_absent(output, "format_detection"):
        lines.append(f"*({_NO_DATA_NOTE})*")
        lines.append("")
    elif fmt.get("error"):
        lines.append(f"Error: {fmt['error']}")
        lines.append("")
    else:
        lines.append(f"- Round-0 probe (single call): detected format `{fmt.get('detected_format', '?')}`, "
                     f"structured tool_calls used: {fmt.get('has_structured_tool_calls')}")
        xml_test = fmt.get("xml_tool_call_test")
        if xml_test:
            if xml_test.get("error"):
                lines.append(f"- Explicit XML tool-call task: ERROR — {_md_escape(xml_test['error'])}")
            else:
                outcome = "PASS" if xml_test.get("pass") else "FAIL"
                lines.append(
                    f"- Explicit XML tool-call task (`<tool_call>` read_file envelope): {outcome}; "
                    f"detected format `{xml_test.get('detected_format', '?')}`, "
                    f"structured tool_calls used: {xml_test.get('structured_tool_calls')}"
                )
        else:
            lines.append("- Explicit XML tool-call task (`<tool_call>` read_file envelope): "
                         f"*({_NO_DATA_NOTE})*")
        lines.append("")
    if _capability_absent(output, "behaviour"):
        lines.append(f"*({_NO_DATA_NOTE})*")
        lines.append("")
    elif behaviour:
        structured = behaviour.get("structured_tool_calls", 0)
        inline     = behaviour.get("inline_json_in_content", 0)
        missing    = behaviour.get("no_call_detected", 0)
        total      = structured + inline + missing
        lines.append(f"- Full probe ({total} tasks): call delivery mode `{behaviour.get('call_delivery_mode', '?')}`")
        lines.append(f"  - Structured tool_calls: {structured}/{total} tasks")
        lines.append(f"  - Inline JSON in content (model ignored the tools API and put "
                     f"the call as JSON text in the message body instead): {inline}/{total} tasks")
        lines.append(f"  - No call detected (neither a structured tool_call nor parseable inline JSON): "
                     f"{missing}/{total} tasks")
        if behaviour.get("note"):
            lines.append(f"- Note: {behaviour['note']}")
        lines.append("")

    elicited = output.get("elicited_names") or {}
    if elicited:
        lines.append("## Elicited tool names")
        lines.append("")
        lines.append("Round 1 asks the model, in free text with no tool schema attached, what "
                     "function/arguments it would use for each task. The prompt never names a "
                     "tool — the model must invent the name itself.")
        lines.append("")
        lines.append("| Operation | Elicitation prompt | Model function name |")
        lines.append("|---|---|---|")
        for op, fn in elicited.items():
            prompt = _md_escape(ELICIT_TASKS.get(op, ""))
            lines.append(f"| {op} | {prompt} | {f'`{fn}`' if fn else '*(none)*'} |")
        lines.append("")

    tools = output.get("inferred_tool_schema") or []
    if tools:
        lines.append("## Inferred tool schema")
        lines.append("")
        for tool in tools:
            fn = tool.get("function") or tool
            name   = fn.get("name", "?")
            desc   = fn.get("description", "")
            params = fn.get("parameters") or {}
            props  = params.get("properties") or {}
            required = set(params.get("required") or [])
            lines.append(f"### `{name}`")
            if desc:
                lines.append("")
                lines.append(desc)
            lines.append("")
            if props:
                lines.append("| Parameter | Type | Required |")
                lines.append("|---|---|---|")
                for pname, pinfo in props.items():
                    ptype = pinfo.get("type", "?") if isinstance(pinfo, dict) else "?"
                    lines.append(f"| {pname} | {ptype} | {'yes' if pname in required else 'no'} |")
            lines.append("")

    if _capability_absent(output, "tsel_test"):
        _append_no_data_section(lines, "## Tool-selection test (`TSEL`)")
    elif tsel_test and "error" not in tsel_test:
        results = tsel_test.get("tsel_results") or {}
        lines.append("## Tool-selection test (`TSEL`)")
        lines.append("")
        lines.append("Each task is run with the full inferred tool schema. PASS means the "
                     "model called the tool assigned to that operation.")
        lines.append("")
        lines.append("| Operation | Result | Expected tool | Called tool | Notes |")
        lines.append("|---|---|---|---|---|")
        for op, result in results.items():
            status = "PASS" if result.get("pass") else "FAIL"
            expected = result.get("expected_function_name") or "*(none)*"
            called = result.get("function_name") or "*(none)*"
            note = _md_escape(result.get("error") or "")
            lines.append(f"| {op} | {status} | `{expected}` | `{called}` | {note} |")
        lines.append("")

    if _capability_absent(output, "agentknit_test"):
        _append_no_data_section(lines, "## Agentknit default-tool compatibility test (`AKDEF`)")
    elif agentknit_test and "error" not in agentknit_test:
        results = agentknit_test.get("akdef_results") or {}
        passed  = agentknit_test.get("akdef_passed", 0)
        total   = agentknit_test.get("akdef_total", 0)
        lines.append("## Agentknit default-tool compatibility test (`AKDEF`)")
        lines.append("")
        lines.append(f"**{passed}/{total} passed** — agentknit's real shipped default tool "
                     "schema (`read_file`/`write_file`/`str_replace`/`exec_shell`, imported "
                     "live from `agentknit._core`) is offered, and each call is dispatched "
                     "through agentknit's real `dispatch()` against a scratch directory. "
                     "PASS requires both the right tool selected and the correct on-disk "
                     "effect.")
        lines.append("")
        lines.append("| Tool | Result | Called tool | Dispatch OK | Effect verified | Notes |")
        lines.append("|---|---|---|---|---|---|")
        for op, r in results.items():
            status = "PASS" if r.get("pass") else "FAIL"
            called = r.get("called_tool") or "*(none)*"
            note   = _md_escape(r.get("error") or "")
            lines.append(f"| {op} | {status} | `{called}` | "
                         f"{r.get('dispatch_ok')} | {r.get('effect_verified')} | {note} |")
        lines.append("")
    elif agentknit_test and agentknit_test.get("error"):
        lines.append("## Agentknit default-tool compatibility test (`AKDEF`)")
        lines.append("")
        lines.append(f"Error: {agentknit_test['error']}")
        lines.append("")

    dispatch = output.get("tool_dispatch") or {}
    if dispatch and "error" not in dispatch:
        lines.append("## Tool dispatch table")
        lines.append("")
        lines.append("| Model tool name | Python function | Param map |")
        lines.append("|---|---|---|")
        for tool_name, entry in dispatch.items():
            param_map = entry.get("param_map") or {}
            param_str = ", ".join(f"{k}→{v}" for k, v in param_map.items()) or "*(none)*"
            generated = " *(generated)*" if entry.get("generated_source") else ""
            lines.append(f"| `{tool_name}` | `{entry.get('python_function', '?')}`{generated} | {param_str} |")
        lines.append("")
    elif dispatch.get("error"):
        lines.append("## Tool dispatch table")
        lines.append("")
        lines.append(f"Error: {dispatch['error']}")
        lines.append("")

    if _capability_absent(output, "quote_test"):
        _append_no_data_section(lines, "## Quote-escaping test (`QUOTE`)")
    elif quote_test and "error" not in quote_test:
        results = quote_test.get("quote_test_results") or {}
        passed  = quote_test.get("quote_test_passed", 0)
        total   = quote_test.get("quote_test_total", 0)
        lines.append("## Quote-escaping test (`QUOTE`)")
        lines.append("")
        lines.append(f"**{passed}/{total} passed** — only the tool relevant to each task is "
                     "advertised (not the full schema), so this isolates quote-escaping "
                     "fidelity from tool-selection behaviour.")
        lines.append("")
        lines.append("| Operation | Isolated schema | Result | Function called | Notes |")
        lines.append("|---|---|---|---|---|")
        for op, r in results.items():
            result = "PASS" if r.get("pass") else "FAIL"
            fn     = r.get("function_name") or "*(none)*"
            note   = _md_escape(r.get("error") or "")
            isolated = "yes" if r.get("isolated") else "no *(fallback: full schema)*"
            lines.append(f"| {op} | {isolated} | {result} | `{fn}` | {note} |")
        lines.append("")
    elif quote_test and quote_test.get("error"):
        lines.append("## Quote-escaping test (`QUOTE`)")
        lines.append("")
        lines.append(f"Error: {quote_test['error']}")
        lines.append("")

    if _capability_absent(output, "token_efficiency_test"):
        _append_no_data_section(lines, "## Token-efficiency test (`GREP`)")
    elif tok_test and "error" not in tok_test:
        results = tok_test.get("token_efficiency_results") or {}
        passed  = tok_test.get("token_efficiency_passed", 0)
        total   = tok_test.get("token_efficiency_total", 0)
        lines.append("## Token-efficiency test (`GREP`)")
        lines.append("")
        lines.append(f"**{passed}/{total} passed** — prefers a filtered/targeted call over "
                     "pulling the entire large file/output into context.")
        lines.append("")
        lines.append("| Operation | Result | Function called | Args | Reason |")
        lines.append("|---|---|---|---|---|")
        for op, r in results.items():
            result = "PASS" if r.get("pass") else "FAIL"
            fn     = r.get("function_name") or "*(none)*"
            args   = _md_escape(json.dumps(r.get("parsed_args") or {}))
            reason = _md_escape(r.get("reason") or "")
            lines.append(f"| {op} | {result} | `{fn}` | {args} | {reason} |")
        lines.append("")
    elif tok_test and tok_test.get("error"):
        lines.append("## Token-efficiency test (`GREP`)")
        lines.append("")
        lines.append(f"Error: {tok_test['error']}")
        lines.append("")

    if _capability_absent(output, "askq_test"):
        _append_no_data_section(lines, "## Ask-user-question phrasing test (`ASKQ`)")
    elif askq_test and "error" not in askq_test:
        results = askq_test.get("askq_results") or {}
        passed  = askq_test.get("askq_passed", 0)
        total   = askq_test.get("askq_total", 0)
        lines.append("## Ask-user-question phrasing test (`ASKQ`)")
        lines.append("")
        lines.append(f"**{passed}/{total} — {_likert_label(passed, total)}** calls its own "
                     "ask_user_question tool across 8 phrasings of the same underlying task "
                     "(full tool schema, one sample per phrasing, no retries).")
        lines.append("")
        lines.append("| Variant | System prompt | Result | Function called |")
        lines.append("|---|---|---|---|")
        for variant, r in results.items():
            result = "ASKED" if r.get("pass") else "SKIPPED"
            fn     = r.get("function_name") or "*(none)*"
            sys_kind = "nudge" if r.get("system") != _ASKQ_SYSTEM_DEFAULT else "default"
            lines.append(f"| {variant} | {sys_kind} | {result} | `{fn}` |")
        lines.append("")
    elif askq_test and askq_test.get("error"):
        lines.append("## Ask-user-question phrasing test (`ASKQ`)")
        lines.append("")
        lines.append(f"Error: {askq_test['error']}")
        lines.append("")

    if _capability_absent(output, "gram_knowledge_test"):
        _append_no_data_section(lines, "## apply_patch grammar-knowledge test (`APPLY_PATCH`)")
    elif gram_knowledge_test and "error" not in gram_knowledge_test:
        results = gram_knowledge_test.get("gram_knowledge_results") or {}
        passed  = gram_knowledge_test.get("gram_knowledge_passed", 0)
        total   = gram_knowledge_test.get("gram_knowledge_total", 0)
        lines.append("## apply_patch grammar-knowledge test (`APPLY_PATCH`)")
        lines.append("")
        lines.append(f"**{passed}/{total} passed** — no tool schema offered; the model is asked "
                     "in free text to produce a raw apply_patch-format patch, parsed against the "
                     "real grammar (not a loose regex). Tests whether the model *knows* the "
                     "syntax, independent of whether the endpoint exposes the tool itself "
                     "(see `~/bin/copilot-notes.md`).")
        lines.append("")
        lines.append("| Operation | Result | Notes |")
        lines.append("|---|---|---|")
        for op, r in results.items():
            result = "PASS" if r.get("pass") else "FAIL"
            note   = _md_escape(r.get("error") or "")
            lines.append(f"| {op} | {result} | {note} |")
        lines.append("")
    elif gram_knowledge_test and gram_knowledge_test.get("error"):
        lines.append("## apply_patch grammar-knowledge test (`APPLY_PATCH`)")
        lines.append("")
        lines.append(f"Error: {gram_knowledge_test['error']}")
        lines.append("")

    if _capability_absent(output, "gram_transport_test"):
        _append_no_data_section(lines, "## Constrained-decoding / custom-tool test (`GRAMT`)")
    elif gram_transport_test and "error" not in gram_transport_test:
        results = gram_transport_test.get("gram_transport_results") or {}
        passed  = gram_transport_test.get("gram_transport_passed", 0)
        total   = gram_transport_test.get("gram_transport_total", 0)
        lines.append("## Constrained-decoding / custom-tool test (`GRAMT`)")
        lines.append("")
        lines.append(f"**{passed}/{total} passed** — sends a real OpenAI `type:\"custom\"` "
                     "freeform tool with `format:{type:\"grammar\", syntax:\"lark\"}`; PASS "
                     "requires a genuine `custom` tool_call back with grammar-valid input "
                     "(not a classic `function` tool_call, and not silently ignored). Tests the "
                     "*endpoint's* transport support, independent of whether the model knows the "
                     "syntax (`APPLY_PATCH`) — see `~/bin/copilot-notes.md`.")
        lines.append("")
        lines.append("| Operation | Result | Tool call type | Notes |")
        lines.append("|---|---|---|---|")
        for op, r in results.items():
            result = "PASS" if r.get("pass") else "FAIL"
            tct    = r.get("tool_call_type") or "*(none)*"
            note   = _md_escape(r.get("error") or "")
            lines.append(f"| {op} | {result} | `{tct}` | {note} |")
        lines.append("")
    elif gram_transport_test and gram_transport_test.get("error"):
        lines.append("## Constrained-decoding / custom-tool test (`GRAMT`)")
        lines.append("")
        lines.append(f"Error: {gram_transport_test['error']}")
        lines.append("")

    if _capability_absent(output, "rjson_test"):
        _append_no_data_section(lines, "## Structured-output test (`RJSON`)")
    elif rjson_test and "error" not in rjson_test:
        results = rjson_test.get("rjson_results") or {}
        passed  = rjson_test.get("rjson_passed", 0)
        total   = rjson_test.get("rjson_total", 0)
        lines.append("## Structured-output test (`RJSON`)")
        lines.append("")
        lines.append(f"**{passed}/{total} passed** — sends a strict "
                     "`response_format:{type:\"json_schema\"}` request with no tool schema; "
                     "PASS requires the endpoint to accept the request and return content "
                     "that parses as JSON conforming to the schema. Tests the *endpoint's* "
                     "structured-output support, independent of tool calling.")
        lines.append("")
        lines.append("| Task | Result | Conformant | Notes |")
        lines.append("|---|---|---|---|")
        for op, r in results.items():
            result = "PASS" if r.get("pass") else "FAIL"
            conf   = "yes" if r.get("schema_conformant") else "no"
            note   = _md_escape(r.get("error") or "")
            lines.append(f"| {op} | {result} | {conf} | {note} |")
        lines.append("")
    elif rjson_test and rjson_test.get("error"):
        lines.append("## Structured-output test (`RJSON`)")
        lines.append("")
        lines.append(f"Error: {rjson_test['error']}")
        lines.append("")

    if _capability_absent(output, "stream_test"):
        _append_no_data_section(lines, "## SSE streaming test (`STRM`)")
    elif stream_test and "error" not in stream_test:
        r      = (stream_test.get("stream_results") or {}).get("basic") or {}
        passed = stream_test.get("stream_passed", 0)
        total  = stream_test.get("stream_total", 0)
        lines.append("## SSE streaming test (`STRM`)")
        lines.append("")
        lines.append(f"**{passed}/{total} passed** — sends `stream:true`; PASS requires more "
                     "than one chunk, non-empty reconstructed content, and a finish_reason "
                     "chunk. FAIL includes the endpoint rejecting `stream:true` outright and "
                     "accepting it but buffering the whole reply into one chunk.")
        lines.append("")
        result = "PASS" if r.get("pass") else "FAIL"
        note   = _md_escape(r.get("error") or "")
        lines.append(f"- Result: {result}")
        lines.append(f"- Chunks: {r.get('chunk_count', '?')}")
        lines.append(f"- Time to first chunk: {r.get('time_to_first_chunk_seconds', '?')}s")
        lines.append(f"- Total time: {r.get('total_seconds', '?')}s")
        lines.append(f"- finish_reason: `{r.get('finish_reason')}`")
        if note:
            lines.append(f"- Notes: {note}")
        lines.append("")
    elif stream_test and stream_test.get("error"):
        lines.append("## SSE streaming test (`STRM`)")
        lines.append("")
        lines.append(f"Error: {stream_test['error']}")
        lines.append("")

    if _capability_absent(output, "reasoning_test"):
        _append_no_data_section(lines, "## Reasoning-tokens & effort-control test (`REASN`)")
    elif reasoning_test and "error" not in reasoning_test:
        results = reasoning_test.get("reason_results") or {}
        passed  = reasoning_test.get("reason_passed", 0)
        total   = reasoning_test.get("reason_total", 0)
        lines.append("## Reasoning-tokens & effort-control test (`REASN`)")
        lines.append("")
        lines.append(f"**{passed}/{total} passed** — `reasoning_tokens_present` checks (no "
                     "special params) whether the reply carries a reasoning trace "
                     "(`reasoning_content`/`reasoning`/`thinking` field, or "
                     "`usage.completion_tokens_details.reasoning_tokens`). "
                     "`effort_control_native` and `effort_control_extra_body` each check "
                     "whether the endpoint accepts one reasoning-effort syntax — the native "
                     "top-level `reasoning_effort` Chat Completions param, and the "
                     "OpenRouter-style `extra_body={'reasoning': {'effort': ...}}` passthrough "
                     "— without erroring. Accepting the parameter is the bar; this does not "
                     "confirm the effort setting changed model behaviour.")
        lines.append("")
        lines.append("| Check | Result | Field | Notes |")
        lines.append("|---|---|---|---|")
        for op, r in results.items():
            result = "PASS" if r.get("pass") else "FAIL"
            field  = r.get("field") or "*(none)*"
            note   = _md_escape(r.get("error") or "")
            lines.append(f"| {op} | {result} | `{field}` | {note} |")
        lines.append("")
    elif reasoning_test and reasoning_test.get("error"):
        lines.append("## Reasoning-tokens & effort-control test (`REASN`)")
        lines.append("")
        lines.append(f"Error: {reasoning_test['error']}")
        lines.append("")

    if _capability_absent(output, "cache_ttl_test"):
        lines.append("## Prompt-cache TTL measurement (`CACH`)")
        lines.append("")
        lines.append(f"*({_NO_DATA_NOTE})*")
        lines.append("")
        lines.append("`CACH` is an opt-in measurement: rerun the probe with "
                     "`--cache-ttl-test` to collect it.")
        lines.append("")
    elif cache_ttl_test and (cache_ttl_test.get("samples") or "error" not in cache_ttl_test):
        samples = cache_ttl_test.get("samples") or {}
        lines.append("## Prompt-cache TTL measurement (`CACH`)")
        lines.append("")
        lines.append("Empirical prompt-cache TTL: the probe primes the cache with a large "
                     "fixed prefix, then re-sends the exact same prefix after increasing "
                     "delays, stopping at the first cold (cache-miss) sample. WARM means "
                     "the provider reported most prompt tokens as cached; COLD means the "
                     "cache entry had expired. An immediate (0-delay) re-send first proves "
                     "the prefix is cacheable at all, and on load-balanced endpoints "
                     "(OpenRouter) the upstream that answered the prime call is pinned for "
                     "every re-send, since a different upstream has a different cache. "
                     "A TTL has no right or wrong value \u2014 this section is informational, "
                     "not pass/fail.")
        lines.append("")
        ttl_min = cache_ttl_test.get("ttl_min_seconds")
        ttl_max = cache_ttl_test.get("ttl_max_seconds")
        provider = cache_ttl_test.get("provider")
        if provider:
            pin_note = ("pinned for every re-send" if cache_ttl_test.get("provider_pinned")
                        else "not pinned")
            lines.append(f"- **Upstream provider reported by the endpoint:** `{provider}` "
                         f"({pin_note})")
            lines.append("")
        if ttl_min is not None and ttl_max is not None:
            lines.append(f"- **Measured TTL: between {ttl_min}s and {ttl_max}s** "
                         f"(midpoint estimate {cache_ttl_test.get('measured_ttl_minutes')} min)")
        elif ttl_min is not None:
            lines.append(f"- **Measured TTL: > {ttl_min}s** (still warm at the longest probed delay)")
        elif ttl_max is not None:
            lines.append("- **Measured TTL: < first probed delay** (already cold on the first re-send)")
        else:
            lines.append("- **TTL: not measurable** (no cache reuse observed, see below)")
        lines.append("")
        lines.append("| Sample | Prompt tokens | Cached tokens | Latency (s) | Provider | Verdict |")
        lines.append("|---|---|---|---|---|---|")
        for label, s in samples.items():
            prompt = s.get("prompt_tokens") or 0
            cached = s.get("cached_tokens")
            if label == "prime":
                verdict = "prime"
            elif label == "confirm_0s":
                verdict = "WARM (no delay)" if (cached is not None and cached >= max(500, 0.5 * prompt)) else "COLD (no delay)"
            elif cached is None:
                verdict = "unknown"
            else:
                verdict = "WARM" if cached >= max(500, 0.5 * prompt) else "COLD"
            lines.append(f"| {label} | {s.get('prompt_tokens', '?')} | "
                         f"{cached if cached is not None else '*(not reported)*'} | "
                         f"{s.get('latency_seconds', '?')} | "
                         f"{s.get('provider') or '\u2014'} | {verdict} |")
        lines.append("")
        mismatch = cache_ttl_test.get("provider_mismatch")
        if mismatch:
            lines.append(f"> The endpoint answered from a different upstream "
                         f"(`{mismatch.get('got')}`) than the one that primed the cache "
                         f"(`{mismatch.get('expected')}`), even with provider pinning: on a "
                         f"load-balanced endpoint a cache miss across upstreams says nothing "
                         f"about cache expiry, so no TTL is reported.")
            lines.append("")
        elif cache_ttl_test.get("cache_observed") is False:
            lines.append(f"> {cache_ttl_test.get('error')}")
            lines.append("")
    elif cache_ttl_test and cache_ttl_test.get("error"):
        lines.append("## Prompt-cache TTL measurement (`CACH`)")
        lines.append("")
        lines.append(f"Error: {cache_ttl_test['error']}")
        lines.append("")

    lines.append("## Missing capabilities")
    lines.append("")
    problems = _find_missing_capabilities(output)
    if problems:
        for p in problems:
            lines.append(f"- {p}")
    else:
        lines.append("N/A")
    lines.append("")

    return re.sub(r"[ \t]+\n", "\n", "\n".join(lines))


def _find_missing_capabilities(output: dict) -> list[str]:
    """Collect problems/gaps found while probing, for the report's tail section."""
    problems: list[str] = []

    if output.get("error"):
        problems.append(f"Probe aborted early: {output['error']}")

    if _capability_absent(output, "format_detection"):
        problems.append(f"`TCALL`: {_NO_DATA_NOTE}.")

    ctx = output.get("context_window")
    if _capability_absent(output, "context_window"):
        problems.append(f"Context window: {_NO_DATA_NOTE}.")
    elif not ctx or ctx.get("tokens") is None:
        problems.append("Context window: no limit recovered for this endpoint; "
                        "record one with --context-tokens.")

    cors_test = _cors_merge(output.get("cors_test") or {})
    if _capability_absent(output, "cors_test"):
        problems.append(f"`CORS`: {_NO_DATA_NOTE}.")
    elif cors_test is None or not cors_test:
        problems.append("`CORS` capability not tested (rerun without --no-cors-test).")
    elif cors_test.get("error"):
        pass  # recorded verbatim in the report; e.g. a local script target
    elif cors_test.get("mode") == "none":
        problems.append("`CORS`: no access-control-allow-origin on any probe -- a "
                        "browser cannot integrate with this endpoint directly.")
    elif cors_test.get("mode") == "blocked":
        problems.append("`CORS`: preflight or actual response rejects browser-direct access.")

    fmt = output.get("format_detection") or {}
    if fmt.get("error"):
        problems.append(f"`TCALL` format detection (round 0) failed: {fmt['error']}")

    elicited = output.get("elicited_names") or {}
    for op, fn in elicited.items():
        if not fn:
            problems.append(f"Could not elicit a function name for operation '{op}' "
                            "(model's free-form answer was unparseable).")

    tsel_test = output.get("tsel_test") or {}
    if _capability_absent(output, "tsel_test"):
        problems.append(f"`TSEL`: {_NO_DATA_NOTE}.")
    for op, result in (tsel_test.get("tsel_results") or {}).items():
        if not result.get("pass"):
            problems.append(f"`TSEL_{op}` FAILED — {result.get('error', 'wrong tool selected')}.")

    dispatch  = output.get("tool_dispatch") or {}
    if dispatch.get("error"):
        problems.append(f"Tool dispatch table build failed: {dispatch['error']}")
    else:
        schema_names    = {(t.get("function") or t).get("name") for t in output.get("inferred_tool_schema") or []}
        dispatched_names = set(dispatch.keys())
        elicited_names   = output.get("elicited_names") or {}
        for name in sorted(schema_names - dispatched_names):
            problems.append(f"Tool `{name}` is in the inferred schema but was never dispatched "
                            "(the model didn't call it with a matching signature during Round 2/3 probing).")

    behaviour = output.get("behaviour") or {}
    if behaviour.get("no_call_detected"):
        problems.append(f"{behaviour['no_call_detected']} probe task(s) produced no detectable "
                        "tool call at all (see probes/<model>/round2_*.json for which ones).")

    quote_test = output.get("quote_test")
    if _capability_absent(output, "quote_test"):
        problems.append(f"`QUOTE`: {_NO_DATA_NOTE}.")
    elif quote_test is None:
        problems.append("`QUOTE` capability not tested (rerun without --no-quote-test).")
    elif quote_test.get("error"):
        problems.append(f"`QUOTE` test failed to run: {quote_test['error']}")
    else:
        for op, r in (quote_test.get("quote_test_results") or {}).items():
            if not r.get("pass"):
                problems.append(f"`QUOTE_{op}` FAILED — {r.get('error', 'unknown reason')}")

    tok_test = output.get("token_efficiency_test")
    if _capability_absent(output, "token_efficiency_test"):
        problems.append(f"`GREP`: {_NO_DATA_NOTE}.")
    elif tok_test is None:
        problems.append("`GREP` capability not tested (rerun without --no-efficiency-test).")
    elif tok_test.get("error"):
        problems.append(f"`GREP` test failed to run: {tok_test['error']}")
    else:
        for op, r in (tok_test.get("token_efficiency_results") or {}).items():
            if not r.get("pass"):
                problems.append(f"`GREP_{op}` FAILED — {r.get('reason', 'unknown reason')}")

    askq_test = output.get("askq_test")
    if _capability_absent(output, "askq_test"):
        problems.append(f"`ASKQ`: {_NO_DATA_NOTE}.")
    elif askq_test is None:
        problems.append("`ASKQ` capability not tested (rerun without --no-askq-test).")
    elif askq_test.get("error"):
        problems.append(f"`ASKQ` test failed to run: {askq_test['error']}")
    else:
        for variant, r in (askq_test.get("askq_results") or {}).items():
            if not r.get("pass"):
                fn = r.get("function_name") or "no tool call"
                problems.append(f"`ASKQ_{variant}` FAILED — called `{fn}` instead of "
                                "asking the user.")

    gram_knowledge_test = output.get("gram_knowledge_test")
    if _capability_absent(output, "gram_knowledge_test"):
        problems.append(f"`APPLY_PATCH`: {_NO_DATA_NOTE}.")
    elif gram_knowledge_test is None:
        problems.append("`APPLY_PATCH` capability not tested (rerun without --no-gram-knowledge-test).")
    elif gram_knowledge_test.get("error"):
        problems.append(f"`APPLY_PATCH` test failed to run: {gram_knowledge_test['error']}")
    else:
        for op, r in (gram_knowledge_test.get("gram_knowledge_results") or {}).items():
            if not r.get("pass"):
                problems.append(f"`APPLY_PATCH_{op}` FAILED — {r.get('error', 'unknown reason')}")

    gram_transport_test = output.get("gram_transport_test")
    if _capability_absent(output, "gram_transport_test"):
        problems.append(f"`GRAMT`: {_NO_DATA_NOTE}.")
    elif gram_transport_test is None:
        problems.append("`GRAMT` capability not tested (rerun without --no-gram-transport-test).")
    elif gram_transport_test.get("error"):
        problems.append(f"`GRAMT` test failed to run: {gram_transport_test['error']}")
    else:
        for op, r in (gram_transport_test.get("gram_transport_results") or {}).items():
            if not r.get("pass"):
                problems.append(f"`GRAMT_{op}` FAILED — {r.get('error', 'unknown reason')}")

    rjson_test = output.get("rjson_test")
    if _capability_absent(output, "rjson_test"):
        problems.append(f"`RJSON`: {_NO_DATA_NOTE}.")
    elif rjson_test is None:
        problems.append("`RJSON` capability not tested (rerun without --no-rjson-test).")
    elif rjson_test.get("error"):
        problems.append(f"`RJSON` test failed to run: {rjson_test['error']}")
    else:
        for op, r in (rjson_test.get("rjson_results") or {}).items():
            if not r.get("pass"):
                problems.append(f"`RJSON_{op}` FAILED — {r.get('error', 'unknown reason')}")

    stream_test = output.get("stream_test")
    if _capability_absent(output, "stream_test"):
        problems.append(f"`STRM`: {_NO_DATA_NOTE}.")
    elif stream_test is None:
        problems.append("`STRM` capability not tested (rerun without --no-stream-test).")
    elif stream_test.get("error"):
        problems.append(f"`STRM` test failed to run: {stream_test['error']}")
    else:
        for op, r in (stream_test.get("stream_results") or {}).items():
            if not r.get("pass"):
                problems.append(f"`STRM_{op}` FAILED — {r.get('error', 'unknown reason')}")

    reasoning_test = output.get("reasoning_test")
    if _capability_absent(output, "reasoning_test"):
        problems.append(f"`REASN`: {_NO_DATA_NOTE}.")
    elif reasoning_test is None:
        problems.append("`REASN` capability not tested (rerun without --no-reasoning-test).")
    elif reasoning_test.get("error"):
        problems.append(f"`REASN` test failed to run: {reasoning_test['error']}")
    else:
        for op, r in (reasoning_test.get("reason_results") or {}).items():
            if not r.get("pass"):
                problems.append(f"`REASN_{op}` FAILED — {r.get('error', 'unknown reason')}")

    agentknit_test = output.get("agentknit_test")
    if _capability_absent(output, "agentknit_test"):
        problems.append(f"`AKDEF`: {_NO_DATA_NOTE}.")
    elif agentknit_test is None:
        problems.append("`AKDEF` capability not tested (rerun without --no-agentknit-test).")
    elif agentknit_test.get("error"):
        problems.append(f"`AKDEF` test failed to run: {agentknit_test['error']}")
    else:
        for op, r in (agentknit_test.get("akdef_results") or {}).items():
            if not r.get("pass"):
                problems.append(f"`AKDEF_{op}` FAILED — {r.get('error', 'unknown reason')}")

    cache_ttl_test = output.get("cache_ttl_test")
    if cache_ttl_test is None:
        pass  # opt-in measurement (--cache-ttl-test); absence is not a gap
    elif cache_ttl_test.get("error"):
        problems.append(f"`CACH` test failed to run: {cache_ttl_test['error']}")

    return problems


# -- main ---------------------------------------------------------------------

def main():
    global ENDPOINT, MODEL, _KEY_NAME, _PROVIDER

    args = parse_args()
    if args.quick_summary:
        quick_summary()
        return
    if args.list_providers:
        if not args.model:
            sys.exit("--list-providers needs --model.")
        providers = openrouter_providers(args.model)
        if not providers:
            sys.exit(f"OpenRouter lists no providers for {args.model}.")
        print(f"{len(providers)} providers serve {args.model} on OpenRouter:\n")
        for p in providers:
            tools = "tools" if p["supports_tools"] else "no tools"
            context = f"{p['context_length']:,}" if p["context_length"] else "?"
            print(f"  --provider {p['slug']:<20} {p['name']:<14} "
                  f"{context:>11} ctx, {tools}")
        return
    _PROVIDER = args.provider
    if args.script:
        ENDPOINT = f"script:{args.script}"
    elif args.endpoint:
        ENDPOINT = args.endpoint
    if args.model:
        MODEL = args.model
    _KEY_NAME = args.key_name

    safe_model = MODEL.replace("/", "_").replace(":", "_")
    server = _safe_server(args.server) if args.server else _known_server_for(ENDPOINT, safe_model)
    if server is None and ENDPOINT.startswith("script:"):
        upstream = _script_upstream(ENDPOINT[len("script:"):])
        if upstream:
            server = _safe_server(upstream)
        else:
            print(f"Warning: {ENDPOINT} did not report an 'x_upstream_endpoint'; "
                  f"filing this run under reports/unknown-server/. Pass --server "
                  f"to name it explicitly.")
    report_dir = _report_dir(ENDPOINT, safe_model, server, _PROVIDER)
    report_dir.mkdir(parents=True, exist_ok=True)
    out_path   = args.output or str(report_dir / f"capabilities_{safe_model}.json")

    # Reports used to live in reports/<model>/; keep reading an existing one so
    # a re-run against the same model still preserves earlier results.
    legacy_path = Path("reports") / safe_model / f"capabilities_{safe_model}.json"
    if args.output is None and not Path(out_path).exists() and legacy_path.exists():
        for legacy in (legacy_path, legacy_path.with_suffix(".md")):
            if legacy.exists():
                legacy.replace(report_dir / legacy.name)
        try:
            legacy_path.parent.rmdir()
        except OSError:
            pass

    if args.render_md_only:
        if not Path(out_path).exists():
            sys.exit(f"Cannot render: {out_path} does not exist. Run the probe first.")
        with open(out_path) as f:
            output = json.load(f)
        md_path = _capabilities_md_path(out_path)
        with open(md_path, "w") as f:
            f.write(render_markdown_report(output))
        print(f"Rendered {md_path} from {out_path} (no probing performed)")
        return

    if args.context_only:
        if not Path(out_path).exists():
            sys.exit(f"Cannot run --context-only: {out_path} does not exist. "
                     "Run the full probe first.")
        _init_probe_dir(safe_model)
        with open(out_path) as f:
            output = json.load(f)
        print(f"Target: {ENDPOINT}")
        print(f"Model:  {MODEL}")
        output["context_window"] = _resolve_context_window(args, _make_target_client)
        md_path = _capabilities_md_path(out_path)
        with open(out_path, "w") as f:
            json.dump(output, f, indent=2)
        with open(md_path, "w") as f:
            f.write(render_markdown_report(output))
        print(f"\nReport written to {out_path} and {md_path}")
        return

    if args.cors_only:
        if not Path(out_path).exists():
            sys.exit(f"Cannot run --cors-only: {out_path} does not exist. "
                     "Run the full probe first.")
        with open(out_path) as f:
            output = json.load(f)
        print(f"Target: {ENDPOINT}")
        print(f"Model:  {MODEL}")
        output["cors_test"] = cors_test_round()
        md_path = _capabilities_md_path(out_path)
        with open(out_path, "w") as f:
            json.dump(output, f, indent=2)
        with open(md_path, "w") as f:
            f.write(render_markdown_report(output))
        print(f"\nReport written to {out_path} and {md_path}")
        return

    if args.tdef_only:
        if Path(out_path).exists():
            with open(out_path) as f:
                output = json.load(f)
        elif _PROVIDER:
            # First pinned run for this provider: there is no report to refresh
            # and nothing to inherit -- every other capability was measured on
            # whichever provider happened to answer, which is not this one. So
            # start an empty report; the untested capabilities render as "no
            # data" rather than borrowing another provider's answers.
            print(f"No report yet for provider {_PROVIDER}; starting one with "
                  f"TDEF only. Run the full probe with --provider {_PROVIDER} "
                  f"to fill in the rest.")
            output = {"model": MODEL, "llmprobe_commit": _llmprobe_commit(),
                      "endpoint": ENDPOINT, "provider": _PROVIDER,
                      "observed_provider": None,
                      "api_type": API_TYPE_LABELS[args.api_type],
                      "status": "ok", "error": None}
        else:
            sys.exit(f"Cannot run --tdef-only: {out_path} does not exist. "
                     "Run the full probe first.")
        print(f"Target: {ENDPOINT}")
        print(f"Model:  {MODEL}")
        api_key = "" if args.script else get_api_key(args.key_name)
        output["tdef_test"] = tdef_test_round(api_key, args)
        if _PROVIDER:
            output["provider"] = _PROVIDER
        output["observed_provider"] = _OBSERVED_PROVIDER or output.get("observed_provider")
        md_path = _capabilities_md_path(out_path)
        with open(out_path, "w") as f:
            json.dump(output, f, indent=2)
        with open(md_path, "w") as f:
            f.write(render_markdown_report(output))
        print(f"\nReport written to {out_path} and {md_path}")
        return

    if args.cache_ttl_only:
        if not Path(out_path).exists():
            sys.exit(f"Cannot run --cache-ttl-only: {out_path} does not exist. "
                     "Run the full probe first.")
        _init_probe_dir(safe_model)
        with open(out_path) as f:
            output = json.load(f)
        if args.script:
            client = ScriptClient(args.script)
        else:
            api_key = get_api_key(args.key_name)
            client  = make_client(api_key)
        print(f"Target: {ENDPOINT}")
        print(f"Model:  {MODEL}")
        output["cache_ttl_test"] = cache_ttl_test_round(client)
        md_path = _capabilities_md_path(out_path)
        with open(out_path, "w") as f:
            json.dump(output, f, indent=2)
        with open(md_path, "w") as f:
            f.write(render_markdown_report(output))
        print(f"\nReport written to {out_path} and {md_path}")
        return

    _init_probe_dir(safe_model)
    previous = _load_previous_report(out_path)

    output: dict = {
        "model":                MODEL,
        "llmprobe_commit":      _llmprobe_commit(),
        "endpoint":             ENDPOINT,
        "provider":             _PROVIDER,
        "observed_provider":    None,
        "server":               report_dir.parent.name,
        "api_type":             API_TYPE_LABELS[args.api_type],
        "status":               "incomplete",
        "error":                None,
        "format_detection":     {},
        "elicited_names":       {},
        "inferred_tool_schema": [],
        "behaviour":            {},
        "tool_dispatch":        {},
        "dispatch_conflicts":   {},
        "tsel_test":            None,
        "quote_test":           None,
        "token_efficiency_test": None,
        "askq_test":            None,
        "gram_knowledge_test":  None,
        "gram_transport_test":  None,
        "rjson_test":           None,
        "stream_test":          None,
        "reasoning_test":       None,
        "agentknit_test":       None,
        "cache_ttl_test":       None,
        "context_window":       None,
        "cors_test":            None,
        "tdef_test":            None,
    }

    md_path = _capabilities_md_path(out_path)

    def save(note: str = ""):
        output["observed_provider"] = _OBSERVED_PROVIDER
        with open(out_path, "w") as f:
            json.dump(output, f, indent=2)
        with open(md_path, "w") as f:
            f.write(render_markdown_report(output))
        msg = f"\nReport written to {out_path} and {md_path}"
        if note:
            msg += f"  ({note})"
        print(msg)

    if args.script:
        api_key = ""
        client = ScriptClient(args.script)
    else:
        api_key = get_api_key(args.key_name)
        client  = make_client(api_key)

    print(f"Target: {ENDPOINT}")
    print(f"Model:  {MODEL}")

    try:
        output["format_detection"] = format_detection_round(client)
    except Exception as e:
        if "No endpoints found that support tool use" in str(e):
            print(f"\nModel does not support tool use -- aborting.")
            raise SystemExit(1)
        if _keep_previous_result(e, previous, "format_detection"):
            output["format_detection"] = previous["format_detection"]
            print(f"\nWARNING in Round 0 (429): {e} -- keeping previous run's result")
        else:
            output["format_detection"] = {"error": str(e)}
            print(f"\nWARNING in Round 0: {e}")

    try:
        elicited = elicit_round(client)
        elicited = _deduplicate_elicited(elicited)
        output["elicited_names"] = {op: v["function_name"] for op, v in elicited.items()}
    except Exception as e:
        output["error"] = f"elicit_round failed: {e}"
        kept = _keep_previous_result(e, previous, "elicited_names")
        if kept:
            _restore_previous_on_429(previous, output)
            output["status"] = "ok"
        print(f"\nERROR in Round 1: {e}")
        if kept:
            print("429: keeping previous run's results")
        save("failed at Round 1" + (" -- previous results kept" if kept else ""))
        raise SystemExit(1)

    initial_tools = build_tool_schema(elicited)

    try:
        probe_calls = probe_round(client, initial_tools)
    except Exception as e:
        output["error"] = f"probe_round failed: {e}"
        kept = _keep_previous_result(e, previous, "behaviour")
        if kept:
            _restore_previous_on_429(previous, output)
            output["status"] = "ok"
        print(f"\nERROR in Round 2: {e}")
        if kept:
            print("429: keeping previous run's results")
        save("failed at Round 2 -- elicited names preserved" + (" -- previous results kept" if kept else ""))
        raise SystemExit(1)

    final_tools  = initial_tools
    final_probes = probe_calls
    behaviour    = behavioural_summary(final_probes)

    output["status"]               = "ok"
    output["inferred_tool_schema"] = final_tools
    output["behaviour"]            = behaviour
    output["tsel_test"]            = tool_selection_test(final_probes, output["elicited_names"])

    section("Final inferred tool schema")
    print(json.dumps(final_tools, indent=2))

    section("Behavioural findings")
    print(json.dumps(behaviour, indent=2))

    try:
        tool_dispatch, dispatch_conflicts = build_tool_dispatch(elicited, final_probes, client)
        output["tool_dispatch"] = tool_dispatch
        output["dispatch_conflicts"] = dispatch_conflicts
        section("Tool dispatch table")
        print(json.dumps(tool_dispatch, indent=2))
        if dispatch_conflicts:
            section("Dispatch conflicts")
            print(json.dumps(dispatch_conflicts, indent=2))
    except Exception as e:
        output["tool_dispatch"] = {"error": str(e)}
        print(f"\nERROR building tool dispatch: {e}")

    if args.quote_test:
        try:
            qt = quote_test_round(client, final_tools, output["elicited_names"])
            output["quote_test"] = qt
        except Exception as e:
            if _keep_previous_result(e, previous, "quote_test"):
                output["quote_test"] = previous["quote_test"]
                print(f"\nERROR in quote-test round (429): {e} -- keeping previous run's result")
            else:
                output["quote_test"] = {"error": str(e)}
                print(f"\nERROR in quote-test round: {e}")

    if args.efficiency_test:
        try:
            et = token_efficiency_test_round(client, final_tools, output.get("tool_dispatch") or {})
            output["token_efficiency_test"] = et
        except Exception as e:
            if _keep_previous_result(e, previous, "token_efficiency_test"):
                output["token_efficiency_test"] = previous["token_efficiency_test"]
                print(f"\nERROR in token-efficiency test round (429): {e} -- keeping previous run's result")
            else:
                output["token_efficiency_test"] = {"error": str(e)}
                print(f"\nERROR in token-efficiency test round: {e}")

    if args.askq_test:
        try:
            ask_tool_name = (elicited.get("ask_user_question") or {}).get("function_name")
            aq = ask_user_question_test_round(client, final_tools, ask_tool_name)
            output["askq_test"] = aq
        except Exception as e:
            if _keep_previous_result(e, previous, "askq_test"):
                output["askq_test"] = previous["askq_test"]
                print(f"\nERROR in ASKQ test round (429): {e} -- keeping previous run's result")
            else:
                output["askq_test"] = {"error": str(e)}
                print(f"\nERROR in ASKQ test round: {e}")

    if args.gram_knowledge_test:
        try:
            gkt = gram_knowledge_test_round(client)
            output["gram_knowledge_test"] = gkt
        except Exception as e:
            if _keep_previous_result(e, previous, "gram_knowledge_test"):
                output["gram_knowledge_test"] = previous["gram_knowledge_test"]
                print(f"\nERROR in APPLY_PATCH test round (429): {e} -- keeping previous run's result")
            else:
                output["gram_knowledge_test"] = {"error": str(e)}
                print(f"\nERROR in APPLY_PATCH test round: {e}")

    if args.gram_transport_test:
        try:
            gtt = gram_transport_test_round(client)
            output["gram_transport_test"] = gtt
        except Exception as e:
            if _keep_previous_result(e, previous, "gram_transport_test"):
                output["gram_transport_test"] = previous["gram_transport_test"]
                print(f"\nERROR in GRAMT test round (429): {e} -- keeping previous run's result")
            else:
                output["gram_transport_test"] = {"error": str(e)}
                print(f"\nERROR in GRAMT test round: {e}")

    if args.rjson_test:
        try:
            rt = response_format_test_round(client)
            output["rjson_test"] = rt
        except Exception as e:
            if _keep_previous_result(e, previous, "rjson_test"):
                output["rjson_test"] = previous["rjson_test"]
                print(f"\nERROR in RJSON test round (429): {e} -- keeping previous run's result")
            else:
                output["rjson_test"] = {"error": str(e)}
                print(f"\nERROR in RJSON test round: {e}")

    if args.stream_test:
        try:
            st = stream_test_round(client)
            output["stream_test"] = st
        except Exception as e:
            if _keep_previous_result(e, previous, "stream_test"):
                output["stream_test"] = previous["stream_test"]
                print(f"\nERROR in STRM test round (429): {e} -- keeping previous run's result")
            else:
                output["stream_test"] = {"error": str(e)}
                print(f"\nERROR in STRM test round: {e}")

    if args.reasoning_test:
        try:
            rst = reasoning_test_round(client)
            output["reasoning_test"] = rst
        except Exception as e:
            if _keep_previous_result(e, previous, "reasoning_test"):
                output["reasoning_test"] = previous["reasoning_test"]
                print(f"\nERROR in REASN test round (429): {e} -- keeping previous run's result")
            else:
                output["reasoning_test"] = {"error": str(e)}
                print(f"\nERROR in REASN test round: {e}")

    if args.agentknit_test:
        try:
            akt = agentknit_default_tools_test_round(client)
            output["agentknit_test"] = akt
        except Exception as e:
            if _keep_previous_result(e, previous, "agentknit_test"):
                output["agentknit_test"] = previous["agentknit_test"]
                print(f"\nERROR in AKDEF test round (429): {e} -- keeping previous run's result")
            else:
                output["agentknit_test"] = {"error": str(e)}
                print(f"\nERROR in AKDEF test round: {e}")

    if args.cache_ttl_test:
        try:
            ct = cache_ttl_test_round(client)
            output["cache_ttl_test"] = ct
        except Exception as e:
            if _keep_previous_result(e, previous, "cache_ttl_test"):
                output["cache_ttl_test"] = previous["cache_ttl_test"]
                print(f"\nERROR in CACH test round (429): {e} -- keeping previous run's result")
            else:
                output["cache_ttl_test"] = {"error": str(e)}
                print(f"\nERROR in CACH test round: {e}")

    if args.context_test or args.context_tokens is not None:
        try:
            output["context_window"] = _resolve_context_window(args, lambda: client)
        except Exception as e:
            if _keep_previous_result(e, previous, "context_window"):
                output["context_window"] = previous["context_window"]
                print(f"\nERROR in CTX test round (429): {e} -- keeping previous run's result")
            else:
                output["context_window"] = {"error": str(e)}
                print(f"\nERROR in CTX test round: {e}")

    if args.cors_test:
        try:
            output["cors_test"] = cors_test_round()
        except Exception as e:
            if _keep_previous_result(e, previous, "cors_test"):
                output["cors_test"] = previous["cors_test"]
                print(f"\nERROR in CORS test round (429): {e} -- keeping previous run's result")
            else:
                output["cors_test"] = {"error": str(e)}
                print(f"\nERROR in CORS test round: {e}")

    if args.tdef_test:
        try:
            output["tdef_test"] = tdef_test_round(api_key, args)
        except Exception as e:
            if _keep_previous_result(e, previous, "tdef_test"):
                output["tdef_test"] = previous["tdef_test"]
                print(f"\nERROR in TDEF test round (429): {e} -- keeping previous run's result")
            else:
                output["tdef_test"] = {"error": str(e)}
                print(f"\nERROR in TDEF test round: {e}")

    save()


if __name__ == "__main__":
    main()

# Capabilities reference

`probe_inference.py` scores each probed model on a fixed set of capabilities.
Every capability has a stable codename (max 5 characters) used as its key in
`tool_schema_<model>.json` and as a heading tag in `tool_schema_<model>.md`.
Codenames do not change across probe runs or script versions, so results are
comparable model-to-model and run-to-run.

Every value in a report comes straight from that run's JSON. When the JSON
carries no data for a capability at all — the report was produced before the
capability existed — the report says `no data, please rerun the probing`
instead of silently dropping the row. A capability the run *deliberately*
skipped (`--no-*`) is different: its key is present but null, and the report
says so and names the flag to drop.

## `TCALL`

**Capability:** Model emits tool invocations via the structured OpenAI
`tool_calls` API field rather than falling back to inline JSON in the
message content (or XML tags, or no detectable call at all).

- **Unit:** tasks passed / tasks run
- **Range:** 0 to N tasks (N = 1 for the round-0 single-call check, N = 8
  for the full multi-op probe)
- **Source:** `format_detection`, `behaviour`

The format-detection round includes both a normal schema-backed call and an
explicit, schema-free XML serialization task. The XML task asks for a
`<tool_call>` envelope containing a `read_file` invocation and passes only
when the envelope and its parsed name/arguments are correct; it is reported
separately in each Markdown report rather than folded into the TCALL score.

## `QUOTE`

**Capability:** Model preserves literal double-quote characters inside JSON
string arguments without dropping or mangling them. Each task advertises
*only* the one tool relevant to it (not the full schema), so a model that
would otherwise pick a different tool for the task (see `GREP`/dispatch
notes) is still exercised on quoting specifically, rather than failing here
for an unrelated tool-selection reason.

- **Unit:** tasks passed / tasks run
- **Range:** 0 to 3 tasks
- **Source:** `quote_test` (enabled with `--quote-test`)

## `GREP`

**Capability:** Model prefers a filtered/targeted call (`grep`, `sed -n`,
`head`, `wc -l`, a dedicated search tool, or a read with offset/limit) over
pulling an entire large file/output into context.

- **Unit:** tasks passed / tasks run
- **Range:** 0 to 6 tasks
- **Source:** `token_efficiency_test` (enabled with `--efficiency-test`)

## `ASKQ`

**Capability:** Model is able to ask users questions — it calls its own
`ask_user_question` tool (rather than guessing or investigating
unilaterally) when a task calls for user input, across 8 phrasings of the
same underlying "should I overwrite or back up this file?" task, all run
against the full tool schema.

- **Unit:** variants passed / variants run, plus a 5-point Likert frequency
  label
- **Range:** 0 to 8 variants → `Never` / `Rarely` / `Sometimes` / `Often` /
  `Always`
- **Source:** `askq_test` (enabled with `--askq-test`)

## `AKDEF`

**Capability:** Model understands and correctly uses agentknit's actual
shipped default tool schema -- `read_file`, `write_file`, `str_replace`,
`exec_shell` -- imported live from `agentknit._core._DEFAULT_TOOL_SCHEMA`,
not reconstructed here. All four tools are offered together, one task per
tool; each resulting call is dispatched through agentknit's real
`dispatch()` (with `_DEFAULT_TOOL_DISPATCH`'s param renames, e.g.
`old_str`/`new_str` -> `old`/`new`) against a disposable scratch directory.
A PASS requires the right tool selected *and* the correct on-disk (or
subprocess) effect -- e.g. `str_replace` must actually flip the target
file's content, `exec_shell`'s dispatched command must exit 0. This is
both a new-model compatibility check (does this model work with
agentknit's defaults at all?) and a regression test to rerun after
fine-tuning a model against agentknit's tool contract, since it tracks
`agentknit._core` directly and needs no manual schema updates when that
default schema changes.

- **Unit:** tasks passed / tasks run (tool selected AND on-disk effect correct)
- **Range:** 0 to 4 tasks
- **Source:** `agentknit_test` (enabled by default; disable with `--no-agentknit-test`; requires `agentknit` importable)

## `TSEL`

**Capability:** Model selects the right tool from the full competing schema
for each operation in Round 2. A pass requires the call's tool name to match
the tool assigned to that operation in the offered schema; calling a different
listed tool, or making no tool call, fails. This is derived from the Round-2
probe calls, not a separate probe round.

- **Unit:** correct tool selections / operations with an offered tool
- **Range:** 0 to 8 ops
- **Source:** `tsel_test` (built from `probe_round()` results; no flag needed)

## `APPLY_PATCH`

**Capability:** Model naturally knows OpenAI's `apply_patch` envelope grammar
from pretraining alone — no tool schema is offered, no
example is shown. The model is asked in free text to produce only the raw
patch, and the result is parsed against the real grammar (not a loose
regex), so this measures grammar knowledge specifically, decoupled from
whether the probing endpoint actually exposes the freeform `apply_patch`
tool.

- **Unit:** tasks passed / tasks run
- **Range:** 0 to 2 tasks
- **Source:** `gram_knowledge_test` (enabled by default; disable with `--no-gram-knowledge-test`)

## `GRAMT`

**Capability:** Endpoint honours a genuine OpenAI custom/freeform tool
(`type:"custom"`, `format:{type:"grammar", syntax:"lark", ...}`) end to end —
a `custom` tool_call comes back with grammar-valid input, rather than the
endpoint falling back to classic function calling or rejecting the request.

- **Unit:** tasks passed / tasks run
- **Range:** 0 to 1 task
- **Source:** `gram_transport_test` (enabled by default; disable with `--no-gram-transport-test`)

## `RJSON`

**Capability:** Endpoint honours strict structured output — a
`response_format:{type:"json_schema"}` request (no tool schema) is accepted
and the reply content parses as JSON conforming to the given schema. Like
`GRAMT`, this is an endpoint/provider feature, not a model behaviour.

- **Unit:** tasks passed / tasks run
- **Range:** 0 to 1 task
- **Source:** `rjson_test` (enabled by default; disable with `--no-rjson-test`)

## `STRM`

**Capability:** Endpoint delivers a `stream:true` response as real
incremental Server-Sent Events rather than rejecting the parameter or
accepting it but buffering the whole reply into a single chunk. PASS
requires more than one `ChatCompletionChunk`, non-empty content
reconstructed from `delta.content` across chunks, and a chunk carrying
`finish_reason`. An endpoint/provider feature, not a model behaviour.

- **Unit:** tasks passed / tasks run
- **Range:** 0 to 1 task
- **Source:** `stream_test` (enabled by default; disable with `--no-stream-test`)

## `REASN`

**Capability:** Two related endpoint/provider checks around reasoning
tokens:

1. `reasoning_tokens_present` — with no special params, does the reply
   carry a reasoning trace? Checked via a `reasoning_content` / `reasoning`
   / `thinking` field on the message, or a non-zero
   `usage.completion_tokens_details.reasoning_tokens`.
2. `effort_control_native` / `effort_control_extra_body` — does the
   endpoint accept a way to tune reasoning effort without a 400? Two wire
   syntaxes are tried: the native top-level `reasoning_effort` Chat
   Completions param (OpenAI o-series/gpt-5), and the OpenRouter-style
   `extra_body={"reasoning": {"effort": ...}}` passthrough many other
   providers proxy. Accepting the parameter is the bar — this does not
   attempt to prove the setting changed model behaviour, only that the
   syntax is honoured rather than rejected.

- **Unit:** checks passed / checks run
- **Range:** 0 to 3 checks
- **Source:** `reasoning_test` (enabled by default; disable with `--no-reasoning-test`)

## `TSEL`

**Capability:** Model calls the right tool when several tools are available.
Each Round-2 task is run against the full inferred schema; the call must match
the tool assigned to that task's operation. See `tsel_test` in the report for
the expected and actual tool names.

## GRAMT detail

The *endpoint* implements OpenAI's real freeform/custom-tool
transport end to end — `type: "custom"` with
`format: {type: "grammar", syntax: "lark", ...}` — rather than only classic
JSON-schema function calling. Sends the actual upstream `apply_patch.lark`
grammar (unmodified, as OpenAI/Codex define it) as a genuine custom tool;
PASS requires a real `custom` tool_call back (not a `function` one, and not
silently dropped/ignored) whose raw `input` text is grammar-valid.

This is a transport/capability question distinct from `APPLY_PATCH`: a
model can score `APPLY_PATCH` 2/2 (know the grammar cold) while its
endpoint scores `GRAMT` 0/1 (can't accept the tool that would let
it use that knowledge natively) — confirmed directly: `gpt-5-mini`'s
Copilot endpoint 400s on this request shape entirely; `gpt-5.6-luna`'s
Responses API returns a genuine `custom_tool_call` once its wrapper's
translator passes `type:"custom"` tools through instead of silently
dropping them. See `~/bin/copilot-notes.md` for the full writeup.

- **Unit:** tasks passed / tasks run
- **Range:** 0 to 1 task
- **Source:** `gram_transport_test` (enabled by default; disable with `--no-gram-transport-test`)

## Notes on scoring

## `CACH`

**Capability:** Endpoint prompt-cache TTL, measured empirically. The probe
primes the cache with a large fixed prefix (~2500 tokens), then re-sends the
exact same prefix (only a trailing question differs) after increasing delays
(30s, 120s, 300s, 600s), stopping at the first cold sample. Warmth is judged
from the provider-reported cached-token count
(`usage.prompt_cache_hit_tokens`, `usage.cached_tokens`,
`usage.prompt_tokens_details.cached_tokens`, ...): a sample is WARM when at
least half of its prompt tokens (and at least 500) come back cached.

- **Unit:** seconds / minutes (a measured range, not a pass/fail score)
- **Range:** bracketed between the last warm and the first cold delay; a TTL
  has no right or wrong value, so this capability is informational only
- **Source:** `cache_ttl_test` (off by default — takes several minutes;
  enable with `--cache-ttl-test`)

## Context window

**Metric:** The model's context-window size, recovered from the endpoint itself
rather than from documentation. Reported as `context_window` in the JSON (not a
codename-scored capability, so it has no `passed/total`), plus a header line and
a `## Context window` section in the Markdown report.

Two endpoint-owned sources are tried in order of trust:

1. **`/models` metadata** (`source: api_metadata`) — providers that publish
   per-model limits are believed first: OpenRouter and Kimi return
   `context_length`, and the Copilot API returns
   `capabilities.limits.{max_context_window_tokens,max_prompt_tokens,max_output_tokens}`
   (the extras are kept in the entry's `metadata`). The official DeepSeek and
   z.ai endpoints publish no limits at all.
2. **The endpoint's own rejection** (`source: api_error_message`) — one
   deliberately oversized prompt is sent and the limit is parsed out of the
   error body ("This model's maximum context length is 1048576 tokens",
   "prompt token count of N exceeds the limit of M"). A rejected request
   produces no tokens, so this costs one request and nothing else. A smaller
   probe is only attempted when the failure was a transport/body-size error
   (413) that never reached the model; a context rejection that names no number
   ends the search, because a smaller prompt could only be *accepted* and yield
   a useless lower bound.

- **Kind:** `context_window` (prompt plus completion, as DeepSeek, OpenRouter,
  Kimi and z.ai describe it) or `max_prompt_tokens` (the Copilot API's *prompt*
  ceiling, which is the binding number for a request). The two are kept apart
  because silently mixing them would make reports incomparable.
- **Not recovered?** An endpoint that neither advertises limits nor names them
  in its errors (z.ai answers an oversized prompt with a bare `Prompt exceeds
  max length`) leaves the entry unknown; record a value by hand with
  `--context-tokens N --context-source ... --context-evidence ...`, which stores
  the number verbatim with its provenance.
- **Escalation guard:** if the oversized prompt is *accepted*, the value is
  recorded as a lower bound (`lower_bound: true`) and nothing larger is sent — a
  prompt that big is billed for real, so the probe never escalates against a
  model with a huge window.
- **Source:** `context_window` (on by default; pass `--no-context-test` to
  skip). `--context-only` refreshes just this field on an existing report.

- All ten capabilities are pass/fail per task; the reported value is
  `passed/total`, not a normalized score. Compare N against the same
  codename's range before comparing two models' fractions.
- `TCALL` is measured twice at different granularity: once as a single
  boolean check in round 0 (`format_detection.has_structured_tool_calls`),
  and once as a per-task breakdown across the 8-op probe
  (`behaviour.structured_tool_calls` / `.inline_json_in_content` /
  `.no_call_detected`). The markdown report shows both under one `TCALL`
  heading.
- `QUOTE`, `GREP`, `ASKQ`, `APPLY_PATCH`, `GRAMT`, `RJSON`, `STRM`, `REASN`, and
  `AKDEF` are on by default (`--quote-test` / `--efficiency-test` /
  `--askq-test` / `--gram-knowledge-test` / `--gram-transport-test` /
  `--rjson-test` / `--stream-test` / `--reasoning-test` /
  `--agentknit-test`); pass the `--no-*` form of any flag to skip it, in
  which case its JSON field is `null` and the markdown report omits the
  section. `TSEL` has no flag — it's always available once the main probe
  runs.
- The context window is data, not a score: it lives in the top-level
  `context_window` field (see "Context window" above), is always attempted
  (`--context-test`, on by default), and is refreshed on an existing report
  with `--context-only`.
- Unlike every other test, `AKDEF` actually executes the model's tool
  calls (writes files, runs a shell command) against a disposable temp
  directory via agentknit's real `dispatch()`, rather than only inspecting
  the call shape. Its per-item failures are tagged `AKDEF_<tool>`.
- `TSEL` is scored directly from the full-schema Round-2 calls. Its per-item
  failures are tagged `TSEL_<op>`, consistently with `QUOTE_<op>`,
  `GREP_<op>`, and `ASKQ_<variant>`.
- A `FAIL` on `GREP` does not mean the model got the *answer* wrong — the
  probe never executes the tool call, so correctness of results is out of
  scope. It only measures whether the model *chose* a token-cheap call
  shape (see `probe_inference.py::_classify_bash_command` /
  `_classify_read_args` for the exact heuristics).
- `ASKQ` deliberately runs each of its 8 phrasings **once**, with no retries.
  This is by design, not a shortcut: retrying to smooth out sampling noise
  would hide the fact that borderline phrasings can flip pass/fail run to
  run even at temperature 0 (observed directly while developing this test —
  the exact same phrasing and schema produced ASKED in one run and SKIPPED
  in another). `ASKQ` measures sensitivity to *phrasing*, across a fixed set
  of phrasings chosen to span explicit-imperative through
  implicit/ambiguous framing; it is not a statistically robust reliability
  estimate for any single phrasing. A low `ASKQ` score means "this model's
  ask-the-user behavior is phrasing-fragile," not "this model never asks."
- `CACH` is the only capability that is off by default: measuring a TTL
  takes several minutes of real waiting (one prime plus the delay of the
  first cold rung). Enable it with `--cache-ttl-test`. Unlike every other
  capability it is informational, not pass/fail, so it never appears in the
  "Missing capabilities" section when skipped.
- `CORS` is the only capability measured without a model behind it: it is a
  property of the endpoint's HTTP layer, not of the model, and its two
  requests carry no Authorization at all, so `--cors-only` refreshes an
  existing report with no credentials needed.

## CORS

**Capability:** Endpoint answers the CORS preflight a browser must clear
before calling it cross-origin, so a plain web page (no backend proxy) can
integrate with it directly. Reported as `cors_test` in the JSON plus a
`CORS` row in the capabilities table and a `## CORS preflight test` section
in the Markdown report.

- **Unit:** the `access-control-allow-origin` answer, verbatim
- **Range:** `*` (wildcard — any origin), the reflected request origin
  (browser-direct works, with or without `access-control-allow-credentials`),
  or no header at all (a browser cannot read responses from this endpoint)
- **Source:** `cors_test` (enabled by default; disable with `--no-cors-test`;
  refresh an existing report with `--cors-only`)

The probe replays exactly the handshake a browser performs for an SDK call:
OPTIONS `/chat/completions` with `Origin` and
`Access-Control-Request-{Method: POST, Headers: authorization, content-type}`,
then a plain Origin-tagged GET `/models` — because some endpoints answer
the preflight yet strip the CORS headers from the actual response, which
breaks the call one request later. `preflight_passed` additionally requires
`access-control-allow-methods` to cover POST and `access-control-allow-headers`
to cover both `authorization` and `content-type`; a preflight missing those
fails in the browser exactly like a missing origin does. The two requests
carry no Authorization, so this round needs no credentials.

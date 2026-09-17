# The Responses API

`POST {base}/responses` — OpenAI's second-generation inference API
([reference](https://platform.openai.com/docs/api-reference/responses)),
introduced 2025, and the surface on which OpenAI ships its newest features:
custom/grammar tools, tool search, server-side conversation state.

Several non-OpenAI vendors expose a Responses-shaped endpoint too, usually
because [Codex](https://developers.openai.com/codex/local-config) requires one
(`wire_api = "responses"`). Those are compatibility layers of varying depth,
and that is where most of Part 2 comes from.

- **[Part 1 — The common API](#part-1--the-common-api)**: what works the same
  everywhere a Responses surface exists at all.
- **[Part 2 — Provider quirks](#part-2--provider-quirks)**: where each vendor's
  layer stops, and what it does instead of saying so.

Part 2 records behaviour observed by calling the endpoints, not behaviour read
from vendor documentation. Where the two disagree, the observation is what is
written down.

---

## Part 1 — The common API

### Transport

```
POST {base}/responses
Authorization: Bearer <key>
Content-Type: application/json
```

The path is not reliably a sibling of the vendor's other routes. Locations
observed:

| Vendor | Responses base | Notes |
|---|---|---|
| OpenAI | `https://api.openai.com/v1` | reference implementation |
| OpenAI Codex (ChatGPT OAuth) | `https://chatgpt.com/backend-api/codex` | full path `.../codex/responses` |
| DeepSeek | `https://api.deepseek.com` (and `/v1`) | `deepseek-flash` only |
| Kimi | `https://api.moonshot.ai/v1`, `https://api.kimi.com/coding/v1` | |
| z.ai | `https://api.z.ai/api/v1` | documented only for Codex, not in the API reference |
| OpenRouter | `https://openrouter.ai/api/v1` | |
| GitHub Copilot | base handed out at auth time | `{api_base}/responses` |

### Request body

```json
{
  "model": "gpt-5.4",
  "instructions": "You are a helpful assistant.",
  "input": [
    {"type": "message", "role": "user",
     "content": [{"type": "input_text", "text": "What is 2+2?"}]}
  ],
  "max_output_tokens": 1024,
  "stream": false,
  "store": false
}
```

| Field | Meaning |
|---|---|
| `model` | required |
| `input` | required — an array of typed items, or a plain string |
| `instructions` | the system prompt, as a top-level field rather than an input item |
| `max_output_tokens` | cap on the generated output |
| `text.format` | structured-output constraint (`json_schema`, `json_object`, `text`) |
| `reasoning.effort` | `minimal` \| `low` \| `medium` \| `high` |
| `tools`, `tool_choice` | see [Tools](#tools) |
| `store` | server-side retention of the response |
| `previous_response_id` | threads onto a stored response instead of resending history |
| `stream` | see [Streaming](#streaming) |

`store` and `previous_response_id` are the pair that makes server-side
threading possible, and the pair least likely to be honoured — see Part 2.

### Input items

`input` is an array of typed items, not of messages. The distinction matters: a
single assistant turn that produced text *and* two tool calls is **three**
items, and the API rejects a `function_call` nested inside a message's
`content`:

```json
{"type": "message", "role": "user",
 "content": [{"type": "input_text", "text": "..."}]}

{"type": "message", "role": "assistant",
 "content": [{"type": "output_text", "text": "..."}]}

{"type": "function_call", "call_id": "call_abc",
 "name": "read_file", "arguments": "{\"path\":\"/tmp/x\"}"}

{"type": "function_call_output", "call_id": "call_abc", "output": "<result>"}
```

The content-part type flips by direction: `input_text` on user and system
turns, `output_text` on assistant turns. Mixing them up is the most common
construction error.

`function_call.arguments` is a JSON-encoded string, and
`function_call_output.output` must be a string too — a structured tool result
has to be serialised before it goes back in.

`call_id` is what ties a call to its output. It is a distinct field from the
output item's own `id`, and it is the one that must round-trip.

### Tools

```json
"tools": [
  {"type": "function",
   "name": "read_file",
   "description": "Read a file from disk.",
   "parameters": {"type": "object", "properties": {...}, "required": [...]}}
]
```

Tool definitions are flat: `type`, `name`, `description` and `parameters` sit
side by side, with no wrapper object.

**Custom / grammar tools**
([guide](https://platform.openai.com/docs/guides/function-calling#custom-tools))
take raw text instead of a JSON object as their argument, optionally
constrained by a Lark or regex grammar with genuine constrained decoding:

```json
{"type": "custom",
 "name": "apply_patch",
 "description": "...",
 "format": {"type": "grammar", "syntax": "lark", "definition": "<lark>"}}
```

Here too the shape is flat — `syntax` and `definition` sit directly under
`format`. The flagship example is `apply_patch`, whose grammar is
[`apply_patch.lark`](https://github.com/openai/codex/blob/main/codex-rs/core/src/tools/handlers/apply_patch.lark)
in `openai/codex`, with a reference host-side implementation in the
[OpenAI cookbook](https://github.com/openai/openai-cookbook/blob/main/examples/gpt-5/apply_patch.py).
The rationale — JSON escaping is expensive and error-prone for multi-line
content, and a grammar makes malformed output structurally impossible rather
than merely unlikely — is in the
[GPT-5 cookbook](https://developers.openai.com/cookbook/examples/gpt-5/gpt-5_new_params_and_tools).

### Output

```json
{
  "id": "resp_...",
  "model": "gpt-5.4",
  "created_at": 1726500000,
  "output": [
    {"type": "message", "role": "assistant",
     "content": [{"type": "output_text", "text": "4"}]},
    {"type": "function_call", "call_id": "call_abc",
     "name": "read_file", "arguments": "{\"path\":\"/tmp/x\"}"},
    {"type": "custom_tool_call", "call_id": "call_def",
     "name": "apply_patch", "status": "completed",
     "input": "*** Begin Patch\n...\n*** End Patch\n"}
  ],
  "usage": {"input_tokens": 12, "output_tokens": 3, "total_tokens": 15,
            "input_tokens_details": {"cached_tokens": 0}}
}
```

`output` is a flat list of items: text and tool calls are siblings, so a reply
carrying both is read by iterating, not by looking for a single field. A
`custom_tool_call` carries `input` as **raw text** — no JSON envelope, no
escaping.

### Token accounting

```json
"usage": {
  "input_tokens": 4893,
  "output_tokens": 52,
  "total_tokens": 4945,
  "input_tokens_details": {"cached_tokens": 2048, "cache_write_tokens": 0},
  "output_tokens_details": {"reasoning_tokens": 180}
}
```

- `input_tokens` is the **grand total** of the prompt; `cached_tokens` is a
  **subset** of it, the part served from cache. Adding them double-counts.
- `cache_write_tokens`, where reported, is the part written into the cache on
  this request.
- `reasoning_tokens` is part of `output_tokens`, not additional to it.
- `total_tokens` = `input_tokens` + `output_tokens`.

Prompt caching is automatic and has no request field
([guide](https://developers.openai.com/api/docs/guides/prompt-caching)); the
counters above are the only evidence it happened. They are also the only
reliable way to verify that an optional request field did anything at all —
see [HTTP 200 proves nothing](#cross-cutting-trap-http-200-proves-nothing).

### Streaming

`stream: true` yields SSE lines carrying typed events:

| Event | Meaning |
|---|---|
| `response.created` | the response object exists, nothing generated yet |
| `response.output_text.delta` | `event.delta` is the next text fragment |
| `response.output_item.done` | `event.item` is one completed output item — this is where tool calls arrive whole |
| `response.completed` | `event.response` is the full final object, `usage` included |
| `response.failed`, `error` | terminal failure |

A client that only wants the final answer can ignore everything but
`response.completed`; a client that needs tool calls as they finish reads
`response.output_item.done`.

---

## Part 2 — Provider quirks

### Cross-cutting trap: HTTP 200 proves nothing

**Silently dropping unknown request fields is the cheap way to build a
compatibility layer**, and most compatibility layers do exactly that. DeepSeek
[documents it outright](https://api-docs.deepseek.com/guides/responses_api/):
unsupported parameters are ignored rather than rejected.

So a `200` on a request carrying a feature flag is not evidence the feature
ran, and the absence of an error message is not a negative result either. Every
finding below that says "accepted, ignored" was established by measuring an
effect — usually input-token counts — not by reading a status code.

### Is the surface even there?

A `404` on `/responses` is a fact about the vendor's deployment, not about the
protocol. Guessing the Responses path from a vendor's other working base URL
fails often enough to be worth avoiding: z.ai, for instance, serves it from a
completely different path prefix than its other routes.

### Deferred tool loading

[Tool search](https://developers.openai.com/api/docs/guides/tools-tool-search)
keeps a tool's parameter schema out of the prompt until the model asks for it
(`gpt-5.4` and later):

```json
"tools": [
  {"type": "function", "name": "warehouse_inventory_lookup",
   "description": "...", "parameters": {...}, "defer_loading": true},
  {"type": "tool_search"}
]
```

A deferred tool's name and description still reach the model; only the
parameter schema is withheld. Loaded tools are appended at the end of the
context, so the cached prefix survives.

Measured by sending the same request three ways — no tools, all schemas inline,
most of them deferred — comparing the endpoint's own reported input tokens, and
then asking for something only a deferred tool can answer:

| Endpoint | Result | Evidence |
|---|---|---|
| OpenRouter | **native** | 4,893 input tokens inline → 1,567 deferred; the model emitted an `openrouter:tool_search` item with query `stock\|inventory\|on-hand\|sku\|warehouse`, then called the deferred tool |
| DeepSeek | accepted, ignored | 4,893 inline vs 4,893 deferred — a difference of exactly zero against a measured schema cost of 4,856 — and the model called a deferred tool directly, with no search step |
| Kimi | rejected | `400 tools.0.defer_loading is not supported` |
| z.ai | unsettled | `429 Weekly/Monthly Limit Exhausted` |

Three findings worth carrying forward:

1. **An aggregator can implement the feature above the provider.** OpenRouter is
   native through Fireworks, Together, Parasail, DeepInfra, Novita and Phala —
   the deferral happens in OpenRouter's own layer, in front of a model
   (`deepseek/deepseek-v4.1-flash`) whose own vendor does not support the
   feature at all.
2. **The search item is vendor-namespaced.** OpenRouter emits
   `"type": "openrouter:tool_search"`, not `"tool_search"`. A client matching a
   fixed string will miss it; match any item type *containing* `tool_search`.
3. **A 200 is worthless here.** The only reliable detector is the input-token
   count, corroborated by whether the model reached a deferred tool without
   searching first.

BaseTen, pinned through OpenRouter, was rate-limited upstream and returned no
answer. That is an error, not a negative — treating a missing control request as
"the schemas stayed in" would produce a confident wrong verdict.

### DeepSeek's published compatibility list

DeepSeek is the only vendor here publishing an explicit accepted/ignored split
([docs](https://api-docs.deepseek.com/guides/responses_api/)):

- **Supported:** `model`, `input`, `instructions`, `stream`, `temperature`,
  `top_p`, `max_output_tokens`, `top_logprobs`, `tools` (function only),
  `tool_choice`, `reasoning.effort`, `user`
- **Ignored — accepted, no effect:** `parallel_tool_calls`, `max_tool_calls`,
  `previous_response_id`, `conversation`, `store`, `background`, `metadata`,
  `include`, `prompt`, `truncation`, `service_tier`, `prompt_cache_key`,
  `stream_options`
- **Tools:** `function` supported; custom `apply_patch` supported; built-ins
  such as `web_search` ignored
- Served for `deepseek-flash` only

The ignored column is the interesting one: `store` and `previous_response_id`
being ignored means **server-side conversation threading silently does
nothing**. Every turn must carry the full history, and a client built on
`previous_response_id` will appear to work while losing all context.

### Kimi's accepted tool types

Kimi's [Responses docs](https://platform.kimi.ai/docs/api/responses) enumerate
what they accept and state that other types are not supported:

- `function`
- `namespace`
- `custom` — **only `apply_patch`**
- `web_search`

`namespace` is the notable one: it is the construct OpenAI recommends deferring
behind, so the *grouping* half of tool search exists on Kimi even though the
deferral half is explicitly rejected (`tools.0.defer_loading is not
supported`). Whether a namespace actually collapses in the prompt or is merely
a grouping convenience is measurable, by the same token-accounting method used
above.

**Trap:** Kimi's `/v1/tools/search` is a hosted **web** search endpoint. It has
nothing to do with `tool_search`. A keyword scan of their docs will hit it; it
is not tool-search support.

### The OpenAI Codex backend (`chatgpt.com/backend-api/codex`)

Reachable with the ChatGPT OAuth token that `codex login` stores locally. Not
the public API, and it differs in ways that break a naive client:

- **`system`-role input items are rejected** — `400 {"detail":"System messages
  are not allowed"}`. The system prompt must go in `instructions`.
- **`temperature` is rejected** — `Unsupported parameter` on the gpt-5.x
  reasoning models.
- **`stream: true` is effectively mandatory**; the final object has to be
  reconstructed from the event stream.
- **`store: false`** is expected.
- Required headers: `Authorization: Bearer <access_token>`,
  `chatgpt-account-id`, `OpenAI-Beta: responses=experimental`,
  `originator: codex_cli_rs`, and a fresh `session_id` (UUID) per request.
- The access token is a JWT; refresh it against
  `https://auth.openai.com/oauth/token` with the Codex client id before its
  `exp`.

Server-side tools such as `code_interpreter` are **not** available on this
backend; they need the public `https://api.openai.com/v1/responses` with a real
API key.

### GitHub Copilot: the backend can be better than the layer in front of it

Copilot's Responses surface supports custom/grammar tools natively. Sent the
real `apply_patch` Lark grammar directly to `{api_base}/responses`, it returns
a genuine call:

```json
{"type": "custom_tool_call", "name": "apply_patch", "status": "completed",
 "input": "*** Begin Patch\n*** Update File: /tmp/test.py\n@@\n-x = 1\n+x = 42\n*** End Patch\n"}
```

The same request routed through an intermediary produced plain prose with the
patch syntax dumped inline — because that intermediary's tool-list filter kept
only entries of `type == "function"` and dropped everything else *before the
request was sent*. A silently-filtering proxy is indistinguishable, from the
caller's side, from an endpoint that does not support the feature.

That is the general hazard: **any capability measured through an intermediary
measures the intermediary**. The same endpoint and the same model went from "no
custom-tool support" to a correct `custom_tool_call` with no change other than
removing the filter.

Copilot also enforces a *prompt* ceiling rather than a context window: `922000`
tokens for `gpt-5.6-luna`, `128000` for `gpt-5-mini`, both recovered from
`exceeds the limit of N` rejections.

### z.ai

The Responses endpoint (`https://api.z.ai/api/v1`) exists but is documented
only in the [Codex setup](https://docs.z.ai/devpack/tool/codex) pages, not in
the [API reference](https://docs.z.ai/api-reference/introduction) — whose
capability index lists Function Calling, Context Caching and Structured Output
and no Responses page at all. Which parameters it honours is therefore doubly
worth confirming by calling it.

Its rate-limit error puts the reset time in the message body (`Your limit will
reset at 2026-09-21 00:27:25`) rather than in a `Retry-After` header.

### Aggregators: one URL is not one endpoint

[OpenRouter](https://openrouter.ai/docs/features/provider-routing) routes a
single model id to a dozen upstream providers running different inference
software under different configuration, and they disagree. Pin routing with:

```json
{"provider": {"order": ["fireworks"], "allow_fallbacks": false}}
```

Even token counts differ by upstream: the same four tool schemas cost 4,856
input tokens through Fireworks, Together, Parasail and DeepInfra, and 4,800
through Novita and Phala — different tokeniser configuration behind one model
id. An unpinned measurement is a measurement of whoever answered.

**Trap:** OpenRouter answers a pin it cannot satisfy with `404 No endpoints
found for <model>` plus a `routing_funnel` — the same status an unimplemented
path returns. Read naively, a broken pin becomes a confident "this surface is
not served here".

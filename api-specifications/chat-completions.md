# The Chat Completions API

`POST {base}/chat/completions` — OpenAI's 2023 wire format
([reference](https://platform.openai.com/docs/api-reference/chat)), and the
format that the endpoints documented here all serve, several of them serving
no other.

This document has two parts:

- **[Part 1 — The common API](#part-1--the-common-api)**: the subset that works,
  unmodified, on every endpoint tested. Safe to depend on.
- **[Part 2 — Provider quirks](#part-2--provider-quirks)**: everything else —
  fields that are rejected, fields that are *silently swallowed*, undocumented
  extensions, and per-provider dialects of the same error.

Part 2 records behaviour observed by calling the endpoints, not behaviour read
from vendor documentation. Where the two disagree, the observation is what is
written down.

---

## Part 1 — The common API

### Transport

```
POST {base}/chat/completions
Authorization: Bearer <key>
Content-Type: application/json
```

`{base}` is the base URL, **not** including `/chat/completions` — the OpenAI
SDKs append that themselves. Providers disagree on whether a `/v1` segment
belongs in the base: `https://api.deepseek.com` and
`https://api.deepseek.com/v1` both work, while
`https://api.z.ai/api/coding/paas/v4` carries its version in the path.

`GET {base}/models` is the one auxiliary route that is universally served. It
returns `{"data": [{"id": ..., ...}, ...]}`. What *else* is in each entry is
provider-specific (see [Model metadata](#model-metadata)).

### Request body

Universally accepted:

| Field | Type | Notes |
|---|---|---|
| `model` | string | Required. |
| `messages` | array | Required. See [Messages](#messages). |
| `max_tokens` | int | Cap on the completion. Deprecated upstream in favour of `max_completion_tokens`, but still the portable spelling. |
| `temperature` | number | 0–2. See the Kimi quirk in Part 2. |
| `top_p` | number | Nucleus sampling. |
| `stream` | bool | See [Streaming](#streaming). |
| `stop` | string \| array | Up to 4 stop sequences. |
| `tools` | array | See [Tools](#tools). |
| `tool_choice` | string \| object | `"auto"` \| `"none"` \| `"required"` \| `{"type":"function","function":{"name":...}}`. |

Accepted-but-unreliable across providers: `n`, `seed`, `logprobs`,
`top_logprobs`, `frequency_penalty`, `presence_penalty`,
`parallel_tool_calls`, `response_format`, `reasoning_effort`, `user`. None of
these should be assumed portable; several are in Part 2.

### Messages

Four roles, all universal:

```json
{"role": "system",    "content": "You are a helpful assistant."}
{"role": "user",      "content": "What is 2+2?"}
{"role": "assistant", "content": "4"}
{"role": "tool",      "content": "<result>", "tool_call_id": "call_abc"}
```

- `content` is a string, or an array of typed parts
  (`{"type":"text","text":...}`, `{"type":"image_url", ...}`) for multimodal
  input. The *string* form is the portable one; part arrays are honoured
  unevenly, and any layer that flattens a conversation into a single prompt
  keeps only the `text` parts.
- An assistant turn that called tools carries `content: null` plus
  `tool_calls`; each subsequent `tool` message answers one call by
  `tool_call_id`.
- A `system` message is placed first. It is *not* universally
  supported — see [Non-conforming backends](#non-http-and-non-conforming-backends)
  in Part 2, where one 400s on it outright.

### Tools

One tool type is portable: `function`
([guide](https://platform.openai.com/docs/guides/function-calling)).

```json
"tools": [
  {"type": "function",
   "function": {
     "name": "read_file",
     "description": "Read a file from disk.",
     "parameters": {"type": "object",
                    "properties": {"path": {"type": "string"}},
                    "required": ["path"]}
   }}
]
```

`parameters` is a JSON Schema object. `strict: true` (schema-constrained tool
arguments) is an OpenAI extension and is *not* portable.

The model answers with structured calls on the message:

```json
{"role": "assistant", "content": null,
 "tool_calls": [
   {"id": "call_abc", "type": "function",
    "function": {"name": "read_file", "arguments": "{\"path\":\"/tmp/x\"}"}}
 ]}
```

`arguments` is a **JSON-encoded string**, not an object. It must be parsed
before use, and it is the model that produced it — so it can be malformed, and
a client that assumes well-formed JSON here will crash on real traffic.

A client must also be prepared for the model *not* to use this channel at all:
some models answer with inline JSON or an XML `<tool_call>` envelope in
`content` instead. That is a model property rather than a transport one, but it
is frequent enough that "no `tool_calls` field" cannot be read as "no tool call
intended".

### Response

```json
{
  "id": "chatcmpl-...",
  "object": "chat.completion",
  "created": 1726500000,
  "model": "deepseek-v4-flash",
  "choices": [
    {"index": 0,
     "message": {"role": "assistant", "content": "4"},
     "finish_reason": "stop"}
  ],
  "usage": {"prompt_tokens": 12, "completion_tokens": 3, "total_tokens": 15}
}
```

`finish_reason` ∈ `stop` | `length` | `tool_calls` | `content_filter`.

Only `choices[0].message` and `usage` are genuinely universal. `object`,
`created` and `id` are omitted by some compatibility layers — GitHub Copilot's
API, for one, omits `object` and `created` — so a strict client-side model of
the response will reject valid replies.

### Token accounting

```json
"usage": {
  "prompt_tokens": 2265,
  "completion_tokens": 52,
  "total_tokens": 2317,
  "prompt_tokens_details": {"cached_tokens": 2048},
  "completion_tokens_details": {"reasoning_tokens": 180}
}
```

- `prompt_tokens` is the **grand total** of the input, and `cached_tokens` is a
  **subset** of it — not an additional amount. Summing the two double-counts
  the cached prefix.
- `total_tokens` = `prompt_tokens` + `completion_tokens`.
- `reasoning_tokens`, where reported, is part of `completion_tokens`.
- The two `*_details` objects are the standard spelling but not a universal
  one; see [Prompt caching](#prompt-caching) for the other field names the same
  number arrives under.
- In a streamed response, `usage` is absent from every chunk unless
  `stream_options: {"include_usage": true}` is sent, and that field is not
  portable.

Token counts are the only lever a client has for verifying that an optional
request field did anything — see
[HTTP 200 proves nothing](#cross-cutting-trap-http-200-proves-nothing).

### Streaming

`stream: true` turns the response into `text/event-stream`:

```
data: {"object":"chat.completion.chunk","choices":[{"index":0,"delta":{"content":"1"},"finish_reason":null}]}
data: {"object":"chat.completion.chunk","choices":[{"index":0,"delta":{},"finish_reason":"stop"}]}
data: [DONE]
```

- The first chunk carries `delta.role`; subsequent ones carry `delta.content`.
- Tool calls stream as `delta.tool_calls[]` with an `index`, and the
  `function.arguments` string arrives in fragments that must be concatenated
  per index.

Accepting `stream: true` is not the same as streaming. Three things separate a
real implementation from a buffered one: more than one chunk, non-empty content
reconstructed from `delta.content` across chunks, and a chunk carrying
`finish_reason`. An endpoint that returns the whole reply as a single chunk
satisfies the HTTP contract and nothing else.

### Errors

The portable shape is:

```json
{"error": {"message": "...", "type": "invalid_request_error",
           "param": null, "code": "invalid_request_error"}}
```

with an HTTP 4xx/5xx status. Beyond that, nothing is portable — see
[Error dialects](#error-dialects-the-same-rejection-in-six-languages).

Rate limiting is `429`. `Retry-After` is sent by GitHub Copilot; z.ai instead
puts the reset timestamp in the message body.

### Model metadata

`GET {base}/models` entries carry a per-model context limit on some providers
and nothing at all on others. The field names seen in the wild, most specific
first: `max_context_window_tokens`, `context_length`, `context_window`,
`context_window_tokens`, `max_context_length`, `max_context_tokens`,
`max_prompt_tokens`, `input_token_limit`.

| Provider | Publishes a limit? | Field |
|---|---|---|
| OpenRouter | yes | `context_length` |
| Kimi (`api.kimi.com/coding/v1`) | yes | `context_length` |
| GitHub Copilot | yes | `capabilities.limits.{max_context_window_tokens,max_prompt_tokens,max_output_tokens}` |
| DeepSeek | no | — |
| z.ai | no | — |

When metadata is absent, the only endpoint-owned source left is the endpoint's
own rejection message, which is itself a per-provider dialect (see
[Context-window rejections](#context-window-rejections)).

---

## Part 2 — Provider quirks

### Cross-cutting trap: HTTP 200 proves nothing

**Silently dropping unknown request fields is the cheap way to build a
compatibility layer**, and most compatibility layers do exactly that. DeepSeek
[documents it outright](https://api-docs.deepseek.com/guides/responses_api/):
unsupported parameters are ignored rather than rejected.

The consequence is that a `200` on a request carrying a feature flag is not
evidence the feature ran. Measured instance: an endpoint that accepts a
schema-withholding flag on four tool definitions bills 4,893 input tokens both
with the schemas present and with three of four marked withheld — a difference
of exactly zero, against a measured schema cost of 4,856 tokens.

Rule of thumb: any optional field must be verified by its *effect* — token
counts, response shape, observable behaviour — never by the absence of an
error.

### Tool types beyond `function`

OpenAI's GPT-5 family added
[custom tools](https://platform.openai.com/docs/guides/function-calling#custom-tools)
— a tool whose argument is raw text, optionally constrained by a Lark or regex
grammar, with genuine constrained decoding. Here the parameters nest under a
`custom` key:

```json
{"type": "custom",
 "custom": {"name": "apply_patch",
            "description": "...",
            "format": {"type": "grammar",
                       "grammar": {"syntax": "lark", "definition": "<lark>"}}}}
```

and the answer comes back as
`{"id":..., "type":"custom", "custom": {"name":..., "input": "<raw text>"}}`.

The flagship example is `apply_patch`, whose grammar is
[`apply_patch.lark`](https://github.com/openai/codex/blob/main/codex-rs/core/src/tools/handlers/apply_patch.lark)
in `openai/codex`; the rationale — JSON escaping is expensive and error-prone
for multi-line content, and a grammar makes malformed output structurally
impossible rather than merely unlikely — is spelled out in the
[GPT-5 cookbook](https://developers.openai.com/cookbook/examples/gpt-5/gpt-5_new_params_and_tools).

**Measured support: zero endpoints.** Every one rejects it, and each in its own
dialect — which is itself the useful finding, because a client cannot
pattern-match on one message:

| Endpoint | HTTP | Message |
|---|---|---|
| DeepSeek | 400 | ``Failed to deserialize the JSON body into the target type: tools[0].type: unknown variant `custom`, expected `function` at line 1 column 343`` |
| Kimi | 400 | `unknown tool type: custom, currently only function and plugin are supported` |
| z.ai | 400 | `{"code":"1214","message":"tools[0].type:type is illegal"}` |
| OpenRouter → DeepInfra | 422 | `Provider returned error` wrapping pydantic's `Input should be 'function'` + `Field required` on `body.tools.0.function` |
| GitHub Copilot | 400 | `Invalid 'tools[0].function.name': empty string.` — validates every entry as if it were a function, whatever `type` says |

Kimi's message names the one non-OpenAI tool type seen in the wild: `plugin`.

The Copilot case is worth singling out because the error is actively
misleading: it names `tools[0].function.name` regardless of what was sent, so
it reads like a malformed function tool rather than an unrecognised tool type.
The backend is pinned to a schema version predating GPT-5 custom tools.

A model that *knows* a grammar can still be driven through a classic function
tool. `gpt-5-mini` degrades gracefully: given a plain
`apply_patch(patch: string)` JSON tool, it JSON-encodes its native patch
envelope as the string value. Knowing the grammar and being able to use it
natively are independent properties — a model can do the first on an endpoint
that cannot carry the second.

### `response_format` and structured output

`response_format: {"type":"json_schema", "json_schema":{"name","strict","schema"}}`
is supported unevenly, and there are **three** distinct behaviours, not two:

| Endpoint | Behaviour |
|---|---|
| Kimi, OpenRouter, GitHub Copilot | accepted, output conforms |
| DeepSeek | rejected: `400 This response_format type is unavailable now` |
| z.ai | **accepted and ignored** — request succeeds, content comes back as the bare word `hello`, not JSON |

The z.ai row is the dangerous one: the request is well-formed, the response is
a 200, and a client that parses the content as JSON crashes at runtime rather
than at request time.

### Reasoning tokens and effort control

Two independent things, both provider-specific.

**Exposure.** A reasoning trace comes back on a non-standard field. Spellings
seen: `reasoning_content`, `reasoning`, `thinking`, `thought`, plus
`usage.completion_tokens_details.reasoning_tokens`.

| Endpoint | Field |
|---|---|
| DeepSeek (all models) | `reasoning_content` |
| Kimi (all models) | `reasoning_content` |
| OpenRouter | `reasoning` |
| GitHub Copilot (`gpt-5-mini`, `gpt-5.6-luna`) | none exposed |

**Effort control.** Two wire syntaxes exist in the wild:

```jsonc
// OpenAI native, top-level param (o-series / gpt-5)
{"reasoning_effort": "high"}

// OpenRouter-style passthrough, proxied by many other providers
{"reasoning": {"effort": "high"}}          // SDK: extra_body=
```

Both were **accepted without error by every endpoint tested**, including ones
that expose no reasoning at all. That is the silent-swallow problem again:
acceptance is the only thing measurable from the wire, and it does not prove
the setting changed anything.

### Temperature

`api.kimi.com/coding/*` **permits only its default temperature of 1**. A client
that standardises on `temperature: 0` for reproducibility must carve out an
exception for it.

The OpenAI Codex backend rejects `temperature` outright with `Unsupported
parameter` on its gpt-5.x reasoning models.

### Prompt caching

Caching is automatic and invisible in the request on every endpoint tested —
there is no field to turn it on. It shows up only in `usage`, under four
different names for the same quantity:

- `usage.prompt_tokens_details.cached_tokens` (the standard spelling)
- `usage.prompt_cache_hit_tokens` (DeepSeek)
- `usage.cached_tokens`
- `usage.cache_read_input_tokens`

A client that reads only one of these silently reports zero cache hits on the
providers that use another.

Measured TTLs, by re-sending an identical ~2,500-token prefix after increasing
delays and watching the reported cached-token count collapse:

| Endpoint | TTL |
|---|---|
| DeepSeek `deepseek-flash`, `deepseek-v4-flash` | > 10 min (still warm at the last rung tested) |
| DeepSeek `deepseek-v4-pro` | no cache reuse even on an immediate re-send |
| Kimi `k3` | > 10 min |
| OpenRouter → DeepInfra | 5–10 min |

### Aggregators: one URL is not one endpoint

[OpenRouter](https://openrouter.ai/docs/features/provider-routing) routes a
single model id to a dozen upstream providers running different inference
software under different configuration. **They disagree**, and two consecutive
unpinned runs of the same model on the same URL have returned different
capability verdicts.

Pin with:

```json
{"provider": {"order": ["fireworks"], "allow_fallbacks": false}}
```

Measured disagreements on one model id (`deepseek/deepseek-v4.1-flash`):

- The same four tool schemas cost **4,856** input tokens through Fireworks,
  Together, Parasail and DeepInfra, and **4,800** through Novita and Phala —
  different tokeniser configuration behind one name.
- The same rejection arrives in a different dialect per provider (below).
- BaseTen was rate-limited upstream and returned no answer at all.

OpenRouter echoes the upstream it picked in a top-level `provider` field on the
response, which is the only way to tell "cache expired" apart from "a different
upstream answered".

**Trap:** OpenRouter answers a pin it cannot satisfy with `404 No endpoints
found for <model>` plus a `routing_funnel` — the same status an unimplemented
path returns. Read naively, a broken pin becomes a confident "this feature is
not served here".

### Error dialects: the same rejection in six languages

One request — a `tools` array whose last entry carries a `type` the endpoint
does not know, and therefore no `function` member — produces structurally
different rejections everywhere:

| Provider | Message |
|---|---|
| DeepSeek | ``tools[4].type: unknown variant `tool_search`, expected `function` `` (serde) |
| Kimi | `unknown tool type: tool_search, currently only function and plugin are supported` |
| z.ai | `1214 tools[4].type:type is illegal` — names the slot, not the field |
| DeepInfra (via OpenRouter) | `422 Field required`, loc `body.tools.4.function` (pydantic) |
| Together | ``missing field `function` `` (serde) |
| Fireworks | `Input should be 'function'` |
| BaseTen | ``unknown variant `tool_search`, expected `function` `` inside OpenRouter's `Provider returned error` |
| Novita | `invalid request error trace_id: ...` — names nothing at all |

The only portable way to classify these is the *structural* fact they share —
the offending entry is the one with no `function` member — not the wording.
Where a message names only an array slot, the index the client itself used is
the only key available.

Novita's message is a reminder that "unclassifiable" is a real outcome: it can
only be recorded as an error, never as a verdict.

### Deferred tool loading is not part of this API

OpenAI's
[tool search](https://developers.openai.com/api/docs/guides/tools-tool-search)
— a `{"type": "tool_search"}` tool entry plus `"defer_loading": true` on the
tools whose schemas should stay out of the prompt until the model asks for them
— is not a Chat Completions feature, and **every endpoint tested rejects it**,
in the dialects listed above. Dated server-tool types such as
`tool_search_tool_bm25_20251119` are rejected here too.

### CORS: can a browser call this directly?

Two legs matter: the `OPTIONS` preflight, which must allow `POST` plus the
`authorization` and `content-type` headers, and the actual response — because
some endpoints answer the preflight and then strip the headers one request
later.

| Endpoint | Verdict |
|---|---|
| OpenRouter | `*` wildcard — browser-direct works from any origin |
| DeepSeek | reflects the request origin, `allow-credentials: true` — browser-direct works |
| Kimi | none — `OPTIONS` 404s, no CORS headers at all |
| z.ai | blocked — preflight 200s but carries only `Vary: Origin`; headers appear on the actual response only |

z.ai is the instructive failure: the preflight looks successful (HTTP 200) yet
omits `access-control-allow-origin`, so the browser never issues the real
request. Checking only the actual response would report it as working.

### Context-window rejections

When `/models` publishes nothing, the fallback is to send one deliberately
oversized prompt and parse the limit out of the refusal. The refusals differ:

| Endpoint | Message | Recovered |
|---|---|---|
| DeepSeek | `This model's maximum context length is 1048576 tokens` | 1,048,576 (context window) |
| GitHub Copilot `gpt-5-mini` | `prompt token count of 300008 exceeds the limit of 128000` | 128,000 (**max prompt**, not window) |
| GitHub Copilot `gpt-5.6-luna` | `exceeds the limit of 922000` | 922,000 (max prompt) |
| z.ai | `Prompt exceeds max length` (code 1261) — **names no number** | nothing; found by bisection at ~1,048,478 |

Two distinct quantities hide here and must not be mixed: a true context window
(prompt plus completion, as DeepSeek, OpenRouter and Kimi describe it) and the
Copilot API's *max prompt tokens*, which is the binding number for a request.

### Non-HTTP and non-conforming backends

Not every "Chat Completions endpoint" is an HTTP endpoint.

- **GitHub Copilot** (`api.individual.githubcopilot.com`) requires an OAuth
  token from the local Copilot config, exchanged at
  `https://api.github.com/copilot_internal/v2/token`, which returns both the
  bearer token *and* the API base URL — so the base URL cannot be known before
  authenticating. Requests additionally need `Copilot-Integration-Id:
  vscode-chat` and editor-identifying headers (`editor-version`,
  `editor-plugin-version`, `user-agent`).
- **Cursor** serves **no `chat/completions` route on any of its hosts** — every
  such path returns 404. Inference runs over HTTP/2
  [Connect-RPC](https://connectrpc.com/docs/protocol), which has no tool-call
  channel of this shape at all: tools are neither declared as `tools` nor
  returned as `tool_calls`.
- **The OpenAI Codex backend** (`chatgpt.com/backend-api/codex`) does not serve
  this route at all, and the differences are not only in the path: it rejects
  `system` messages outright (`400 {"detail":"System messages are not
  allowed"}`), rejects `temperature`, requires streaming, and needs the headers
  `OpenAI-Beta: responses=experimental`, `originator: codex_cli_rs`,
  `chatgpt-account-id` and a per-request `session_id`.

When a local process stands between the caller and the real server, the server
that actually answered is not recoverable from the request — a Copilot base URL,
for instance, is handed out at auth time. A practical convention is for such a
process to add a non-standard top-level `x_upstream_endpoint` field naming the
real base URL, which the caller strips before parsing.

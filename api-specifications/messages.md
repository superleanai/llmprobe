# The Anthropic Messages API

`POST {base}/v1/messages` — Anthropic's native API
([reference](https://docs.claude.com/en/api/messages)), and the surface every
vendor emulates when they want Claude Code to work against their models.

Every emulation observed so far is a **model-slot proxy**: it accepts Claude
model names and routes them to the vendor's own models. That single design
choice is the source of most of Part 2.

- **[Part 1 — The common API](#part-1--the-common-api)**: the shape itself.
- **[Part 2 — Provider quirks](#part-2--provider-quirks)**: what the
  compatibility layers keep, drop, and silently ignore.

Part 2 records behaviour observed by calling the endpoints, not behaviour read
from vendor documentation. Where the two disagree, the observation is what is
written down.

---

## Part 1 — The common API

### Transport

```
POST {base}/v1/messages
x-api-key: <key>
anthropic-version: 2023-06-01
content-type: application/json
```

`anthropic-version` is **mandatory** and has been `2023-06-01` throughout.
Optional features are gated behind `anthropic-beta: <feature>,<feature>`.

Auth has two accepted forms: `x-api-key` for API keys, and
`Authorization: Bearer <token>` for OAuth tokens (what the official SDK's
`auth_token=` argument sends).

Compatibility layers hang the surface off `/anthropic` on the vendor's own
host:

| Vendor | Messages base |
|---|---|
| Anthropic | `https://api.anthropic.com` |
| DeepSeek | `https://api.deepseek.com/anthropic` |
| z.ai | `https://api.z.ai/api/anthropic` |
| Kimi | `https://api.moonshot.ai/anthropic` — **not** on `api.kimi.com` |
| OpenRouter | not served (404) |

The host is not always the one the vendor's other services live on, so this
base URL has to be read from the vendor's documentation rather than derived.

### Request body

```json
{
  "model": "claude-haiku-4-5-20251001",
  "max_tokens": 4096,
  "system": "You are a helpful assistant.",
  "messages": [
    {"role": "user", "content": "What is 2+2?"}
  ]
}
```

Three rules are mandatory and are the usual source of `400`s:

1. **`max_tokens` is required.** There is no default.
2. **There is no `system` role.** The system prompt is a top-level field — a
   string, or an array of text blocks.
3. **Roles strictly alternate** `user` / `assistant`. Consecutive same-role
   messages must be merged into one, which matters most for tool results,
   below.

Other fields: `temperature`, `top_p`, `top_k`, `stop_sequences`, `stream`,
`tools`, `tool_choice`, `thinking`, `metadata`.

### Content blocks

`content` is a string or an array of typed blocks. The block types that matter:

```json
{"type": "text", "text": "..."}
{"type": "thinking", "thinking": "...", "signature": "..."}
{"type": "tool_use", "id": "toolu_abc", "name": "read_file",
                     "input": {"path": "/tmp/x"}}
{"type": "tool_result", "tool_use_id": "toolu_abc",
                        "content": "<result>", "is_error": false}
```

- **`tool_use.input` is a JSON object**, already parsed — not a string to
  decode.
- **`tool_result` blocks live on a `user` turn.** There is no dedicated role for
  them. Several results answering parallel calls go into **one** user message as
  several blocks; sending them as separate messages breaks the alternation
  rule.
- A failed tool sets `is_error: true` on its `tool_result` block rather than
  reporting the failure some other way.

### Tools

```json
"tools": [
  {"name": "read_file",
   "description": "Read a file from disk.",
   "input_schema": {"type": "object",
                    "properties": {"path": {"type": "string"}},
                    "required": ["path"]}}
]
```

A client tool is a name, a description and an `input_schema` (JSON Schema).
There is no type tag and no wrapper object.

`tool_choice` is an object: `{"type":"auto"}`, `{"type":"any"}`,
`{"type":"tool","name":...}`, `{"type":"none"}`.

**Server tools** are a separate family, identified by a dated `type` string and
executed by Anthropic rather than by the caller — e.g.
`{"type": "web_search_20250305", "name": "web_search"}`
([list](https://docs.claude.com/en/docs/agents-and-tools/tool-use/overview)).
Compatibility layers whitelist these by exact string, which makes their
rejection messages unusually informative (see Part 2).

### Response

```json
{
  "id": "msg_...",
  "type": "message",
  "role": "assistant",
  "model": "claude-haiku-4-5-20251001",
  "content": [
    {"type": "text", "text": "4"},
    {"type": "tool_use", "id": "toolu_abc", "name": "read_file",
     "input": {"path": "/tmp/x"}}
  ],
  "stop_reason": "tool_use",
  "stop_sequence": null,
  "usage": {"input_tokens": 12, "output_tokens": 3,
            "cache_creation_input_tokens": 0, "cache_read_input_tokens": 0}
}
```

`content` is a list of blocks: text and tool calls are siblings, so a reply
carrying both is read by iterating.

`stop_reason` ∈ `end_turn` | `max_tokens` | `stop_sequence` | `tool_use` | 
`pause_turn` | `refusal`. `tool_use` is the one that means the turn is not
over: the caller must run the tools and send the results back.

### Token accounting

```json
"usage": {
  "input_tokens": 217,
  "output_tokens": 52,
  "cache_creation_input_tokens": 0,
  "cache_read_input_tokens": 2048
}
```

The critical property: **`input_tokens` counts only the uncached input.** Cache
reads and cache writes are *separate, additional* counters, not subsets. The
total input the request consumed is:

```
input_tokens + cache_read_input_tokens + cache_creation_input_tokens
```

Treating `input_tokens` as the whole prompt understates it by the entire cached
prefix — usually the majority of a long conversation.

There is no `total_tokens` field; add the counters yourself. Thinking tokens,
where reported, appear under `output_tokens_details` and are part of
`output_tokens`.

In a streamed response the initial `usage` arrives on `message_start` and is
completed by `message_delta` at the end.

### Prompt caching

Caching is [**explicit**](https://docs.claude.com/en/docs/build-with-claude/prompt-caching):
nothing is cached unless the request says so. A `cache_control` marker on a
content block caches the whole prefix up to and including that block.

```json
{"type": "text", "text": "<long stable prefix>",
 "cache_control": {"type": "ephemeral"}}
```

Up to four breakpoints per request. The placement that works for an agentic
loop:

- on the **system** prompt — static across turns;
- on the **last tool** definition — caches the whole tool array;
- on the **last block of the last message** — caches the conversation prefix
  incrementally, so each turn re-reads everything before it from cache.

A client that omits these markers turns a cacheable workload into a fully
billed one, with no warning anywhere in the response.

### Extended thinking

```json
"thinking": {"type": "enabled", "budget_tokens": 8000}
```

`max_tokens` must exceed `budget_tokens`, or the request is rejected.

Thinking arrives as `thinking` blocks carrying a `signature`, which must be
echoed back verbatim on the next turn for multi-turn thinking to validate.
Some variants are beta-gated, e.g.
`anthropic-beta: interleaved-thinking-2025-05-14`.

### Streaming

`stream: true` yields named SSE events:

| Event | Payload |
|---|---|
| `message_start` | the message envelope, with initial `usage` |
| `content_block_start` | a new block opens, with its index and type |
| `content_block_delta` | `text_delta`, `input_json_delta` (tool arguments, as text fragments), or `thinking_delta` |
| `content_block_stop` | that block is complete |
| `message_delta` | `stop_reason` and final `usage.output_tokens` |
| `message_stop` | end |
| `ping`, `error` | keepalive; terminal failure |

Tool arguments stream as `input_json_delta` **partial JSON text**, which must
be concatenated per block index and parsed only once the block stops — a
mid-stream fragment is not valid JSON.

---

## Part 2 — Provider quirks

### Cross-cutting trap: HTTP 200 proves nothing

Silently dropping unknown request fields is the cheap way to build a
compatibility layer, and most compatibility layers do exactly that. A `200` on
a request carrying an optional field is therefore not evidence the field did
anything — it has to be verified by its effect. The `cache_control` case below
is the canonical example.

### The model name is a routing key, not a model

Every emulation maps Claude model names onto its own models:

| Vendor | Mapping |
|---|---|
| DeepSeek | `claude-opus*` → `deepseek-v4-pro`; `claude-haiku*`, `claude-sonnet*` → `deepseek-flash`; **unknown names fall through to `deepseek-flash`** |
| z.ai | GLM-5.3 on the opus *and* sonnet slots; GLM-5.3-Flash on haiku |

DeepSeek's fallthrough is the dangerous half: there is no such thing as an
unrecognised model on that endpoint, so a typo in a model id is served
successfully by the wrong model, with no error. The model id has to be chosen
deliberately for each endpoint, never carried over from somewhere else.

### DeepSeek's published unsupported list

DeepSeek is the only vendor publishing an explicit list for its Anthropic
surface ([docs](https://api-docs.deepseek.com/guides/anthropic_api)).
Unsupported:

- the **`anthropic-beta` header** (except for the Files API) — so every
  beta-gated feature, including interleaved thinking, is off
- `container`, `mcp_servers`
- `top_k`, and `top_p` outside thinking mode
- **`cache_control` is ignored** — the markers are accepted and do nothing, so
  a client that placed its four breakpoints carefully gets no caching and no
  warning
- unsupported content types: documents, search results, redacted thinking, code
  execution results, MCP tools

`cache_control` being *ignored* rather than rejected is the exact failure the
cross-cutting trap describes: the request is well-formed, the response is a
200, and the feature did not happen. The only way to notice is that
`cache_read_input_tokens` never rises.

### Server tools: the rejection names the whitelist

Sending a server tool DeepSeek does not implement:

```
400 Failed to deserialize the JSON body into the target type:
    tools[4]: unknown variant `tool_search_tool_bm25_20251119`,
    expected `web_search_20250305` or `web_search_20260209`
```

The error enumerates the entire set of server tools that layer accepts — two
dated `web_search` versions and nothing else. That is more information than the
vendor's own documentation carries, and it is the cheapest way to discover the
whitelist of any serde-based compatibility layer.

### Deferred tool loading

[Tool search](https://docs.claude.com/en/docs/agents-and-tools/tool-use/tool-search-tool)
keeps a tool's schema out of the prompt until the model asks for it:

```json
"tools": [
  {"name": "warehouse_inventory_lookup", "description": "...",
   "input_schema": {...}, "defer_loading": true},
  {"type": "tool_search_tool_bm25_20251119", "name": "tool_search_tool_bm25"}
]
```

There is also a regex variant (`tool_search_tool_regex_20251119` /
`tool_search_tool_regex`). Results come back as `tool_search_tool_result`
content blocks.

Two rules the API enforces:

- the search tool itself must not be deferred;
- **at least one tool must stay non-deferred**, else `400 All tools have
  defer_loading set`. Marking every tool deferred produces a rejection that
  looks like "unsupported" and is not.

**Observed on compatibility layers: no support anywhere.**

| Endpoint | Result |
|---|---|
| DeepSeek `/anthropic` | rejected — the server tool is unknown (see above) |
| Kimi | not applicable — `api.kimi.com/anthropic` 404s; the surface lives on `api.moonshot.ai` |
| z.ai `/api/anthropic` | unsettled — `429 [1310][Weekly/Monthly Limit Exhausted...]` |
| OpenRouter | not applicable — no Anthropic surface |

### Kimi: two hosts, and the docs point at the other one

The coding-plan host `api.kimi.com` serves **no** Anthropic surface — `404`.
The documented base is `https://api.moonshot.ai/anthropic`
([overview](https://platform.kimi.ai/docs/api/overview)), a different host
entirely. Deriving the URL from whichever Kimi host is already working
concludes "not supported" when the right answer is "not here".

### z.ai: errors, and where the docs aren't

The Anthropic endpoint exists but is documented in the
[Claude Code setup](https://docs.z.ai/devpack/tool/claude) pages rather than
the [API reference](https://docs.z.ai/api-reference/introduction). Its
rate-limit error carries a vendor code and a trace id inside the message rather
than in headers:

```
429 rate_limit_error 1310
[1310][Weekly/Monthly Limit Exhausted. Your limit will reset at 2026-09-21 00:27:25][20260916133221582d8f5b8065466b]
```

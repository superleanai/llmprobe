# The Cursor Connect-RPC API

Cursor's inference runs over
**[Connect-RPC](https://connectrpc.com/docs/protocol)** — an HTTP/2 protocol
with protobuf *or* JSON payloads — on hosts separate from its documented REST
API, which serves no completion route at all (every such path returns 404).
There is no vendor specification for the RPC surface; this document records
behaviour observed by calling the endpoints and by reading the schema out of
the shipped CLI bundle.

This document has two parts:

- **[Part 1 — The protocol](#part-1--the-protocol)**: auth, framing, encoding,
  and the message envelope. Verified by live calls.
- **[Part 2 — Traps](#part-2--traps)**: behaviours that break a client written
  against Part 1, each with the test that established it.

Verified 2026-09-17 against CLI build `2026.09.15-d2fe57e`. Cursor changes
hosts and RPCs between releases; re-test before trusting any of it.

---

## Part 1 — The protocol

### Hosts

Three surfaces, one credential. Which package a host serves matters more than
which generation it belongs to.

| Host | HTTP | Serves |
|---|---|---|
| `api.cursor.com` | 1.1 | REST admin/cloud APIs (`/v1/me`, `/v1/models`, `/v1/agents`). JSON. No inference. |
| `api2.cursor.sh` | 1.1 + h2 | Auth, `aiserver.v1.*`, **and** `agent.v1.AgentService` |
| `agentn.global.api5.cursor.sh` | **h2 only** | `agent.v1.AgentService`. What the CLI actually calls. |
| `agentn.api5.cursor.sh`, `agent.api5.cursor.sh` | **h2 only** | `agent.v1.AgentService` |

The `api5` hosts reject HTTP/1.1 at the connection level with `illegal request
line`, so a client that does not negotiate h2 sees them as dead. They do not
serve `aiserver.v1` (404).

### Auth

Two-legged. The long-lived `crsr_` API key is never sent to an RPC:

```http
POST https://api2.cursor.sh/auth/exchange_user_api_key
Authorization: Bearer crsr_<key>
Content-Type: application/json

{}
```

returns `{"accessToken": "<JWT>", ...}`, valid about an hour. Every RPC carries
that JWT as `Authorization: Bearer <jwt>`.

`POST https://api2.cursor.sh/oauth/token` also exists and is the refresh path
for IDE OAuth sessions rather than for API keys.

### Encoding

Connect picks the content type by **RPC kind**, not by payload format. Both
protobuf and JSON are accepted, so a working client needs neither `.proto`
files nor a protobuf runtime.

| RPC kind | Content-Type | Body |
|---|---|---|
| Unary | `application/json` | the bare JSON message |
| Unary | `application/proto` | the bare protobuf message |
| Streaming | `application/connect+json` | framed JSON |
| Streaming | `application/connect+proto` | framed protobuf |

Field names in JSON are lowerCamelCase of the protobuf field names
(`conversation_state` → `conversationState`).

### Framing

Streaming bodies — in both directions — are sequences of length-prefixed
frames:

```
[flags: 1 byte][length: 4 bytes big-endian][payload: length bytes]
```

`flags == 0` is a normal message. `flags == 2` is the trailing end-of-stream
frame; it carries `{}` on success or `{"error": {...}}` on failure. A stream
that ends with flag 2 and an error still arrived as **HTTP 200** — see
[HTTP 200 proves nothing](#http-200-proves-nothing).

### Headers

Only two are enforced:

```http
Authorization: Bearer <jwt>
Content-Type: <per the table above>
```

The real client also sends `connect-protocol-version: 1`,
`x-cursor-client-type`, `x-cursor-client-version`, `x-ghost-mode`,
`x-request-id`, `x-original-request-id` and `x-blob-encryption-key`. Sending
them is harmless and is what this document's examples do, but none are
required, and none are validated (see [Part 2](#the-client-identifying-headers-are-not-validated)).

### Model catalogue

```http
POST {host}/agent.v1.AgentService/GetUsableModels
Content-Type: application/json

{}
```

returns `{"models": [{"modelId": "...", "displayName": "...", "aliases": [...]},
...]}` — 223 entries, which are the *flattened* variants
(`claude-opus-5-thinking-high-fast`).

The REST API's `GET https://api.cursor.com/v1/models` returns the same grid
*parameterised*: 39 base ids plus knobs (`thinking`, `context`,
`effort`/`reasoning`, `fast`) whose cross-product the CLI ids spell out. Both
naming schemes are accepted where a model id is expected.

### Inference: `agent.v1.AgentService/Run`

The live inference RPC is **bidirectional streaming**:

```http
POST https://agentn.global.api5.cursor.sh/agent.v1.AgentService/Run
Content-Type: application/connect+json
```

Every frame in each direction is a `oneof` envelope.

**`AgentClientMessage`** (client → server):

| # | Field |
|---|---|
| 1 | `runRequest` |
| 2 | `execClientMessage` |
| 3 | `kvClientMessage` |
| 4 | `conversationAction` |
| 5 | `execClientControlMessage` |
| 6 | `interactionResponse` |
| 7 | `clientHeartbeat` |
| 8 | `prewarmRequest` |

**`AgentServerMessage`** (server → client):

| # | Field |
|---|---|
| 1 | `interactionUpdate` |
| 2 | `execServerMessage` |
| 3 | `conversationCheckpointUpdate` |
| 4 | `kvServerMessage` |
| 5 | `execServerControlMessage` |
| 7 | `interactionQuery` |
| 8 | `ttftBreakdown` |

### The opening request

`AgentRunRequest` has 33 fields; this is the subset the CLI actually sends for
a plain prompt, recovered by decoding a captured request:

```json
{"runRequest": {
  "conversationState": {},
  "mcpTools": {},
  "action": {"userMessageAction": {"userMessage": {
      "text": "Reply with exactly PONG",
      "messageId": "<uuid>",
      "selectedContext": {},
      "mode": 2
  }}},
  "conversationId": "<uuid>",
  "conversationGroupId": "<uuid>",
  "runId": "<uuid>",
  "requestedModel": {"modelId": "composer-2.5",
                     "params": [{"id": "fast", "value": "false"}]},
  "excludeWorkspaceContext": true
}}
```

The prompt travels as **plain text** in
`action.userMessageAction.userMessage.text`. `conversationState` may be empty
on the first turn.

### Output

Assistant text arrives as `interactionUpdate.textDelta.text` — protobuf path
`f1.f1.f1`. Reasoning arrives as `interactionUpdate.thinkingDelta`
(`f1.f4.f1`). Other `interactionUpdate` cases worth handling:
`toolCallStarted`, `toolCallCompleted`, `tokenDelta`, `turnEnded`, and
`heartbeat` — which is emitted every ~10 s and carries nothing.

### Tools

Tools are not declared as schemas in the request, and calls do not come back
as a field on the reply. Instead the server drives an execution loop against
the client:

```
server → AgentServerMessage.execServerMessage {id, execId, <name>Args}
client → AgentClientMessage.execClientMessage {id, execId, <name>Result}
```

`ExecServerMessage` has a `oneof` of ~30 tool argument types, including
`shellArgs`, `readArgs`, `writeArgs`, `grepArgs`, `lsArgs`, `fetchArgs`,
`mcpArgs`, `subagentArgs`, `computerUseArgs`, `diagnosticsArgs` and
`requestContextArgs`. `ExecClientMessage` mirrors each with a `*Result`.

The client's own tool declarations travel in the `requestContextResult` reply,
under `RequestContext.tools`. The first thing the server asks for on every run
is `requestContextArgs`; nothing else proceeds until it is answered.

---

## Part 2 — Traps

### HTTP 200 proves nothing

The status line is written before generation starts, so a deprecated RPC, a
malformed request and a successful generation all return `200`. The verdict is
in the trailing flag-2 frame:

```json
{"error":{"code":"unimplemented","message":"Error",
 "details":[{"type":"aiserver.v1.ErrorDetails",
             "debug":{"error":"ERROR_DEPRECATED", ...}}]}}
```

A client that checks `response.status_code` and stops has learned nothing.

### 415 means the wrong RPC kind, not the wrong payload format

Measured on `AiService/AvailableModels`, a unary RPC:

| Content-Type | Body | Result |
|---|---|---|
| `application/json` | JSON | **200**, full model list |
| `application/connect+json` | JSON | **415** — streaming content type on a unary RPC |
| `application/proto` | JSON | **500** |

So a 415 here is a kind mismatch. JSON payloads are served, and no protobuf
runtime is needed to call a unary RPC.

### The client-identifying headers are not validated

Tested three ways on a unary RPC:

- `Authorization` + `Content-Type` alone → **200**. Every
  `x-cursor-*` header the real client sends can be omitted.
- `x-cursor-client-version: cli-0.0.0-bogus` → **200**. No version gate.
- a bogus `x-cursor-checksum` → **200**. The value is not verified.

### Absence from the CLI bundle is not evidence of death

`aiserver.v1.ChatService` appears nowhere in the `cursor-agent` bundle, yet the
server still serves `ChatService/StreamUnifiedChatWithTools` — it rejects a bad
body with *"first message is not a streamUnifiedChatRequest"*, which is an
answer from a live handler. Conversely, several RPCs that *are* in the bundle
are dead server-side.

Current status of the older `aiserver.v1` chat RPCs:

| RPC | Status |
|---|---|
| `ChatService/StreamUnifiedChatWithTools` | alive |
| `AiService/AvailableModels`, `AiService/ServerTime` | alive |
| `AiService/StreamChat` | `unimplemented / ERROR_DEPRECATED` |
| `AiService/StreamComposer` | `"streamComposer is deprecated"` |
| `AiService/StreamChatToolformer` | `unimplemented / Deprecated` |
| `ChatService/GetPromptDryRun` | `501 unimplemented` |

### `Run` is bidirectional, and that rules out most HTTP clients

The client must keep its request body open and write frames *while* reading the
response — the server's first act is to ask for `requestContextArgs`, and it
will not generate until answered. Synchronous HTTP clients that finish sending
the request before reading the response (including `httpx`'s sync client)
deadlock here: the call hangs with no response headers at all.

`RunSSE` (server-streaming) and `RunPoll` exist as fallbacks, but a plain
`RunSSE` call with a complete-looking `AgentRunRequest` streams only
`heartbeat` frames and then closes with
`deadline_exceeded — "Timeout waiting for next item."`: it is still waiting for
client input that a one-shot request cannot supply.

### It is an agent harness, not a completion endpoint

Even a three-word prompt costs ~7k cached input tokens, because the request
carries Cursor's own system prompt, rules, skills and tool definitions. There
is no "raw model" mode. Anything measuring model behaviour through this API is
measuring the harness too.

### There is no supported way to capture the traffic

The wire is protobuf by default and there is no JSON-dump flag. Three
obstacles:

- `CURSOR_API_ENDPOINT` / `-e` redirects the bootstrap RPCs but **not** the
  inference call, which goes to `api5` regardless.
- The CLI launcher runs Node with `--use-system-ca`, so `NODE_EXTRA_CA_CERTS`
  is ignored. Setting `AGENT_CLI_CREDENTIAL_STORE=file` makes the launcher skip
  that flag, after which a proxy CA is trusted.
- A streaming RPC never produces a "completed response", so a proxy addon that
  captures in a response hook records nothing. Capture in the request hook and
  let response bodies stream through.

### The schema is in the bundle

The CLI ships `@bufbuild/protobuf` messages in a compact form —
`static $(){return["AgentRunRequest|1 conversation_state #0|…", Ref, …]}` —
where the string names the fields and the trailing array resolves the `#N`
references. This is the authoritative schema for a given build. Resolution must
be positional: each webpack chunk reuses the same minified identifiers, so a
global identifier map produces confidently wrong type names.

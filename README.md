# llmprobe

Probe an LLM endpoint for tool-calling behaviour and API-surface
capabilities by actually calling it, instead of trusting vendor docs.

Results are written as JSON + Markdown per model under `reports/<model>/`,
so different models and endpoints can be compared side by side.

## Capabilities

| Codename | What it measures |
|---|---|
| `TCALL` | Structured `tool_calls` vs. inline JSON/XML/no call |
| `QUOTE` | Literal double-quotes preserved inside JSON string args |
| `GREP` | Prefers filtered/targeted calls over pulling whole files/outputs into context |
| `ASKQ` | How strongly task phrasing drives use of an `ask_user_question` tool |
| `APPLY_PATCH` | Model's own knowledge of OpenAI's `apply_patch` grammar (no tool schema) |
| `GRAMT` | Endpoint's transport support for real grammar-constrained custom tools |
| `RJSON` | Endpoint honours strict `response_format:{type:"json_schema"}` |
| `STRM` | Endpoint delivers real incremental SSE chunks under `stream:true` |
| `REASN` | Reasoning tokens exposed, and whether reasoning-effort syntax is accepted |
| `TSEL` | Model calls the right tool in a list |
| `AKDEF` | Model uses agentknit's real default tools (`read_file`/`write_file`/`str_replace`/`exec_shell`) correctly end to end |

Full definitions, units, and ranges: [CAPABILITIES.md](CAPABILITIES.md).

## Install

```bash
pip install openai lark keyring
```

API keys are read from an environment variable or, if unset, from the
system keyring (`keyring get login2 <key-name>`).

## Usage

```bash
python3 probe_inference.py \
  --endpoint https://api.deepseek.com \
  --model deepseek-v4-flash \
  --key-name deepseek_api_key
```

`--endpoint` is the API's base URL, not the full `chat/completions` path
(the `openai` client appends that itself).

This runs the full probe — tool-name/parameter elicitation plus every
capability test below — and writes:

- `reports/<model>/capabilities_<model>.json`
- `reports/<model>/capabilities_<model>.md`

Every extra capability test is on by default; pass its `--no-*` flag to
skip one, e.g. `--no-stream-test`. Local wrapper scripts (auth handled
internally, non-OpenAI transport) can be probed in place of an HTTP
endpoint with `--script path/to/script.py`; the script must read one
Chat-Completions JSON payload from stdin and print one JSON response to
stdout.

`AKDEF` needs `agentknit` importable (a sibling checkout on `sys.path`,
editable-installed); if it isn't, the test reports an error and every
other capability still runs. Skip it explicitly with `--no-agentknit-test`.

`CACH` (empirical prompt-cache TTL measurement) is the one test that is
off by default — it takes several minutes of real waiting. Enable it with
`--cache-ttl-test`.

Run `python3 probe_inference.py --help` for every flag.

### Re-render a report without re-probing

```bash
python3 probe_inference.py --model deepseek-v4-flash --render-md
```

### Compare tool-calling support across all probed models

```bash
python3 probe_inference.py --quick-summary
```


## Reports

Each `reports/<model>/capabilities_<model>.md` includes a capabilities
summary table, the inferred tool schema, the tool-dispatch table, and a
per-capability breakdown with pass/fail detail. `probes/<model>/` (not
checked in) holds the raw request/response JSON for every probe call, for
debugging.

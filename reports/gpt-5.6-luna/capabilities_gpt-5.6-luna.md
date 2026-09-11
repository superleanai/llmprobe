# Model capability probe: gpt-5.6-luna

- **Endpoint:** script:/home/martin/bin/copilot-gpt-5.6-luna.py
- **API type:** OpenAI Completions

## Capabilities summary

See `CAPABILITIES.md` for what each codename measures, its unit, and its range.

| Codename | Value |
|---|---|
| `TCALL` | 8/8 |
| `QUOTE` | 3/3 |
| `GREP` | 6/6 |
| `ASKQ` | 4/8 (Sometimes) |
| `APPLY_PATCH` | 2/2 |
| `GRAMT` | 1/1 |
| `RJSON` | 1/1 |
| `STRM` | 0/1 |
| `REASN` | 2/3 |
| `AKDEF` | 4/4 |
| `TSEL` | 6/8 |

## Format detection & call delivery (`TCALL`)

- Round-0 probe (single call): detected format `structured_tool_calls`, structured tool_calls used: True

- Full probe (8 tasks): call delivery mode `structured_tool_calls`
  - Structured tool_calls: 8/8 tasks
  - Inline JSON in content (model ignored the tools API and put the call as JSON text in the message body instead): 0/8 tasks
  - No call detected (neither a structured tool_call nor parseable inline JSON): 0/8 tasks
- Note: Model correctly uses the structured tool_calls API field.

## Elicited tool names

Round 1 asks the model, in free text with no tool schema attached, what function/arguments it would use for each task. The prompt never names a tool — the model must invent the name itself.

| Operation | Elicitation prompt | Model function name |
|---|---|---|
| read_file | You need to read the contents of the file /etc/hostname. What function would you call and what arguments would you pass? Respond with a JSON object like: {"function_name": "...", "arguments": {"arg1": "val1", ...}} | `read_file` |
| write_file | You need to write the text 'hello world' to the file /tmp/test.txt. What function would you call and what arguments would you pass? Respond with a JSON object like: {"function_name": "...", "arguments": {"arg1": "val1", ...}} | `write_file` |
| update_file | The file /tmp/test.py already exists and contains Python code. You need to make a targeted edit: replace the exact string 'x = 1' with 'x = 42', without rewriting the whole file. What function would you call and what arguments would you pass? Respond with a JSON object like: {"function_name": "...", "arguments": {"arg1": "val1", ...}} | `replace_in_file` |
| execute_bash | You need to run the shell command `ls -la /tmp`. What function would you call and what arguments would you pass? Respond with a JSON object like: {"function_name": "...", "arguments": {"arg1": "val1", ...}} | `run_shell_command` |
| ask_user_question | You need to ask the user a clarifying question: 'Should I overwrite the existing file, or create a backup first?' with options 'Overwrite' and 'Backup'. What function would you call and what arguments would you pass? Respond with a JSON object like: {"function_name": "...", "arguments": {"arg1": "val1", ...}} | `ask_user_question` |
| list_directory | You need to list all files and subdirectories inside /tmp. What function would you call and what arguments would you pass? Respond with a JSON object like: {"function_name": "...", "arguments": {"arg1": "val1", ...}} | `list_directory` |
| search_files | You need to find every line containing the string 'def main' in any file under /tmp/myproject (search recursively). What function would you call and what arguments would you pass? Respond with a JSON object like: {"function_name": "...", "arguments": {"arg1": "val1", ...}} | `grep` |
| glob | You need to find all Python source files (matching *.py) anywhere under /tmp/myproject, recursively. What function would you call and what arguments would you pass? Respond with a JSON object like: {"function_name": "...", "arguments": {"arg1": "val1", ...}} | `glob.glob` |

## Inferred tool schema

### `read_file`

Perform the 'read_file' operation.

| Parameter | Type | Required |
|---|---|---|
| path | string | yes |

### `write_file`

Perform the 'write_file' operation.

| Parameter | Type | Required |
|---|---|---|
| path | string | yes |
| content | string | yes |

### `replace_in_file`

Perform the 'update_file' operation.

| Parameter | Type | Required |
|---|---|---|
| path | string | yes |
| old_string | string | yes |
| new_string | string | yes |

### `run_shell_command`

Perform the 'execute_bash' operation.

| Parameter | Type | Required |
|---|---|---|
| command | string | yes |

### `ask_user_question`

Perform the 'ask_user_question' operation.

| Parameter | Type | Required |
|---|---|---|
| question | string | yes |

### `list_directory`

Perform the 'list_directory' operation.

| Parameter | Type | Required |
|---|---|---|
| path | string | yes |

### `grep`

Perform the 'search_files' operation.

| Parameter | Type | Required |
|---|---|---|
| pattern | string | yes |
| path | string | yes |
| recursive | boolean | yes |

### `glob_glob`

Perform the 'glob' operation.

| Parameter | Type | Required |
|---|---|---|
| pathname | string | yes |
| recursive | boolean | yes |

## Tool-selection test (`TSEL`)

Each task is run with the full inferred tool schema. PASS means the model called the tool assigned to that operation.

| Operation | Result | Expected tool | Called tool | Notes |
|---|---|---|---|---|
| read_file | PASS | `read_file` | `read_file` |  |
| write_file | PASS | `write_file` | `write_file` |  |
| update_file | FAIL | `replace_in_file` | `read_file` | called 'read_file' instead of 'replace_in_file' |
| execute_bash | PASS | `run_shell_command` | `run_shell_command` |  |
| ask_user_question | PASS | `ask_user_question` | `ask_user_question` |  |
| list_directory | PASS | `list_directory` | `list_directory` |  |
| search_files | PASS | `grep` | `grep` |  |
| glob | FAIL | `glob.glob` | `glob_glob` | called 'glob_glob' instead of 'glob.glob' |

## Agentknit default-tool compatibility test (`AKDEF`)

**4/4 passed** — agentknit's real shipped default tool schema (`read_file`/`write_file`/`str_replace`/`exec_shell`, imported live from `agentknit._core`) is offered, and each call is dispatched through agentknit's real `dispatch()` against a scratch directory. PASS requires both the right tool selected and the correct on-disk effect.

| Tool | Result | Called tool | Dispatch OK | Effect verified | Notes |
|---|---|---|---|---|---|
| read_file | PASS | `read_file` | True | True |  |
| write_file | PASS | `write_file` | True | True |  |
| str_replace | PASS | `str_replace` | True | True |  |
| exec_shell | PASS | `exec_shell` | True | True |  |

## Tool dispatch table

| Model tool name | Python function | Param map |
|---|---|---|
| `read_file` | `t_read` | path→path |
| `write_file` | `t_write` | path→path, content→content |
| `run_shell_command` | `t_run` | command→command |
| `ask_user_question` | `t_ask_user` | question→question |
| `list_directory` | `t_list_dir` | path→path |
| `grep` | `t_search` | pattern→pattern, path→path |
| `glob_glob` | `t_read` | pathname→path |

## Quote-escaping test (`QUOTE`)

**3/3 passed** — only the tool relevant to each task is advertised (not the full schema), so this isolates quote-escaping fidelity from tool-selection behaviour.

| Operation | Isolated schema | Result | Function called | Notes |
|---|---|---|---|---|
| write_file | yes | PASS | `write_file` |  |
| execute_bash | yes | PASS | `run_shell_command` |  |
| update_file | yes | PASS | `replace_in_file` |  |

## Token-efficiency test (`GREP`)

**6/6 passed** — prefers a filtered/targeted call over pulling the entire large file/output into context.

| Operation | Result | Function called | Args | Reason |
|---|---|---|---|---|
| large_log_grep | PASS | `run_shell_command` | {"command": "grep -F 'FATAL ERROR' /var/log/app.log"} | command includes a filtering tool (grep/awk/sed/head/tail/wc/cut) |
| count_occurrences | PASS | `run_shell_command` | {"command": "grep -F -c '203.0.113.42' /var/data/access.log"} | command includes a filtering tool (grep/awk/sed/head/tail/wc/cut) |
| specific_line | PASS | `run_shell_command` | {"command": "sed -n '48213p' /opt/build/output.txt"} | command includes a filtering tool (grep/awk/sed/head/tail/wc/cut) |
| check_string_exists | PASS | `run_shell_command` | {"command": "grep -q 'DeprecationWarning' /var/log/build.log && echo yes \|\| echo no"} | command includes a filtering tool (grep/awk/sed/head/tail/wc/cut) |
| function_definition_search | PASS | `grep` | {"pattern": "^def train_model\\b", "path": "/repo/src/model.py", "recursive": false} | dedicated search tool used -- inherently filtered |
| process_output_filter | PASS | `run_shell_command` | {"command": "ps aux \| grep -i '[p]ython'"} | command includes a filtering tool (grep/awk/sed/head/tail/wc/cut) |

## Ask-user-question phrasing test (`ASKQ`)

**4/8 — Sometimes** calls its own ask_user_question tool across 8 phrasings of the same underlying task (full tool schema, one sample per phrasing, no retries).

| Variant | System prompt | Result | Function called |
|---|---|---|---|
| original_probe | default | SKIPPED | `*(none)*` |
| imperative_must | default | ASKED | `ask_user_question` |
| explicit_tool_name | default | ASKED | `ask_user_question` |
| first_person_unsure | default | SKIPPED | `*(none)*` |
| ambiguous_no_ask_word | default | SKIPPED | `*(none)*` |
| destructive_warning | default | SKIPPED | `*(none)*` |
| options_style | default | ASKED | `ask_user_question` |
| system_nudge | nudge | ASKED | `ask_user_question` |

## apply_patch grammar-knowledge test (`APPLY_PATCH`)

**2/2 passed** — no tool schema offered; the model is asked in free text to produce a raw apply_patch-format patch, parsed against the real grammar (not a loose regex). Tests whether the model *knows* the syntax, independent of whether the endpoint exposes the tool itself (see `~/bin/copilot-notes.md`).

| Operation | Result | Notes |
|---|---|---|
| update_file | PASS |  |
| add_file | PASS |  |

## Constrained-decoding / custom-tool test (`GRAMT`)

**1/1 passed** — sends a real OpenAI `type:"custom"` freeform tool with `format:{type:"grammar", syntax:"lark"}`; PASS requires a genuine `custom` tool_call back with grammar-valid input (not a classic `function` tool_call, and not silently ignored). Tests the *endpoint's* transport support, independent of whether the model knows the syntax (`APPLY_PATCH`) — see `~/bin/copilot-notes.md`.

| Operation | Result | Tool call type | Notes |
|---|---|---|---|
| apply_patch | PASS | `custom` |  |

## Structured-output test (`RJSON`)

**1/1 passed** — sends a strict `response_format:{type:"json_schema"}` request with no tool schema; PASS requires the endpoint to accept the request and return content that parses as JSON conforming to the schema. Tests the *endpoint's* structured-output support, independent of tool calling.

| Task | Result | Conformant | Notes |
|---|---|---|---|
| json_schema | PASS | yes |  |

## SSE streaming test (`STRM`)

**0/1 passed** — sends `stream:true`; PASS requires more than one chunk, non-empty reconstructed content, and a finish_reason chunk. FAIL includes the endpoint rejecting `stream:true` outright and accepting it but buffering the whole reply into one chunk.

- Result: FAIL
- Chunks: 0
- Time to first chunk: Nones
- Total time: Nones
- finish_reason: `None`
- Notes: request failed: 'tuple' object has no attribute 'choices'

## Reasoning-tokens & effort-control test (`REASN`)

**2/3 passed** — `reasoning_tokens_present` checks (no special params) whether the reply carries a reasoning trace (`reasoning_content`/`reasoning`/`thinking` field, or `usage.completion_tokens_details.reasoning_tokens`). `effort_control_native` and `effort_control_extra_body` each check whether the endpoint accepts one reasoning-effort syntax — the native top-level `reasoning_effort` Chat Completions param, and the OpenRouter-style `extra_body={'reasoning': {'effort': ...}}` passthrough — without erroring. Accepting the parameter is the bar; this does not confirm the effort setting changed model behaviour.

| Check | Result | Field | Notes |
|---|---|---|---|
| reasoning_tokens_present | FAIL | `*(none)*` |  |
| effort_control_native | PASS | `*(none)*` |  |
| effort_control_extra_body | PASS | `*(none)*` |  |

## Missing capabilities

- `TSEL_update_file` FAILED — called 'read_file' instead of 'replace_in_file'.
- `TSEL_glob` FAILED — called 'glob_glob' instead of 'glob.glob'.
- Tool `replace_in_file` is in the inferred schema but was never dispatched (the model didn't call it with a matching signature during Round 2/3 probing).
- `ASKQ_original_probe` FAILED — called `no tool call` instead of asking the user.
- `ASKQ_first_person_unsure` FAILED — called `no tool call` instead of asking the user.
- `ASKQ_ambiguous_no_ask_word` FAILED — called `no tool call` instead of asking the user.
- `ASKQ_destructive_warning` FAILED — called `no tool call` instead of asking the user.
- `STRM_basic` FAILED — request failed: 'tuple' object has no attribute 'choices'
- `REASN_reasoning_tokens_present` FAILED — None

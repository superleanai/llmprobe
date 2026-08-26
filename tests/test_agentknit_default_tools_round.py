"""End-to-end (mocked client) test for agentknit_default_tools_test_round.

Exercises the full AKDEF round -- schema offered, call extracted, dispatched
through agentknit's real dispatch() -- against a fake client that always
answers correctly, to prove the plumbing (not just _akdef_verify_effect)
scores 4/4 when the model behaves.
"""
import json
import re
import sys
from pathlib import Path

import openai

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import probe_inference as pi


def _tool_call_response(name: str, arguments: dict) -> "openai.types.chat.ChatCompletion":
    return openai.types.chat.ChatCompletion.model_validate({
        "id": "test", "object": "chat.completion", "created": 0, "model": "test-model",
        "choices": [{
            "index": 0,
            "finish_reason": "tool_calls",
            "message": {
                "role": "assistant",
                "content": None,
                "tool_calls": [{
                    "id": "call_1", "type": "function",
                    "function": {"name": name, "arguments": json.dumps(arguments)},
                }],
            },
        }],
    })


class _FakeCompletions:
    def create(self, **kwargs):
        user_text = kwargs["messages"][-1]["content"]
        if (m := re.search(r"read the file (\S+) and", user_text)):
            return _tool_call_response("read_file", {"path": m.group(1)})
        if (m := re.search(r"to the file (\S+)\.", user_text)):
            return _tool_call_response("write_file", {"path": m.group(1), "content": "hello world\n"})
        if (m := re.search(r"In the file (\S+),", user_text)):
            return _tool_call_response("str_replace", {
                "path": m.group(1), "old_str": "x = 1", "new_str": "x = 42",
            })
        if (m := re.search(r"run `([^`]+)`", user_text)):
            return _tool_call_response("exec_shell", {"command": m.group(1)})
        raise AssertionError(f"unrecognised AKDEF task: {user_text!r}")


class _FakeChat:
    def __init__(self):
        self.completions = _FakeCompletions()


class _FakeClient:
    def __init__(self):
        self.chat = _FakeChat()


def test_akdef_round_all_pass_with_a_well_behaved_model():
    result = pi.agentknit_default_tools_test_round(_FakeClient())
    assert "error" not in result
    assert result["akdef_passed"] == 4
    assert result["akdef_total"] == 4
    for op, r in result["akdef_results"].items():
        assert r["pass"], f"{op}: {r}"


def test_akdef_round_flags_wrong_tool_selection():
    class _WrongToolCompletions(_FakeCompletions):
        def create(self, **kwargs):
            resp = super().create(**kwargs)
            # Force every call to claim it read a file, regardless of task.
            resp.choices[0].message.tool_calls[0].function.name = "read_file"
            return resp

    client = _FakeClient()
    client.chat.completions = _WrongToolCompletions()
    result = pi.agentknit_default_tools_test_round(client)
    assert result["akdef_passed"] < result["akdef_total"]
    assert not result["akdef_results"]["write_file"]["tool_selected"]
    assert result["akdef_results"]["write_file"]["error"] == "called 'read_file' instead of 'write_file'"

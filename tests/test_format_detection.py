"""Tests for round-0 format detection, including explicit XML serialization."""
import sys
from pathlib import Path
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import probe_inference as pi


def _response(*, content="", tool_calls=None):
    return SimpleNamespace(
        choices=[SimpleNamespace(message=SimpleNamespace(content=content, tool_calls=tool_calls))]
    )


def test_format_detection_records_explicit_valid_xml(monkeypatch):
    responses = iter([
        _response(tool_calls=[SimpleNamespace()]),
        _response(content='<tool_call>{"name":"read_file","arguments":{"path":"/etc/hostname"}}</tool_call>'),
    ])
    monkeypatch.setattr(pi, "chat", lambda *args, **kwargs: next(responses))
    monkeypatch.setattr(pi, "_save_probe", lambda *args, **kwargs: None)

    result = pi.format_detection_round(object())

    assert result["detected_format"] == "structured_tool_calls"
    assert result["xml_tool_call_test"] == {
        "pass": True,
        "detected_format": "xml_tool_call",
        "structured_tool_calls": False,
        "function_name": "read_file",
        "arguments": {"path": "/etc/hostname"},
        "raw_content_snippet": '<tool_call>{"name":"read_file","arguments":{"path":"/etc/hostname"}}</tool_call>',
        "error": None,
    }

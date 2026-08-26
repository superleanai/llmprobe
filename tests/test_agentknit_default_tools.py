"""Unit tests for the AKDEF agentknit default-tool compatibility round."""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import probe_inference as pi


def test_verify_effect_write_file(tmp_path):
    write_path = tmp_path / "w.txt"
    write_path.write_text("hello world\n")
    assert pi._akdef_verify_effect("write_file", "OK: wrote 12 bytes",
                                    str(write_path), str(tmp_path / "u.py"))

    write_path.write_text("wrong content\n")
    assert not pi._akdef_verify_effect("write_file", "OK: wrote 12 bytes",
                                        str(write_path), str(tmp_path / "u.py"))


def test_verify_effect_str_replace(tmp_path):
    update_path = tmp_path / "u.py"
    update_path.write_text("x = 42\n")
    assert pi._akdef_verify_effect("str_replace", "OK: replaced 1 of 1 occurrence(s)",
                                    str(tmp_path / "w.txt"), str(update_path))

    update_path.write_text("x = 1\n")
    assert not pi._akdef_verify_effect("str_replace", "OK: replaced 1 of 1 occurrence(s)",
                                        str(tmp_path / "w.txt"), str(update_path))


def test_verify_effect_read_file():
    assert pi._akdef_verify_effect("read_file", "<abc12345>hello agentknit\n</abc12345>",
                                    "/tmp/w", "/tmp/u")
    assert not pi._akdef_verify_effect("read_file", "<abc12345>something else\n</abc12345>",
                                        "/tmp/w", "/tmp/u")


def test_verify_effect_exec_shell():
    ok = json.dumps({"stdout": "", "stderr": "", "returncode": 0})
    bad = json.dumps({"stdout": "", "stderr": "no such file", "returncode": 2})
    assert pi._akdef_verify_effect("exec_shell", ok, "/tmp/w", "/tmp/u")
    assert not pi._akdef_verify_effect("exec_shell", bad, "/tmp/w", "/tmp/u")
    assert not pi._akdef_verify_effect("exec_shell", "not json", "/tmp/w", "/tmp/u")


def test_agentknit_default_tool_schema_matches_agentknit():
    from agentknit._core import _DEFAULT_TOOL_DISPATCH, _DEFAULT_TOOL_SCHEMA
    names = {t["function"]["name"] for t in _DEFAULT_TOOL_SCHEMA}
    assert names == {"read_file", "write_file", "str_replace", "exec_shell"}
    assert set(_DEFAULT_TOOL_DISPATCH.keys()) == names

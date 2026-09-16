"""Tests for the reports/<server>/<model>/ layout."""
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import probe_inference as pi


def test_server_slug_is_the_host_of_an_http_endpoint():
    assert pi._safe_server("https://api.kimi.com/coding/v1") == "api.kimi.com"
    assert pi._safe_server("https://openrouter.ai/api/v1") == "openrouter.ai"


def test_bare_hostname_is_accepted_as_given():
    assert pi._safe_server("githubcopilot.com") == "githubcopilot.com"


def test_a_script_target_is_not_itself_a_server():
    assert pi._safe_server("script:/home/x/bin/copilot-completions.py") == "unknown-server"
    assert pi._safe_server("") == "unknown-server"


def test_server_slug_never_contains_path_separators():
    for endpoint in ("https://host:8080/v1", "a/b c", "weird"):
        slug = pi._safe_server(endpoint)
        assert slug and "/" not in slug and ":" not in slug


def test_report_dir_nests_model_under_server():
    assert pi._report_dir("https://api.z.ai/api/coding/paas/v4", "glm-5.3") \
        == Path("reports") / "api.z.ai" / "glm-5.3"


def test_report_dir_prefers_an_explicit_server():
    assert pi._report_dir("script:/x.py", "gpt-5-mini", "api.githubcopilot.com") \
        == Path("reports") / "api.githubcopilot.com" / "gpt-5-mini"


def test_same_model_on_two_servers_gets_two_directories():
    a = pi._report_dir("https://api.deepseek.com/v1", "deepseek-v4-pro")
    b = pi._report_dir("https://openrouter.ai/api/v1", "deepseek-v4-pro")
    assert a != b


# -- upstream discovery through a wrapper script -------------------------------

WRAPPER = """#!/usr/bin/env python3
import json, sys
json.load(sys.stdin)
print(json.dumps({
    "id": "x", "model": "m",
    "choices": [{"index": 0, "finish_reason": "stop",
                 "message": {"role": "assistant", "content": "hi"}}],
    %s
}))
"""


def _write_wrapper(tmp_path, extra):
    path = tmp_path / "wrapper.py"
    path.write_text(WRAPPER % extra)
    return str(path)


def test_wrapper_reporting_its_upstream_names_the_server(tmp_path):
    script = _write_wrapper(tmp_path, '"x_upstream_endpoint": "https://api.individual.githubcopilot.com"')

    upstream = pi._script_upstream(script)

    assert upstream == "https://api.individual.githubcopilot.com"
    assert pi._safe_server(upstream) == "api.individual.githubcopilot.com"


def test_wrapper_without_an_upstream_field_yields_no_server(tmp_path):
    assert pi._script_upstream(_write_wrapper(tmp_path, '"object": "chat.completion"')) is None


def test_upstream_field_is_stripped_before_the_response_is_parsed(tmp_path):
    script = _write_wrapper(tmp_path, '"x_upstream_endpoint": "https://up.example"')

    resp = pi._ScriptChatCompletions(script).create(model="m", messages=[])

    assert resp.choices[0].message.content == "hi"
    assert not hasattr(resp, "x_upstream_endpoint")


def test_rerun_reuses_the_server_directory_of_an_earlier_report(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    d = tmp_path / "reports" / "api.individual.githubcopilot.com" / "gpt-5-mini"
    d.mkdir(parents=True)
    (d / "capabilities_gpt-5-mini.json").write_text(
        json.dumps({"endpoint": "script:/home/x/copilot.py", "model": "gpt-5-mini"}))

    assert pi._known_server_for("script:/home/x/copilot.py", "gpt-5-mini") \
        == "api.individual.githubcopilot.com"
    assert pi._known_server_for("script:/home/x/other.py", "gpt-5-mini") is None


def test_known_server_lookup_survives_a_missing_reports_tree(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    assert pi._known_server_for("https://api.kimi.com/coding/v1", "k3") is None
    assert not os.path.exists(tmp_path / "reports")

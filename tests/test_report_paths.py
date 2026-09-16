"""Tests for the reports/<server>/<model>/ layout."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import probe_inference as pi


def test_server_slug_is_the_host_of_an_http_endpoint():
    assert pi._safe_server("https://api.kimi.com/coding/v1") == "api.kimi.com"
    assert pi._safe_server("https://openrouter.ai/api/v1") == "openrouter.ai"


def test_server_slug_of_a_script_target_uses_the_script_name():
    assert pi._safe_server("script:/home/x/bin/copilot-gpt-5-mini-completions.py") \
        == "script_copilot-gpt-5-mini-completions"


def test_server_slug_never_contains_path_separators():
    for endpoint in ("https://host:8080/v1", "script:/a/b/c d.py", "", "weird"):
        slug = pi._safe_server(endpoint)
        assert slug and "/" not in slug and ":" not in slug


def test_report_dir_nests_model_under_server():
    assert pi._report_dir("https://api.z.ai/api/coding/paas/v4", "glm-5.3") \
        == Path("reports") / "api.z.ai" / "glm-5.3"


def test_same_model_on_two_servers_gets_two_directories():
    a = pi._report_dir("https://api.deepseek.com/v1", "deepseek-v4-pro")
    b = pi._report_dir("https://openrouter.ai/api/v1", "deepseek-v4-pro")
    assert a != b

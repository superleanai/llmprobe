"""Tests for the CORS preflight probe and its report rendering."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import probe_inference as pi


class _Headers(dict):
    """Case-insensitive header map, like http.client.HTTPMessage."""

    def get(self, name, default=None):  # noqa: A003 - mirrors email.message API
        for key, value in self.items():
            if key.lower() == str(name).lower():
                return value
        return default


def _preflight_headers(origin="*", methods="POST, OPTIONS",
                       allowed="Authorization, Content-Type"):
    return _Headers({
        "Access-Control-Allow-Origin": origin,
        "Access-Control-Allow-Methods": methods,
        "Access-Control-Allow-Headers": allowed,
    })


def test_classify_reads_access_control_headers(monkeypatch):
    headers = _Headers({"Access-Control-Allow-Origin": "*",
                        "Access-Control-Allow-Credentials": "true",
                        "Access-Control-Expose-Headers": "request-id"})

    classified = pi._cors_classify(headers)

    assert classified["allow_origin"] == "*"
    assert classified["allow_methods"] is None
    assert classified["allow_credentials"] is True
    assert classified["expose_headers"] == "request-id"


def test_wildcard_headers_kept_verbatim():
    headers = _Headers(dict(_preflight_headers(), Vary="Origin"))

    detail = pi._cors_wildcard(headers)

    assert detail["access-control-allow-origin"] == "*"
    assert detail["vary"] == "Origin"
    assert "access-control-expose-headers" not in detail


def test_preflight_pass_requires_origin_post_and_both_headers(monkeypatch):
    sent: list[dict] = []

    def fake_send(req):
        sent.append({"url": req.full_url, "method": req.get_method(),
                     "headers": {k: v for k, v in req.header_items()}})
        return 200, _preflight_headers()

    monkeypatch.setattr(pi, "_cors_send", fake_send)
    pi.ENDPOINT = "https://api.example.com/v1"

    result = pi._cors_run()
    pre = result["preflight"]

    assert sent[0]["url"] == "https://api.example.com/v1/chat/completions"
    assert sent[0]["method"] == "OPTIONS"
    assert sent[0]["headers"]["Origin"] == pi._CORS_REQUEST_ORIGIN
    assert sent[0]["headers"]["Access-control-request-method"] == "POST"
    assert pre["pass"] is True


def test_preflight_fails_when_authorization_header_not_allowed(monkeypatch):
    monkeypatch.setattr(pi, "_cors_send",
                        lambda req: (200, _preflight_headers(allowed="Content-Type")))
    pi.ENDPOINT = "https://api.example.com/v1"

    result = pi._cors_run()

    assert result["preflight"]["pass"] is False


def test_reduce_detects_wildcard_from_any_leg():
    cors = {"preflight": {"allow_origin": "*", "pass": True},
            "actual_response": {"allow_origin": "*", "pass": True}}

    reduced = pi._cors_reduce(cors)

    assert reduced["mode"] == "wildcard"
    assert reduced["allow_origin"] == "*"


def test_reduce_detects_reflected_origin():
    cors = {"preflight": {"allow_origin": "https://llmprobe.example", "pass": True,
                          "allow_credentials": True},
            "actual_response": {"allow_origin": "https://llmprobe.example", "pass": True}}

    reduced = pi._cors_reduce(cors)

    assert reduced["mode"] == "reflected"
    assert reduced["allow_origin"] == "https://llmprobe.example"
    assert reduced["allow_credentials"] is True


def test_reduce_reports_none_when_no_origin_anywhere():
    cors = {"preflight": {"allow_origin": None, "pass": False},
            "actual_response": {"allow_origin": None, "pass": False}}

    reduced = pi._cors_reduce(cors)

    assert reduced["mode"] == "none"
    assert reduced["allow_origin"] is None


def test_report_header_and_section_show_wildcard():
    report = pi.render_markdown_report({
        "model": "m",
        "cors_test": {"preflight": {"allow_origin": "*", "status": 204,
                                    "allow_methods": "POST,OPTIONS",
                                    "allow_headers": "Authorization,Content-Type",
                                    "pass": True},
                      "actual_response": {"allow_origin": "*", "status": 200,
                                          "pass": True}},
    })

    assert "- **CORS:** `*` (any origin)" in report
    assert "| `CORS` | `*` (any origin) |" in report
    assert "## CORS preflight test (`CORS`)" in report
    assert "any web page can call this endpoint" in report


def test_report_says_no_data_when_cors_key_absent():
    report = pi.render_markdown_report({"model": "m"})

    assert f"- **CORS:** *({pi._NO_DATA_NOTE})*" in report
    assert f"| `CORS` | *({pi._NO_DATA_NOTE})* |" in report
    assert "## CORS preflight test (`CORS`)\n\n*(no data, please rerun the probing)*" in report
    assert "- `CORS`: no data, please rerun the probing." in report


def test_report_key_present_but_none_says_not_run():
    report = pi.render_markdown_report({"model": "m", "cors_test": None})

    assert "- **CORS:** *(not run — rerun without `--no-cors-test`)*" in report
    assert "- `CORS` capability not tested (rerun without --no-cors-test)." in report


def test_report_lists_missing_cors_as_problem_when_no_origin():
    report = pi.render_markdown_report({
        "model": "m",
        "cors_test": {"preflight": {"allow_origin": None, "status": 200, "pass": False},
                      "actual_response": {"allow_origin": None, "status": 404,
                                          "pass": False}},
    })

    assert "`CORS`: no access-control-allow-origin" in report


def test_report_renders_error_for_script_target():
    report = pi.render_markdown_report({
        "model": "m",
        "cors_test": {"error": "CORS does not apply: the probe target is a local "
                               "script wrapper, not an HTTP endpoint"},
    })

    assert "Error: CORS does not apply" in report

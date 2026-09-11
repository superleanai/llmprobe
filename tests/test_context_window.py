"""Tests for context-window recovery and its report rendering."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import probe_inference as pi


def test_parse_limit_from_deepseek_style_message():
    limit, evidence = pi._ctx_parse_limit(
        "This model's maximum context length is 1048576 tokens. However, you "
        "requested 1200035 tokens (1200031 in the messages, 4 in the completion).")

    assert limit == 1048576
    assert "1048576" in evidence


def test_parse_limit_from_copilot_style_message():
    limit, evidence = pi._ctx_parse_limit(
        "prompt token count of 1200001 exceeds the limit of 922000")

    assert limit == 922000
    assert "922000" in evidence


def test_parse_limit_returns_none_when_no_number_is_named():
    """z.ai rejects an oversized prompt without ever naming its limit."""
    assert pi._ctx_parse_limit('{"error":{"code":"1261","message":"Prompt exceeds max length"}}') is None


def test_parse_limit_ignores_small_integers():
    """A stray small number must not be mistaken for a token limit."""
    assert pi._ctx_parse_limit("request failed after 429 retries") is None


def test_find_model_entry_matches_exact_id():
    payload = {"data": [{"id": "k3", "context_length": 1048576},
                        {"id": "k3-256k", "context_length": 262144}]}

    assert pi._ctx_find_model_entry(payload, "k3")["context_length"] == 1048576
    assert pi._ctx_find_model_entry(payload, "k3-256k")["context_length"] == 262144


def test_find_model_entry_matches_vendor_prefixed_id():
    payload = {"data": [{"id": "deepseek/deepseek-v4.1-flash", "context_length": 1048576}]}

    assert pi._ctx_find_model_entry(payload, "deepseek/deepseek-v4.1-flash") is not None


def test_metadata_limits_flattens_nested_copilot_shape():
    entry = {"id": "gpt-5-mini", "capabilities": {"limits": {
        "max_context_window_tokens": 264000,
        "max_prompt_tokens": 128000,
        "max_output_tokens": 64000,
        "vision": {"max_prompt_images": 1},
    }}}

    limits = pi._ctx_metadata_limits(entry)

    assert limits == {"max_context_window_tokens": 264000,
                      "max_prompt_tokens": 128000,
                      "max_output_tokens": 64000}


def test_report_header_and_section_show_context_window():
    report = pi.render_markdown_report({
        "model": "m",
        "context_window": {"tokens": 1048576, "kind": "context_window",
                           "source": "api_error_message",
                           "evidence": "maximum context length is 1048576"},
    })

    assert "- **Context window:** 1,048,576 tokens (context_window, from the endpoint's rejection message)" in report
    assert "## Context window" in report
    assert "| Tokens | 1,048,576 |" in report


def test_report_says_no_data_when_context_window_key_absent():
    report = pi.render_markdown_report({"model": "m"})

    assert "- **Context window:** *(no data, please rerun the probing)*" in report
    assert "Context window: no data, please rerun the probing." in report


def test_report_marks_lower_bound_as_such():
    report = pi.render_markdown_report({
        "model": "m",
        "context_window": {"tokens": 1200000, "kind": "context_window",
                           "source": "accepted_oversized_prompt", "lower_bound": True,
                           "evidence": "endpoint accepted a ~1200000-token prompt"},
    })

    assert "lower bound" in report
    assert "| Confidence | lower bound" in report

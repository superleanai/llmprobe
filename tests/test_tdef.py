"""Tests for the deferred-tool-loading probe (TDEF) and its report rendering."""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import probe_inference as pi


def _args(**overrides):
    defaults = {"tdef_responses_base": None, "tdef_anthropic_base": None,
                "tdef_anthropic_model": None}
    return argparse.Namespace(**dict(defaults, **overrides))


# -- tool set ------------------------------------------------------------------

def test_probe_tool_set_never_defers_everything():
    """Anthropic 400s on `All tools have defer_loading set` -- avoid it."""
    tools = pi._tdef_tools()

    assert any(t["deferrable"] for t in tools)
    assert any(not t["deferrable"] for t in tools)


def test_probe_tool_schemas_are_large_enough_to_measure():
    """The inline/deferred token difference must beat usage-counter noise."""
    tools = pi._tdef_tools()
    schema_chars = sum(len(json.dumps(t["parameters"])) for t in tools
                       if t["deferrable"])

    # ~4 chars/token, and the round needs hundreds of tokens per deferred tool.
    assert schema_chars / 4 > 900


# -- surface URL derivation ----------------------------------------------------

def test_anthropic_base_guessed_per_vendor_layout():
    assert pi._tdef_anthropic_base("https://api.deepseek.com") == \
        "https://api.deepseek.com/anthropic"
    assert pi._tdef_anthropic_base("https://api.moonshot.ai/v1") == \
        "https://api.moonshot.ai/anthropic"
    assert pi._tdef_anthropic_base("https://api.z.ai/api/paas/v4") == \
        "https://api.z.ai/api/anthropic"


# -- payload shapes ------------------------------------------------------------

def test_deferred_payload_carries_the_surface_specific_search_tool():
    tools = pi._tdef_tools()

    anthropic = pi._tdef_surface_payload("anthropic-messages", "m", tools, "hi", "deferred")
    responses = pi._tdef_surface_payload("openai-responses", "m", tools, "hi", "deferred")
    chat = pi._tdef_surface_payload("openai-completions", "m", tools, "hi", "deferred")

    assert anthropic["tools"][-1]["type"] == "tool_search_tool_bm25_20251119"
    assert responses["tools"][-1] == {"type": "tool_search"}
    assert chat["tools"][-1] == {"type": "tool_search"}
    assert responses["input"] == "hi"
    assert anthropic["messages"] == [{"role": "user", "content": "hi"}]
    # Anthropic names the schema field input_schema, not parameters.
    assert "input_schema" in anthropic["tools"][0]
    assert anthropic["tools"][0]["defer_loading"] is True


def test_inline_and_bare_payloads_carry_no_deferral():
    tools = pi._tdef_tools()

    inline = pi._tdef_surface_payload("openai-responses", "m", tools, "hi", "inline")
    bare = pi._tdef_surface_payload("openai-responses", "m", tools, "hi", "bare")

    assert all("defer_loading" not in t for t in inline["tools"])
    assert all(t.get("type") != "tool_search" for t in inline["tools"])
    assert "tools" not in bare


# -- verdict from token accounting --------------------------------------------

def test_matching_token_counts_mean_the_field_was_swallowed():
    verdict, detail = pi._tdef_classify_tokens(bare=100, inline=1700, deferred=1700)

    assert verdict == "accepted-but-ignored"
    assert "1600" in detail


def test_large_token_drop_is_the_positive_signal():
    verdict, _ = pi._tdef_classify_tokens(bare=100, inline=1700, deferred=300)

    assert verdict == "native"


def test_no_usage_reported_yields_no_token_verdict():
    verdict, detail = pi._tdef_classify_tokens(bare=None, inline=None, deferred=None)

    assert verdict is None
    assert "no input-token usage" in detail


# -- per-surface probe ---------------------------------------------------------

def _fake_post(responses):
    """Return a _tdef_post stand-in serving `responses` in order, plus a log."""
    sent: list[dict] = []

    def fake(url, payload, headers):
        sent.append({"url": url, "payload": payload, "headers": headers})
        return responses[len(sent) - 1]

    return fake, sent


def test_400_naming_the_field_is_a_clean_rejection(monkeypatch):
    fake, sent = _fake_post([(400, {"error": {"message": "unknown field defer_loading"}})])
    monkeypatch.setattr(pi, "_tdef_post", fake)

    result = pi._tdef_probe_surface("openai-responses", "https://x/responses", "m", {})

    assert result["verdict"] == "rejected"
    assert result["rejected_component"] == "defer_loading"
    assert len(sent) == 1  # a clean negative ends the probe


def test_rejection_says_which_half_of_the_feature_was_refused():
    """Whitelisting tool types and dropping `defer_loading` are separate failures."""
    assert pi._tdef_rejected_component("unknown variant `tool_search`") == "tool_search"
    assert pi._tdef_rejected_component("defer_loading is not supported") == "defer_loading"
    assert pi._tdef_rejected_component("defer_loading and tool_search rejected") == "both"
    assert pi._tdef_rejected_component("model is overloaded") is None
    # z.ai names neither half and only points at the offending array slot.
    assert pi._tdef_rejected_component("1214 tools[4].type:type is illegal",
                                       search_index=4) == "tool_search"
    assert pi._tdef_rejected_component("tools.0.something is not supported",
                                       search_index=4,
                                       deferred_indices=(0, 1, 2)) == "defer_loading"


def test_404_means_the_surface_is_absent(monkeypatch):
    fake, _ = _fake_post([(404, "<html>not found</html>")])
    monkeypatch.setattr(pi, "_tdef_post", fake)

    result = pi._tdef_probe_surface("openai-responses", "https://x/responses", "m", {})

    assert result["verdict"] == "n/a"


def test_200_with_unchanged_token_count_is_accepted_but_ignored(monkeypatch):
    ok = {"usage": {"prompt_tokens": 1700}}
    fake, sent = _fake_post([
        (200, ok),                                  # deferred
        (200, ok),                                  # inline
        (200, {"usage": {"prompt_tokens": 100}}),   # bare
        (200, {"choices": [{"message": {"tool_calls": [
            {"function": {"name": "warehouse_inventory_lookup"}}]}}]}),
    ])
    monkeypatch.setattr(pi, "_tdef_post", fake)

    result = pi._tdef_probe_surface("openai-completions", "https://x/c", "m", {})

    assert result["verdict"] == "accepted-but-ignored"
    assert result["reachability"] == "direct_call"
    assert result["input_tokens"] == {"bare": 100, "inline": 1700, "deferred": 1700}
    assert len(sent) == 4


def test_token_drop_plus_search_round_trip_is_native(monkeypatch):
    fake, _ = _fake_post([
        (200, {"usage": {"input_tokens": 320}}),
        (200, {"usage": {"input_tokens": 1800}}),
        (200, {"usage": {"input_tokens": 120}}),
        (200, {"content": [{"type": "tool_search_tool_result", "content": []}]}),
    ])
    monkeypatch.setattr(pi, "_tdef_post", fake)

    result = pi._tdef_probe_surface("anthropic-messages", "https://x/v1/messages", "m", {})

    assert result["verdict"] == "native"
    assert result["reachability"] == "search_then_call"


def test_search_round_trip_outranks_flat_token_counts(monkeypatch):
    """Some layers synthesise usage rather than measure it; the search call wins."""
    fake, _ = _fake_post([
        (200, {"usage": {"input_tokens": 1800}}),
        (200, {"usage": {"input_tokens": 1800}}),
        (200, {"usage": {"input_tokens": 100}}),
        (200, {"output": [{"type": "tool_search_call", "id": "ts_1"}]}),
    ])
    monkeypatch.setattr(pi, "_tdef_post", fake)

    result = pi._tdef_probe_surface("openai-responses", "https://x/responses", "m", {})

    assert result["verdict"] == "native"
    assert "overridden" in result["token_accounting"]


# -- aggregation and rendering -------------------------------------------------

def test_search_step_detected_whatever_the_vendor_calls_it():
    """OpenRouter namespaces the item type; Anthropic uses a result block."""
    assert pi._tdef_search_used(
        {"output": [{"type": "openrouter:tool_search", "query": "sku"}]})
    assert pi._tdef_search_used(
        {"content": [{"type": "tool_search_tool_result", "content": []}]})
    assert not pi._tdef_search_used({"output": [{"type": "message"}]})


def test_echoed_request_tools_are_not_mistaken_for_a_search_step():
    """Several endpoints echo the request's `tools` back in the response."""
    body = {"tools": [{"type": "tool_search"}],
            "output": [{"type": "message", "content": []}]}

    assert not pi._tdef_search_used(body)


def test_overall_verdict_takes_the_best_surface():
    surfaces = {"a": {"verdict": "n/a"}, "b": {"verdict": "accepted-but-ignored"},
                "c": {"verdict": "native"}}

    assert pi._tdef_overall(surfaces) == "native"
    assert pi._tdef_overall({}) == "n/a"


def test_table_cell_lists_every_surface():
    tdef = {"verdict": "accepted-but-ignored",
            "surfaces": {"openai-completions": {"verdict": "accepted-but-ignored"},
                         "openai-responses": {"verdict": "n/a"}}}

    cell = pi._tdef_cell(tdef)

    assert "accepted-but-ignored" in cell
    assert "openai-responses: n/a" in cell


def test_markdown_report_renders_the_tdef_section():
    output = {"model": "m", "endpoint": "https://x", "status": "ok",
              "tdef_test": {"verdict": "accepted-but-ignored", "surfaces": {
                  "openai-completions": {
                      "verdict": "accepted-but-ignored",
                      "input_tokens": {"bare": 100, "inline": 1700, "deferred": 1700},
                      "reachability": "direct_call",
                      "token_accounting": "tool schemas cost 1600 input tokens inline"},
                  "anthropic-messages": {"verdict": "n/a",
                                         "evidence": "HTTP 404: not served here"}}}}

    md = pi.render_markdown_report(output)

    assert "## Deferred tool loading (`TDEF`)" in md
    assert "| `TDEF` |" in md
    assert "100 / 1700 / 1700" in md
    assert "HTTP 404: not served here" in md


def test_markdown_distinguishes_never_run_from_skipped():
    absent = pi.render_markdown_report({"model": "m", "endpoint": "e", "status": "ok"})
    skipped = pi.render_markdown_report({"model": "m", "endpoint": "e", "status": "ok",
                                         "tdef_test": None})

    assert f"| `TDEF` | *({pi._NO_DATA_NOTE})* |" in absent
    assert "--no-tdef-test" in skipped


# -- round entry point ---------------------------------------------------------

def test_local_script_target_is_not_an_http_surface(monkeypatch):
    monkeypatch.setattr(pi, "ENDPOINT", "script:/home/x/completions.py")

    result = pi.tdef_test_round("key", _args())

    assert "does not apply" in result["error"]


def test_round_probes_all_three_surfaces_with_the_right_auth(monkeypatch):
    monkeypatch.setattr(pi, "ENDPOINT", "https://api.deepseek.com")
    monkeypatch.setattr(pi, "MODEL", "deepseek-v4-flash")
    seen: list[tuple] = []

    def fake_probe(surface, url, model, headers):
        seen.append((surface, url, model, headers))
        return {"verdict": "n/a", "evidence": ""}

    monkeypatch.setattr(pi, "_tdef_probe_surface", fake_probe)

    result = pi.tdef_test_round("sk-test", _args(tdef_anthropic_model="claude-opus-4"))

    urls = {s[0]: s[1] for s in seen}
    assert urls["openai-completions"] == "https://api.deepseek.com/chat/completions"
    assert urls["openai-responses"] == "https://api.deepseek.com/responses"
    assert urls["anthropic-messages"] == "https://api.deepseek.com/anthropic/v1/messages"
    anthropic = [s for s in seen if s[0] == "anthropic-messages"][0]
    assert anthropic[2] == "claude-opus-4"
    assert anthropic[3]["x-api-key"] == "sk-test"
    assert anthropic[3]["anthropic-version"] == pi._TDEF_ANTHROPIC_VERSION
    assert result["verdict"] == "n/a"

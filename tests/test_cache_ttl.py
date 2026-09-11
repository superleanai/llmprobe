"""Tests for the CACH prompt-cache TTL measurement round."""
import sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import probe_inference as pi


def _resp(prompt_tokens, cached, provider=None):
    payload = {"prompt_tokens": prompt_tokens, "completion_tokens": 3}
    if cached is not None:
        payload["prompt_cache_hit_tokens"] = cached
    usage = SimpleNamespace(
        prompt_tokens=prompt_tokens,
        completion_tokens=3,
        model_dump=lambda: dict(payload),
    )
    dump = {"usage": dict(payload)}
    if provider is not None:
        dump["provider"] = provider
    return SimpleNamespace(usage=usage, model_dump=lambda: dict(dump))


def _client(responses, calls=None):
    """A duck-typed client whose chat.completions.create pops canned responses."""
    def create(**kw):
        if calls is not None:
            calls.append(kw)
        return responses.pop(0)
    completions = SimpleNamespace(create=create)
    return SimpleNamespace(chat=SimpleNamespace(completions=completions))


def test_cached_tokens_extraction_variants():
    assert pi._cach_cached_tokens(_resp(100, 42)) == 42
    # prompt_tokens_details.cached_tokens fallback
    usage = SimpleNamespace(model_dump=lambda: {
        "prompt_tokens": 100,
        "prompt_tokens_details": {"cached_tokens": 77},
    })
    assert pi._cach_cached_tokens(SimpleNamespace(usage=usage)) == 77
    # no cache fields at all
    usage = SimpleNamespace(model_dump=lambda: {"prompt_tokens": 100})
    assert pi._cach_cached_tokens(SimpleNamespace(usage=usage)) is None
    assert pi._cach_cached_tokens(SimpleNamespace(usage=None)) is None


def test_ttl_bracketed_between_last_warm_and_first_cold():
    responses = [
        _resp(2600, 0),     # prime
        _resp(2600, 2500),  # immediate re-send: warm (prefix is cacheable)
        _resp(2600, 2500),  # delay 30s: warm
        _resp(2600, 0),     # delay 120s: cold -> stop
    ]
    with patch.object(pi.time, "sleep", lambda s: None):
        result = pi.cache_ttl_test_round(client=_client(responses))

    assert result["cache_observed"] is True
    assert result["ttl_min_seconds"] == 30
    assert result["ttl_max_seconds"] == 120
    assert set(result["samples"]) == {"prime", "confirm_0s", "delay_30s", "delay_120s"}


def test_no_cache_reported_is_informational_error():
    responses = [_resp(2600, None),  # prime
                 _resp(2600, None)]  # immediate re-send: still no cached tokens
    with patch.object(pi.time, "sleep", lambda s: None):
        result = pi.cache_ttl_test_round(client=_client(responses))

    assert result["cache_observed"] is False
    assert "error" in result


def test_no_cache_reuse_on_immediate_resend_reports_no_ttl():
    """A cold 0-delay re-send means the prefix is never served from cache, so no
    TTL may be claimed (this is what an unpinned load-balanced endpoint looks like)."""
    responses = [_resp(2600, 0, provider="SiliconFlow"),
                 _resp(2600, 0, provider="Morph")]
    with patch.object(pi.time, "sleep", lambda s: None):
        result = pi.cache_ttl_test_round(client=_client(responses))

    assert result["cache_observed"] is False
    assert result["ttl_min_seconds"] is None
    assert result["ttl_max_seconds"] is None
    assert result["provider"] == "SiliconFlow"


def test_still_warm_at_longest_delay():
    responses = [_resp(2600, 0), _resp(2600, 2500)]
    responses += [_resp(2600, 2500) for _ in pi._CACH_DELAYS_SECONDS]
    with patch.object(pi.time, "sleep", lambda s: None):
        result = pi.cache_ttl_test_round(client=_client(responses))

    assert result["ttl_max_seconds"] is None
    assert result["ttl_min_seconds"] == pi._CACH_DELAYS_SECONDS[-1]
    assert result["measured_ttl_seconds"] is None


def test_provider_is_pinned_on_every_resend():
    """OpenRouter-style load balancing must be neutralised: every re-send carries
    the provider that primed the cache, and the reported provider is recorded."""
    calls: list[dict] = []
    responses = [_resp(2600, 0, provider="Novita"),
                 _resp(2600, 2500, provider="Novita")]
    responses += [_resp(2600, 2500, provider="Novita") for _ in pi._CACH_DELAYS_SECONDS]
    with patch.object(pi.time, "sleep", lambda s: None):
        result = pi.cache_ttl_test_round(client=_client(responses, calls))

    assert result["provider"] == "Novita"
    assert result["provider_pinned"] == "Novita"
    # the prime call carries no pin; every later call pins the primed provider
    assert "extra_body" not in calls[0]
    for call in calls[1:]:
        assert call["extra_body"] == {"provider": {"order": ["Novita"],
                                                   "allow_fallbacks": False}}
    assert result["samples"]["delay_30s"]["provider"] == "Novita"


def test_provider_switch_despite_pin_is_not_a_ttl():
    """A cache miss served by a *different* upstream says nothing about expiry."""
    responses = [_resp(2600, 0, provider="SiliconFlow"),
                 _resp(2600, 2500, provider="SiliconFlow"),   # confirm: warm
                 _resp(2600, 0, provider="GMICloud")]         # delay 30s: other upstream
    with patch.object(pi.time, "sleep", lambda s: None):
        result = pi.cache_ttl_test_round(client=_client(responses))

    assert result["cache_observed"] is False
    assert result["provider_mismatch"] == {"expected": "SiliconFlow", "got": "GMICloud"}
    assert result["ttl_min_seconds"] is None and result["ttl_max_seconds"] is None


def test_cach_provider_extraction():
    assert pi._cach_provider(_resp(10, 0, provider="Morph")) == "Morph"
    assert pi._cach_provider(_resp(10, 0)) is None


def test_fixed_prefix_is_large_and_deterministic():
    a = pi._cach_build_prefix()
    b = pi._cach_build_prefix()
    assert a == b
    assert len(a) // 4 >= pi._CACH_TOKEN_TARGET

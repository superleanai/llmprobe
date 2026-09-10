"""Tests for the CACH prompt-cache TTL measurement round."""
import sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import probe_inference as pi


def _resp(prompt_tokens, cached):
    usage = SimpleNamespace(
        prompt_tokens=prompt_tokens,
        completion_tokens=3,
        model_dump=lambda: (
            {"prompt_tokens": prompt_tokens, "completion_tokens": 3,
             "prompt_cache_hit_tokens": cached}
            if cached is not None else
            {"prompt_tokens": prompt_tokens, "completion_tokens": 3}
        ),
    )
    return SimpleNamespace(usage=usage)


def _client(responses):
    """A duck-typed client whose chat.completions.create pops canned responses."""
    completions = SimpleNamespace(create=lambda **kw: responses.pop(0))
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
        _resp(2600, 2500),  # delay 30s: warm
        _resp(2600, 2400),  # delay 120s: warm
        _resp(2600, 0),     # delay 300s: cold -> stop
    ]
    with patch.object(pi.time, "sleep", lambda s: None):
        result = pi.cache_ttl_test_round(client=_client(responses))

    assert result["cache_observed"] is True
    assert result["ttl_min_seconds"] == 120
    assert result["ttl_max_seconds"] == 300
    assert result["measured_ttl_seconds"] == 210
    assert set(result["samples"]) == {"prime", "delay_30s", "delay_120s", "delay_300s"}


def test_no_cache_reported_is_informational_error():
    responses = [_resp(2600, None),  # prime
                 _resp(2600, None)]  # delay 30s: still no cached tokens -> stop
    with patch.object(pi.time, "sleep", lambda s: None):
        result = pi.cache_ttl_test_round(client=_client(responses))

    assert result["cache_observed"] is False
    assert "error" in result


def test_still_warm_at_longest_delay():
    responses = [_resp(2600, 0)] + [_resp(2600, 2500) for _ in pi._CACH_DELAYS_SECONDS]
    with patch.object(pi.time, "sleep", lambda s: None):
        result = pi.cache_ttl_test_round(client=_client(responses))

    assert result["ttl_max_seconds"] is None
    assert result["ttl_min_seconds"] == pi._CACH_DELAYS_SECONDS[-1]
    assert result["measured_ttl_seconds"] is None


def test_fixed_prefix_is_large_and_deterministic():
    a = pi._cach_build_prefix()
    b = pi._cach_build_prefix()
    assert a == b
    assert len(a) // 4 >= pi._CACH_TOKEN_TARGET

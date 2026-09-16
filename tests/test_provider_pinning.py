"""Tests for pinning an aggregator's routing to one upstream provider."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import probe_inference as pi


# -- routing block -------------------------------------------------------------

def test_routing_block_disables_fallbacks():
    """A silent reroute would measure a different server than the one named."""
    assert pi._provider_routing("deepinfra/fp8") == {
        "order": ["deepinfra/fp8"], "allow_fallbacks": False}


# -- report layout -------------------------------------------------------------

def test_pinned_runs_nest_under_the_provider():
    pinned = pi._report_dir("https://openrouter.ai/api/v1", "m", None, "deepinfra/fp8")

    assert pinned == Path("reports/openrouter.ai/m/deepinfra_fp8")


def test_unpinned_runs_keep_the_existing_two_level_path():
    """Existing reports must not move when provider pinning is available."""
    unpinned = pi._report_dir("https://openrouter.ai/api/v1", "m")

    assert unpinned == Path("reports/openrouter.ai/m")


def test_provider_slug_is_filesystem_safe():
    assert pi._safe_provider("deepinfra/fp8") == "deepinfra_fp8"
    assert pi._safe_provider("///") == "unknown-provider"


# -- pinned client -------------------------------------------------------------

class _FakeCompletions:
    def __init__(self):
        self.calls: list[dict] = []

    def create(self, **kwargs):
        self.calls.append(kwargs)
        return "response"


class _FakeChat:
    def __init__(self):
        self.completions = _FakeCompletions()


class _FakeClient:
    def __init__(self):
        self.chat = _FakeChat()
        self.models = "models-api"


def test_pinned_client_adds_the_provider_block_to_every_call():
    inner = _FakeClient()

    client = pi.PinnedClient(inner, "fireworks")
    client.chat.completions.create(model="m", messages=[])

    assert inner.chat.completions.calls[0]["extra_body"]["provider"] == {
        "order": ["fireworks"], "allow_fallbacks": False}


def test_pinned_client_preserves_a_call_s_own_extra_body():
    inner = _FakeClient()

    client = pi.PinnedClient(inner, "fireworks")
    client.chat.completions.create(model="m", messages=[],
                                   extra_body={"reasoning": {"effort": "low"}})

    extra = inner.chat.completions.calls[0]["extra_body"]
    assert extra["reasoning"] == {"effort": "low"}
    assert extra["provider"]["order"] == ["fireworks"]


def test_pinned_client_delegates_everything_else():
    """`/models` lookups and the like must keep working untouched."""
    client = pi.PinnedClient(_FakeClient(), "fireworks")

    assert client.models == "models-api"


# -- observed provider ---------------------------------------------------------

class _Resp:
    def __init__(self, extra):
        self.model_extra = extra


def test_observed_provider_read_from_the_response():
    assert pi._observed_provider(_Resp({"provider": "DeepInfra"})) == "DeepInfra"
    assert pi._observed_provider({"provider": "Fireworks"}) == "Fireworks"
    assert pi._observed_provider(_Resp({})) is None


def test_report_flags_a_pin_that_did_not_hold():
    line = pi._md_provider_line({"provider": "fireworks",
                                 "observed_provider": "DeepInfra"})

    assert "did not hold" in line


def test_report_accepts_a_pin_that_held():
    line = pi._md_provider_line({"provider": "deepinfra/fp8",
                                 "observed_provider": "DeepInfra"})

    assert "pinned, fallbacks disabled" in line


def test_unpinned_aggregator_report_says_the_result_is_not_reproducible():
    line = pi._md_provider_line({"endpoint": "https://openrouter.ai/api/v1"})

    assert "not pinned" in line
    assert "--provider" in line


def test_single_provider_endpoint_gets_no_provider_line():
    assert pi._md_provider_line({"endpoint": "https://api.deepseek.com"}) == ""


# -- listings ------------------------------------------------------------------

def test_quick_summary_label_names_the_provider():
    assert pi._summary_label({"model": "m", "provider": "fireworks"}, "?") == "m @ fireworks"
    assert pi._summary_label({"model": "m"}, "?") == "m"

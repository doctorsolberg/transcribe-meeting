import pytest

from transcribe_meeting import readable as R
from transcribe_meeting.readable import openai_client_options


def test_openai_client_uses_optional_compatible_base_url(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    monkeypatch.setenv("OPENAI_BASE_URL", "https://provider.test/v1")

    assert openai_client_options() == {
        "api_key": "test-key",
        "base_url": "https://provider.test/v1",
    }


@pytest.mark.parametrize("base_url", [None, "", "  \t"])
def test_openai_client_explicitly_passes_the_default_endpoint(monkeypatch, base_url):
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    if base_url is None:
        monkeypatch.delenv("OPENAI_BASE_URL", raising=False)
    else:
        monkeypatch.setenv("OPENAI_BASE_URL", base_url)

    assert openai_client_options() == {
        "api_key": "test-key",
        "base_url": "https://api.openai.com/v1",
    }


@pytest.mark.parametrize(
    "url",
    [
        "http://provider.test/v1",
        "http://localhost:8000/v1",
        "http://127.1/v1",
        "http://0x7f000001/v1",
        "http://[::ffff:127.0.0.1]/v1",
        "ftp://provider.test/v1",
        "https:/provider.test/v1",
        "https://private-value@provider.test/v1",
        "https://user:private-value@provider.test/v1",
        "https://@provider.test/v1",
        "https://provider.test/v1?key=private-value",
        "https://provider.test/v1?",
        "https://provider.test/v1#private-value",
        "https://provider.test/v1#",
        "https://provider.test:bad/v1",
        "https://provider.test:65536/v1",
        "https://provider.test:/v1",
        "https://provider.test:0/v1",
        "https://[2001:db8::1]extra/v1",
        "https://provider.test\\other.test/v1",
        "https://provider.test/v 1",
        "https://provider.test\n/v1",
    ],
)
def test_base_url_policy_rejects_unsafe_or_malformed_values(monkeypatch, url):
    monkeypatch.setenv("OPENAI_BASE_URL", url)
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    with pytest.raises((RuntimeError, ValueError), match="OPENAI_BASE_URL") as error:
        openai_client_options()
    assert "private-value" not in str(error.value)


@pytest.mark.parametrize(
    ("url", "host"),
    [
        ("http://127.0.0.1:8080/v1/private-path", "127.0.0.1:8080"),
        ("http://127.20.1.2/v1", "127.20.1.2"),
        ("http://[::1]:8080/v1", "[::1]:8080"),
        ("https://[::1]/v1", "[::1]"),
    ],
)
def test_readiness_allows_explicit_literal_keyless_loopback(monkeypatch, url, host):
    monkeypatch.setenv("OPENAI_BASE_URL", url)
    monkeypatch.setenv("OPENAI_API_KEY", "  ")
    state = R.readable_readiness()
    assert (state.ready, state.host, state.diagnostic) == (True, host, None)
    assert openai_client_options() == {"api_key": "", "base_url": url}


def test_readiness_rejects_missing_key_for_remote_endpoint(monkeypatch):
    monkeypatch.setenv("OPENAI_BASE_URL", "https://provider.test/private-path")
    monkeypatch.setenv("OPENAI_API_KEY", "  ")
    state = R.readable_readiness()
    assert not state.ready
    assert state.host == "provider.test"
    assert "OPENAI_API_KEY" in state.diagnostic
    assert "private-path" not in repr(state)


def test_readiness_checks_request_settings_without_exposing_bad_values(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    monkeypatch.delenv("OPENAI_BASE_URL", raising=False)
    monkeypatch.setenv("TRANSCRIBE_READABLE_API", "private-value")
    state = R.readable_readiness()
    assert not state.ready
    assert state.host == "api.openai.com"
    assert "TRANSCRIBE_READABLE_API" in state.diagnostic
    assert "private-value" not in repr(state)
    assert "test-key" not in repr(state)

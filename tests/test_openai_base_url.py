from transcribe_meeting.readable import openai_client_options


def test_openai_client_uses_optional_compatible_base_url(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    monkeypatch.setenv("OPENAI_BASE_URL", "https://provider.test/v1")

    assert openai_client_options() == {
        "api_key": "test-key",
        "base_url": "https://provider.test/v1",
    }


def test_openai_client_keeps_default_endpoint_when_base_url_is_missing(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    monkeypatch.delenv("OPENAI_BASE_URL", raising=False)

    assert openai_client_options() == {"api_key": "test-key"}

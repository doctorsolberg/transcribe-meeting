from transcribe_meeting.cli import _resolved_run_config


def _config(**overrides):
    values = {
        "language": "ru",
        "audio_stream": None,
        "speakers": None,
        "min_speakers": 1,
        "max_speakers": 8,
        "clean": ["raw", "light"],
        "profile_name": "balanced",
        "cpu_threads": None,
        "ffmpeg_threads": None,
        "chunk_minutes": None,
        "interactive_names": True,
        "resume": True,
        "keep_working_audio": False,
        "asr_preprocess": "none",
        "asr_compression": "none",
        "context_prompt": "  тема  ",
        "llm_model": "gpt-5.6-terra",
    }
    values.update(overrides)
    return _resolved_run_config(**values)


def test_startup_configuration_resolves_balanced_defaults():
    config = _config()

    assert config["profile"] == "balanced"
    assert config["chunk_minutes"] is None
    assert config["context_prompt"] == "тема"
    assert config["cpu_threads"] >= 1
    assert config["ffmpeg_threads"] >= 1
    assert config["diarization_provider"] == "foxnose"
    assert config["diarization_model"] == "diarize/wespeaker-voxceleb-resnet34-LM"
    assert config["asr_provider"] == "local"
    assert config["asr_backend"] == "gigaam"


def test_startup_configuration_resolves_pyannote_diarization_model():
    config = _config(diarization_provider="pyannote")

    assert config["diarization_provider"] == "pyannote"
    assert config["diarization_model"] == "pyannote/speaker-diarization-community-1"


def test_startup_configuration_caps_groq_chunks_to_free_tier_limit():
    """Chunk capping is keyed off any groq model actually present in asr_models."""
    config = _config(asr_models=["groq-whisper-large-v3"], chunk_minutes=30)

    assert config["chunk_minutes"] == 10
    assert config["asr_provider"] == "groq"


def test_startup_configuration_does_not_cap_chunks_for_an_all_local_list():
    config = _config(chunk_minutes=30)

    assert config["chunk_minutes"] is None


def test_startup_configuration_caps_chunks_for_a_mixed_local_and_groq_list():
    config = _config(
        asr_models=["multilingual_large_ctc", "v3_rnnt", "groq-whisper-large-v3"],
        chunk_minutes=30,
    )

    assert config["chunk_minutes"] == 10
    assert config["asr_provider"] == "mixed"


def test_startup_configuration_defaults_to_the_two_default_asr_models():
    config = _config()

    assert config["asr_models"] == ["multilingual_large_ctc", "v3_rnnt"]


def test_startup_configuration_accepts_an_explicit_single_model():
    config = _config(asr_models=["v3_rnnt"])

    assert config["asr_models"] == ["v3_rnnt"]


def test_default_readable_model_is_luna():
    """Owner's choice: luna edits more freely, and the change log keeps it reviewable."""
    import inspect

    from transcribe_meeting.cli import run

    assert inspect.signature(run).parameters["llm_model"].default.default == "gpt-5.6-luna"


def test_startup_configuration_resolves_deepgram_diarization_model():
    config = _config(diarization_provider="deepgram")

    assert config["diarization_provider"] == "deepgram"
    assert config["diarization_model"] == "deepgram/nova-3"


def test_startup_configuration_labels_a_deepgram_only_asr_run_correctly():
    """A single-provider deepgram run must not fall through to the "mixed" label —
    only a genuine mix of providers should say that."""
    config = _config(asr_models=["deepgram-nova-3"])

    assert config["asr_provider"] == "deepgram"
    assert config["asr_backend"] == "deepgram"
    # deepgram never chunks, so
    # chunk_minutes must not get the groq free-tier cap applied to it.
    assert config["chunk_minutes"] is None


def test_startup_configuration_does_not_cap_chunks_for_a_deepgram_only_list():
    config = _config(asr_models=["deepgram-nova-3"], chunk_minutes=30)

    assert config["chunk_minutes"] is None

import pytest

from transcribe_meeting.asr import (
    ASR_MODELS,
    DEEPGRAM_API_MODEL,
    DEEPGRAM_ASR_MODEL_NAME,
    DEFAULT_ASR_MODELS,
    GROQ_ASR_MODEL_NAME,
    GROQ_MODEL,
    degrade_deepgram_without_key,
    degrade_groq_without_key,
    validate_asr_models,
)
from transcribe_meeting.cli import _asr_models_for_provider, _parse_asr_models


def test_default_models_are_two_known_local_registry_entries():
    assert len(DEFAULT_ASR_MODELS) == 2
    assert all(name in ASR_MODELS for name in DEFAULT_ASR_MODELS)
    assert all(ASR_MODELS[name]["provider"] == "local" for name in DEFAULT_ASR_MODELS)


def test_registry_has_exactly_one_groq_entry():
    groq_entries = [name for name, info in ASR_MODELS.items() if info["provider"] == "groq"]
    assert groq_entries == [GROQ_ASR_MODEL_NAME]
    assert ASR_MODELS[GROQ_ASR_MODEL_NAME]["api_model"] == GROQ_MODEL


def test_unknown_model_name_is_rejected():
    with pytest.raises(ValueError, match="Unknown ASR models"):
        validate_asr_models(["not_a_real_model"])


def test_more_than_three_models_is_rejected():
    with pytest.raises(ValueError, match="No more than three"):
        validate_asr_models(["v3_rnnt", "v3_ctc", "v2_rnnt", "v2_ctc"])


def test_duplicate_model_is_rejected():
    with pytest.raises(ValueError, match="must not repeat"):
        validate_asr_models(["v3_rnnt", "v3_rnnt"])


def test_empty_model_list_is_rejected():
    with pytest.raises(ValueError):
        validate_asr_models([])


def test_more_than_one_groq_model_is_rejected():
    # There is only one groq entry in the registry, so this can only be triggered by
    # duplicating it — which the duplicate check would also catch first. Still worth
    # asserting the groq-specific message exists and fires before duplicates matter
    # in a hypothetical multi-groq registry.
    with pytest.raises(ValueError):
        validate_asr_models([GROQ_ASR_MODEL_NAME, GROQ_ASR_MODEL_NAME])


def test_valid_single_double_and_triple_selections_pass():
    validate_asr_models(["v3_rnnt"])
    validate_asr_models(["multilingual_large_ctc", "v3_rnnt"])
    validate_asr_models(["multilingual_large_ctc", "v3_rnnt", GROQ_ASR_MODEL_NAME])


def test_cli_parser_reports_bad_parameter_for_unknown_model():
    import typer

    with pytest.raises(typer.BadParameter):
        _parse_asr_models("v3_rnnt,made_up_model")


def test_provider_flag_rejects_a_mismatched_explicit_groq_model_list():
    import typer

    with pytest.raises(typer.BadParameter, match="groq"):
        _asr_models_for_provider("groq", ["v3_rnnt"], provider_explicit=True, models_explicit=True)


def test_provider_flag_groq_default_invocation_is_not_broken_by_the_default_models():
    """--asr-provider groq without --asr-models must still resolve to the groq model."""
    resolved = _asr_models_for_provider(
        "groq",
        ["multilingual_large_ctc", "v3_rnnt"],
        provider_explicit=True,
        models_explicit=False,
    )

    assert resolved == [GROQ_ASR_MODEL_NAME]


def test_provider_flag_local_rejects_a_mismatched_explicit_groq_model():
    import typer

    with pytest.raises(typer.BadParameter, match="local"):
        _asr_models_for_provider(
            "local",
            ["v3_rnnt", GROQ_ASR_MODEL_NAME],
            provider_explicit=True,
            models_explicit=True,
        )


def test_provider_flag_not_explicit_lets_asr_models_mix_providers_freely():
    """The main new-usage path: default --asr-provider, explicit mixed --asr-models."""
    resolved = _asr_models_for_provider(
        "local",
        ["multilingual_large_ctc", "v3_rnnt", GROQ_ASR_MODEL_NAME],
        provider_explicit=False,
        models_explicit=True,
    )

    assert resolved == ["multilingual_large_ctc", "v3_rnnt", GROQ_ASR_MODEL_NAME]


def test_local_keeps_both_models_when_provider_not_explicit():
    assert _asr_models_for_provider(
        "local",
        ["multilingual_large_ctc", "v3_rnnt"],
        provider_explicit=False,
        models_explicit=False,
    ) == ["multilingual_large_ctc", "v3_rnnt"]


def test_degrade_drops_groq_model_without_a_key_and_keeps_locals():
    kept, dropped = degrade_groq_without_key(
        ["multilingual_large_ctc", "v3_rnnt", GROQ_ASR_MODEL_NAME], has_key=False
    )

    assert kept == ["multilingual_large_ctc", "v3_rnnt"]
    assert dropped == [GROQ_ASR_MODEL_NAME]


def test_degrade_drops_nothing_when_key_is_present():
    kept, dropped = degrade_groq_without_key(
        ["multilingual_large_ctc", GROQ_ASR_MODEL_NAME], has_key=True
    )

    assert kept == ["multilingual_large_ctc", GROQ_ASR_MODEL_NAME]
    assert dropped == []


def test_degrade_can_leave_the_list_empty_when_groq_was_the_only_model():
    kept, dropped = degrade_groq_without_key([GROQ_ASR_MODEL_NAME], has_key=False)

    assert kept == []
    assert dropped == [GROQ_ASR_MODEL_NAME]


def test_registry_has_exactly_one_deepgram_entry():
    deepgram_entries = [name for name, info in ASR_MODELS.items() if info["provider"] == "deepgram"]
    assert deepgram_entries == [DEEPGRAM_ASR_MODEL_NAME]
    assert ASR_MODELS[DEEPGRAM_ASR_MODEL_NAME]["api_model"] == DEEPGRAM_API_MODEL


def test_more_than_one_deepgram_model_is_rejected():
    with pytest.raises(ValueError):
        validate_asr_models([DEEPGRAM_ASR_MODEL_NAME, DEEPGRAM_ASR_MODEL_NAME])


def test_deepgram_can_be_combined_with_local_and_groq_models():
    validate_asr_models(["v3_rnnt", DEEPGRAM_ASR_MODEL_NAME])
    validate_asr_models(["v3_rnnt", GROQ_ASR_MODEL_NAME, DEEPGRAM_ASR_MODEL_NAME])


def test_degrade_deepgram_drops_the_model_without_a_key_and_keeps_locals():
    kept, dropped = degrade_deepgram_without_key(
        ["multilingual_large_ctc", DEEPGRAM_ASR_MODEL_NAME], has_key=False
    )

    assert kept == ["multilingual_large_ctc"]
    assert dropped == [DEEPGRAM_ASR_MODEL_NAME]


def test_degrade_deepgram_keeps_the_model_when_a_key_is_present():
    kept, dropped = degrade_deepgram_without_key(
        ["multilingual_large_ctc", DEEPGRAM_ASR_MODEL_NAME], has_key=True
    )

    assert kept == ["multilingual_large_ctc", DEEPGRAM_ASR_MODEL_NAME]
    assert dropped == []


def test_degrade_deepgram_does_not_touch_a_groq_model():
    """Each network provider is degraded independently."""
    kept, dropped = degrade_deepgram_without_key(
        [GROQ_ASR_MODEL_NAME, DEEPGRAM_ASR_MODEL_NAME], has_key=False
    )

    assert kept == [GROQ_ASR_MODEL_NAME]
    assert dropped == [DEEPGRAM_ASR_MODEL_NAME]

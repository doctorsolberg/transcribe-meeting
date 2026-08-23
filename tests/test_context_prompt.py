from transcribe_meeting.core import make_asr_prompt


def test_context_prompt_and_glossary_are_explicit_asr_guidance():
    prompt = make_asr_prompt(
        [{"canonical": "OpenAI", "variants": ["Опен АИ"]}],
        "Тема: планирование продукта. Цель: согласовать релиз.",
    )

    assert "не добавляй фактов" in prompt
    assert "планирование продукта" in prompt
    assert "OpenAI" in prompt


def test_empty_context_does_not_add_artificial_asr_prompt():
    assert make_asr_prompt([], "   ") == ""

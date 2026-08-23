from transcribe_meeting.cleaning import glossary_replace, light_clean


def test_light_removes_only_explicit_fillers_and_exact_repetitions():
    """No trailing period any more: an utterance is a fragment, and render.blocks
    joins consecutive same-speaker utterances into one block, where only the whole
    block (see finish_light_block in render.py) can decide sentence-final punctuation."""
    assert light_clean("эм ну ну Open AI Open AI") == "ну Open AI"
    assert light_clean("ну вот значит короче") == "ну вот значит короче"


def test_glossary_only_replaces_explicit_variants():
    terms = [{"canonical": "OpenAI", "variants": ["Open AI"]}]
    assert glossary_replace("Open AI и OpenA1", terms) == "OpenAI и OpenA1"

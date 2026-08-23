from transcribe_meeting.schemas import ReadableItem, ReadablePayload


def test_strict_readable_schema_requires_every_item_property():
    schema = ReadablePayload.model_json_schema()
    item = schema["$defs"]["ReadableItem"]

    assert set(item["required"]) == set(item["properties"])
    assert (
        ReadableItem.model_validate(
            {"utterance_id": "u1", "readable_text": "Текст", "concerns": []}
        ).concerns
        == []
    )

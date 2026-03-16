from inbox_vault.llm import extract_first_json


def test_extract_first_json_basic():
    text = 'prefix {"category":"Important","importance":9,"action":"reply","summary":"x"} suffix'
    obj = extract_first_json(text)
    assert obj is not None
    assert obj["category"] == "Important"


def test_extract_first_json_none_when_missing():
    assert extract_first_json("no json here") is None


def test_extract_first_json_skips_invalid_then_parses_valid_object():
    text = 'bad {not: json} prefix then valid {"ok": true, "n": 2}'
    obj = extract_first_json(text)
    assert obj == {"ok": True, "n": 2}


def test_extract_first_json_handles_nested_objects_and_braces_in_strings():
    text = 'prefix {"a":{"b":1},"note":"brace: } and literal"} suffix'
    obj = extract_first_json(text)
    assert obj is not None
    assert obj["a"]["b"] == 1
    assert obj["note"].startswith("brace")

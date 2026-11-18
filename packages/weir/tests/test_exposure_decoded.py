"""The exposure scan's decoded view (docs/scan.md, "JSON escapes").

JSON lets a writer escape any character as `\\uXXXX`, so a key can sit in
an exported string with some of its letters escaped. Where a string is
JSON, or holds a JSON object or array, the scan also reads it with its
JSON escapes undone, by the decoded view the canary judge reads
(`weir.json_escapes`).

Every expectation here is derived, never restated: an escaped export must
scan, gauge and report exactly as the same export written plainly does,
and the plain export is run through the same command to say what that is.
Escaped text is built with the standard library and the judge's test
helper (canary_fixtures.u_escaped), never with the decoder under test. The
view's bound is read from docs/scan.md.
"""

from __future__ import annotations

import json
import re
import time
from collections.abc import Callable
from pathlib import Path

import canary_fixtures as fx
import pytest
from click.testing import CliRunner

from weir.adapters.otel import decode_input
from weir.adapters.otel.exposure import _json_texts, scan_surface
from weir.canary.judge import _json_spans
from weir.catalog import DEFAULT_CATALOG
from weir.cli.main import main
from weir.exposure import ExposureScan, classify_exposure

_DOC = (Path(__file__).parents[3] / "docs" / "scan.md").read_text(encoding="utf-8")


def _bound() -> int:
    found = re.search(r"unescaped again, up to \*\*(\d+)\*\* times", _DOC)
    assert found, "docs/scan.md states the decoded view's bound"
    return int(found.group(1))


BOUND = _bound()

# Fake keys, one per key-shaped class, each eligible as written plainly.
_KEYS = {
    "openai": "sk-proj-Qh7Rk2Ls9Vn4Xb6Zt1Wc8Mp3Jd5Fg0Yu2Ae7Ri4",
    "anthropic": "sk-ant-api03-" + "Qh7Rk2Ls9Vn4Xb6Zt1Wc8Mp3Jd5Fg0Yu2Ae7Ri4" * 2,
    "github": "ghp_Qh7Rk2Ls9Vn4Xb6Zt1Wc8Mp3Jd5Fg0Yu2Ae7",
    "aws": "AKIAQH7RK2LS9VN4XB6Z",
}
_KEY = _KEYS["openai"]

# Which characters of a key are written `\uXXXX`: any subset may be.
_PATTERNS: dict[str, Callable[[int, int], bool]] = {
    "first": lambda i, n: i == 0,
    "last": lambda i, n: i == n - 1,
    "middle": lambda i, n: i == n // 2,
    "every-fifth": lambda i, n: i % 5 == 0,
    "every": lambda i, n: True,
}


def _escaped(value: str, pattern: str = "first", *, upper: bool = False) -> str:
    picks = _PATTERNS[pattern]
    text = fx.u_escaped(value, lambda i: picks(i, len(value)), upper=upper)
    assert value not in text
    return text


def _attr(key: str, value: object) -> dict[str, object]:
    wrapped = value if isinstance(value, dict) else {"stringValue": value}
    return {"key": key, "value": wrapped}


def _spans(*attributes: dict[str, object], events: list[object] | None = None) -> str:
    span: dict[str, object] = {
        "traceId": "11" * 16,
        "spanId": "aa" * 8,
        "name": "execute_tool get_weather",
        "attributes": [
            _attr("gen_ai.operation.name", "execute_tool"),
            _attr("gen_ai.tool.name", "get_weather"),
            *attributes,
        ],
    }
    if events is not None:
        span["events"] = events
    return json.dumps({"resourceSpans": [{"scopeSpans": [{"spans": [span]}]}]})


def _logs(record: dict[str, object]) -> str:
    return json.dumps({"resourceLogs": [{"scopeLogs": [{"logRecords": [record]}]}]})


def _scan(body: str) -> ExposureScan:
    return classify_exposure(scan_surface(decode_input(body.encode())), DEFAULT_CATALOG)


def _cli(body: str) -> dict[str, object]:
    """Everything a user sees: scan's exit code and text, its HTML report,
    and the gauge's exposure lines."""
    runner = CliRunner()
    with runner.isolated_filesystem():
        Path("export.json").write_text(body, encoding="utf-8")
        scan = runner.invoke(main, ["scan", "export.json", "--report", "r.html"])
        html = Path("r.html").read_text(encoding="utf-8")
        gauge = runner.invoke(main, ["gauge", "export.json"])
    assert scan.exception is None or isinstance(scan.exception, SystemExit), scan.output
    assert gauge.exit_code == 0, gauge.output
    return {
        "exit": scan.exit_code,
        "scan": scan.output,
        "html": html,
        "gauge": [ln for ln in gauge.output.splitlines() if ln.startswith(("exposure", "  "))],
    }


# --- found the same as the plain value --------------------------------------

# Where a JSON text sits, as (description, export for a JSON text). The
# `{}` in each document is replaced by the key, escaped or plain.
_DOCUMENTS: dict[str, str] = {
    "tool-call arguments": '{"query": "weather", "note": "{}"}',
    "under a credential field name": '{"query": "weather", "api_key": "{}"}',
    "a bearer header under a credential field name": '{"password": "Bearer {}"}',
    "more than the key under a credential field name": '{"api_key": "{} trailing words"}',
    "a JSON string": '"{}"',
    "an array": '[1, null, "{}"]',
    "an object key": '{"{}": true}',
    "nested": '{"a": {"b": ["{}"]}}',
    "an object inside other text": 'calling get_weather with {"note": "{}"} now',
    "an array inside other text": 'args=["weather", "{}"]',
}

_PLACES: dict[str, Callable[[str], str]] = {
    "gen_ai.tool.call.arguments": lambda t: _spans(_attr("gen_ai.tool.call.arguments", t)),
    "a vendor attribute": lambda t: _spans(_attr("acme.agent.payload", t)),
    "an array element": lambda t: _spans(
        _attr("acme.list", {"arrayValue": {"values": [{"stringValue": "x"}, {"stringValue": t}]}})
    ),
    "a kvlist entry": lambda t: _spans(
        _attr("acme.map", {"kvlistValue": {"values": [_attr("inner", t)]}})
    ),
    "an event attribute": lambda t: _spans(
        events=[{"name": "tool.call", "attributes": [_attr("payload", t)]}]
    ),
    "a log record body": lambda t: _logs({"body": {"stringValue": t}}),
    "a log record attribute": lambda t: _logs({"attributes": [_attr("payload", t)]}),
}


def _fill(document: str, value: str) -> str:
    return document.replace("{}", value)


@pytest.mark.parametrize("document", sorted(_DOCUMENTS))
@pytest.mark.parametrize("place", sorted(_PLACES))
def test_an_escaped_key_scans_the_same_as_the_plain_key(place: str, document: str) -> None:
    export = _PLACES[place]
    plain = _cli(export(_fill(_DOCUMENTS[document], _KEY)))
    assert plain["exit"] == 1, plain["scan"]
    escaped = _cli(export(_fill(_DOCUMENTS[document], _escaped(_KEY))))
    assert escaped == plain


@pytest.mark.parametrize("upper", [False, True])
@pytest.mark.parametrize("pattern", sorted(_PATTERNS))
@pytest.mark.parametrize("kind", sorted(_KEYS))
@pytest.mark.parametrize("field", ["note", "api_key"])
def test_any_letters_of_any_key_may_be_escaped(
    kind: str, pattern: str, upper: bool, field: str
) -> None:
    document = '{"query": "weather", "' + field + '": "{}"}'
    key = _KEYS[kind]
    plain = _scan(_spans(_attr("gen_ai.tool.call.arguments", _fill(document, key))))
    assert [h.eligible for h in plain.hits] == [True], plain.hits
    escaped = _scan(
        _spans(
            _attr(
                "gen_ai.tool.call.arguments", _fill(document, _escaped(key, pattern, upper=upper))
            )
        )
    )
    assert escaped == plain


def _nested(value: str, unescapings: int) -> str:
    """A JSON string that holds `value` only after `unescapings`
    unescapings: its first letter escaped, then quoted again by json."""
    text = '"' + _escaped(value) + '"'
    for _ in range(unescapings - 1):
        text = json.dumps(text)
    return text


def _plain_nested(value: str, unescapings: int) -> str:
    """The same nesting with the value written plainly."""
    text = '"' + value + '"'
    for _ in range(unescapings - 1):
        text = json.dumps(text)
    return text


@pytest.mark.parametrize("unescapings", [2, 3, BOUND])
def test_json_held_as_a_string_inside_json_is_read_too(unescapings: int) -> None:
    plain = _cli(_spans(_attr("acme.payload", _plain_nested(_KEY, unescapings))))
    assert plain["exit"] == 1, plain["scan"]
    escaped = _cli(_spans(_attr("acme.payload", _nested(_KEY, unescapings))))
    assert escaped == plain


@pytest.mark.parametrize("wrap", ["as is", "held as a string inside JSON"])
def test_letters_escaped_in_a_token_value_read_as_those_letters(wrap: str) -> None:
    """Under `token` a value needs a letter and a digit of its own; in JSON,
    `\\u0041` is the letter A, so it scans as the value written plainly."""

    def export(document: str) -> str:
        text = document if wrap == "as is" else json.dumps({"body": document})
        return _spans(_attr("probe", text))

    plain = _scan(export('{"token": "AB123456789"}'))
    assert plain.hits, "the value written plainly is a match"
    assert _scan(export('{"token": "\\u0041\\u0042123456789"}')) == plain


# --- one finding, not two ---------------------------------------------------


@pytest.mark.parametrize(
    "document",
    [
        '{"q": "line\\nbreak", "note": "{plain}"}',  # an escape elsewhere
        '{"a": "{plain}", "b": "{escaped}"}',  # written both ways
        '{"api_key": "{plain}", "note": "{escaped}"}',
        '["{escaped}", "{escaped}"]',
    ],
)
def test_a_key_found_as_written_and_decoded_is_one_finding(document: str) -> None:
    def written(escaped: str) -> str:
        return _spans(
            _attr(
                "gen_ai.tool.call.arguments",
                document.replace("{plain}", _KEY).replace("{escaped}", escaped),
            )
        )

    plain = _cli(written(_KEY))
    escaped = _cli(written(_escaped(_KEY, "middle")))
    assert escaped == plain
    assert len(_scan(written(_escaped(_KEY, "middle"))).hits) == 1


@pytest.mark.parametrize(
    "document",
    [
        '{"a": "{key}", "b": "{key}ZZA"{escape}}',
        '{"a": "{key}", "b": "Authorization: Bearer {key}XYZ"{escape}}',
        '{"body": "{\\"a\\": \\"{key}\\", \\"b\\": \\"{key}ZZA\\"}"{escape}}',
    ],
)
def test_a_longer_key_that_contains_another_is_not_the_same_finding(document: str) -> None:
    """Two distinct keys, one the start of the other, are two findings, and
    an escape elsewhere in the string, which makes the scan read its decoded
    view too, leaves both standing."""

    def written(escape: str) -> str:
        return _spans(
            _attr(
                "gen_ai.tool.call.arguments",
                document.replace("{key}", _KEY).replace("{escape}", escape),
            )
        )

    plain = _scan(written(""))
    assert len({h.fingerprint for h in plain.hits if h.eligible}) == 2, plain.hits
    assert _scan(written(', "c": "x\\ny"')) == plain


# A key of one class with a key of another inside it: plain text holds both.
_KEY_IN_KEY = [
    "sk-" + _KEYS["github"],
    "sk-proj-Qh7Rk2Ls9Vn4Xb6Zt1Wc8-" + _KEYS["aws"],
]


@pytest.mark.parametrize("value", _KEY_IN_KEY)
def test_a_key_inside_a_key_of_another_class_stands_beside_an_escape(value: str) -> None:
    """With no escape the string is read as written only, as before the
    view; an escape elsewhere must leave both keys standing."""

    def written(escape: str) -> str:
        return _spans(_attr("gen_ai.tool.call.arguments", '{"a": "' + value + '"' + escape + "}"))

    plain = _scan(written(""))
    assert len({h.source_class for h in plain.hits if h.eligible}) == 2, plain.hits
    assert _scan(written(', "b": "x\\ny"')) == plain


@pytest.mark.parametrize("pattern", ["first", "middle"])
@pytest.mark.parametrize("value", _KEY_IN_KEY)
def test_an_escaped_key_with_a_key_of_another_class_inside_it(value: str, pattern: str) -> None:
    def written(text: str) -> str:
        return _spans(_attr("gen_ai.tool.call.arguments", '{"a": "' + text + '"}'))

    """Both keys are found, as in the plain export; the decoded key is
    listed after the one as written, so the order is not compared."""

    def found(scan: ExposureScan) -> list[tuple[str, bool, str | None]]:
        return sorted((h.source_class, h.eligible, h.fingerprint) for h in scan.hits)

    plain = _scan(written(value))
    assert len({h.source_class for h in plain.hits if h.eligible}) == 2, plain.hits
    assert found(_scan(written(_escaped(value, pattern)))) == found(plain)


def _keys_and_longer_keys(count: int, escaped: bool) -> tuple[str, int]:
    """Distinct keys, each also written with more after it, and one escape
    elsewhere in the string or none."""
    keys = [_KEY[:-6] + f"{i:06d}" for i in range(count)]
    document = {f"a{i}": key for i, key in enumerate(keys)}
    document |= {f"b{i}": key + "ZZA" for i, key in enumerate(keys)}
    document["c"] = "x\ny" if escaped else "x"
    return json.dumps(document), 2 * count


def _keys_cut_short_alike(count: int, escaped: bool) -> tuple[str, int]:
    """Distinct keys that share their start, each with a letter after it
    written as an escape or plainly, so that as written every key is cut
    short to the same value."""
    cut = _KEY.index("Fg0")
    letter = "\\u0046" if escaped else "F"
    items = [f'"a{i}": "{_KEY[:cut]}{letter}g0Yu2A{i:06d}"' for i in range(count)]
    return "{" + ", ".join(items) + "}", count


@pytest.mark.parametrize("shape", [_keys_and_longer_keys, _keys_cut_short_alike])
def test_many_keys_beside_escapes_cost_about_what_they_cost_without_them(
    shape: Callable[[int, bool], tuple[str, int]],
) -> None:
    """One string of many distinct keys: escapes make the scan read the
    decoded view too, which may cost a few times the plain scan, never a
    cost that grows with the square of the keys."""

    def cost(escaped: bool) -> float:
        text, distinct = shape(4000, escaped)
        body = _spans(_attr("gen_ai.tool.call.arguments", text))
        timings = []
        for _ in range(3):
            began = time.perf_counter()
            scan = _scan(body)
            timings.append(time.perf_counter() - began)
        assert len({h.fingerprint for h in scan.hits if h.eligible}) == distinct
        return min(timings)

    plain = cost(False)
    assert cost(True) < 10 * plain + 0.5


def test_the_view_counts_no_extra_string() -> None:
    plain = _scan(_spans(_attr("gen_ai.tool.call.arguments", '{"note": "' + _KEY + '"}')))
    escaped = _scan(
        _spans(_attr("gen_ai.tool.call.arguments", '{"note": "' + _escaped(_KEY) + '"}'))
    )
    assert escaped.strings_scanned == plain.strings_scanned
    assert escaped.values_not_scanned == plain.values_not_scanned == 0


# --- what the view does not read --------------------------------------------

# Text that holds no JSON is read as written, as before the view.
_NOT_JSON = [
    "key={}",
    '{"cut": "{}',  # JSON cut short before it closes
    "tool said {} and stopped",
    "\\q\\x {}",
]


@pytest.mark.parametrize("text", _NOT_JSON)
def test_text_that_is_not_json_is_not_read_unescaped(text: str) -> None:
    scan = _scan(_spans(_attr("acme.payload", _fill(text, _escaped(_KEY, "every")))))
    assert [h for h in scan.hits if h.eligible] == []
    assert scan.values_not_scanned == 0


@pytest.mark.parametrize(
    "text",
    [
        *_DOCUMENTS.values(),
        *_NOT_JSON,
        '{"a": "x\\u0041"} and [1, "\\n"] then {"cut": "\\u0041',
        '"\\u00e9" trailing',
        "plain text with no escape {}",
        "[" * 5000 + '"\\u0041"' + "]" * 5000,  # past the decoder's depth limit
    ],
)
def test_the_scan_picks_the_json_text_the_judge_picks(text: str) -> None:
    """One rule for both detectors: the same JSON texts of one string."""
    filled = _fill(text, _escaped(_KEY))
    data = filled.encode("utf-8", errors="surrogatepass")
    assert _json_texts(filled) == [
        s.decode("utf-8", errors="surrogateescape") for s in _json_spans(data)
    ]


# --- the bound --------------------------------------------------------------


def test_the_bound_is_the_last_level_read() -> None:
    plain = _cli(_spans(_attr("acme.payload", _plain_nested(_KEY, BOUND))))
    escaped = _cli(_spans(_attr("acme.payload", _nested(_KEY, BOUND))))
    assert escaped == plain
    assert "not scanned" not in str(escaped)


def test_text_escaped_past_the_bound_is_counted_as_not_scanned() -> None:
    deep = _nested(_KEY, BOUND + 1)
    body = _spans(_attr("acme.payload", deep), _attr("acme.again", deep))
    unbounded = _scan(_spans())
    scan = _scan(body)
    assert scan.hits == []
    # Each string escaped too deep is one value not scanned.
    assert scan.values_not_scanned == unbounded.values_not_scanned + 2
    out = _cli(body)
    line = "exposure scan: 2 values not scanned - not in a shape the scan reads"
    assert line in str(out["scan"]).splitlines()
    assert out["exit"] == 3
    gauge = out["gauge"]
    assert isinstance(gauge, list)
    assert line in gauge
    # Never reported clean: the gauge does not say "no credential-shaped
    # values" over a value it could not read.
    assert not [ln for ln in gauge if "no credential-shaped values" in ln], gauge


def test_a_key_beside_text_escaped_past_the_bound_is_still_found() -> None:
    deep = _nested(_KEY, BOUND + 1)
    plain = _scan(_spans(_attr("acme.key", _KEY)))
    scan = _scan(_spans(_attr("acme.key", _KEY), _attr("acme.payload", deep)))
    assert scan.hits == plain.hits
    assert scan.values_not_scanned == plain.values_not_scanned + 1

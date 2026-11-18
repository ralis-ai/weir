"""Span attributes read both the camelCase and snake_case spellings of the
OTLP AnyValue wrappers. Every value here is fake."""

import json
import random
import re
import time
from collections.abc import Callable

import pytest

from weir.adapters.otel import decode_input
from weir.adapters.otel.exposure import scan_surface
from weir.catalog import DEFAULT_CATALOG
from weir.exposure import ExposureHit, classify_exposure
from weir.schema.exposure import ExposureSurface

_KEY = "sk-proj-Qh7Rk2Ls9Vn4Xb6Zt1Wc8Mp3Jd5Fg0Yu2Ae7Ri4"


def _scan(doc):
    surface = scan_surface(decode_input(json.dumps(doc).encode()))
    return classify_exposure(surface, DEFAULT_CATALOG)


def _span_doc(attributes, *, span_id="aa" * 8):
    return {
        "resourceSpans": [
            {"scopeSpans": [{"spans": [{"spanId": span_id, "attributes": attributes}]}]}
        ]
    }


# -- snake_case string, array and kvlist span attribute ----------------------


def test_snake_case_string_value_in_span_attribute() -> None:
    doc = _span_doc([{"key": "k", "value": {"string_value": _KEY}}])
    scan = _scan(doc)
    assert any(h.source_class == "openai_api_key" and h.eligible for h in scan.hits)
    assert scan.values_not_scanned == 0


def test_snake_case_array_value_in_span_attribute() -> None:
    doc = _span_doc([{"key": "k", "value": {"array_value": {"values": [{"string_value": _KEY}]}}}])
    scan = _scan(doc)
    assert any(h.source_class == "openai_api_key" and h.eligible for h in scan.hits)
    assert scan.values_not_scanned == 0


def test_snake_case_kvlist_value_in_span_attribute() -> None:
    doc = _span_doc(
        [
            {
                "key": "k",
                "value": {
                    "kvlist_value": {"values": [{"key": "inner", "value": {"string_value": _KEY}}]}
                },
            }
        ]
    )
    scan = _scan(doc)
    assert any(h.source_class == "openai_api_key" and h.eligible for h in scan.hits)
    assert scan.values_not_scanned == 0


# -- snake_case number and boolean carriers hold no string -------------------


@pytest.mark.parametrize("wrapper", ["int_value", "double_value", "bool_value"])
def test_snake_case_scalar_wrapper_holds_no_string(wrapper: str) -> None:
    doc = _span_doc([{"key": "k", "value": {wrapper: "42"}}])
    surface = scan_surface(decode_input(json.dumps(doc).encode()))
    texts = [s.text for s in surface.strings if not s.key]
    assert texts == []


# -- nested kvlist_value inside array_value ----------------------------------


def test_nested_kvlist_inside_snake_case_array() -> None:
    doc = _span_doc(
        [
            {
                "key": "k",
                "value": {
                    "array_value": {
                        "values": [
                            {
                                "kvlist_value": {
                                    "values": [{"key": "inner", "value": {"string_value": _KEY}}]
                                }
                            }
                        ]
                    }
                },
            }
        ]
    )
    scan = _scan(doc)
    assert any(h.source_class == "openai_api_key" for h in scan.hits)


# -- mixed camelCase and snake_case in one export ----------------------------


def test_mixed_camel_and_snake_case_in_one_export() -> None:
    doc = {
        "resourceSpans": [
            {
                "scopeSpans": [
                    {
                        "spans": [
                            {
                                "spanId": "aa" * 8,
                                "attributes": [
                                    {"key": "a", "value": {"stringValue": _KEY}},
                                ],
                            },
                            {
                                "spanId": "bb" * 8,
                                "attributes": [
                                    {"key": "b", "value": {"string_value": _KEY}},
                                ],
                            },
                        ]
                    }
                ]
            }
        ]
    }
    scan = _scan(doc)
    hits = [h for h in scan.hits if h.source_class == "openai_api_key"]
    assert len(hits) == 2
    assert {h.span_ref for h in hits} == {"aa" * 8, "bb" * 8}
    assert scan.values_not_scanned == 0


# -- snake_case in a span event attribute ------------------------------------


def test_snake_case_in_span_event_attribute() -> None:
    doc = {
        "resourceSpans": [
            {
                "scopeSpans": [
                    {
                        "spans": [
                            {
                                "spanId": "aa" * 8,
                                "events": [
                                    {
                                        "name": "log",
                                        "attributes": [
                                            {"key": "msg", "value": {"string_value": _KEY}}
                                        ],
                                    }
                                ],
                            }
                        ]
                    }
                ]
            }
        ]
    }
    scan = _scan(doc)
    assert any(h.source_class == "openai_api_key" for h in scan.hits)


# -- resource attributes read with the same wrappers as their context --------


def test_resource_attributes_read_snake_case_under_span_context() -> None:
    doc = {
        "resourceSpans": [
            {
                "resource": {"attributes": [{"key": "r", "value": {"string_value": _KEY}}]},
                "scopeSpans": [{"spans": [{"spanId": "aa" * 8}]}],
            }
        ]
    }
    scan = _scan(doc)
    assert any(h.source_class == "openai_api_key" for h in scan.hits)


# -- timing guard: repeated input stays close to linear ----------------------

_SIZE = 800_000
_ORDINARY = json.dumps(
    {"role": "user", "content": "Summarize the report, then list three risks.", "n": [1, 2]}
)
_REPEATS = 3
_MAX_RATIO = 4.0


def _surface(text: str) -> ExposureSurface:
    doc = {
        "resourceSpans": [
            {
                "scopeSpans": [
                    {
                        "spans": [
                            {
                                "spanId": "ab" * 8,
                                "attributes": [{"key": "probe", "value": {"stringValue": text}}],
                            }
                        ]
                    }
                ]
            }
        ]
    }
    return scan_surface(decode_input(json.dumps(doc).encode()))


def _best_time(run: Callable[[], list[ExposureHit]]) -> float:
    best = float("inf")
    for _ in range(_REPEATS):
        start = time.perf_counter()
        run()
        best = min(best, time.perf_counter() - start)
    return best


@pytest.mark.parametrize(
    "unit",
    [
        "password=correcthorsebatterystaple&",
        "api_key='fake-key-01234567890abcdef'&",
    ],
)
def test_repeated_matches_scan_like_ordinary_input(unit: str) -> None:
    text = unit * (_SIZE // len(unit))
    slow = _surface(text)
    ordinary = _surface((_ORDINARY * (_SIZE // len(_ORDINARY) + 1))[: len(text)])

    def scan(surface: ExposureSurface) -> Callable[[], list[ExposureHit]]:
        return lambda: classify_exposure(surface, DEFAULT_CATALOG).hits

    scan(ordinary)()
    ratio = _best_time(scan(slow)) / _best_time(scan(ordinary))
    assert ratio < _MAX_RATIO, f"{ratio:.1f}x slower than ordinary input of the same size"


# -- differential: optimized pruning matches naive reference -----------------


def _naive_subsumed(cand, candidates):
    """O(n^2) reference: the original logic before the bisect optimization."""
    if not cand.has_group:
        return False
    return any(
        other.eligible
        and other.spec is not cand.spec
        and cand.value_start <= other.start < cand.value_end
        for other in candidates
    )


def test_optimized_pruning_matches_naive_reference() -> None:
    from weir.exposure.classify import _candidates

    rng = random.Random(42)
    classes = [
        (spec, re.compile(spec.content_pattern), 0)
        for spec in DEFAULT_CATALOG.sources
        if spec.exposure
    ]

    for _ in range(20):
        parts = []
        for _ in range(rng.randint(5, 30)):
            parts.append(f"password={rng.randbytes(16).hex()}&")
            chars = "abcdefghijklmnopqrstuvwxyz0123456789"
            fake: str = "".join(rng.choices(chars, k=36))
            parts.append(f"api_key='sk-proj-{fake}'&")
            pad: bytes = rng.randbytes(rng.randint(0, 40))
            parts.append(pad.hex())
        text = "".join(parts)
        candidates = _candidates(text, classes)
        naive = [_naive_subsumed(c, candidates) for c in candidates]
        actual = [c.subsumed for c in candidates]
        assert actual == naive, f"mismatch on text of length {len(text)}"

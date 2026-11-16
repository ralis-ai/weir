"""A long backslash run after a credential-field key scans at the speed of
ordinary input of the same size.

The value of a credential field must not end inside a backslash run that
escapes its closing quote, so the pattern looks at the end of the run. A
pattern that looks again from every backslash re-reads the run once per
value character, which made a key followed by a long run of backslashes
scan many times slower than any other input of its size. The guard times
the real scan of both inputs and compares them, so it holds on a slow or a
fast machine alike.
"""

from __future__ import annotations

import json
import time
from collections.abc import Callable

import pytest

from weir.adapters.otel import decode_input
from weir.adapters.otel.exposure import scan_surface
from weir.catalog import DEFAULT_CATALOG
from weir.exposure import ExposureHit, classify_exposure
from weir.schema.exposure import ExposureSurface

_SIZE = 100_000
# Ordinary telemetry text: a JSON chat message, repeated to the same size.
_ORDINARY = json.dumps(
    {"role": "user", "content": "Summarize the report, then list three risks.", "n": [1, 2]}
)
# A pattern that re-reads the run once per value character scans the run
# input more than 10 times slower than _ORDINARY; a linear one stays within
# about 1.5 times.
_MAX_RATIO = 4.0
_REPEATS = 3


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


_KEYS = {
    "key-value": "password=",
    "key-value-quoted": 'api_key="',
    "json-double-quoted": '{"password": "',
    "json-single-quoted": "{'client_secret': '",
}
_RUNS = {
    "to-end": lambda n: "\\" * n,
    "before-quote": lambda n: "\\" * n + "\"'",
    "before-escape-letter": lambda n: "\\" * n + "n",
    "after-letters": lambda n: "abcdefgh" + "\\" * n,
}


@pytest.mark.parametrize("run", sorted(_RUNS))
@pytest.mark.parametrize("key", sorted(_KEYS))
def test_a_backslash_run_after_a_credential_key_scans_like_ordinary_input(
    key: str, run: str
) -> None:
    head = _KEYS[key]
    slow = _surface(head + _RUNS[run](_SIZE))
    ordinary = _surface(head + (_ORDINARY * (_SIZE // len(_ORDINARY) + 1))[:_SIZE])
    assert len(slow.strings[0].text) >= len(ordinary.strings[0].text)

    def scan(surface: ExposureSurface) -> Callable[[], list[ExposureHit]]:
        return lambda: classify_exposure(surface, DEFAULT_CATALOG).hits

    # Warm the compiled-pattern cache so neither timing pays for it.
    scan(ordinary)()
    ratio = _best_time(scan(slow)) / _best_time(scan(ordinary))
    assert ratio < _MAX_RATIO, f"{ratio:.1f}x slower than ordinary input of the same size"


# Input built to make the URL classes re-read it: a scheme start at every
# dot, a SAS parameter that could open a sig search at every `&`, a sig
# whose sibling search could run to the end, userinfo colons with no `@`.
_URL_RUNS = {
    "scheme-starts": lambda n: "a." * (n // 2),
    "sas-siblings": lambda n: "se=a&" * (n // 5),
    "sig-params": lambda n: "sig=a&" * (n // 6),
    "userinfo-colons": lambda n: "a://b:" * (n // 6),
    "query-names": lambda n: "?X-Amz-Signature=&" * (n // 18),
}


@pytest.mark.parametrize("run", sorted(_URL_RUNS))
def test_url_shaped_input_scans_like_ordinary_input(run: str) -> None:
    text = _URL_RUNS[run](_SIZE)
    slow = _surface(text)
    ordinary = _surface((_ORDINARY * (_SIZE // len(_ORDINARY) + 1))[: len(text)])

    def scan(surface: ExposureSurface) -> Callable[[], list[ExposureHit]]:
        return lambda: classify_exposure(surface, DEFAULT_CATALOG).hits

    scan(ordinary)()
    ratio = _best_time(scan(slow)) / _best_time(scan(ordinary))
    assert ratio < _MAX_RATIO, f"{ratio:.1f}x slower than ordinary input of the same size"

"""Exposure classification (spec 2026-09-06 section 2): a scanned surface plus
the catalog -> value-free hits. Pure, and a sidecar beside the flow engine
rather than a stage of it.

The strings arrive already enumerated by `weir.adapters.otel.exposure`, which
is why this module needs no wire model and never imports `weir.adapters` - the
"Analysis-path purity" contract holds it to that. Each string is read as
written and through its decoded view (docs/scan.md, "JSON escapes"), which
the walk hands over beside it.

This is also the one place the matched values and the names they sit under
are both in hand, so it is where a span id, span name or location key that
holds a matched value becomes `[withheld]` (docs/scan.md, "Why you can trust it").
Every renderer downstream prints hits as they are and cannot get a name back.
"""

from __future__ import annotations

import hashlib
import re
from collections.abc import Callable
from typing import NamedTuple

import msgspec

from weir._text_safety import flatten_untrusted
from weir.canary import needles
from weir.canary._types import RUN
from weir.canary.text import WITHHELD
from weir.catalog import (
    Catalog,
    SourceSpec,
    assignment_prefix,
    class_prefix,
    is_verbatim_eligible,
    matched_value,
)
from weir.exposure._types import ExposureHit, ExposureScan
from weir.json_escapes import LEVELS, decoded_view
from weir.schema.exposure import ExposureSurface, ScannedString, Segment, format_location

FINGERPRINT_VERSION = 1

# A value this short, plus its own last4, could be reconstructed from the
# hit alone - see the last4= line in classify_exposure below.
_MIN_LAST4_VALUE_LEN = 12

# A needle at least this long is looked up by its last this-many bytes.
_TAIL = 8

_REGEX_META = frozenset(".^$*+?{}[]\\|()")

# The leading boundaries a key-shaped content_pattern may open with: the
# escape-aware one (a word boundary, or right after a JSON \n, \t or \r
# escape) and the plain word boundary. Longest first.
_LEADING_BOUNDARIES = ("(?:\\b|(?<=\\\\[ntr]))", "\\b")


class _RawHit(msgspec.Struct, frozen=True):
    """One match after subsumption, before triage/eligible dedupe. Carries
    the value, which the public ExposureHit deliberately does not."""

    record_index: int
    span_ref: str
    span_name: str
    log_record: int | None
    path: tuple[Segment, ...]
    key: bool
    location: str
    # How many unescapings gave the text the hit is in: 0 as written.
    unescapings: int
    start: int
    spec: SourceSpec
    value: str
    prefix: str
    eligible: bool


class _Candidate(NamedTuple):
    """One match at one (span, location). `value_start`, `value_end`,
    `has_group` and `subsumed` exist only to decide subsumption and claims
    (`_claims`) at THIS location, so they never make it into `_RawHit`."""

    spec: SourceSpec
    start: int
    value: str
    prefix: str
    eligible: bool
    value_start: int
    value_end: int
    has_group: bool
    subsumed: bool = False


def _fingerprint(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()[:12]


def _literal_prefix(pattern: str) -> str:
    """The leading LITERAL run of a content_pattern. Two classes can match at
    one offset (`sk-` and `sk-ant-`); the longer literal prefix is the more
    specific class and wins. A leading boundary (`_LEADING_BOUNDARIES`) is
    skipped and scanning stops at the first metacharacter, so `(?:AKIA|ASIA)...`
    and the `(?i)`-prefixed credential_field pattern both yield "" and lose
    every tie - correct, since neither is a prefix refinement of anything."""
    for boundary in _LEADING_BOUNDARIES:
        if pattern.startswith(boundary):
            pattern = pattern[len(boundary):]
            break
    out: list[str] = []
    for ch in pattern:
        if ch in _REGEX_META:
            break
        out.append(ch)
    return "".join(out)


def _has_floor(spec: SourceSpec) -> bool:
    e = spec.eligibility
    return e.structure_class is not None or e.pattern is not None or e.min_length is not None


def _subsumed(cand: _Candidate, candidates: list[_Candidate]) -> bool:
    """`api_key='sk-proj-...'` matches credential_field at the key name and
    openai_api_key at the value, so the offset rule does not dedupe them. Drop
    the assignment-shaped hit when an eligible hit of ANOTHER class lies
    inside its value group: the verdict-grade hit is the stronger claim about
    the same bytes, and keeping both would double every location in the
    report and the gauge.

    `candidates` is always the list for one (span, location) - one
    `_raw_hits_for` call, one `ScannedString` - so unlike the batch-wide scan
    this used to run against, span/location never need comparing here."""
    if not cand.has_group:
        return False
    return any(
        other.eligible
        and other.spec is not cand.spec
        and cand.value_start <= other.start < cand.value_end
        for other in candidates
    )


_Classes = list[tuple[SourceSpec, re.Pattern[str], int]]


def _candidates(text: str, classes: _Classes) -> list[_Candidate]:
    """Every match in one text, each flagged `subsumed` when `_subsumed`
    drops it: `_claims` reads a subsumed match to drop the same claim
    found with fewer unescapings."""
    # Class assignment happens BEFORE eligibility and is final: at one offset
    # the longest literal prefix wins, so one value yields one hit per
    # location, never two. An sk-ant- value that fails the strict Anthropic
    # pattern stays an anthropic_api_key MISS; it is never re-graded under
    # openai_api_key. A miss must stay visible as a miss. Do not "fix" this.
    # A tie between two classes that BOTH have a zero-length literal prefix
    # (e.g. `aws_access_key_id`'s `(?:AKIA|ASIA)...` against the
    # `(?i)`-prefixed `credential_field`) falls back to catalog list order -
    # no bundled pair collides at one offset, so this is untested by design,
    # not unhandled.
    best: dict[int, tuple[int, SourceSpec, re.Match[str]]] = {}
    for spec, pattern, prefix_len in classes:
        for match in pattern.finditer(text):
            incumbent = best.get(match.start())
            if incumbent is None or prefix_len > incumbent[0]:
                best[match.start()] = (prefix_len, spec, match)

    candidates: list[_Candidate] = []
    for start in sorted(best):
        _, spec, match = best[start]
        has_group = match.re.groups > 0
        if has_group:
            # An assignment-shaped class captures the SECRET in group 1; the
            # text before it is the key name, which is the display prefix.
            value = matched_value(match.group(1) or "", spec)
            prefix = assignment_prefix(match.group(0)[: match.start(1) - match.start(0)])
            value_start, value_end = match.span(1)
        else:
            value = matched_value(match.group(0), spec)
            prefix = class_prefix(value)
            value_start, value_end = match.span(0)
        if spec.exposure_prefix is not None:
            prefix = spec.exposure_prefix
        eligible = is_verbatim_eligible(value, spec)
        if (
            not eligible
            and not _has_floor(spec)
            and any(re.fullmatch(p, value) is not None for p in spec.eligibility.reject_patterns)
        ):
            # A triage-only class declares no floor, so there is no weaker
            # state to demote a rejected value to: a masked `api_key='***'` is
            # excluded outright rather than parked in the review queue forever.
            continue
        candidates.append(
            _Candidate(spec, start, value, prefix, eligible, value_start, value_end, has_group)
        )

    return [cand._replace(subsumed=_subsumed(cand, candidates)) for cand in candidates]


# An escape cut short at the end of a match: a class's length bound can end
# a match inside one.
_CUT_ESCAPE = re.compile(r"\\(?:u[0-9a-fA-F]{0,3})?\Z")


def _plain(value: str) -> str:
    """`value` with its JSON escapes undone as far as the decoded view goes,
    less an escape cut short at its end."""
    texts = decoded_view(value.encode("utf-8", errors="surrogatepass")).texts
    plain = texts[-1].decode("utf-8", errors="surrogatepass") if texts else value
    return _CUT_ESCAPE.sub("", plain)


# Text cut short reads plainly like the whole text but for its end: an
# escape is at most 12 characters (a surrogate pair), so each unescaping
# changes at most 11 more characters before the cut.
_CUT_MARGIN = 11 * LEVELS


def _plain_from(text: str, start: int, length: int) -> str:
    """`text` from `start` read plainly, at least `length` characters of it
    when it has them: decoded over a window, never the whole text."""
    span = length + _CUT_MARGIN
    while start + span < len(text):
        plain = _plain(text[start : start + span])
        if len(plain) >= length + _CUT_MARGIN:
            return plain[: len(plain) - _CUT_MARGIN]
        span *= 2
    return _plain(text[start:])


def _claims(found: list[tuple[int, str, _Candidate]]) -> list[tuple[int, _Candidate]]:
    """One string's matches across its texts, one per claim. Read plainly
    (`_plain`), a credential is the same claim in every text that holds it:
    it is kept at the fewest unescapings that hold it, so the value as
    written stands wherever the text itself holds it (a token with `\\/` for
    `/` keeps that spelling). And a match the decoded view reads further
    is the claim its own class makes there: an escape inside a key cuts the
    key short as written (`sk-proj-Qh7R\\u006b...` matches up to the
    escape), and under a credential field name the field match as written
    is subsumed only where read plainly the key lies inside its value. So a
    match with an escape in or right after its value is dropped when a text
    with more unescapings holds a reading of its class whose value starts
    with its value - longer, or equal and subsumed or eligible where the
    match is not - and its own text, read plainly, holds that reading over
    it. A longer credential elsewhere that merely starts with its value, or
    one of another class, leaves it standing."""
    if all(level == 0 for level, _, _ in found):
        return [(level, cand) for level, _, cand in found if not cand.subsumed]
    read = [(level, text, cand, _plain(cand.value)) for level, text, cand in found]
    # Per class, each plain value read with the most unescapings that hold
    # it at all, subsumed, and eligible, and every length a value has; and
    # the fewest unescapings that hold each eligible claim.
    readings: dict[str, dict[str, list[int]]] = {}
    lengths: dict[str, set[int]] = {}
    fewest: dict[tuple[str, str], int] = {}
    for level, _, cand, value in read:
        name = cand.spec.name
        deepest = readings.setdefault(name, {}).setdefault(value, [-1, -1, -1])
        deepest[0] = max(deepest[0], level)
        if cand.subsumed:
            deepest[1] = max(deepest[1], level)
        if cand.eligible:
            deepest[2] = max(deepest[2], level)
        lengths.setdefault(name, set()).add(len(value))
        if cand.eligible and not cand.subsumed:
            claim = (name, value)
            fewest[claim] = min(level, fewest.get(claim, level))

    def covered(text: str, cand: _Candidate, level: int, value: str) -> bool:
        name = cand.spec.name
        tail = _plain_from(text, cand.value_start, max(lengths[name]))
        if not tail.startswith(value):
            return False
        for length in lengths[name]:
            if length < len(value) or length > len(tail):
                continue
            deepest = readings[name].get(tail[:length])
            if deepest is None:
                continue
            if length > len(value):
                if deepest[0] > level:
                    return True
            elif deepest[1] > level or (not cand.eligible and deepest[2] > level):
                return True
        return False

    kept: list[tuple[int, _Candidate]] = []
    for level, text, cand, value in read:
        if cand.subsumed:
            continue
        if cand.eligible and fewest[(cand.spec.name, value)] < level:
            continue
        if "\\" in text[cand.value_start : cand.value_end + 1] and covered(
            text, cand, level, value
        ):
            continue
        kept.append((level, cand))
    return kept


def _raw_hits_for(scanned: ScannedString, classes: _Classes) -> list[_RawHit]:
    """The string's matches as written, then those of its decoded view
    (`ScannedString.decoded`), each text read the same way and reported at
    the string's own span and location."""
    texts = [(0, scanned.text)] + [
        (level, text) for view in scanned.decoded for level, text in enumerate(view, start=1)
    ]
    found = [
        (level, read, cand)
        for level, text in texts
        for read in [text if scanned.header is None else f"{scanned.header}: {text}"]
        for cand in _candidates(read, classes)
    ]
    return [
        _RawHit(
            record_index=scanned.record_index, span_ref=scanned.span_ref,
            span_name=scanned.span_name, log_record=scanned.log_record,
            path=scanned.path, key=scanned.key, location=scanned.location, unescapings=level, start=cand.start,
            spec=cand.spec, value=cand.value, prefix=cand.prefix, eligible=cand.eligible,
        )
        for level, cand in _claims(found)
    ]


def _withholding(values: set[str]) -> Callable[[str], str]:
    """Text that holds one of `values`, in any of its encoded forms, or any
    run of 6 consecutive characters of one, prints as `[withheld]`. The text
    is checked as written, without undoing JSON escapes. The flattened form
    is checked as well, since that is what is printed.

    A scan can match a value per span, so each needle is looked up by its
    last `_TAIL` bytes, or a shorter one by its length, and each distinct
    string is checked once: the cost is the length of the strings checked,
    not that times every needle of the scan."""
    runs = frozenset(value[i : i + RUN] for value in values for i in range(len(value) - RUN + 1))
    short: dict[int, set[bytes]] = {}
    by_tail: dict[bytes, set[bytes]] = {}
    for value in values:
        for needle, _ in needles(value):
            if len(needle) < _TAIL:
                short.setdefault(len(needle), set()).add(needle)
            else:
                by_tail.setdefault(needle[-_TAIL:], set()).add(needle)

    def holds_value(text: str) -> bool:
        if any(text[i : i + RUN] in runs for i in range(len(text) - RUN + 1)):
            return True
        data = text.encode("utf-8", errors="surrogatepass")
        if any(
            data[i : i + size] in found
            for size, found in short.items()
            for i in range(len(data) - size + 1)
        ):
            return True
        return any(
            data.endswith(needle, 0, end)
            for end in range(_TAIL, len(data) + 1)
            for needle in by_tail.get(data[end - _TAIL : end], ())
        )

    printed: dict[str, str] = {}

    def withhold(text: str) -> str:
        if text not in printed:
            held = holds_value(text) or holds_value(flatten_untrusted(text))
            printed[text] = WITHHELD if held else text
        return printed[text]

    return withhold


def classify_exposure(surface: ExposureSurface, catalog: Catalog) -> ExposureScan:
    classes = [
        (spec, re.compile(spec.content_pattern), len(_literal_prefix(spec.content_pattern)))
        for spec in catalog.sources
        if spec.exposure
    ]

    raw: list[_RawHit] = []
    for scanned in surface.strings:
        raw.extend(_raw_hits_for(scanned, classes))

    # Every value this scan matched, eligible or triage. A span id is
    # lowercased off the wire, so for it each value counts in lower case too.
    values = {hit.value for hit in raw}
    withhold = _withholding(values)
    withhold_id = _withholding(values | {value.lower() for value in values})
    # Counts key on these ordinals, in hit order, never on the printed text:
    # two locations can both print as `attributes.[withheld]`.
    span_keys: dict[str, int] = {}
    site_keys: dict[tuple[str, int | None, str], int] = {}
    hits: list[ExposureHit] = []
    # Dedupe key, both grades: one claim per (record, location, class, value).
    # record_index not span_ref - two spans may legally share a spanId (retries,
    # broken instrumentation, or an attacker-set id), and a security finding
    # must not vanish into a duplicate. The fingerprint distinguishes two
    # different secrets at one location, and is None for triage hits, which
    # collapses this to the (span, location, class) rule triage needs.
    seen: set[tuple[int, str, str, str | None]] = set()
    # Within a location, the text as written first: where the decoded view
    # finds the same claim again, the one as written is kept.
    for hit in sorted(raw, key=lambda h: (h.record_index, h.location, h.unescapings, h.start)):
        fingerprint = _fingerprint(hit.value) if hit.eligible else None
        key = (hit.record_index, hit.location, hit.spec.name, fingerprint)
        if key in seen:
            continue
        seen.add(key)
        hits.append(ExposureHit(
            span_ref=withhold_id(hit.span_ref),
            span_name=withhold(hit.span_name),
            location=format_location(hit.path, withhold, key=hit.key),
            span_key=span_keys.setdefault(hit.span_ref, len(span_keys)),
            site_key=site_keys.setdefault(
                (hit.span_ref, hit.log_record, hit.location), len(site_keys)
            ),
            source_class=hit.spec.name,
            eligible=hit.eligible,
            value_len=len(hit.value),
            prefix=hit.prefix,
            # Never emit a suffix that could reconstruct a short value: the
            # bundled classes are all >= 20 chars, but a contributed catalog
            # sets its own floor and G4 must not depend on it.
            last4=(
                hit.value[-4:]
                if hit.eligible and len(hit.value) >= _MIN_LAST4_VALUE_LEN
                else None
            ),
            fingerprint=fingerprint,
            fingerprint_v=FINGERPRINT_VERSION,
            log_record=hit.log_record,
        ))

    return ExposureScan(
        applicable=surface.applicable,
        spans_scanned=surface.spans_scanned,
        strings_scanned=len(surface.strings),
        hits=hits,
        log_records_scanned=surface.log_records_scanned,
        batches_not_scanned=surface.batches_not_scanned,
        values_not_scanned=surface.values_not_scanned,
    )

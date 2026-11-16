"""Credentials inside URLs: a password in a URL's userinfo, and the
credential query parameters of signed URLs (X-Amz-Security-Token,
X-Amz-Signature, X-Goog-Signature, an Azure SAS `sig`).

Every case runs the real scan over a synthetic OTLP span. The expectations
come from the planted value itself (its decoded length, its last four
characters, the class's own `exposure_prefix`), never from a typed copy of
what the scan prints. Every value is fake: random letters and digits made
for this file, never a real credential.
"""

from __future__ import annotations

import json
from pathlib import Path
from urllib.parse import quote

import pytest
from click.testing import CliRunner

from weir.adapters.otel import decode_input
from weir.adapters.otel.exposure import scan_surface
from weir.canary._types import RUN
from weir.catalog import DEFAULT_CATALOG
from weir.cli.main import main
from weir.evaluate.exposure import evaluate_exposure
from weir.exposure import ExposureHit, classify_exposure
from weir.inventory.classify import value_shape
from weir.inventory.project import scan_text
from weir.rules_commons import load_rules

_SPEC = {spec.name: spec for spec in DEFAULT_CATALOG.sources}

_PASSWORD = "Pq7Zt2Lm9Xc4Vb8N"
# Reserved characters a URL must percent-encode, so the scan has to judge
# the decoded password, not its escapes.
_PASSWORD_RESERVED = "p@ss:w0rd/Xy9Q#k"
# A session token is long base64; `+` and `/` reach the URL as %2B and %2F.
_SESSION_TOKEN = ("IQoJb3JpZ2luX2VjEFAKQh7Rk2Ls9Vn4Xb6Zt1Wc8Mp3Jd5Fg0Yu2Ae7Ri4+/"
                  "Tn8Sv3Hq6Wm1Kc5Lp9Bz2Ny7Gd4Fx0Ju8Ra3Ue6Io1Ok5Pm2Ql9Ws4Ev7Rt0"
                  "Yb3Hn6Mj9Kl2Za5Xs8Cd1Vf4Bg7Nh0Mj3Ki6Lo9Pa2Qs5Wd8Ef1Rg4Th7Yj0Uk3")
_AMZ_SIGNATURE = "3f9a7c1e5b2d8f4a6c0e9b1d7f3a5c8e2b4d6f8a0c1e3b5d7f9a2c4e6b8d0f1a"
_GOOG_SIGNATURE = ("7c1e5b2d8f4a6c0e9b1d7f3a5c8e2b4d6f8a0c1e3b5d7f9a2c4e6b8d0f1a3f9a"
                   "1e5b2d8f4a6c0e9b1d7f3a5c8e2b4d6f8a0c1e3b5d7f9a2c4e6b8d0f1a3f9a7c")
# Base64 of a 32-byte HMAC: 43 characters and one `=`.
_SAS_SIGNATURE = "Qh7Rk2Ls9Vn4Xb6Zt1Wc8Mp3Jd5Fg0Yu2Ae7Ri4Tn+/="

_AMZ_QUERY = ("X-Amz-Algorithm=AWS4-HMAC-SHA256&X-Amz-Date=20260101T000000Z"
              "&X-Amz-Expires=900&X-Amz-SignedHeaders=host")
_SAS_PARAMS = "sv=2022-11-02&ss=b&srt=o&sp=r&se=2026-01-02T00%3A00%3A00Z&spr=https"

# name: (text the span carries, the class that must find it, the value it judges)
_CASES: dict[str, tuple[str, str, str]] = {
    "userinfo": (f"postgres://svc_user:{_PASSWORD}@db.internal.example:5432/app",
                 "url_with_password", _PASSWORD),
    "userinfo-encoded": (f"redis://:{quote(_PASSWORD_RESERVED, safe='')}@cache.example:6379/0",
                         "url_with_password", _PASSWORD_RESERVED),
    "amz-security-token": (
        f"https://bucket.s3.amazonaws.com/o.txt?{_AMZ_QUERY}"
        f"&X-Amz-Security-Token={quote(_SESSION_TOKEN, safe='')}",
        "aws_url_security_token", _SESSION_TOKEN),
    "amz-signature": (f"https://bucket.s3.amazonaws.com/o.txt?{_AMZ_QUERY}"
                      f"&X-Amz-Signature={_AMZ_SIGNATURE}", "aws_url_signature", _AMZ_SIGNATURE),
    "goog-signature": (
        "https://storage.googleapis.com/bkt/o.txt?X-Goog-Algorithm=GOOG4-RSA-SHA256"
        f"&X-Goog-Expires=900&X-Goog-SignedHeaders=host&X-Goog-Signature={_GOOG_SIGNATURE}",
        "google_url_signature", _GOOG_SIGNATURE),
    "sas-sig-last": (f"https://acct.blob.core.windows.net/c/b.txt?{_SAS_PARAMS}"
                     f"&sig={quote(_SAS_SIGNATURE, safe='')}", "azure_sas_signature",
                     _SAS_SIGNATURE),
    "sas-sig-first": (f"https://acct.blob.core.windows.net/c/b.txt?sig="
                      f"{quote(_SAS_SIGNATURE, safe='')}&sp=r&sv=2022-11-02&sr=b",
                      "azure_sas_signature", _SAS_SIGNATURE),
    "sas-connection-string": (
        "BlobEndpoint=https://acct.blob.core.windows.net/;SharedAccessSignature="
        f"{_SAS_PARAMS}&sig={quote(_SAS_SIGNATURE, safe='')}",
        "azure_sas_signature", _SAS_SIGNATURE),
}


def _export(*texts: str, name: str = "http.request") -> str:
    return json.dumps({"resourceSpans": [{"scopeSpans": [{"spans": [{
        "spanId": "ab" * 8, "name": name,
        "attributes": [{"key": f"app.url{i}", "value": {"stringValue": text}}
                       for i, text in enumerate(texts)],
    }]}]}]})


def _hits(*texts: str) -> list[ExposureHit]:
    surface = scan_surface(decode_input(_export(*texts).encode("utf-8")))
    return classify_exposure(surface, DEFAULT_CATALOG).hits


def _scan_cli(*texts: str, name: str = "http.request"):
    runner = CliRunner()
    with runner.isolated_filesystem():
        Path("export.json").write_text(_export(*texts, name=name), encoding="utf-8")
        return runner.invoke(main, ["scan", "export.json"])


def _assert_withheld(output: str, secret: str) -> None:
    assert not any(secret[i : i + RUN] in output for i in range(len(secret) - RUN + 1))


@pytest.mark.parametrize("case", sorted(_CASES))
def test_a_url_credential_is_one_verdict_grade_hit_of_its_class(case: str) -> None:
    text, source_class, secret = _CASES[case]
    found = [h for h in _hits(text) if h.source_class == source_class]
    assert len(found) == 1
    hit = found[0]
    assert hit.eligible
    assert hit.value_len == len(secret)
    assert hit.last4 == secret[-4:]
    assert hit.prefix == _SPEC[source_class].exposure_prefix
    findings = evaluate_exposure(
        classify_exposure(scan_surface(decode_input(_export(text).encode())), DEFAULT_CATALOG),
        load_rules(),
    )
    assert any(f.source_class == source_class and f.is_verdict_grade for f in findings)


@pytest.mark.parametrize("case", sorted(_CASES))
def test_the_scan_fails_and_never_prints_the_credential(case: str) -> None:
    text, source_class, secret = _CASES[case]
    result = _scan_cli(text)
    assert result.exit_code == 1
    assert f"{_SPEC[source_class].exposure_prefix}…{secret[-4:]}" in result.output
    _assert_withheld(result.output, secret)


def test_a_span_name_holding_the_password_is_withheld() -> None:
    text, _, secret = _CASES["userinfo"]
    result = _scan_cli(text, name=f"connect {text}")
    assert "[withheld]" in result.output
    _assert_withheld(result.output, secret)


def test_the_user_name_of_a_url_is_never_printed() -> None:
    # The user name is arbitrary text and can itself be a token; only the
    # class's fixed head is printed before the password's last four.
    token = "ghp_Qh7Rk2Ls9Vn4Xb6Zt1Wc8Mp3Jd5Fg0Yu2Ae7R"
    user = "svc_reader_9f2c"
    for url in (f"https://{token}:{_PASSWORD}@git.example/acme/repo.git",
                f"postgres://{user}:{_PASSWORD}@db.example/app"):
        result = _scan_cli(url)
        assert result.exit_code == 1
        assert user not in result.output
        _assert_withheld(result.output, token)
        _assert_withheld(result.output, _PASSWORD)


def test_an_aws_session_token_replaces_the_key_name_review_line() -> None:
    # On its own the parameter name only reads as the key name `Token`, a
    # review-queue line whose value runs on past `&` to the end of the query.
    # The signed-URL classes judge the real value; the key-name line goes.
    text = _CASES["amz-security-token"][0]
    assert {h.source_class for h in _hits(text)} == {"aws_url_security_token"}
    bare = text.split("?", 1)[1].split("&X-Amz-Security-Token", 1)[1]
    assert bare.startswith("=")
    assert [h.source_class for h in _hits("X-Amz-Security-Token" + bare)] == ["credential_field"]


# --- the userinfo password threshold ----------------------------------------


@pytest.mark.parametrize("password", [
    pytest.param("Pq7Zt2Lm9Xc4", id="12-chars-12-distinct"),
    pytest.param("Pq7Zt2Lm9Xc", id="11-chars"),
    pytest.param("aabbccddeefg", id="12-chars-7-distinct"),
    pytest.param("aabbccddeefgh", id="13-chars-8-distinct"),
    # Separators inside a password are part of it: every character counts,
    # none is stripped as a class head.
    pytest.param("ab_cd_ef_gh_ij12", id="separators-count"),
    pytest.param("hunter2", id="short"),
])
def test_a_userinfo_password_is_verdict_grade_from_12_chars_with_8_distinct(
    password: str,
) -> None:
    strong = len(password) >= 12 and len(set(password)) >= 8
    found = [h for h in _hits(f"postgres://app:{password}@db.example:5432/app")
             if h.source_class == "url_with_password"]
    assert len(found) == 1
    hit = found[0]
    assert hit.eligible is strong
    assert hit.value_len == len(password)
    if not strong:
        # The review queue keeps no part of a weak password: no last four
        # characters and no fingerprint to look it up by.
        assert hit.last4 is None
        assert hit.fingerprint is None
    result = _scan_cli(f"postgres://app:{password}@db.example:5432/app")
    assert result.exit_code == (1 if strong else 0)
    _assert_withheld(result.output, password)


_WRITTEN_PLACEHOLDERS = [
    "${DB_PASSWORD}", "$DB_PASSWORD", "{{db_password}}", "<password>", "***", "xxxxxxxx",
    "password", "PASSWORD", "Passwd", "pass", "pwd", "secret", "your_password", "mypassword",
    "changeme", "example", "placeholder", "REDACTED", "[REDACTED]", "null", "None",
    "%(password)s", "%s",
]
_PLACEHOLDERS = [
    *_WRITTEN_PLACEHOLDERS,
    *sorted({quote(p, safe="") for p in _WRITTEN_PLACEHOLDERS} - set(_WRITTEN_PLACEHOLDERS)),
]


@pytest.mark.parametrize("password", _PLACEHOLDERS)
def test_an_empty_or_placeholder_password_is_never_reported(password: str) -> None:
    url = f"postgres://app:{password}@db.example:5432/app"
    assert _hits(url) == []
    assert _scan_cli(url).exit_code == 0
    assert value_shape(url) is None
    assert scan_text(url + "\n", rel="notes.txt") == []


@pytest.mark.parametrize("url", [
    "postgres://app:@db.example:5432/app", "postgres://app@db.example/app",
    "https://db.example:5432/app", "ssh://git@host.example:22/repo.git",
    # RFC 3986 3.2: a port followed by `?` or `#` ends the authority.
    "http://localhost:3000?email=jane.doe@example.com",
    "http://shop.example:8080?ref=partner&contact=sales@example.com",
    "http://localhost:3000#to=jane.doe@example.com",
    # RFC 3986 3.2.2: `[` and `]` enclose an IP literal host, never userinfo.
    "http://[::1]:3000?email=jane.doe@example.com",
    "http://[2001:db8::1]:8443?contact=sales@example.com",
    "http://[::1]:3000#to=jane.doe@example.com",
])
def test_a_url_with_no_password_is_not_a_credential(url: str) -> None:
    assert _hits(url) == []
    assert value_shape(url) is None
    assert scan_text(url + "\n", rel="notes.txt") == []


# libpq and SQLAlchemy's `make_url` read the password up to the `@`, so an
# unencoded `#` or `?` is part of it.
@pytest.mark.parametrize("password", ["Kq7#pX2mZr9vLw4t", "Kq7?pX2mZr9vLw4t"])
@pytest.mark.parametrize("host", ["db.example:5432", "[::1]:5432"])
def test_a_password_holding_a_hash_or_question_mark_is_found(password: str, host: str) -> None:
    url = f"postgres://app:{password}@{host}/app"
    found = [h for h in _hits(url) if h.source_class == "url_with_password"]
    assert len(found) == 1
    assert found[0].eligible
    assert value_shape(url) == "url_with_password"
    assert [h.value for h in scan_text(f"DATABASE_URL={url}\n", rel=".env")] == [url]


# --- signed URLs: only in their signed-URL shape ----------------------------


@pytest.mark.parametrize("text", [
    # A bare `sig` is common outside a SAS token: it counts only beside sv= or se=.
    pytest.param(f"https://api.example/v1/items?sig={quote(_SAS_SIGNATURE, safe='')}&page=2",
                 id="sig-without-sas-params"),
    pytest.param(f"sig={_SAS_SIGNATURE}", id="sig-alone"),
    pytest.param(f"https://a.example/x?sv=2022-11-02 https://b.example/y?sig={_SAS_SIGNATURE}",
                 id="sv-in-another-url"),
    pytest.param(f"https://a.example/x?sv=2022-11-02&xsig={_SAS_SIGNATURE}", id="xsig"),
    # The Amz and Goog names count only as query parameters.
    pytest.param(f"X-Amz-Signature={_AMZ_SIGNATURE}", id="amz-not-a-query-param"),
    pytest.param(f"header X-Goog-Signature: {_GOOG_SIGNATURE}", id="goog-header"),
    pytest.param(f"https://b.example/o?xX-Amz-Signature={_AMZ_SIGNATURE}", id="amz-glued"),
])
def test_a_signed_url_name_outside_its_shape_is_not_reported(text: str) -> None:
    assert _hits(text) == []


@pytest.mark.parametrize(("source_class", "text"), [
    ("aws_url_signature", "https://b.example/o?X-Amz-Signature=" + "0" * 64),
    ("aws_url_signature", "https://b.example/o?X-Amz-Signature=" + _AMZ_SIGNATURE[:63]),
    ("google_url_signature", "https://b.example/o?X-Goog-Signature=" + _AMZ_SIGNATURE[:40]),
    ("aws_url_security_token", "https://b.example/o?X-Amz-Security-Token=" + "FAKE" * 30),
    ("azure_sas_signature", f"https://a.example/c?{_SAS_PARAMS}&sig=" + _SAS_SIGNATURE[:-2]),
])
def test_a_signed_url_value_that_misses_its_shape_goes_to_the_review_queue(
    source_class: str, text: str
) -> None:
    found = [h for h in _hits(text) if h.source_class == source_class]
    assert len(found) == 1
    assert not found[0].eligible


# --- the inventory recognizes the same URLs ----------------------------------


@pytest.mark.parametrize("url", [
    *(_CASES[c][0] for c in ("userinfo", "userinfo-encoded")),
    "postgres://app:hunter2@db.example:5432/app",
    "mysql://root:Pq7Zt2Lm9Xc4@127.0.0.1:3306/",
    *(f"postgres://app:{p}@db.example:5432/app" for p in _PLACEHOLDERS),
    "postgres://app:@db.example:5432/app",
])
def test_the_inventory_and_the_scan_agree_on_a_url_with_a_password(url: str) -> None:
    scan_finds = any(h.source_class == "url_with_password" for h in _hits(url))
    assert (value_shape(url) == "url_with_password") is scan_finds
    project = [h for h in scan_text(f"url: {url}\n", rel="config.yml") if h.value == url]
    assert bool(project) is scan_finds


@pytest.mark.parametrize("sig", [_SAS_SIGNATURE, "FAKE" * 5], ids=["real-sig", "fake-sig"])
@pytest.mark.parametrize("lead", ["", "?"], ids=["bare", "query"])
def test_the_inventory_lists_a_sas_token_by_its_key_name(sig: str, lead: str) -> None:
    line = f"AZURE_STORAGE_SAS_TOKEN={lead}{_SAS_PARAMS}&sig={sig}\n"
    assert "AZURE_STORAGE_SAS_TOKEN" in {h.name for h in scan_text(line, rel=".env")}

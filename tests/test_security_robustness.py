"""
Security and robustness patterns.

Covers: secrets module, hashlib + HMAC, bcrypt password hashing,
timezone-aware datetime, Decimal for money, mutable default argument
antipattern, contextlib.suppress / closing, resource cleanup.
"""

from __future__ import annotations

import hashlib
import hmac
import math
import secrets
import time
from contextlib import closing, suppress
from datetime import datetime, timedelta, timezone
from decimal import ROUND_DOWN, ROUND_HALF_UP, Decimal
from io import StringIO

import bcrypt
import pytest


# ===========================================================================
# secrets  —  cryptographically strong random values
# ===========================================================================
# Use `secrets` for tokens, passwords, salts — NOT `random`.

def test_secrets_token_hex():
    token = secrets.token_hex(32)    # 32 bytes → 64 hex chars
    assert len(token) == 64
    assert all(char in "0123456789abcdef" for char in token)


def test_secrets_token_urlsafe():
    token = secrets.token_urlsafe(24)  # 24 bytes → ~32 URL-safe chars
    assert len(token) >= 24            # base64url adds padding


def test_secrets_tokens_are_unique():
    tokens = {secrets.token_hex(16) for _ in range(100)}
    assert len(tokens) == 100          # vanishingly unlikely to collide


def test_secrets_compare_digest_timing_safe():
    """
    Use secrets.compare_digest (or hmac.compare_digest) for token comparison —
    it runs in constant time, preventing timing-oracle attacks.
    Never use == for secret comparison.
    """
    token    = secrets.token_hex(32)
    correct  = token
    wrong    = secrets.token_hex(32)

    assert secrets.compare_digest(correct, token)
    assert not secrets.compare_digest(wrong, token)


def test_secrets_choice_for_passphrase():
    wordlist = ["correct", "horse", "battery", "staple", "paper", "clip"]
    passphrase = " ".join(secrets.choice(wordlist) for _ in range(4))
    assert len(passphrase.split()) == 4
    assert all(word in wordlist for word in passphrase.split())


# ===========================================================================
# hashlib  —  deterministic digests (NOT for passwords — see bcrypt below)
# ===========================================================================

def test_hashlib_sha256():
    data    = b"Hello, world!"
    digest  = hashlib.sha256(data).hexdigest()
    assert len(digest) == 64
    # Deterministic
    assert digest == hashlib.sha256(data).hexdigest()
    # Different input → different digest
    assert digest != hashlib.sha256(b"Hello, World!").hexdigest()


def test_hashlib_file_integrity(tmp_path):
    """Verify a downloaded file hasn't been tampered with."""
    content   = b"package contents"
    expected  = hashlib.sha256(content).hexdigest()

    filepath  = tmp_path / "pkg.tar.gz"
    filepath.write_bytes(content)

    actual = hashlib.sha256(filepath.read_bytes()).hexdigest()
    assert actual == expected


def test_hashlib_incremental():
    """Hash large data in chunks without loading it all into memory."""
    h = hashlib.sha256()
    h.update(b"chunk1")
    h.update(b"chunk2")
    h.update(b"chunk3")
    streaming_digest = h.hexdigest()

    one_shot = hashlib.sha256(b"chunk1chunk2chunk3").hexdigest()
    assert streaming_digest == one_shot


# ===========================================================================
# HMAC  —  message authentication codes (verify origin + integrity)
# ===========================================================================

def sign(message: bytes, key: bytes) -> str:
    return hmac.new(key, message, hashlib.sha256).hexdigest()


def verify(message: bytes, key: bytes, signature: str) -> bool:
    expected = sign(message, key)
    return hmac.compare_digest(expected, signature)


def test_hmac_sign_and_verify():
    key  = secrets.token_bytes(32)
    message  = b'{"user_id": 42, "role": "admin"}'
    sig  = sign(message, key)

    assert verify(message, key, sig)
    assert not verify(b"tampered", key, sig)


def test_hmac_different_keys_different_sigs():
    message  = b"data"
    key1   = secrets.token_bytes(32)
    key2   = secrets.token_bytes(32)
    assert sign(message, key1) != sign(message, key2)


# ===========================================================================
# bcrypt  —  password hashing (adaptive, slow by design)
# ===========================================================================
# Rule: NEVER store plaintext passwords.
#       NEVER use sha256/md5 for passwords — use bcrypt/argon2/scrypt.

def hash_password(plaintext: str) -> bytes:
    return bcrypt.hashpw(plaintext.encode(), bcrypt.gensalt())


def check_password(plaintext: str, hashed: bytes) -> bool:
    return bcrypt.checkpw(plaintext.encode(), hashed)


def test_bcrypt_hash_and_verify():
    hashed = hash_password("s3cr3tP@ss!")
    assert check_password("s3cr3tP@ss!", hashed)
    assert not check_password("wrong", hashed)


def test_bcrypt_different_salts():
    """Same password → different hash each time (random salt embedded)."""
    h1 = hash_password("password")
    h2 = hash_password("password")
    assert h1 != h2              # different salts
    assert check_password("password", h1)
    assert check_password("password", h2)


def test_bcrypt_hash_is_bytes():
    hashed = hash_password("test")
    assert isinstance(hashed, bytes)
    assert hashed.startswith(b"$2b$")   # bcrypt prefix


# ===========================================================================
# Timezone-aware datetime
# ===========================================================================
# Rule: always use timezone-aware datetimes.
#       Store / transmit in UTC; convert to local only for display.

def test_datetime_utc_now():
    now = datetime.now(timezone.utc)
    assert now.tzinfo is not None
    assert now.tzinfo.utcoffset(now).total_seconds() == 0


def test_datetime_naive_vs_aware():
    naive  = datetime(2025, 1, 1, 12, 0)
    aware  = datetime(2025, 1, 1, 12, 0, tzinfo=timezone.utc)

    assert naive.tzinfo is None
    assert aware.tzinfo is not None

    with pytest.raises(TypeError):
        _ = aware - naive    # can't subtract naive from aware


def test_datetime_timezone_conversion():
    from datetime import timezone as tz
    utc_plus5 = timezone(timedelta(hours=5))

    utc_time   = datetime(2025, 6, 1, 8, 0, tzinfo=timezone.utc)
    local_time = utc_time.astimezone(utc_plus5)

    assert local_time.hour == 13   # 08:00 UTC = 13:00 UTC+5
    assert local_time.tzinfo == utc_plus5


def test_datetime_isoformat_and_parse():
    dt  = datetime(2025, 3, 15, 10, 30, 0, tzinfo=timezone.utc)
    iso = dt.isoformat()                  # "2025-03-15T10:30:00+00:00"
    assert "+" in iso

    parsed = datetime.fromisoformat(iso)  # Python 3.11+ handles +00:00 suffix
    assert parsed == dt


def test_datetime_duration_arithmetic():
    start = datetime(2025, 1, 1, tzinfo=timezone.utc)
    end   = datetime(2025, 1, 8, tzinfo=timezone.utc)

    delta = end - start
    assert delta.days == 7
    assert delta.total_seconds() == 7 * 24 * 3600

    two_weeks_later = start + timedelta(weeks=2)
    assert two_weeks_later.day == 15


# ===========================================================================
# Decimal  —  exact arithmetic for money
# ===========================================================================
# Rule: never use float for money — rounding errors accumulate.

def test_decimal_vs_float_precision():
    float_result   = 0.1 + 0.2           # 0.30000000000000004
    decimal_result = Decimal("0.1") + Decimal("0.2")   # exactly 0.3

    assert float_result != 0.3
    assert decimal_result == Decimal("0.3")


def test_decimal_money_rounding():
    price    = Decimal("19.99")
    tax_rate = Decimal("0.0875")   # 8.75%
    tax      = (price * tax_rate).quantize(Decimal("0.01"), rounding=ROUND_HALF_UP)
    total    = price + tax

    assert tax   == Decimal("1.75")
    assert total == Decimal("21.74")


def test_decimal_rounding_modes():
    CENTS = Decimal("0.01")
    assert Decimal("2.345").quantize(CENTS, rounding=ROUND_HALF_UP) == Decimal("2.35")
    assert Decimal("2.344").quantize(CENTS, rounding=ROUND_HALF_UP) == Decimal("2.34")
    assert Decimal("2.349").quantize(CENTS, rounding=ROUND_DOWN)    == Decimal("2.34")


def test_decimal_sum_of_prices():
    prices = [Decimal("9.99"), Decimal("24.99"), Decimal("4.99")]
    total  = sum(prices, Decimal("0"))
    assert total == Decimal("39.97")


# ===========================================================================
# Mutable default argument antipattern
# ===========================================================================

# BAD — the list is shared across ALL calls:
def bad_append(item, lst=[]):    # noqa: B006  (ruff B006 catches this)
    lst.append(item)
    return lst


def test_mutable_default_antipattern():
    """The bug: default list is created once and reused forever."""
    r1 = bad_append("a")
    r2 = bad_append("b")
    # r1 and r2 are the SAME list object
    assert r1 is r2
    assert r1 == ["a", "b"]   # "a" is still there!


# GOOD — use None as sentinel:
def good_append(item, lst=None):
    if lst is None:
        lst = []
    lst.append(item)
    return lst


def test_mutable_default_fixed():
    r1 = good_append("a")
    r2 = good_append("b")
    assert r1 == ["a"]
    assert r2 == ["b"]
    assert r1 is not r2


# The same trap with dicts:
def bad_config(key, value, cfg={}):   # noqa: B006
    cfg[key] = value
    return cfg


def test_mutable_default_dict():
    bad_config("x", 1)
    result = bad_config("y", 2)
    assert result == {"x": 1, "y": 2}   # "x" leaked in from previous call!


# ===========================================================================
# contextlib  —  suppress, closing
# ===========================================================================

def test_suppress_specific_exception():
    """suppress() swallows the named exception — great for optional cleanup."""
    result = []
    with suppress(FileNotFoundError):
        open("/nonexistent/path/file.txt")   # raises but is suppressed
        result.append("opened")

    assert result == []   # block short-circuited at the raise


def test_suppress_only_named_exception():
    """Exceptions NOT in the suppress list still propagate."""
    with pytest.raises(ValueError):
        with suppress(FileNotFoundError):
            raise ValueError("this must propagate")


def test_closing_ensures_cleanup():
    """closing() wraps objects that have .close() but no __exit__."""
    buf = StringIO("hello world")
    with closing(buf) as f:
        content = f.read()
    # After the with-block, f.close() was called
    assert content == "hello world"
    assert f.closed

"""
Advanced pytest patterns.

Covers: caplog, capsys, freezegun, hypothesis,
pytest.mark (skip/skipif/xfail/custom), pytest.warns,
snapshot / golden-file testing.

Builds on test_testing_helpers.py (mocks, fixtures, tmp_path, parametrize).
"""

from __future__ import annotations

import json
import logging
import time
import warnings
from datetime import date, datetime
from pathlib import Path

import pytest
from freezegun import freeze_time
from hypothesis import assume, given, settings as h_settings
from hypothesis import strategies as st


# ===========================================================================
# caplog  —  capture log records emitted during a test
# ===========================================================================

logger = logging.getLogger("myapp.service")


def process_order(order_id: str, amount: float) -> str:
    if amount <= 0:
        logger.warning("order %s has zero/negative amount: %s", order_id, amount)
        return "rejected"
    logger.info("processing order %s for $%.2f", order_id, amount)
    return "accepted"


def test_caplog_captures_info(caplog):
    with caplog.at_level(logging.INFO, logger="myapp.service"):
        result = process_order("ORD-1", 49.99)

    assert result == "accepted"
    assert len(caplog.records) == 1
    assert caplog.records[0].levelname == "INFO"
    assert "ORD-1" in caplog.records[0].message


def test_caplog_captures_warning(caplog):
    with caplog.at_level(logging.WARNING, logger="myapp.service"):
        result = process_order("ORD-2", 0.0)

    assert result == "rejected"
    assert "zero/negative" in caplog.text
    assert any(r.levelno == logging.WARNING for r in caplog.records)


def test_caplog_no_unexpected_logs(caplog):
    with caplog.at_level(logging.ERROR, logger="myapp.service"):
        process_order("ORD-3", 10.0)
    assert caplog.records == []   # no ERROR-level logs


# ===========================================================================
# capsys  —  capture sys.stdout / sys.stderr
# (capfd captures at the file-descriptor level, works for C extensions too)
# ===========================================================================

def print_report(items: list[str]) -> None:
    print(f"Report ({len(items)} items):")
    for item in items:
        print(f"  - {item}")
    if not items:
        print("  (empty)", file=__import__("sys").stderr)


def test_capsys_stdout(capsys):
    print_report(["alpha", "beta"])
    out, err = capsys.readouterr()
    assert "Report (2 items)" in out
    assert "alpha" in out
    assert err == ""


def test_capsys_stderr(capsys):
    print_report([])
    out, err = capsys.readouterr()
    assert "(empty)" in err


def test_capsys_multiple_captures(capsys):
    print("first")
    out1, _ = capsys.readouterr()   # drain
    print("second")
    out2, _ = capsys.readouterr()   # only captures since last readouterr
    assert "first"  in out1
    assert "second" in out2
    assert "first"  not in out2


# ===========================================================================
# freezegun  —  freeze or travel through time
# ===========================================================================

def days_until_deadline(deadline: date) -> int:
    return (deadline - date.today()).days


def is_market_open() -> bool:
    """Markets open Mon–Fri 09:30–16:00 EST (simplified)."""
    now = datetime.now()
    return now.weekday() < 5 and 9 <= now.hour < 16


@freeze_time("2025-01-15")
def test_freeze_time_decorator():
    assert date.today() == date(2025, 1, 15)
    assert days_until_deadline(date(2025, 2, 14)) == 30


def test_freeze_time_context_manager():
    with freeze_time("2025-06-02 10:00:00"):  # Monday
        assert is_market_open() is True

    with freeze_time("2025-06-02 20:00:00"):  # Monday after hours
        assert is_market_open() is False


def test_freeze_time_tick():
    """tick=True lets time advance normally from the frozen start point."""
    with freeze_time("2025-01-01 00:00:00", tick=True):
        t0 = datetime.now()
        time.sleep(0.01)
        t1 = datetime.now()
        assert t1 > t0


def test_freeze_time_travel():
    """Move forward by incrementing the frozen instant."""
    from freezegun import freeze_time as ft

    with ft("2025-03-01") as frozen:
        assert date.today() == date(2025, 3, 1)
        frozen.move_to("2025-12-31")
        assert date.today() == date(2025, 12, 31)


# ===========================================================================
# hypothesis  —  property-based testing
# ===========================================================================
# Instead of hand-picked examples, Hypothesis generates hundreds of inputs
# and shrinks failures to a minimal reproducing case.

def encode_decode(s: str) -> str:
    """Round-trip: base64 encode then decode."""
    import base64
    return base64.b64decode(base64.b64encode(s.encode())).decode()


@given(st.text())
def test_hypothesis_encode_decode_roundtrip(s):
    assert encode_decode(s) == s


def is_palindrome(s: str) -> bool:
    return s == s[::-1]


@given(st.text(alphabet=st.characters(whitelist_categories=("L",)), min_size=1))
def test_hypothesis_palindrome_reversed_is_palindrome(s):
    """If we construct a palindrome, it must be detected as one."""
    palindrome = s + s[::-1]
    assert is_palindrome(palindrome)


def add(a: int, b: int) -> int:
    return a + b


@given(st.integers(), st.integers())
def test_hypothesis_addition_commutative(a, b):
    assert add(a, b) == add(b, a)


@given(st.integers(), st.integers(), st.integers())
def test_hypothesis_addition_associative(a, b, c):
    assert add(add(a, b), c) == add(a, add(b, c))


@given(st.lists(st.integers(), min_size=1))
def test_hypothesis_sorted_is_sorted(lst):
    result = sorted(lst)
    assert result == sorted(result)        # idempotent
    assert sorted(result, reverse=True) == result[::-1]


@given(st.lists(st.floats(allow_nan=False, allow_infinity=False), min_size=1))
def test_hypothesis_max_is_in_list(lst):
    assume(len(lst) > 0)
    m = max(lst)
    assert m in lst
    assert all(x <= m for x in lst)


@given(st.dictionaries(st.text(), st.integers()))
def test_hypothesis_json_roundtrip(d):
    assert json.loads(json.dumps(d)) == d


# ===========================================================================
# pytest.mark  —  skip, skipif, xfail, custom marks
# ===========================================================================

@pytest.mark.skip(reason="feature not yet implemented")
def test_mark_skip():
    assert False   # never runs


@pytest.mark.skipif(
    condition=True,   # in real code: sys.platform == "win32" etc.
    reason="skipped on this platform",
)
def test_mark_skipif():
    assert False   # never runs


@pytest.mark.xfail(reason="known bug #123", strict=False)
def test_mark_xfail_known_failure():
    assert 1 == 2   # expected to fail → shown as xfail, not ERROR


@pytest.mark.xfail(reason="known crash #456", strict=True)
def test_mark_xfail_strict_still_failing():
    """
    strict=True: the test MUST stay failing.
    If it ever passes, pytest reports XPASS and fails the suite —
    your signal to remove the xfail mark and celebrate the fix.
    """
    raise RuntimeError("still broken")   # expected failure → shows as xfail


# Register custom mark in conftest.py; shown here for reference
# pytestmark = pytest.mark.slow  ← marks entire module
@pytest.mark.slow
def test_mark_custom_slow():
    """Run with: pytest -m slow  or exclude: pytest -m 'not slow'"""
    time.sleep(0)   # stand-in for a long test
    assert True


# ===========================================================================
# pytest.warns
# ===========================================================================

def deprecated_function(x: int) -> int:
    warnings.warn(
        "deprecated_function() is deprecated; use new_function() instead.",
        DeprecationWarning,
        stacklevel=2,
    )
    return x * 2


def test_pytest_warns_deprecation():
    with pytest.warns(DeprecationWarning, match="deprecated_function"):
        result = deprecated_function(5)
    assert result == 10


def test_pytest_warns_captures_multiple():
    with pytest.warns(UserWarning) as warning_list:
        warnings.warn("first",  UserWarning)
        warnings.warn("second", UserWarning)

    messages = [str(w.message) for w in warning_list]
    assert "first"  in messages
    assert "second" in messages


# ===========================================================================
# Snapshot / golden-file testing
# ===========================================================================
# Pattern: on first run, write the output to a file.
#          On subsequent runs, compare against that file.
# Libraries: syrupy, pytest-snapshot — or roll your own (shown below).

def generate_report(data: list[dict]) -> str:
    lines = ["# Sales Report", ""]
    for row in data:
        lines.append(f"- {row['name']}: ${row['revenue']:,.2f}")
    lines.append("")
    lines.append(f"Total: ${sum(r['revenue'] for r in data):,.2f}")
    return "\n".join(lines)


SAMPLE_DATA = [
    {"name": "Widget", "revenue": 12_500.00},
    {"name": "Gadget", "revenue": 8_750.50},
]


def test_snapshot_golden_file(tmp_path):
    """
    Minimal golden-file pattern — no extra library needed.
    Delete the .golden file to regenerate it.
    """
    golden_dir  = Path(__file__).parent / "goldens"
    golden_file = golden_dir / "sales_report.txt"
    golden_dir.mkdir(exist_ok=True)

    actual = generate_report(SAMPLE_DATA)

    if not golden_file.exists():
        golden_file.write_text(actual)
        pytest.skip("golden file created — re-run to verify")

    expected = golden_file.read_text()
    assert actual == expected, (
        "Output changed. If intentional, delete the golden file to regenerate:\n"
        f"  {golden_file}"
    )

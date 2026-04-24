"""
Error handling and observability.

Covers: custom exception hierarchies, exception chaining, stdlib logging,
structlog structured logging, warnings, tenacity retry/backoff.
"""

from __future__ import annotations

import logging
import warnings
from typing import Any

import pytest
import structlog
from tenacity import (
    RetryError,
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_fixed,
    before_log,
    after_log,
)


# ===========================================================================
# Custom exception hierarchy
# ===========================================================================
# Rule: create a base exception per library/domain; users catch the base.

class AppError(Exception):
    """Base for all application exceptions.  Never raise this directly."""


class ValidationError(AppError):
    """Input failed validation."""
    def __init__(self, field: str, message: str):
        self.field = field
        super().__init__(f"{field}: {message}")


class NotFoundError(AppError):
    """Requested resource does not exist."""
    def __init__(self, resource: str, id: Any):
        self.resource = resource
        self.id = id
        super().__init__(f"{resource} with id={id!r} not found")


class ConflictError(AppError):
    """Operation conflicts with current state."""


class ExternalServiceError(AppError):
    """Upstream / third-party service failure."""
    def __init__(self, service: str, status_code: int | None = None):
        self.service = service
        self.status_code = status_code
        message = f"{service} unavailable"
        if status_code:
            message += f" (HTTP {status_code})"
        super().__init__(message)


def test_exception_hierarchy_isinstance():
    err = NotFoundError("User", 42)
    assert isinstance(err, NotFoundError)
    assert isinstance(err, AppError)     # base catch works
    assert isinstance(err, Exception)


def test_exception_carries_metadata():
    err = ValidationError("email", "invalid format")
    assert err.field == "email"
    assert "email" in str(err)
    assert "invalid format" in str(err)


def test_catch_base_class():
    """Library consumers only need to catch AppError."""
    def find_user(uid: int):
        raise NotFoundError("User", uid)

    with pytest.raises(AppError):      # catches subclass
        find_user(99)


# ===========================================================================
# Exception chaining — raise X from Y
# ===========================================================================

def fetch_config(path: str) -> dict:
    try:
        raise FileNotFoundError(f"no such file: {path}")
    except FileNotFoundError as e:
        # __cause__ explicitly set; "During handling of the above..."
        raise AppError(f"Could not load config from {path}") from e


def suppress_context_example():
    try:
        int("not-a-number")
    except ValueError:
        # raise X from None suppresses the context (no "During handling...")
        raise AppError("internal error") from None


def test_exception_chaining_cause():
    with pytest.raises(AppError) as exc_info:
        fetch_config("/etc/missing.toml")
    assert exc_info.value.__cause__ is not None
    assert isinstance(exc_info.value.__cause__, FileNotFoundError)


def test_exception_suppressed_context():
    with pytest.raises(AppError) as exc_info:
        suppress_context_example()
    assert exc_info.value.__cause__   is None
    assert exc_info.value.__context__ is not None   # still set, just suppressed
    assert exc_info.value.__suppress_context__ is True


# ===========================================================================
# ExceptionGroup / except*  (Python 3.11+)
# ===========================================================================

def validate_all(data: dict) -> None:
    errors = []
    if not data.get("name"):
        errors.append(ValidationError("name", "required"))
    if not data.get("email"):
        errors.append(ValidationError("email", "required"))
    if data.get("age", 0) < 0:
        errors.append(ValidationError("age", "must be non-negative"))
    if errors:
        raise ExceptionGroup("validation failed", errors)


def test_exception_group_full_payload():
    with pytest.raises(ExceptionGroup) as exc_info:
        validate_all({"age": -1})

    eg = exc_info.value
    assert len(eg.exceptions) == 3   # name + email + age


def test_exception_group_except_star():
    try:
        validate_all({"age": -1})
    except* ValidationError as eg:
        fields = [error.field for error in eg.exceptions]
        assert "name"  in fields
        assert "email" in fields
        assert "age"   in fields


# ===========================================================================
# stdlib logging
# ===========================================================================

def test_logging_levels_and_caplog(caplog):
    log = logging.getLogger("myapp.orders")

    with caplog.at_level(logging.DEBUG, logger="myapp.orders"):
        log.debug("debug detail")
        log.info("order created: %s", "ORD-1")
        log.warning("low stock: %s", "Widget")
        log.error("payment failed: %s", "ORD-2")

    assert len(caplog.records) == 4
    levels = [record.levelname for record in caplog.records]
    assert levels == ["DEBUG", "INFO", "WARNING", "ERROR"]


def test_logging_does_not_propagate_below_threshold(caplog):
    log = logging.getLogger("myapp.quiet")
    with caplog.at_level(logging.ERROR, logger="myapp.quiet"):
        log.debug("this should be filtered out")
        log.info("so should this")
        log.error("but not this")
    assert len(caplog.records) == 1


def test_logging_extra_context(caplog):
    """Attach structured context to a log record via extra={}."""
    log = logging.getLogger("myapp.http")
    with caplog.at_level(logging.INFO, logger="myapp.http"):
        log.info("request", extra={"method": "GET", "path": "/api/users", "status": 200})

    record = caplog.records[0]
    assert record.method == "GET"     # type: ignore[attr-defined]
    assert record.status == 200       # type: ignore[attr-defined]


# ===========================================================================
# structlog  —  structured, context-rich logging
# ===========================================================================

def test_structlog_capture_logs():
    """structlog.testing.capture_logs() captures bound context as dicts."""
    with structlog.testing.capture_logs() as captured:
        log = structlog.get_logger()
        log.info("user.login",  user_id=42,    ip="1.2.3.4")
        log.error("auth.failed", user_id=99,   reason="bad_password")

    assert len(captured) == 2

    login = captured[0]
    assert login["event"]   == "user.login"
    assert login["user_id"] == 42
    assert login["log_level"] == "info"

    auth_fail = captured[1]
    assert auth_fail["reason"] == "bad_password"


def test_structlog_bound_logger():
    """bind() creates a child logger with pre-attached context."""
    with structlog.testing.capture_logs() as captured:
        request_log = structlog.get_logger().bind(
            request_id="req-abc", user_id=7
        )
        request_log.info("request.start",  path="/api/orders")
        request_log.info("request.finish", status=200)

    assert all(entry["request_id"] == "req-abc" for entry in captured)
    assert all(entry["user_id"]    == 7         for entry in captured)
    assert captured[1]["status"] == 200


# ===========================================================================
# warnings  —  deprecation notices, runtime cautions
# ===========================================================================

def old_api(x: int) -> int:
    warnings.warn(
        "old_api() is deprecated since v2.0; use new_api() instead.",
        DeprecationWarning,
        stacklevel=2,
    )
    return x * 2


class ExperimentalFeature:
    def __init__(self):
        warnings.warn(
            "ExperimentalFeature is not production-ready.",
            UserWarning,
            stacklevel=2,
        )


def test_deprecation_warning_is_raised():
    with pytest.warns(DeprecationWarning, match="deprecated since v2.0"):
        result = old_api(5)
    assert result == 10


def test_user_warning_is_raised():
    with pytest.warns(UserWarning, match="not production-ready"):
        ExperimentalFeature()


def test_warnings_can_be_silenced():
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        result = old_api(3)   # no warning raised
    assert result == 6


# ===========================================================================
# tenacity  —  retry / backoff
# ===========================================================================

# ---- basic retry -----------------------------------------------------------

_call_count = 0

@retry(stop=stop_after_attempt(3), wait=wait_fixed(0))
def flaky_service() -> str:
    global _call_count
    _call_count += 1
    if _call_count < 3:
        raise ConnectionError("transient failure")
    return "ok"


def test_retry_succeeds_on_third_attempt():
    global _call_count
    _call_count = 0
    result = flaky_service()
    assert result == "ok"
    assert _call_count == 3


# ---- exhausted retries raises RetryError -----------------------------------

@retry(stop=stop_after_attempt(2), wait=wait_fixed(0))
def always_fails() -> str:
    raise ConnectionError("always broken")


def test_retry_raises_after_attempts():
    with pytest.raises(RetryError):
        always_fails()


# ---- retry only on specific exceptions ------------------------------------

@retry(
    stop=stop_after_attempt(4),
    wait=wait_fixed(0),
    retry=retry_if_exception_type(ConnectionError),   # NOT on ValueError
    reraise=True,     # re-raise original exception instead of RetryError
)
def type_selective(exc_type: type) -> str:
    raise exc_type("error")


def test_retry_selective_exception_type():
    """ConnectionError is retried; ValueError bubbles immediately."""
    with pytest.raises(ConnectionError):   # ConnectionError: retried then re-raised
        type_selective(ConnectionError)

    with pytest.raises(ValueError):        # ValueError: not retried, raised instantly
        type_selective(ValueError)


# ---- retry with logging callbacks ------------------------------------------

def test_retry_with_logging(caplog):
    attempt_log = logging.getLogger("tenacity")
    counter = {"count": 0}

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_fixed(0),
        before=before_log(attempt_log, logging.DEBUG),
        after=after_log(attempt_log,  logging.DEBUG),
    )
    def tracked():
        counter["count"] += 1
        if counter["count"] < 3:
            raise IOError("not yet")
        return "done"

    with caplog.at_level(logging.DEBUG, logger="tenacity"):
        result = tracked()

    assert result == "done"
    assert counter["count"] == 3

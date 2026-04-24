"""
Testing helpers — mocks, stubs, fixtures, parametrize.
The fixtures defined here are local to this file; shared fixtures go in conftest.py.
"""

import pytest
from unittest.mock import MagicMock, patch, call


# ---------------------------------------------------------------------------
# MagicMock — replace collaborators without real implementations
# ---------------------------------------------------------------------------

class EmailService:
    def send(self, to: str, subject: str, body: str) -> bool:
        raise NotImplementedError("would hit real SMTP")


class UserNotifier:
    def __init__(self, email_service: EmailService):
        self._email = email_service

    def notify_signup(self, user_email: str):
        self._email.send(
            to=user_email,
            subject="Welcome!",
            body="Thanks for signing up.",
        )


def test_magicmock_verifies_call():
    mock_email = MagicMock(spec=EmailService)
    notifier = UserNotifier(mock_email)
    notifier.notify_signup("alice@example.com")

    mock_email.send.assert_called_once_with(
        to="alice@example.com",
        subject="Welcome!",
        body="Thanks for signing up.",
    )


def test_magicmock_return_value():
    mock_email = MagicMock(spec=EmailService)
    mock_email.send.return_value = True

    result = mock_email.send("x@y.com", "hi", "body")
    assert result is True


def test_magicmock_side_effect_raises():
    mock_email = MagicMock(spec=EmailService)
    mock_email.send.side_effect = ConnectionError("SMTP down")

    with pytest.raises(ConnectionError):
        mock_email.send("x@y.com", "hi", "body")


def test_magicmock_call_count_and_args_list():
    mock_fn = MagicMock(return_value="ok")
    mock_fn("a", 1)
    mock_fn("b", 2)

    assert mock_fn.call_count == 2
    assert mock_fn.call_args_list == [call("a", 1), call("b", 2)]


# ---------------------------------------------------------------------------
# patch — monkey-patching via decorator or context manager
# ---------------------------------------------------------------------------

import time as _time_module   # reference for patching

def get_timestamp() -> float:
    return _time_module.time()


@patch("tests.test_testing_helpers._time_module.time", return_value=1_000_000.0)
def test_patch_decorator(mock_time):
    ts = get_timestamp()
    assert ts == 1_000_000.0
    mock_time.assert_called_once()


def test_patch_context_manager():
    with patch("tests.test_testing_helpers._time_module.time", return_value=42.0):
        assert get_timestamp() == 42.0
    # patch is removed after the with-block
    assert get_timestamp() != 42.0


# ---------------------------------------------------------------------------
# patch.object — patch a method on an existing object / class
# ---------------------------------------------------------------------------

class PaymentGateway:
    def charge(self, amount: float) -> dict:
        raise NotImplementedError

class OrderService:
    def __init__(self, gateway: PaymentGateway):
        self.gateway = gateway

    def place_order(self, amount: float) -> str:
        result = self.gateway.charge(amount)
        return "success" if result.get("status") == "ok" else "failed"


def test_patch_object():
    gateway = PaymentGateway()
    with patch.object(gateway, "charge", return_value={"status": "ok"}):
        svc = OrderService(gateway)
        assert svc.place_order(99.99) == "success"


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def sample_users():
    """In-memory user list — shared setup, isolated per test."""
    return [
        {"id": 1, "name": "Alice", "active": True},
        {"id": 2, "name": "Bob",   "active": False},
        {"id": 3, "name": "Carol", "active": True},
    ]


def test_fixture_filters_active(sample_users):
    active = [user for user in sample_users if user["active"]]
    assert len(active) == 2
    assert active[0]["name"] == "Alice"


def test_fixture_is_independent(sample_users):
    # Mutations here don't bleed into other tests — each test gets a fresh copy
    sample_users.clear()
    assert sample_users == []


# ---------------------------------------------------------------------------
# tmp_path — built-in fixture for temporary files
# ---------------------------------------------------------------------------

def test_tmp_path_file_io(tmp_path):
    data_file = tmp_path / "data.txt"
    data_file.write_text("hello\nworld\n")

    lines = data_file.read_text().splitlines()
    assert lines == ["hello", "world"]


# ---------------------------------------------------------------------------
# capfd — capture stdout / stderr
# ---------------------------------------------------------------------------

def test_capfd_captures_print(capfd):
    print("hello from test")
    out, err = capfd.readouterr()
    assert out.strip() == "hello from test"
    assert err == ""


# ---------------------------------------------------------------------------
# parametrize — data-driven tests (replaces RSpec shared_examples + let)
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("n,expected", [
    (0, True),
    (1, True),
    (4, True),
    (9, True),
    (10, False),
    (16, True),
])
def test_parametrize_is_perfect_square(n, expected):
    import math
    result = math.isqrt(n) ** 2 == n
    assert result == expected


# ---------------------------------------------------------------------------
# Fixture with scope — "session" fixtures are created once per test run
# ---------------------------------------------------------------------------

@pytest.fixture(scope="session")
def expensive_resource():
    """Simulate a one-time setup (e.g., loading a ML model)."""
    return {"model": "loaded", "vocab_size": 50000}


def test_session_fixture_a(expensive_resource):
    assert expensive_resource["vocab_size"] == 50000


def test_session_fixture_b(expensive_resource):
    assert expensive_resource["model"] == "loaded"

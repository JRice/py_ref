"""
Idiomatic Python patterns with no direct Ruby equivalent.
context managers, descriptors, protocols, dataclasses, type hints, etc.
"""

from __future__ import annotations

import contextlib
import copy
from dataclasses import dataclass, field, KW_ONLY
from typing import Protocol, TypeVar, Generic, Iterator, overload


# ---------------------------------------------------------------------------
# Context managers — the "resource block" pattern
# ---------------------------------------------------------------------------

class Timer:
    """Measure elapsed time for a block of code."""
    import time as _t

    def __enter__(self):
        self._start = self._t.monotonic()
        return self

    def __exit__(self, *_):
        self.elapsed = self._t.monotonic() - self._start

    @property
    def ms(self):
        return self.elapsed * 1000


def test_context_manager_class():
    with Timer() as timer:
        total = sum(range(10_000))
    assert total == 49995000
    assert timer.elapsed >= 0


@contextlib.contextmanager
def managed_list() -> Iterator[list]:
    """contextlib.contextmanager turns a generator into a context manager."""
    lst: list = []
    try:
        yield lst
    finally:
        lst.clear()


def test_contextmanager_decorator():
    with managed_list() as items:
        items.extend([1, 2, 3])
        assert items == [1, 2, 3]
    assert items == []   # cleaned up in finally


# ---------------------------------------------------------------------------
# @property and descriptors — computed attributes, like Ruby attr_accessor
# ---------------------------------------------------------------------------

class Temperature:
    def __init__(self, celsius: float):
        self._c = celsius

    @property
    def celsius(self) -> float:
        return self._c

    @celsius.setter
    def celsius(self, value: float):
        if value < -273.15:
            raise ValueError("Below absolute zero")
        self._c = value

    @property
    def fahrenheit(self) -> float:
        return self._c * 9 / 5 + 32


def test_property():
    t = Temperature(100)
    assert t.fahrenheit == 212.0
    t.celsius = 0
    assert t.fahrenheit == 32.0

    import pytest
    with pytest.raises(ValueError):
        t.celsius = -300


# ---------------------------------------------------------------------------
# __dunder__ methods — operator overloading
# ---------------------------------------------------------------------------

@dataclass
class Vector2D:
    x: float
    y: float

    def __add__(self, other: Vector2D) -> Vector2D:
        return Vector2D(self.x + other.x, self.y + other.y)

    def __mul__(self, scalar: float) -> Vector2D:
        return Vector2D(self.x * scalar, self.y * scalar)

    def __abs__(self) -> float:
        return (self.x**2 + self.y**2) ** 0.5

    def __repr__(self) -> str:
        return f"Vector2D({self.x}, {self.y})"


def test_dunder_operators():
    v1 = Vector2D(1.0, 2.0)
    v2 = Vector2D(3.0, 4.0)
    assert v1 + v2 == Vector2D(4.0, 6.0)
    assert v1 * 2 == Vector2D(2.0, 4.0)
    assert abs(Vector2D(3.0, 4.0)) == 5.0


# ---------------------------------------------------------------------------
# Protocols — structural typing (duck typing with type checker support)
# ---------------------------------------------------------------------------

class Drawable(Protocol):
    def draw(self) -> str: ...


class Circle:
    def draw(self) -> str:
        return "○"

class Square:
    def draw(self) -> str:
        return "□"

class Triangle:
    """Does NOT implement Drawable — no draw method."""
    pass


def render_all(shapes: list[Drawable]) -> str:
    return " ".join(shape.draw() for shape in shapes)


def test_protocol_structural_typing():
    shapes = [Circle(), Square(), Circle()]
    assert render_all(shapes) == "○ □ ○"
    # No inheritance required — any class with .draw() satisfies Drawable


# ---------------------------------------------------------------------------
# Dataclass features
# ---------------------------------------------------------------------------

@dataclass
class Config:
    host: str
    port: int = 5432
    KW_ONLY: ... = KW_ONLY  # all following fields are keyword-only
    debug: bool = False
    tags: list[str] = field(default_factory=list)  # mutable default!

    def __post_init__(self):
        if not self.host:
            raise ValueError("host required")


def test_dataclass_defaults_and_validation():
    config = Config("localhost")
    assert config.port == 5432
    assert config.debug is False
    assert config.tags == []

    config2 = Config("prod.server", 5433, debug=True, tags=["prod"])
    assert config2.tags == ["prod"]

    # Mutable default_factory means each instance gets its own list
    assert config.tags is not config2.tags


def test_dataclass_frozen():
    @dataclass(frozen=True)
    class Point:
        x: float
        y: float

    p = Point(1.0, 2.0)
    import pytest
    with pytest.raises(Exception):   # FrozenInstanceError
        p.x = 99.0


# ---------------------------------------------------------------------------
# Generics
# ---------------------------------------------------------------------------

T = TypeVar("T")

class Stack(Generic[T]):
    def __init__(self):
        self._items: list[T] = []

    def push(self, item: T) -> None:
        self._items.append(item)

    def pop(self) -> T:
        return self._items.pop()

    def __len__(self) -> int:
        return len(self._items)


def test_generic_stack():
    stack: Stack[int] = Stack()
    stack.push(1)
    stack.push(2)
    assert stack.pop() == 2
    assert len(stack) == 1


# ---------------------------------------------------------------------------
# copy — shallow vs deep
# ---------------------------------------------------------------------------

def test_shallow_vs_deep_copy():
    original = {"name": "Alice", "scores": [10, 20, 30]}

    shallow = copy.copy(original)
    shallow["scores"].append(99)
    assert original["scores"] == [10, 20, 30, 99]   # inner list is shared!

    deep = copy.deepcopy(original)
    deep["scores"].append(999)
    assert original["scores"] == [10, 20, 30, 99]   # NOT affected


# ---------------------------------------------------------------------------
# Exception chaining and groups (Python 3.11+)
# ---------------------------------------------------------------------------

def test_exception_chaining():
    import pytest

    def load_config(path: str):
        try:
            raise FileNotFoundError(f"{path} not found")
        except FileNotFoundError as e:
            raise RuntimeError("Config load failed") from e

    with pytest.raises(RuntimeError) as exc_info:
        load_config("/etc/myapp.conf")

    assert exc_info.value.__cause__ is not None
    assert "not found" in str(exc_info.value.__cause__)


def test_exception_group():
    """ExceptionGroup (Python 3.11+) — collect multiple failures."""
    errors = [ValueError("bad field"), TypeError("wrong type")]
    try:
        raise ExceptionGroup("validation errors", errors)
    except* ValueError as eg:
        assert len(eg.exceptions) == 1
    except* TypeError as eg:
        assert len(eg.exceptions) == 1


# ---------------------------------------------------------------------------
# match statement (Python 3.10+) — structural pattern matching
# ---------------------------------------------------------------------------

def classify_event(event: dict) -> str:
    match event:
        case {"type": "click", "button": "left"}:
            return "left-click"
        case {"type": "click", "button": button}:
            return f"click:{button}"
        case {"type": "keypress", "key": str(key)}:
            return f"key:{key}"
        case {"type": event_type}:
            return f"unknown:{event_type}"
        case _:
            return "invalid"


def test_match_statement():
    assert classify_event({"type": "click", "button": "left"}) == "left-click"
    assert classify_event({"type": "click", "button": "right"}) == "click:right"
    assert classify_event({"type": "keypress", "key": "Enter"}) == "key:Enter"
    assert classify_event({"type": "scroll"}) == "unknown:scroll"
    assert classify_event({}) == "invalid"


# ---------------------------------------------------------------------------
# String tricks
# ---------------------------------------------------------------------------

def test_string_formatting():
    name, score = "Alice", 98.5
    assert f"{name} scored {score:.1f}%" == "Alice scored 98.5%"
    assert f"{1_000_000:,}" == "1,000,000"
    assert f"{'left':<10}|" == "left      |"
    assert f"{'right':>10}|" == "     right|"
    assert f"{42:08b}" == "00101010"     # binary, zero-padded


def test_string_methods():
    s = "  Hello, World!  "
    assert s.strip() == "Hello, World!"
    assert s.strip().lower() == "hello, world!"
    assert "World" in s
    assert s.strip().replace("World", "Python") == "Hello, Python!"

    csv_line = "alice,30,engineer"
    fields = csv_line.split(",")
    assert fields == ["alice", "30", "engineer"]
    assert "-".join(fields) == "alice-30-engineer"

"""
Python data model deep cuts.

Covers: attrs, NamedTuple, TypedDict, the full dunder suite,
NewType, Literal, TypeVar-with-bound, cached_property.

Skip if you're looking for: dataclasses (test_datatypes.py),
Protocols (test_pythonisms.py / test_patterns.py),
context managers / generators / itertools (test_pythonisms.py / test_loops.py).
"""

from __future__ import annotations

import math
from functools import cached_property, total_ordering
from typing import (
    Callable, ClassVar, Generator, Literal, NamedTuple,
    NewType, Protocol, TypeVar, runtime_checkable,
)
from typing import TypedDict

import attrs


# ===========================================================================
# attrs  —  like dataclasses but older, richer, and still widely used
# ===========================================================================
# attrs gives you: slots by default, validators, converters, on_setattr hooks.
# Use @attrs.define (modern) rather than @attr.s (legacy).

@attrs.define
class Vector:
    x: float = attrs.field(validator=attrs.validators.instance_of(float))
    y: float = attrs.field(validator=attrs.validators.instance_of(float))

    def magnitude(self) -> float:
        return math.hypot(self.x, self.y)


def test_attrs_basic():
    v = Vector(3.0, 4.0)
    assert v.magnitude() == 5.0
    assert repr(v) == "Vector(x=3.0, y=4.0)"   # __repr__ generated


def test_attrs_validator():
    import pytest
    with pytest.raises(TypeError):
        Vector("not a float", 1.0)   # instance_of validator fires


@attrs.define(frozen=True)
class RGB:
    r: int = attrs.field(validator=[
        attrs.validators.instance_of(int),
        attrs.validators.in_(range(256)),
    ])
    g: int = attrs.field(validator=attrs.validators.in_(range(256)))
    b: int = attrs.field(validator=attrs.validators.in_(range(256)))

    def hex(self) -> str:
        return f"#{self.r:02x}{self.g:02x}{self.b:02x}"


def test_attrs_frozen_and_validator():
    import pytest
    c = RGB(255, 128, 0)
    assert c.hex() == "#ff8000"
    with pytest.raises(attrs.exceptions.FrozenInstanceError):
        c.r = 0
    with pytest.raises(ValueError):
        RGB(256, 0, 0)   # 256 not in range(256)


@attrs.define
class Config:
    host: str = "localhost"
    port: int = attrs.field(
        default=8080,
        converter=int,        # coerce strings to int at construction time
    )
    tags: list[str] = attrs.Factory(list)   # mutable default via Factory

def test_attrs_converter_and_factory():
    cfg = Config(port="9000")   # string coerced to int
    assert cfg.port == 9000
    cfg.tags.append("prod")
    cfg2 = Config()
    assert cfg2.tags == []      # separate list per instance


# ===========================================================================
# NamedTuple  —  immutable, indexable, unpackable, yet typed
# ===========================================================================

class Point(NamedTuple):
    x: float
    y: float

    def distance_to(self, other: "Point") -> float:
        return math.hypot(self.x - other.x, self.y - other.y)


class Movie(NamedTuple):
    title: str
    year:  int
    rating: float = 0.0   # default values allowed


def test_namedtuple_basic():
    p1 = Point(0.0, 0.0)
    p2 = Point(3.0, 4.0)
    assert p1.distance_to(p2) == 5.0
    assert p2[0] == 3.0           # positional indexing
    x, y = p2                     # unpacking
    assert x == 3.0


def test_namedtuple_immutable():
    import pytest
    p = Point(1.0, 2.0)
    with pytest.raises(AttributeError):
        p.x = 99.0


def test_namedtuple_as_dict_and_replace():
    m = Movie("Dune", 2021, 8.0)
    assert m._asdict() == {"title": "Dune", "year": 2021, "rating": 8.0}
    sequel = m._replace(title="Dune: Part Two", year=2024)
    assert sequel.title == "Dune: Part Two"
    assert m.title == "Dune"    # original unchanged


def test_namedtuple_sorting():
    movies = [Movie("Z", 2000, 7.0), Movie("A", 2020, 9.5), Movie("M", 2010, 6.0)]
    by_rating = sorted(movies, key=lambda m: m.rating, reverse=True)
    assert by_rating[0].title == "A"


# ===========================================================================
# TypedDict  —  typed shapes for dicts (common in APIs / JSON data)
# ===========================================================================

class Address(TypedDict):
    street: str
    city:   str
    zip:    str


class User(TypedDict, total=False):   # total=False → all keys optional
    id:      int
    name:    str
    email:   str
    address: Address              # nested TypedDict


def test_typeddict_construction():
    user: User = {"id": 1, "name": "Alice"}   # email/address omitted — fine with total=False
    assert user["name"] == "Alice"
    assert "email" not in user


def test_typeddict_nested():
    user: User = {
        "id": 2,
        "name": "Bob",
        "address": {"street": "1 Main St", "city": "Springfield", "zip": "12345"},
    }
    assert user["address"]["city"] == "Springfield"


def test_typeddict_is_just_a_dict():
    # TypedDict is purely a type-checker hint — at runtime it's a plain dict
    user: User = {"id": 3, "name": "Carol"}
    assert type(user) is dict
    user["unknown_key"] = "ignored by type checker, fine at runtime"  # type: ignore[typeddict-unknown-key]


# ===========================================================================
# Full dunder suite
# ===========================================================================

@total_ordering   # only need __eq__ + one of __lt__/__le__/__gt__/__ge__
class Version:
    """Semantic version: 1.2.3"""

    def __init__(self, version_str: str):
        self._parts = tuple(int(p) for p in version_str.split("."))

    def __repr__(self) -> str:          # unambiguous; used in repr(), REPL
        return f"Version({'.'.join(str(p) for p in self._parts)!r})"

    def __str__(self) -> str:           # human-readable; used in print(), str()
        return ".".join(str(p) for p in self._parts)

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, Version):
            return NotImplemented
        return self._parts == other._parts

    def __lt__(self, other: "Version") -> bool:
        if not isinstance(other, Version):
            return NotImplemented
        return self._parts < other._parts

    def __hash__(self) -> int:          # required when defining __eq__
        return hash(self._parts)

    def __len__(self) -> int:           # len(version) = number of parts
        return len(self._parts)

    def __contains__(self, part: int) -> bool:
        return part in self._parts


def test_dunder_repr_and_str():
    v = Version("1.2.3")
    assert repr(v) == "Version('1.2.3')"
    assert str(v) == "1.2.3"


def test_dunder_ordering():
    v1 = Version("1.0.0")
    v2 = Version("1.2.0")
    v3 = Version("2.0.0")
    assert v1 < v2 < v3
    assert v3 > v1
    assert sorted([v3, v1, v2]) == [v1, v2, v3]    # uses __lt__
    assert max(v1, v2, v3) == v3


def test_dunder_hash_in_set():
    versions = {Version("1.0.0"), Version("1.0.0"), Version("2.0.0")}
    assert len(versions) == 2   # deduped via __hash__ + __eq__


def test_dunder_len_and_contains():
    v = Version("3.11.4")
    assert len(v) == 3
    assert 11 in v
    assert 99 not in v


# ===========================================================================
# NewType  —  distinct types that share the same runtime representation
# ===========================================================================
# Prevents mixing up e.g. user IDs and product IDs at the type-checker level.
# Zero cost at runtime — NewType is just the identity function.

UserId    = NewType("UserId",    int)
ProductId = NewType("ProductId", int)
Email     = NewType("Email",     str)


def get_user(uid: UserId) -> dict:
    return {"id": uid, "name": "Alice"}


def test_newtype_runtime_is_just_base_type():
    uid = UserId(42)
    assert isinstance(uid, int)     # same runtime type
    assert uid + 1 == 43            # arithmetic works
    # Type checker would flag: get_user(ProductId(42)) — wrong type
    # But at runtime there's no error; NewType is purely static


def test_newtype_prevents_accidental_mixing():
    # This test documents intent; violations caught by mypy/pyright, not pytest.
    uid: UserId    = UserId(1)
    pid: ProductId = ProductId(1)
    assert uid == pid    # same int value
    # mypy would flag: get_user(pid)  # Argument of type "ProductId" not assignable to "UserId"


# ===========================================================================
# Literal  —  restrict a value to specific constants
# ===========================================================================

Direction  = Literal["north", "south", "east", "west"]
LogLevel   = Literal["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]
HttpMethod = Literal["GET", "POST", "PUT", "DELETE", "PATCH"]


def move(steps: int, direction: Direction) -> tuple[int, int]:
    deltas = {"north": (0, 1), "south": (0, -1), "east": (1, 0), "west": (-1, 0)}
    dx, dy = deltas[direction]
    return (dx * steps, dy * steps)


def test_literal_values():
    assert move(3, "north") == (0, 3)
    assert move(2, "east")  == (2, 0)


def test_literal_with_overload():
    """Literal enables precise overload dispatch in type stubs."""
    def parse(raw: str, as_type: Literal["int", "float", "str"]):
        return {"int": int, "float": float, "str": str}[as_type](raw)

    assert parse("42",   "int")   == 42
    assert parse("3.14", "float") == 3.14
    assert parse("hi",   "str")   == "hi"


# ===========================================================================
# TypeVar with bound  —  generic functions constrained to a type family
# ===========================================================================

class Comparable(Protocol):
    def __lt__(self, other: "Comparable") -> bool: ...

C = TypeVar("C", bound=Comparable)


def clamp(value: C, lo: C, hi: C) -> C:
    """Constrain value to [lo, hi]. Works for any Comparable type."""
    if value < lo:
        return lo
    if hi < value:
        return hi
    return value


def test_typevar_bound_int():
    assert clamp(5, 1, 10) == 5
    assert clamp(-3, 1, 10) == 1
    assert clamp(99, 1, 10) == 10


def test_typevar_bound_float_and_str():
    assert clamp(3.14, 0.0, 2.0) == 2.0
    assert clamp("m", "a", "z") == "m"
    assert clamp("!", "a", "z") == "a"


# ===========================================================================
# cached_property  —  lazy, computed-once attribute
# ===========================================================================

class DataSet:
    def __init__(self, values: list[float]):
        self.values = values
        self._compute_count = 0

    @cached_property
    def stats(self) -> dict:
        """Computed once on first access, then cached on the instance."""
        self._compute_count += 1
        n = len(self.values)
        mean = sum(self.values) / n
        variance = sum((x - mean) ** 2 for x in self.values) / n
        return {"n": n, "mean": mean, "std": math.sqrt(variance)}


def test_cached_property_computed_once():
    ds = DataSet([2.0, 4.0, 4.0, 4.0, 5.0, 5.0, 7.0, 9.0])
    _ = ds.stats
    _ = ds.stats   # second access — should NOT recompute
    assert ds._compute_count == 1
    assert round(ds.stats["mean"], 4) == 5.0
    assert round(ds.stats["std"],  4) == 2.0


def test_cached_property_stored_in_instance_dict():
    ds = DataSet([1.0, 2.0, 3.0])
    assert "stats" not in ds.__dict__   # not yet computed
    _ = ds.stats
    assert "stats" in ds.__dict__       # now cached directly on instance


def test_cached_property_invalidation():
    # To invalidate: delete from __dict__
    ds = DataSet([1.0, 2.0, 3.0])
    _ = ds.stats
    del ds.__dict__["stats"]
    ds.values.append(4.0)
    new_stats = ds.stats
    assert new_stats["n"] == 4
    assert ds._compute_count == 2    # recomputed once after invalidation

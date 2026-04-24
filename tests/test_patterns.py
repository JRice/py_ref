"""
OO design patterns — idiomatic Python edition.

Patterns included (all common in real Python codebases):
  Observer      — event bus with typed listeners
  Command       — undo/redo with callable commands
  State         — enum-driven state machine
  Strategy      — swappable algorithms via callables / Protocol
  Iterator      — custom __iter__ + generator shortcut
  Template Method — ABC with abstract steps
  Visitor       — functools.singledispatch (the Pythonic Visitor)
  Mixin         — multiple inheritance for composable behaviour
  Registry      — decorator-based plugin registry
  Descriptor    — __get__/__set__ for ORM-style field validation

Patterns intentionally omitted:
  Memento       — just use pickle / dataclasses.asdict
  Interpreter   — use the `ast` stdlib module or a parsing library
  Mediator      — rare; usually an event bus (Observer) suffices
  Chain of Responsibility — Python middleware pipelines cover this
"""

from __future__ import annotations

import abc
import functools
from collections import defaultdict
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Any, Callable, ClassVar, Protocol, TypeVar


# ===========================================================================
# Observer — event bus
# ===========================================================================
# Java needs a full listener interface + registration boilerplate.
# Python: a dict of {event_name: [callables]}.

class EventBus:
    def __init__(self):
        self._listeners: defaultdict[str, list[Callable]] = defaultdict(list)

    def subscribe(self, event: str, handler: Callable) -> None:
        self._listeners[event].append(handler)

    def unsubscribe(self, event: str, handler: Callable) -> None:
        self._listeners[event].remove(handler)

    def publish(self, event: str, **payload) -> None:
        for handler in self._listeners[event]:
            handler(**payload)

    def on(self, event: str):
        """Decorator sugar: @bus.on('user.signup')"""
        def decorator(func):
            self.subscribe(event, func)
            return func
        return decorator


def test_observer_basic():
    bus = EventBus()
    log = []

    @bus.on("order.placed")
    def send_confirmation(order_id, amount):
        log.append(f"email:{order_id}")

    @bus.on("order.placed")
    def update_inventory(order_id, amount):
        log.append(f"inventory:{order_id}")

    bus.publish("order.placed", order_id="ORD-1", amount=99.99)

    assert log == ["email:ORD-1", "inventory:ORD-1"]


def test_observer_unsubscribe():
    bus = EventBus()
    calls = []

    handler = lambda **kwargs: calls.append(kwargs)
    bus.subscribe("tick", handler)
    bus.publish("tick", n=1)
    bus.unsubscribe("tick", handler)
    bus.publish("tick", n=2)

    assert len(calls) == 1
    assert calls[0]["n"] == 1


def test_observer_no_listeners_is_silent():
    bus = EventBus()
    bus.publish("ghost.event", x=1)   # no subscribers — should not raise


# ===========================================================================
# Command — undo/redo stack
# ===========================================================================
# In Python a "command" is just a callable + optional undo callable.
# No need for a Command base class.

@dataclass
class CommandHistory:
    _done: list[tuple[Callable, Callable]] = field(default_factory=list)

    def execute(self, do: Callable, undo: Callable) -> None:
        do()
        self._done.append((do, undo))

    def undo(self) -> None:
        if self._done:
            _, undo = self._done.pop()
            undo()


def test_command_text_editor():
    text = ["Hello"]

    history = CommandHistory()

    # Append " World"
    history.execute(
        do=lambda: text.append(" World"),
        undo=lambda: text.pop(),
    )
    assert text == ["Hello", " World"]

    # Append "!"
    history.execute(
        do=lambda: text.append("!"),
        undo=lambda: text.pop(),
    )
    assert text == ["Hello", " World", "!"]

    history.undo()
    assert text == ["Hello", " World"]

    history.undo()
    assert text == ["Hello"]

    history.undo()   # nothing to undo — safe no-op
    assert text == ["Hello"]


# ===========================================================================
# State — enum-driven state machine
# ===========================================================================
# Python: enum + dict mapping (state, event) -> (new_state, action).
# Much lighter than the classic "one class per state" GoF approach.

class OrderState(Enum):
    PENDING    = auto()
    PROCESSING = auto()
    SHIPPED    = auto()
    DELIVERED  = auto()
    CANCELLED  = auto()


@dataclass
class Order:
    id: str
    state: OrderState = OrderState.PENDING
    _history: list[str] = field(default_factory=list, repr=False)

    # (current_state, event) -> next_state
    _TRANSITIONS: ClassVar[dict[tuple[OrderState, str], OrderState]] = {
        (OrderState.PENDING,    "confirm"):  OrderState.PROCESSING,
        (OrderState.PROCESSING, "ship"):     OrderState.SHIPPED,
        (OrderState.SHIPPED,    "deliver"):  OrderState.DELIVERED,
        (OrderState.PENDING,    "cancel"):   OrderState.CANCELLED,
        (OrderState.PROCESSING, "cancel"):   OrderState.CANCELLED,
    }

    def trigger(self, event: str) -> None:
        key = (self.state, event)
        if key not in self._TRANSITIONS:
            raise ValueError(f"Cannot '{event}' from state {self.state.name}")
        self.state = self._TRANSITIONS[key]
        self._history.append(f"{event} -> {self.state.name}")


def test_state_happy_path():
    order = Order("ORD-42")
    order.trigger("confirm")
    order.trigger("ship")
    order.trigger("deliver")
    assert order.state == OrderState.DELIVERED
    assert order._history == [
        "confirm -> PROCESSING",
        "ship -> SHIPPED",
        "deliver -> DELIVERED",
    ]


def test_state_cancellation():
    order = Order("ORD-7")
    order.trigger("confirm")
    order.trigger("cancel")
    assert order.state == OrderState.CANCELLED


def test_state_invalid_transition():
    import pytest
    order = Order("ORD-99")
    order.trigger("confirm")
    order.trigger("ship")
    with pytest.raises(ValueError, match="Cannot 'confirm'"):
        order.trigger("confirm")   # can't re-confirm a shipped order


# ===========================================================================
# Strategy — swappable algorithms
# ===========================================================================
# In Java: Strategy interface + ConcreteStrategy classes.
# In Python: just pass a callable (or a Protocol for type-checker support).

class DiscountStrategy(Protocol):
    def __call__(self, price: float, quantity: int) -> float: ...


def no_discount(price: float, quantity: int) -> float:
    return price * quantity

def bulk_discount(price: float, quantity: int) -> float:
    factor = 0.9 if quantity >= 10 else 1.0
    return price * quantity * factor

def vip_discount(price: float, quantity: int) -> float:
    return price * quantity * 0.8


@dataclass
class PriceCalculator:
    strategy: DiscountStrategy = no_discount

    def total(self, price: float, quantity: int) -> float:
        return round(self.strategy(price, quantity), 2)


def test_strategy_swappable():
    calc = PriceCalculator()
    assert calc.total(10.0, 5) == 50.0

    calc.strategy = bulk_discount
    assert calc.total(10.0, 10) == 90.0   # 10% off

    calc.strategy = vip_discount
    assert calc.total(10.0, 5) == 40.0    # 20% off


def test_strategy_lambda_inline():
    # Strategies don't need names — any callable works
    calc = PriceCalculator(strategy=lambda price, qty: price * qty * 0.5)
    assert calc.total(20.0, 3) == 30.0


# ===========================================================================
# Iterator — custom __iter__ / __next__
# ===========================================================================
# Python's iterator protocol is built-in; generators make it even simpler.

class InOrderTraversal:
    """Iterate a binary search tree in sorted order."""

    @dataclass
    class Node:
        val: int
        left:  Any = None
        right: Any = None

    def __init__(self, root):
        self._root = root

    def __iter__(self):
        # Generator replaces a full iterator class with __next__ + stack state
        yield from self._inorder(self._root)

    def _inorder(self, node):
        if node is None:
            return
        yield from self._inorder(node.left)
        yield node.val
        yield from self._inorder(node.right)


def test_iterator_bst():
    N = InOrderTraversal.Node
    #       4
    #      / \
    #     2   6
    #    / \ / \
    #   1  3 5  7
    root = N(4, N(2, N(1), N(3)), N(6, N(5), N(7)))
    assert list(InOrderTraversal(root)) == [1, 2, 3, 4, 5, 6, 7]


class CountDown:
    """Classic __iter__ / __next__ pair — useful to understand the protocol."""
    def __init__(self, start: int):
        self._n = start

    def __iter__(self):
        return self

    def __next__(self):
        if self._n <= 0:
            raise StopIteration
        self._n -= 1
        return self._n + 1


def test_iterator_countdown():
    assert list(CountDown(3)) == [3, 2, 1]
    assert sum(CountDown(5)) == 15


# ===========================================================================
# Template Method — ABC with abstract steps
# ===========================================================================
# Python uses abc.ABC; prefer composition when inheritance would be deep.

class DataPipeline(abc.ABC):
    """ETL skeleton: subclasses fill in the blanks."""

    def run(self) -> list:
        raw  = self.extract()
        data = self.transform(raw)
        return self.load(data)

    @abc.abstractmethod
    def extract(self) -> list: ...

    @abc.abstractmethod
    def transform(self, data: list) -> list: ...

    def load(self, data: list) -> list:
        # Default load: return as-is (subclasses may override)
        return data


class UpperCasePipeline(DataPipeline):
    def extract(self):
        return ["hello", "world", "python"]

    def transform(self, data):
        return [text.upper() for text in data]


class EvenNumberPipeline(DataPipeline):
    def extract(self):
        return list(range(10))

    def transform(self, data):
        return [number for number in data if number % 2 == 0]

    def load(self, data):
        return {"values": data, "count": len(data)}  # override load


def test_template_method_uppercase():
    assert UpperCasePipeline().run() == ["HELLO", "WORLD", "PYTHON"]


def test_template_method_even_numbers():
    result = EvenNumberPipeline().run()
    assert result == {"values": [0, 2, 4, 6, 8], "count": 5}


def test_template_method_abstract_enforcement():
    import pytest
    with pytest.raises(TypeError):
        DataPipeline()   # can't instantiate — abstract methods not implemented


# ===========================================================================
# Visitor — functools.singledispatch
# ===========================================================================
# GoF Visitor requires accept() on every node class — tedious.
# Python answer: singledispatch dispatches on the *first argument's type*,
# keeping all visitor logic in one place with no changes to node classes.

@dataclass
class Num:
    value: float

@dataclass
class Add:
    left: Any
    right: Any

@dataclass
class Mul:
    left: Any
    right: Any

@dataclass
class Neg:
    operand: Any


@functools.singledispatch
def evaluate(node) -> float:
    raise TypeError(f"Unknown node type: {type(node)}")

@evaluate.register
def _(node: Num) -> float:
    return node.value

@evaluate.register
def _(node: Add) -> float:
    return evaluate(node.left) + evaluate(node.right)

@evaluate.register
def _(node: Mul) -> float:
    return evaluate(node.left) * evaluate(node.right)

@evaluate.register
def _(node: Neg) -> float:
    return -evaluate(node.operand)


def test_visitor_singledispatch_expression():
    # (3 + 4) * -(2)  =  7 * -2  =  -14
    expr = Mul(Add(Num(3), Num(4)), Neg(Num(2)))
    assert evaluate(expr) == -14.0


def test_visitor_singledispatch_unknown_type():
    import pytest
    with pytest.raises(TypeError, match="Unknown node type"):
        evaluate("not a node")


# Second visitor over the same nodes — no changes to node classes needed
@functools.singledispatch
def to_str(node) -> str:
    raise TypeError

@to_str.register
def _(node: Num) -> str:
    return str(node.value)

@to_str.register
def _(node: Add) -> str:
    return f"({to_str(node.left)} + {to_str(node.right)})"

@to_str.register
def _(node: Mul) -> str:
    return f"({to_str(node.left)} * {to_str(node.right)})"

@to_str.register
def _(node: Neg) -> str:
    return f"(-{to_str(node.operand)})"


def test_visitor_second_visitor_same_nodes():
    expr = Mul(Add(Num(3), Num(4)), Neg(Num(2)))
    assert to_str(expr) == "((3 + 4) * (-2))"


# ===========================================================================
# Mixin — multiple inheritance for composable behaviour
# ===========================================================================
# Python-specific; no GoF equivalent.
# Rule: mixins are narrow, stateless (no __init__), and named *Mixin.

class TimestampMixin:
    """Adds created_at / updated_at tracking to any dataclass-like object."""
    from datetime import datetime, timezone

    def touch(self):
        self.updated_at = self.datetime.now(self.timezone.utc)

    def age_seconds(self) -> float:
        return (self.datetime.now(self.timezone.utc) - self.created_at).total_seconds()


class JsonMixin:
    """Adds .to_json() / .from_json() to any class with a __dict__."""
    import json as _json

    def to_json(self) -> str:
        return self._json.dumps(
            {k: v for k, v in self.__dict__.items() if not k.startswith("_")},
            default=str,
        )

    @classmethod
    def from_json(cls, s: str):
        return cls(**cls._json.loads(s))


class ReprMixin:
    def __repr__(self):
        attrs = ", ".join(f"{k}={v!r}" for k, v in self.__dict__.items())
        return f"{type(self).__name__}({attrs})"


@dataclass
class Product(JsonMixin, ReprMixin):
    name: str
    price: float
    category: str


def test_mixin_json_round_trip():
    p = Product("Widget", 9.99, "hardware")
    serialized = p.to_json()
    restored   = Product.from_json(serialized)
    assert restored.name     == "Widget"
    assert restored.price    == 9.99
    assert restored.category == "hardware"


def test_mixin_repr():
    p = Product("Gadget", 24.99, "electronics")
    assert "Gadget" in repr(p)
    assert "24.99"  in repr(p)


def test_mixin_multiple_independent():
    # Both mixins can be used without knowing about each other
    class Tagged(JsonMixin, ReprMixin):
        def __init__(self, tag, count):
            self.tag = tag
            self.count = count

    t = Tagged("python", 42)
    assert '"python"' in t.to_json()
    assert "Tagged("  in repr(t)


# ===========================================================================
# Registry — decorator-based plugin system
# ===========================================================================
# Python-specific; replaces service locators, factories, and IoC containers.
# Used in: Flask routes, Click commands, Celery tasks, pytest plugins, etc.

class FormatRegistry:
    """Register serialisers by name; look them up at runtime."""

    def __init__(self):
        self._registry: dict[str, Callable] = {}

    def register(self, name: str):
        def decorator(func: Callable) -> Callable:
            self._registry[name] = func
            return func
        return decorator

    def serialize(self, name: str, data: Any) -> str:
        if name not in self._registry:
            raise KeyError(f"No serialiser registered for '{name}'")
        return self._registry[name](data)

    @property
    def formats(self) -> list[str]:
        return list(self._registry)


formats = FormatRegistry()

@formats.register("json")
def _json_serializer(data) -> str:
    import json
    return json.dumps(data)

@formats.register("csv_row")
def _csv_serializer(data: dict) -> str:
    return ",".join(str(value) for value in data.values())

@formats.register("pipe")
def _pipe_serializer(data: dict) -> str:
    return "|".join(str(value) for value in data.values())


def test_registry_dispatch():
    record = {"name": "Alice", "score": 42}
    assert formats.serialize("json",    record) == '{"name": "Alice", "score": 42}'
    assert formats.serialize("csv_row", record) == "Alice,42"
    assert formats.serialize("pipe",    record) == "Alice|42"


def test_registry_unknown_format():
    import pytest
    with pytest.raises(KeyError, match="xml"):
        formats.serialize("xml", {})


def test_registry_new_format_at_runtime():
    @formats.register("keys_only")
    def _(data: dict) -> str:
        return ",".join(data.keys())

    assert formats.serialize("keys_only", {"a": 1, "b": 2}) == "a,b"
    assert "keys_only" in formats.formats


# ===========================================================================
# Descriptor — __get__ / __set__ for field-level validation
# ===========================================================================
# Descriptors power Python's property, classmethod, staticmethod, and ORMs
# like SQLAlchemy and Django models.  Write one when a @property would be
# duplicated across many classes.

class Validated:
    """
    Generic descriptor: validates a value with a user-supplied predicate
    and stores it per-instance in the instance's __dict__.
    """
    def __set_name__(self, owner, name):
        self._name = name

    def __init__(self, predicate: Callable[[Any], bool], error_msg: str):
        self._predicate = predicate
        self._error_msg = error_msg

    def __get__(self, obj, objtype=None):
        if obj is None:
            return self           # class-level access returns the descriptor
        return obj.__dict__.get(self._name)

    def __set__(self, obj, value):
        if not self._predicate(value):
            raise ValueError(f"{self._name!r}: {self._error_msg} (got {value!r})")
        obj.__dict__[self._name] = value


class BankAccount:
    owner   = Validated(lambda value: isinstance(value, str) and value, "must be a non-empty string")
    balance = Validated(lambda value: isinstance(value, (int, float)) and value >= 0, "must be >= 0")

    def __init__(self, owner: str, balance: float = 0.0):
        self.owner   = owner
        self.balance = balance

    def deposit(self, amount: float):
        self.balance += amount   # descriptor validates on each assignment

    def withdraw(self, amount: float):
        self.balance = self.balance - amount   # raises if it goes negative


def test_descriptor_valid_usage():
    acc = BankAccount("Alice", 100.0)
    acc.deposit(50.0)
    assert acc.balance == 150.0
    acc.withdraw(30.0)
    assert acc.balance == 120.0


def test_descriptor_rejects_negative_balance():
    import pytest
    acc = BankAccount("Bob", 10.0)
    with pytest.raises(ValueError, match="balance"):
        acc.withdraw(50.0)   # would set balance to -40


def test_descriptor_rejects_empty_owner():
    import pytest
    with pytest.raises(ValueError, match="owner"):
        BankAccount("")


def test_descriptor_class_access_returns_descriptor():
    assert isinstance(BankAccount.balance, Validated)

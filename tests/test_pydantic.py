"""
Pydantic v2 — the de-facto standard for data validation in Python.

Covers: BaseModel, Field validators, model_validator, nested models,
serialization, ValidationError, discriminated unions, BaseSettings.
"""

from __future__ import annotations

import json
import os
from enum import StrEnum
from typing import Annotated, Any, Literal

import pytest
from pydantic import (
    BaseModel,
    EmailStr,
    Field,
    HttpUrl,
    ValidationError,
    computed_field,
    field_validator,
    model_validator,
)
from pydantic import field_serializer
from pydantic_settings import BaseSettings, SettingsConfigDict


# ===========================================================================
# Basic model
# ===========================================================================

class User(BaseModel):
    id:    int
    name:  str = Field(min_length=1, max_length=50)
    email: EmailStr
    age:   int  = Field(ge=0, le=150)   # ge=greater-or-equal, le=less-or-equal
    score: float = 0.0                  # optional with default


def test_pydantic_valid_model():
    u = User(id=1, name="Alice", email="alice@example.com", age=30)
    assert u.name == "Alice"
    assert u.score == 0.0      # default applied
    assert u.id == 1


def test_pydantic_coercion():
    # Pydantic coerces types where unambiguous
    u = User(id="42", name="Bob", email="bob@example.com", age="25")  # type: ignore
    assert u.id == 42      # str → int
    assert u.age == 25


def test_pydantic_validation_error_details():
    with pytest.raises(ValidationError) as exc_info:
        User(id=1, name="", email="not-an-email", age=200)

    errors = exc_info.value.errors()
    fields_with_errors = {error["loc"][0] for error in errors}
    assert "name"  in fields_with_errors   # too short
    assert "email" in fields_with_errors   # invalid format
    assert "age"   in fields_with_errors   # > 150


# ===========================================================================
# Field validators  and  model validators
# ===========================================================================

class Product(BaseModel):
    name:  str
    price: float
    sku:   str

    @field_validator("price")
    @classmethod
    def price_must_be_positive(cls, v: float) -> float:
        if v <= 0:
            raise ValueError("price must be positive")
        return round(v, 2)

    @field_validator("sku")
    @classmethod
    def sku_must_be_uppercase(cls, v: str) -> str:
        return v.upper()   # coerce rather than reject — validators can transform

    @model_validator(mode="after")
    def name_cannot_contain_sku(self) -> "Product":
        if self.sku.lower() in self.name.lower():
            raise ValueError("product name must not contain the SKU")
        return self


def test_field_validator_coercion():
    p = Product(name="Widget", price=9.999, sku="wgt-01")
    assert p.price == 10.0      # rounded to 2dp
    assert p.sku   == "WGT-01"  # uppercased


def test_field_validator_rejection():
    with pytest.raises(ValidationError, match="price must be positive"):
        Product(name="Freebie", price=0, sku="FREE")


def test_model_validator():
    with pytest.raises(ValidationError, match="must not contain the SKU"):
        Product(name="Widget WGT-01 Pro", price=9.99, sku="wgt-01")


# ===========================================================================
# Nested models and Optional fields
# ===========================================================================

class Address(BaseModel):
    street: str
    city:   str
    country: str = "US"


class Order(BaseModel):
    order_id:    str
    customer:    User
    ship_to:     Address
    items:       list[str]
    notes:       str | None = None   # Optional — None by default


def test_nested_model():
    order = Order(
        order_id="ORD-1",
        customer={"id": 1, "name": "Alice", "email": "a@b.com", "age": 30},
        ship_to={"street": "1 Main St", "city": "Boston"},
        items=["Widget", "Gadget"],
    )
    assert order.customer.name == "Alice"
    assert order.ship_to.country == "US"    # default applied in nested model
    assert order.notes is None


def test_nested_model_validation_propagates():
    with pytest.raises(ValidationError) as exc_info:
        Order(
            order_id="ORD-2",
            customer={"id": 1, "name": "", "email": "bad", "age": 30},
            ship_to={"street": "x", "city": "y"},
            items=[],
        )
    fields = {str(error["loc"]) for error in exc_info.value.errors()}
    # Errors are on nested paths like ("customer", "name")
    assert any("customer" in field for field in fields)


# ===========================================================================
# Serialization / deserialization
# ===========================================================================

class BlogPost(BaseModel):
    title:   str
    body:    str
    tags:    list[str] = []
    published: bool = False


def test_model_dump():
    post = BlogPost(title="Hello", body="World", tags=["python"])
    d = post.model_dump()
    assert d == {"title": "Hello", "body": "World", "tags": ["python"], "published": False}


def test_model_dump_exclude_and_include():
    post = BlogPost(title="Hello", body="Secret", tags=[])
    d = post.model_dump(exclude={"body"})
    assert "body" not in d
    assert "title" in d


def test_model_dump_json_and_parse():
    post = BlogPost(title="Round-trip", body="Test", tags=["a", "b"])
    raw  = post.model_dump_json()
    assert isinstance(raw, str)
    restored = BlogPost.model_validate_json(raw)
    assert restored == post


def test_model_validate_from_dict():
    raw = {"title": "From Dict", "body": "...", "tags": [], "published": True}
    post = BlogPost.model_validate(raw)
    assert post.published is True


# ===========================================================================
# computed_field  —  derived values included in serialization
# ===========================================================================

class Rectangle(BaseModel):
    width:  float = Field(gt=0)
    height: float = Field(gt=0)

    @computed_field
    @property
    def area(self) -> float:
        return self.width * self.height

    @computed_field
    @property
    def perimeter(self) -> float:
        return 2 * (self.width + self.height)


def test_computed_field():
    r = Rectangle(width=3.0, height=4.0)
    assert r.area      == 12.0
    assert r.perimeter == 14.0

    d = r.model_dump()
    assert d["area"]      == 12.0   # included in serialization
    assert d["perimeter"] == 14.0


# ===========================================================================
# Discriminated union  —  polymorphic shapes with a "type" tag
# ===========================================================================

class Cat(BaseModel):
    kind:   Literal["cat"] = "cat"
    name:   str
    indoor: bool = True


class Dog(BaseModel):
    kind:    Literal["dog"] = "dog"
    name:    str
    breed:   str


class Parrot(BaseModel):
    kind:    Literal["parrot"] = "parrot"
    name:    str
    can_talk: bool = False


Pet = Annotated[Cat | Dog | Parrot, Field(discriminator="kind")]


class Household(BaseModel):
    address: str
    pets:    list[Pet]


def test_discriminated_union():
    data = {
        "address": "1 Main St",
        "pets": [
            {"kind": "cat",    "name": "Whiskers"},
            {"kind": "dog",    "name": "Rex", "breed": "Labrador"},
            {"kind": "parrot", "name": "Polly", "can_talk": True},
        ],
    }
    h = Household.model_validate(data)
    assert isinstance(h.pets[0], Cat)
    assert isinstance(h.pets[1], Dog)
    assert h.pets[2].can_talk is True   # type: ignore[union-attr]


def test_discriminated_union_invalid_kind():
    with pytest.raises(ValidationError):
        Household.model_validate({
            "address": "x",
            "pets": [{"kind": "fish", "name": "Nemo"}],
        })


# ===========================================================================
# Custom field serializer
# ===========================================================================

from datetime import datetime, timezone

class Event(BaseModel):
    name:       str
    occurs_at:  datetime

    @field_serializer("occurs_at")
    def serialize_dt(self, dt: datetime) -> str:
        return dt.isoformat()


def test_field_serializer():
    evt = Event(name="Deploy", occurs_at=datetime(2025, 6, 1, 12, 0, tzinfo=timezone.utc))
    d = evt.model_dump()
    assert d["occurs_at"] == "2025-06-01T12:00:00+00:00"


# ===========================================================================
# BaseSettings  —  config from env vars / .env files
# ===========================================================================

class AppSettings(BaseSettings):
    model_config = SettingsConfigDict(
        env_prefix="APP_",   # reads APP_HOST, APP_PORT, etc.
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )
    host:     str   = "localhost"
    port:     int   = 8080
    debug:    bool  = False
    api_key:  str   = "dev-key"


def test_settings_defaults():
    s = AppSettings()
    assert s.host  == "localhost"
    assert s.port  == 8080
    assert s.debug is False


def test_settings_env_override(monkeypatch):
    monkeypatch.setenv("APP_HOST",  "prod.server.com")
    monkeypatch.setenv("APP_PORT",  "443")
    monkeypatch.setenv("APP_DEBUG", "true")

    s = AppSettings()
    assert s.host  == "prod.server.com"
    assert s.port  == 443
    assert s.debug is True


def test_settings_validation_from_env(monkeypatch):
    monkeypatch.setenv("APP_PORT", "not-a-number")
    with pytest.raises(ValidationError):
        AppSettings()

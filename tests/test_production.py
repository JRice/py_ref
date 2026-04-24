"""
Common production tasks.

Covers: argparse CLI, Typer CLI, python-dotenv env loading,
FastAPI with TestClient (CRUD + Pydantic), SQLAlchemy 2.0 ORM.

Alembic migrations and background workers (Celery/ARQ) are excluded here
because they require external processes; the patterns are well-documented
at alembic.sqlalchemy.org and arq-docs.helpmanual.io respectively.
"""

from __future__ import annotations

import argparse
import os
from io import StringIO
from typing import Annotated

import pytest
import typer
from typer.testing import CliRunner

from fastapi import Depends, FastAPI, HTTPException, Query
from fastapi.testclient import TestClient
from pydantic import BaseModel, Field
from sqlalchemy import ForeignKey, String, create_engine, select
from sqlalchemy.orm import (
    DeclarativeBase,
    Mapped,
    Session,
    mapped_column,
    relationship,
)


# ===========================================================================
# argparse  —  stdlib CLI builder
# ===========================================================================

def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="myapp",
        description="A sample CLI",
    )
    p.add_argument("input",               help="input file path")
    p.add_argument("-o", "--output",      default="out.txt", help="output path")
    p.add_argument("-n", "--count",       type=int, default=10)
    p.add_argument("-v", "--verbose",     action="store_true")
    p.add_argument("--format",            choices=["json", "csv", "tsv"], default="json")

    sub = p.add_subparsers(dest="command")
    run_cmd = sub.add_parser("run",  help="execute a job")
    run_cmd.add_argument("--dry-run", action="store_true")

    return p


def test_argparse_defaults():
    p = build_parser()
    args = p.parse_args(["data.csv"])
    assert args.input   == "data.csv"
    assert args.output  == "out.txt"
    assert args.count   == 10
    assert args.verbose is False
    assert args.format  == "json"


def test_argparse_flags():
    p = build_parser()
    args = p.parse_args(["-v", "--count", "25", "--format", "csv", "data.csv"])
    assert args.verbose is True
    assert args.count   == 25
    assert args.format  == "csv"


def test_argparse_subcommand():
    p = build_parser()
    args = p.parse_args(["data.csv", "run", "--dry-run"])
    assert args.command  == "run"
    assert args.dry_run  is True


def test_argparse_invalid_choice():
    p = build_parser()
    with pytest.raises(SystemExit):
        p.parse_args(["data.csv", "--format", "xml"])   # xml not in choices


# ===========================================================================
# Typer  —  modern Click-based CLI with type-annotation magic
# ===========================================================================

app = typer.Typer()


@app.command()
def greet(
    name: str,
    times: Annotated[int,  typer.Option("--times", "-n", help="Repetitions")] = 1,
    shout: Annotated[bool, typer.Option("--shout/--no-shout")]                 = False,
):
    message = f"Hello, {name}!"
    if shout:
        message = message.upper()
    for _ in range(times):
        typer.echo(message)


@app.command()
def add(a: float, b: float):
    """Add two numbers."""
    typer.echo(str(a + b))


runner = CliRunner()


def test_typer_basic():
    result = runner.invoke(app, ["greet", "Alice"])
    assert result.exit_code == 0
    assert "Hello, Alice!" in result.output


def test_typer_options():
    result = runner.invoke(app, ["greet", "Bob", "--times", "3", "--shout"])
    assert result.exit_code == 0
    lines = [line for line in result.output.strip().splitlines() if line]
    assert len(lines) == 3
    assert all("BOB" in line for line in lines)


def test_typer_subcommand():
    result = runner.invoke(app, ["add", "2.5", "3.5"])
    assert result.exit_code == 0
    assert "6.0" in result.output


def test_typer_help():
    result = runner.invoke(app, ["--help"])
    assert result.exit_code == 0
    assert "greet" in result.output


# ===========================================================================
# python-dotenv  —  load .env into os.environ
# ===========================================================================

from dotenv import dotenv_values, load_dotenv


def test_dotenv_load_from_file(tmp_path, monkeypatch):
    env_file = tmp_path / ".env"
    env_file.write_text("DATABASE_URL=sqlite:///dev.db\nDEBUG=true\nPORT=5432\n")

    # monkeypatch to isolate from real environment
    monkeypatch.delenv("DATABASE_URL", raising=False)

    load_dotenv(env_file, override=True)
    assert os.getenv("DATABASE_URL") == "sqlite:///dev.db"
    assert os.getenv("DEBUG")        == "true"
    assert os.getenv("PORT")         == "5432"


def test_dotenv_values_without_modifying_environment(tmp_path):
    env_file = tmp_path / ".env"
    env_file.write_text("SECRET=s3cr3t\nAPP_ENV=test\n")

    values = dotenv_values(env_file)   # dict, does NOT touch os.environ
    assert values["SECRET"] == "s3cr3t"
    assert os.getenv("SECRET") is None   # not set in real env


# ===========================================================================
# FastAPI  —  async web framework; test with TestClient (no server needed)
# ===========================================================================

# ---- Pydantic schemas -----

class ItemCreate(BaseModel):
    name:  str = Field(min_length=1)
    price: float = Field(gt=0)


class ItemResponse(ItemCreate):
    id: int


# ---- In-memory store ------

_items_db: dict[int, dict] = {}
_next_id = 0

def reset_store():
    global _next_id
    _items_db.clear()
    _next_id = 0

def get_db():
    """Dependency — in production this would yield a real DB session."""
    yield _items_db


# ---- App ------

api = FastAPI(title="Items API")


@api.get("/items", response_model=list[ItemResponse])
def list_items(
    db=Depends(get_db),
    skip: int = Query(0, ge=0),
    limit: int = Query(10, ge=1, le=100),
):
    items = list(db.values())[skip: skip + limit]
    return items


@api.get("/items/{item_id}", response_model=ItemResponse)
def get_item(item_id: int, db=Depends(get_db)):
    if item_id not in db:
        raise HTTPException(404, detail="item not found")
    return db[item_id]


@api.post("/items", response_model=ItemResponse, status_code=201)
def create_item(body: ItemCreate, db=Depends(get_db)):
    global _next_id
    _next_id += 1
    record = {"id": _next_id, **body.model_dump()}
    db[_next_id] = record
    return record


@api.put("/items/{item_id}", response_model=ItemResponse)
def update_item(item_id: int, body: ItemCreate, db=Depends(get_db)):
    if item_id not in db:
        raise HTTPException(404, detail="item not found")
    db[item_id].update(body.model_dump())
    return db[item_id]


@api.delete("/items/{item_id}", status_code=204)
def delete_item(item_id: int, db=Depends(get_db)):
    if item_id not in db:
        raise HTTPException(404, detail="item not found")
    del db[item_id]


@pytest.fixture(autouse=True)
def _reset_items_store():
    reset_store()
    yield
    reset_store()


client = TestClient(api)


def test_fastapi_create_and_get():
    resp = client.post("/items", json={"name": "Widget", "price": 9.99})
    assert resp.status_code == 201
    data = resp.json()
    assert data["id"] == 1
    assert data["name"] == "Widget"

    resp2 = client.get(f"/items/{data['id']}")
    assert resp2.status_code == 200
    assert resp2.json()["price"] == 9.99


def test_fastapi_list_with_pagination():
    for i in range(5):
        client.post("/items", json={"name": f"item-{i}", "price": float(i + 1)})

    resp = client.get("/items?skip=2&limit=2")
    assert resp.status_code == 200
    items = resp.json()
    assert len(items) == 2
    assert items[0]["name"] == "item-2"


def test_fastapi_update():
    client.post("/items", json={"name": "Old", "price": 1.0})
    resp = client.put("/items/1", json={"name": "New", "price": 2.0})
    assert resp.status_code == 200
    assert resp.json()["name"] == "New"


def test_fastapi_delete():
    client.post("/items", json={"name": "Temp", "price": 5.0})
    resp = client.delete("/items/1")
    assert resp.status_code == 204
    assert client.get("/items/1").status_code == 404


def test_fastapi_404():
    resp = client.get("/items/999")
    assert resp.status_code == 404
    assert resp.json()["detail"] == "item not found"


def test_fastapi_validation_error():
    resp = client.post("/items", json={"name": "", "price": -1})
    assert resp.status_code == 422   # Unprocessable Entity


# ===========================================================================
# SQLAlchemy 2.0 ORM  —  modern mapped_column + Session style
# ===========================================================================

class Base(DeclarativeBase):
    pass


class Author(Base):
    __tablename__ = "authors"
    id:   Mapped[int] = mapped_column(primary_key=True)
    name: Mapped[str] = mapped_column(String(100))
    books: Mapped[list["Book"]] = relationship(back_populates="author")


class Book(Base):
    __tablename__ = "books"
    id:        Mapped[int]  = mapped_column(primary_key=True)
    title:     Mapped[str]  = mapped_column(String(200))
    year:      Mapped[int]
    author_id: Mapped[int]  = mapped_column(ForeignKey("authors.id"))
    author:    Mapped[Author] = relationship(back_populates="books")


@pytest.fixture
def db_session():
    engine = create_engine("sqlite:///:memory:", echo=False)
    Base.metadata.create_all(engine)
    with Session(engine) as session:
        yield session
    Base.metadata.drop_all(engine)


def test_sqlalchemy_insert_and_query(db_session):
    author = Author(name="Frank Herbert")
    db_session.add(author)
    db_session.flush()   # get the auto-generated id without committing

    book1 = Book(title="Dune",             year=1965, author=author)
    book2 = Book(title="Dune Messiah",     year=1969, author=author)
    db_session.add_all([book1, book2])
    db_session.commit()

    result = db_session.scalars(select(Book).order_by(Book.year)).all()
    assert len(result) == 2
    assert result[0].title == "Dune"
    assert result[0].author.name == "Frank Herbert"


def test_sqlalchemy_filter_and_update(db_session):
    author = Author(name="Ursula K. Le Guin")
    db_session.add(author)
    db_session.flush()
    db_session.add(Book(title="The Left Hand of Darkness", year=1969, author=author))
    db_session.commit()

    # Update
    book = db_session.scalars(select(Book).where(Book.year == 1969)).one()
    book.year = 1970   # attribute mutation is tracked by the Session
    db_session.commit()

    refreshed = db_session.get(Book, book.id)
    assert refreshed.year == 1970


def test_sqlalchemy_relationship_eager_load(db_session):
    from sqlalchemy.orm import selectinload

    a1 = Author(name="Isaac Asimov")
    db_session.add(a1)
    db_session.flush()
    db_session.add_all([
        Book(title="Foundation",         year=1951, author=a1),
        Book(title="Foundation and Empire", year=1952, author=a1),
    ])
    db_session.commit()

    authors = db_session.scalars(
        select(Author).options(selectinload(Author.books))
    ).all()
    assert len(authors[0].books) == 2


def test_sqlalchemy_delete(db_session):
    author = Author(name="Temp Author")
    db_session.add(author)
    db_session.commit()

    db_session.delete(author)
    db_session.commit()

    assert db_session.get(Author, author.id) is None

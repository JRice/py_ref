"""
Database CRUD in four flavors:
  SQL     — sqlite3 (stdlib)
  NoSQL   — TinyDB (document store, file-based)
  Graph   — NetworkX (in-memory)
  Vector  — NumPy ANN with cosine similarity (in-memory)
"""

import sqlite3
import pytest
import math
import numpy as np
from pathlib import Path


# ===========================================================================
# SQL — sqlite3
# ===========================================================================

@pytest.fixture
def db():
    """In-memory SQLite database, torn down after each test."""
    conn = sqlite3.connect(":memory:")
    conn.row_factory = sqlite3.Row    # rows behave like dicts
    conn.execute("""
        CREATE TABLE products (
            id      INTEGER PRIMARY KEY AUTOINCREMENT,
            name    TEXT NOT NULL,
            price   REAL NOT NULL,
            stock   INTEGER DEFAULT 0
        )
    """)
    conn.commit()
    yield conn
    conn.close()


def test_sql_insert_and_select(db):
    db.execute("INSERT INTO products (name, price, stock) VALUES (?, ?, ?)",
               ("Widget", 9.99, 100))
    db.execute("INSERT INTO products (name, price, stock) VALUES (?, ?, ?)",
               ("Gadget", 24.99, 50))
    db.commit()

    rows = db.execute("SELECT * FROM products ORDER BY price").fetchall()
    assert len(rows) == 2
    assert rows[0]["name"] == "Widget"
    assert rows[1]["price"] == 24.99


def test_sql_update_and_delete(db):
    db.execute("INSERT INTO products (name, price, stock) VALUES (?, ?, ?)",
               ("Widget", 9.99, 100))
    db.commit()

    db.execute("UPDATE products SET price = ? WHERE name = ?", (7.99, "Widget"))
    db.commit()
    row = db.execute("SELECT price FROM products WHERE name = ?", ("Widget",)).fetchone()
    assert row["price"] == 7.99

    db.execute("DELETE FROM products WHERE name = ?", ("Widget",))
    db.commit()
    assert db.execute("SELECT COUNT(*) FROM products").fetchone()[0] == 0


def test_sql_parameterized_query_prevents_injection(db):
    # Never use f-strings for SQL — always use ? placeholders
    malicious = "'; DROP TABLE products; --"
    db.execute("INSERT INTO products (name, price) VALUES (?, ?)", (malicious, 1.0))
    db.commit()
    count = db.execute("SELECT COUNT(*) FROM products").fetchone()[0]
    assert count == 1   # table still exists; input stored literally


def test_sql_transaction_rollback(db):
    try:
        with db:   # context manager auto-commits or rolls back
            db.execute("INSERT INTO products (name, price) VALUES (?, ?)", ("A", 1.0))
            raise ValueError("simulated error")
    except ValueError:
        pass

    count = db.execute("SELECT COUNT(*) FROM products").fetchone()[0]
    assert count == 0   # rolled back


# ===========================================================================
# NoSQL — TinyDB (JSON document store)
# ===========================================================================

from tinydb import TinyDB, Query
from tinydb.storages import MemoryStorage


@pytest.fixture
def tdb():
    db = TinyDB(storage=MemoryStorage)
    yield db
    db.close()


def test_nosql_insert_and_search(tdb):
    tdb.insert({"user": "alice", "tags": ["admin", "beta"], "score": 42})
    tdb.insert({"user": "bob",   "tags": ["beta"],          "score": 17})
    tdb.insert({"user": "carol", "tags": ["admin"],         "score": 95})

    Q = Query()
    admins = tdb.search(Q.tags.any(["admin"]))
    assert {r["user"] for r in admins} == {"alice", "carol"}

    high_scorers = tdb.search(Q.score > 40)
    assert len(high_scorers) == 2


def test_nosql_update_and_remove(tdb):
    tdb.insert({"user": "alice", "active": True})
    Q = Query()

    tdb.update({"active": False}, Q.user == "alice")
    assert tdb.get(Q.user == "alice")["active"] is False

    tdb.remove(Q.user == "alice")
    assert tdb.get(Q.user == "alice") is None


def test_nosql_upsert(tdb):
    Q = Query()
    tdb.upsert({"user": "alice", "score": 10}, Q.user == "alice")  # insert
    tdb.upsert({"user": "alice", "score": 20}, Q.user == "alice")  # update
    assert tdb.get(Q.user == "alice")["score"] == 20
    assert len(tdb.all()) == 1


# ===========================================================================
# Graph — NetworkX
# ===========================================================================

import networkx as nx


def test_graph_social_network():
    G = nx.Graph()
    G.add_edges_from([
        ("Alice", "Bob"),
        ("Alice", "Carol"),
        ("Bob", "Dave"),
        ("Carol", "Dave"),
        ("Dave", "Eve"),
    ])

    assert nx.has_path(G, "Alice", "Eve")
    shortest = nx.shortest_path(G, "Alice", "Eve")
    assert len(shortest) == 4    # Alice → Bob/Carol → Dave → Eve (3 hops)

    # Degree centrality — who is most connected?
    centrality = nx.degree_centrality(G)
    most_central = max(centrality, key=centrality.get)
    assert most_central == "Dave"   # connects to Bob, Carol, Eve


def test_graph_directed_dependency():
    """Package dependency graph — can we install in topological order?"""
    G = nx.DiGraph()
    G.add_edges_from([
        ("numpy", "scipy"),
        ("numpy", "pandas"),
        ("scipy", "scikit-learn"),
        ("pandas", "scikit-learn"),
    ])

    assert nx.is_directed_acyclic_graph(G)
    install_order = list(nx.topological_sort(G))
    assert install_order.index("numpy") < install_order.index("scipy")
    assert install_order.index("scipy") < install_order.index("scikit-learn")


def test_graph_weighted_shortest_path():
    G = nx.Graph()
    G.add_weighted_edges_from([
        ("A", "B", 4),
        ("A", "C", 2),
        ("C", "B", 1),
        ("B", "D", 5),
    ])
    length = nx.shortest_path_length(G, "A", "D", weight="weight")
    assert length == 8   # A→C (2) + C→B (1) + B→D (5)


# ===========================================================================
# Vector — NumPy cosine-similarity ANN
# ===========================================================================

def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))


def ann_search(query: np.ndarray, vectors: np.ndarray, k: int) -> list[int]:
    """Brute-force k-nearest-neighbours by cosine similarity."""
    sims = [cosine_similarity(query, v) for v in vectors]
    return sorted(range(len(sims)), key=lambda i: sims[i], reverse=True)[:k]


def test_vector_ann_search():
    """
    Mock 'documents' as 3-d embeddings; find the two most similar to a query.
    Think of each dimension as a topic score: [tech, sports, cooking].
    """
    docs = np.array([
        [0.9, 0.1, 0.0],   # doc 0: mostly tech
        [0.8, 0.2, 0.0],   # doc 1: mostly tech (very similar to 0)
        [0.1, 0.9, 0.0],   # doc 2: mostly sports
        [0.0, 0.1, 0.9],   # doc 3: mostly cooking
        [0.5, 0.5, 0.0],   # doc 4: tech+sports
    ], dtype=float)

    # Normalise to unit vectors (required for cosine sim to equal dot product)
    norms = np.linalg.norm(docs, axis=1, keepdims=True)
    docs = docs / norms

    query = np.array([0.85, 0.15, 0.0])   # a tech-heavy query
    query = query / np.linalg.norm(query)

    top2 = ann_search(query, docs, k=2)
    # Should return the two tech-heavy docs (0 and 1) in some order
    assert set(top2) == {0, 1}


def test_vector_cosine_identical():
    v = np.array([1.0, 2.0, 3.0])
    assert math.isclose(cosine_similarity(v, v), 1.0)


def test_vector_cosine_orthogonal():
    a = np.array([1.0, 0.0])
    b = np.array([0.0, 1.0])
    assert math.isclose(cosine_similarity(a, b), 0.0)

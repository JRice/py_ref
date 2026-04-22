"""
Analytical / search databases.

  DuckDB    — in-process OLAP engine; the "analytical SQLite".
              Columnar storage, window functions, direct file queries,
              Parquet round-trips.  Think: local Data Lake.

  SQLite    — FTS5 virtual tables give you a real inverted index
  FTS5        with BM25 ranking, phrase search, and prefix search,
              all through stdlib sqlite3.
"""

import sqlite3
import duckdb
import json
import pytest
from pathlib import Path


# ===========================================================================
# DuckDB — OLAP / Data Lake patterns
# ===========================================================================

@pytest.fixture
def duck():
    """In-memory DuckDB connection, isolated per test."""
    con = duckdb.connect()
    yield con
    con.close()


@pytest.fixture
def sales_duck(duck):
    """Seed a mock sales fact table."""
    duck.execute("""
        CREATE TABLE sales AS
        SELECT * FROM (VALUES
            ('2024-01', 'North', 'Widget',  120, 9.99),
            ('2024-01', 'South', 'Widget',   80, 9.99),
            ('2024-01', 'North', 'Gadget',   30, 24.99),
            ('2024-02', 'North', 'Widget',  150, 9.99),
            ('2024-02', 'South', 'Gadget',   60, 24.99),
            ('2024-02', 'East',  'Widget',   90, 9.99),
            ('2024-03', 'North', 'Gadget',   45, 24.99),
            ('2024-03', 'East',  'Widget',  200, 9.99),
            ('2024-03', 'South', 'Widget',  110, 9.99)
        ) t(month, region, product, units, price)
    """)
    return duck


def test_duck_group_by_aggregation(sales_duck):
    """Classic OLAP: revenue by product."""
    rows = sales_duck.execute("""
        SELECT product,
               SUM(units)              AS total_units,
               ROUND(SUM(units * price), 2) AS revenue
        FROM   sales
        GROUP  BY product
        ORDER  BY revenue DESC
    """).fetchall()

    assert rows[0][0] == "Widget"           # Widget outsells Gadget in revenue
    assert rows[0][1] == 750                # total Widget units
    assert rows[1][0] == "Gadget"


def test_duck_window_running_total(sales_duck):
    """Window function: cumulative units sold per product over time."""
    rows = sales_duck.execute("""
        SELECT month,
               product,
               units,
               SUM(units) OVER (
                   PARTITION BY product
                   ORDER BY month
                   ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW
               ) AS running_total
        FROM   sales
        WHERE  region = 'North'
        ORDER  BY product, month
    """).fetchall()

    # North / Widget: 120 → 270 (120+150)
    widget_rows = [r for r in rows if r[1] == "Widget"]
    assert widget_rows[0][3] == 120
    assert widget_rows[1][3] == 270


def test_duck_window_rank(sales_duck):
    """RANK() within each month to find the best-selling region."""
    rows = sales_duck.execute("""
        SELECT month, region,
               SUM(units) AS units,
               RANK() OVER (PARTITION BY month ORDER BY SUM(units) DESC) AS rnk
        FROM   sales
        GROUP  BY month, region
        ORDER  BY month, rnk
    """).fetchall()

    # January: North sold 120+30=150, South sold 80 → North ranks 1
    jan = [r for r in rows if r[0] == "2024-01"]
    assert jan[0][1] == "North"
    assert jan[0][3] == 1


def test_duck_window_lag_month_over_month(sales_duck):
    """LAG() to compute month-over-month unit change per product."""
    rows = sales_duck.execute("""
        WITH monthly AS (
            SELECT month, product, SUM(units) AS units
            FROM   sales
            GROUP  BY month, product
        )
        SELECT month, product, units,
               LAG(units) OVER (PARTITION BY product ORDER BY month) AS prev_units,
               units - LAG(units) OVER (PARTITION BY product ORDER BY month) AS delta
        FROM   monthly
        ORDER  BY product, month
    """).fetchall()

    gadget = [r for r in rows if r[1] == "Gadget"]
    assert gadget[0][3] is None       # no previous month
    assert gadget[1][4] == gadget[1][2] - gadget[0][2]   # delta = units - prev


def test_duck_query_csv_directly(duck, tmp_path):
    """
    Data Lake pattern: DuckDB can query CSV / Parquet files on disk
    without loading them into a table first.
    """
    csv_file = tmp_path / "events.csv"
    csv_file.write_text(
        "user_id,event,ts\n"
        "1,login,2024-01-01\n"
        "2,login,2024-01-01\n"
        "1,purchase,2024-01-02\n"
        "3,login,2024-01-03\n"
        "2,purchase,2024-01-03\n"
    )

    result = duck.execute(f"""
        SELECT event, COUNT(*) AS cnt
        FROM   read_csv_auto('{csv_file.as_posix()}')
        GROUP  BY event
        ORDER  BY cnt DESC
    """).fetchall()

    assert result[0] == ("login", 3)
    assert result[1] == ("purchase", 2)


def test_duck_parquet_round_trip(duck, tmp_path):
    """Write a query result to Parquet, re-query it — the Data Lake loop."""
    duck.execute("""
        CREATE TABLE products AS
        SELECT * FROM (VALUES
            (1, 'Widget', 9.99,  'hardware'),
            (2, 'Gadget', 24.99, 'hardware'),
            (3, 'eBook',  4.99,  'digital'),
            (4, 'Course', 99.99, 'digital')
        ) t(id, name, price, category)
    """)

    parquet_path = tmp_path / "products.parquet"
    duck.execute(f"COPY products TO '{parquet_path.as_posix()}' (FORMAT PARQUET)")

    assert parquet_path.exists()

    rows = duck.execute(f"""
        SELECT   category, ROUND(AVG(price), 2) AS avg_price
        FROM     read_parquet('{parquet_path.as_posix()}')
        GROUP BY category
        ORDER BY avg_price DESC
    """).fetchall()

    assert rows[0][0] == "digital"       # Course + eBook average > hardware
    assert rows[0][1] == pytest.approx(52.49)


def test_duck_json_unnest(duck):
    """
    DuckDB speaks JSON natively — useful for querying semi-structured logs
    without a schema migration.
    """
    duck.execute("""
        CREATE TABLE raw_logs (payload JSON);
        INSERT INTO raw_logs VALUES
            ('{"user": "alice", "tags": ["admin", "beta"]}'),
            ('{"user": "bob",   "tags": ["beta"]}'),
            ('{"user": "carol", "tags": ["admin"]}')
    """)

    rows = duck.execute("""
        SELECT payload->>'user' AS user,
               COUNT(tag.value) AS tag_count
        FROM   raw_logs,
               json_each(payload->'tags') AS tag
        GROUP  BY user
        ORDER  BY user
    """).fetchall()

    assert dict(rows)["alice"] == 2
    assert dict(rows)["bob"]   == 1


# ===========================================================================
# SQLite FTS5 — inverted index / full-text search
# ===========================================================================

@pytest.fixture
def fts_db():
    """In-memory SQLite with an FTS5 virtual table of article snippets."""
    con = sqlite3.connect(":memory:")
    con.execute("""
        CREATE VIRTUAL TABLE articles USING fts5(
            title,
            body,
            tokenize = 'porter unicode61'   -- Porter stemming + Unicode
        )
    """)
    con.executemany("INSERT INTO articles VALUES (?, ?)", [
        ("Python asyncio guide",
         "asyncio enables concurrent IO-bound tasks using coroutines and event loops"),
        ("Introduction to DuckDB",
         "DuckDB is an in-process analytical database optimised for OLAP workloads"),
        ("Python type hints tutorial",
         "Type hints improve code readability and enable static analysis with mypy"),
        ("Full-text search with SQLite",
         "SQLite FTS5 provides inverted index search with BM25 ranking built in"),
        ("Async web scraping with httpx",
         "httpx supports asyncio making concurrent web scraping easy with async HTTP"),
        ("Understanding database indexes",
         "Database indexes speed up queries by maintaining sorted data structures"),
    ])
    con.commit()
    yield con
    con.close()


def test_fts5_basic_search(fts_db):
    """Simple keyword search across all columns."""
    rows = fts_db.execute(
        "SELECT title FROM articles WHERE articles MATCH 'asyncio'"
    ).fetchall()
    titles = [r[0] for r in rows]
    assert "Python asyncio guide" in titles
    assert "Async web scraping with httpx" in titles


def test_fts5_bm25_ranking(fts_db):
    """
    ORDER BY rank returns results in BM25 relevance order.
    SQLite's rank values are negative floats — more negative = more relevant.
    """
    rows = fts_db.execute("""
        SELECT title, rank
        FROM   articles
        WHERE  articles MATCH 'python'
        ORDER  BY rank
    """).fetchall()

    titles = [r[0] for r in rows]
    # Both Python articles should appear; rank is a float
    assert any("Python" in t for t in titles)
    assert all(isinstance(r[1], float) for r in rows)
    assert rows[0][1] < rows[-1][1]   # first result is most relevant (most negative)


def test_fts5_phrase_search(fts_db):
    """Quoted phrases must appear as a contiguous sequence of tokens."""
    rows = fts_db.execute(
        'SELECT title FROM articles WHERE articles MATCH \'"inverted index"\''
    ).fetchall()
    assert len(rows) == 1
    assert rows[0][0] == "Full-text search with SQLite"


def test_fts5_prefix_search(fts_db):
    """Trailing * matches any word starting with the prefix."""
    rows = fts_db.execute(
        "SELECT title FROM articles WHERE articles MATCH 'concurr*'"
    ).fetchall()
    # "concurrent" in asyncio guide, "concurrent" in httpx article
    assert len(rows) == 2


def test_fts5_column_filter(fts_db):
    """Restrict a term to a specific column with column:term syntax."""
    # 'database' appears in titles AND bodies — narrow to title only
    rows = fts_db.execute(
        "SELECT title FROM articles WHERE articles MATCH 'title:database'"
    ).fetchall()
    titles = [r[0] for r in rows]
    assert any("database" in t.lower() for t in titles)
    # Body-only mention in asyncio guide should NOT appear
    assert "Python asyncio guide" not in titles


def test_fts5_boolean_operators(fts_db):
    """AND / OR / NOT operators for compound queries."""
    rows = fts_db.execute(
        "SELECT title FROM articles WHERE articles MATCH 'python AND type'"
    ).fetchall()
    assert len(rows) == 1
    assert "type hints" in rows[0][0].lower()

    rows_or = fts_db.execute(
        "SELECT title FROM articles WHERE articles MATCH 'DuckDB OR asyncio'"
    ).fetchall()
    assert len(rows_or) >= 2   # DuckDB article + at least one asyncio article


def test_fts5_snippet(fts_db):
    """snippet() returns the matching excerpt with highlights — like search result previews."""
    rows = fts_db.execute("""
        SELECT title,
               snippet(articles, 1, '[', ']', '...', 8) AS excerpt
        FROM   articles
        WHERE  articles MATCH 'inverted'
        ORDER  BY rank
    """).fetchall()

    assert len(rows) == 1
    title, excerpt = rows[0]
    assert "[inverted]" in excerpt    # matched term is wrapped in our markers


def test_fts5_stemming(fts_db):
    """Porter stemmer means 'optimise' matches 'optimised' in the index."""
    rows = fts_db.execute(
        "SELECT title FROM articles WHERE articles MATCH 'optimise'"
    ).fetchall()
    assert len(rows) == 1
    assert "DuckDB" in rows[0][0]

"""
Data / science-adjacent Python.

Covers: NumPy arrays, Pandas DataFrames, Polars expressions,
PyArrow tables + Parquet, Matplotlib (headless), scikit-learn Pipeline.

All tests run fully offline; matplotlib uses the Agg (non-GUI) backend.
"""

from __future__ import annotations

import io
import math

import matplotlib
matplotlib.use("Agg")   # must set before importing pyplot
import matplotlib.pyplot as plt

import numpy as np
import pandas as pd
import polars as pl
import pyarrow as pa
import pyarrow.parquet as pq
import pytest
from sklearn.datasets import load_iris
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


# ===========================================================================
# NumPy  —  arrays, broadcasting, fancy indexing, vectorised ops
# ===========================================================================

def test_numpy_array_creation():
    a = np.array([1, 2, 3, 4, 5])
    assert a.dtype == np.int64  or a.dtype == np.int32   # platform-dependent
    assert a.shape == (5,)
    assert a.sum() == 15

    z = np.zeros((3, 4), dtype=float)
    assert z.shape == (3, 4)
    assert z.sum() == 0.0

    eye = np.eye(3)
    assert eye[0, 0] == 1.0 and eye[0, 1] == 0.0


def test_numpy_slicing_and_indexing():
    a = np.arange(10)   # [0..9]
    assert list(a[2:5])   == [2, 3, 4]
    assert list(a[::2])   == [0, 2, 4, 6, 8]
    assert list(a[::-1])  == [9, 8, 7, 6, 5, 4, 3, 2, 1, 0]

    # Fancy indexing with index array
    idx = np.array([1, 4, 7])
    assert list(a[idx]) == [1, 4, 7]

    # Boolean mask
    mask   = a > 5
    assert list(a[mask]) == [6, 7, 8, 9]


def test_numpy_2d_slicing():
    m = np.arange(12).reshape(3, 4)
    assert m[1, 2] == 6             # row 1, col 2
    assert list(m[:, 0]) == [0, 4, 8]   # first column
    assert list(m[0, :]) == [0, 1, 2, 3]  # first row


def test_numpy_broadcasting():
    """Broadcasting: shapes are aligned from the right; size-1 dims expand."""
    row    = np.array([[1, 2, 3]])      # shape (1, 3)
    col    = np.array([[10], [20]])     # shape (2, 1)
    result = row + col                  # broadcasts to (2, 3)
    assert result.shape == (2, 3)
    assert result[0, 0] == 11
    assert result[1, 2] == 23


def test_numpy_vectorised_vs_loop():
    """NumPy ufuncs avoid Python-level loops — crucial for performance."""
    data = np.random.default_rng(42).standard_normal(10_000)

    # Python loop
    def python_rmse(arr):
        return math.sqrt(sum(x**2 for x in arr) / len(arr))

    # NumPy vectorised
    np_rmse = lambda array: np.sqrt(np.mean(array**2))

    assert math.isclose(python_rmse(data), np_rmse(data), rel_tol=1e-9)


def test_numpy_aggregations():
    a = np.array([[1, 2, 3], [4, 5, 6]])
    assert a.sum()          == 21
    assert a.sum(axis=0).tolist() == [5, 7, 9]   # column sums
    assert a.sum(axis=1).tolist() == [6, 15]      # row sums
    assert a.mean()         == 3.5
    assert a.max()          == 6
    assert a.argmax()       == 5    # flat index of maximum


def test_numpy_linear_algebra():
    A = np.array([[1, 2], [3, 4]], dtype=float)
    b = np.array([5, 6], dtype=float)
    x = np.linalg.solve(A, b)                      # solves Ax = b
    assert np.allclose(A @ x, b)                   # verify: A * solution ≈ b

    vals, vecs = np.linalg.eig(A)
    # Eigenvalue equation: A @ v = λ * v
    for eigenvalue, eigenvector in zip(vals, vecs.T):
        assert np.allclose(A @ eigenvector, eigenvalue * eigenvector)


def test_numpy_where_and_clip():
    a = np.array([-3, -1, 0, 2, 5, 8])
    clipped = np.clip(a, 0, 5)
    assert list(clipped) == [0, 0, 0, 2, 5, 5]

    labels = np.where(a >= 0, "pos", "neg")
    assert list(labels) == ["neg", "neg", "pos", "pos", "pos", "pos"]


# ===========================================================================
# Pandas  —  DataFrame operations
# ===========================================================================

@pytest.fixture
def sales_df():
    return pd.DataFrame({
        "month":    ["Jan","Jan","Feb","Feb","Mar","Mar"],
        "region":   ["North","South","North","South","North","South"],
        "product":  ["Widget","Widget","Gadget","Widget","Widget","Gadget"],
        "units":    [120, 80, 30, 90, 150, 45],
        "price":    [9.99, 9.99, 24.99, 9.99, 9.99, 24.99],
    })


def test_pandas_basic_ops(sales_df):
    assert len(sales_df) == 6
    assert list(sales_df.columns) == ["month","region","product","units","price"]
    assert sales_df["units"].sum() == 515
    assert sales_df["price"].mean() == pytest.approx(14.99, abs=0.01)


def test_pandas_filter(sales_df):
    north = sales_df[sales_df["region"] == "North"]
    assert len(north) == 3
    assert north["units"].sum() == 300

    high_value = sales_df[(sales_df["units"] > 100) & (sales_df["price"] < 15)]
    assert len(high_value) == 2


def test_pandas_assign_computed_column(sales_df):
    df = sales_df.assign(revenue=lambda df: df["units"] * df["price"])
    assert df["revenue"].iloc[0] == pytest.approx(120 * 9.99)
    assert "revenue" not in sales_df.columns   # original unchanged


def test_pandas_groupby(sales_df):
    summary = (
        sales_df
        .assign(revenue=lambda df: df["units"] * df["price"])
        .groupby("region")["revenue"]
        .sum()
        .reset_index()
        .sort_values("revenue", ascending=False)
    )
    assert summary.iloc[0]["region"] == "North"


def test_pandas_pivot_table(sales_df):
    pivot = sales_df.pivot_table(
        values="units", index="month", columns="region", aggfunc="sum"
    )
    assert pivot.loc["Jan", "North"] == 120
    assert pivot.loc["Feb", "South"] == 90


def test_pandas_merge():
    df_a = pd.DataFrame({"id": [1, 2, 3], "name": ["Alice","Bob","Carol"]})
    df_b = pd.DataFrame({"id": [1, 2, 4], "score": [90, 85, 70]})

    inner = pd.merge(df_a, df_b, on="id", how="inner")
    assert len(inner) == 2     # only ids 1,2 in both

    left  = pd.merge(df_a, df_b, on="id", how="left")
    assert len(left) == 3
    assert pd.isna(left.loc[left["id"] == 3, "score"].values[0])


def test_pandas_fillna_and_dropna():
    df = pd.DataFrame({"a": [1, None, 3], "b": [None, 2, 3]})
    filled  = df.fillna(0)
    dropped = df.dropna()

    assert filled.isna().sum().sum() == 0
    assert len(dropped) == 1   # only row 2 has no NaN


def test_pandas_apply(sales_df):
    df = sales_df.copy()
    df["label"] = df["units"].apply(lambda units: "high" if units >= 100 else "low")
    high_count = (df["label"] == "high").sum()
    assert high_count == 2   # units >= 100: [120, 150]


# ===========================================================================
# Polars  —  fast DataFrame library with lazy evaluation
# ===========================================================================

def test_polars_basic():
    df = pl.DataFrame({
        "name":  ["Alice", "Bob", "Carol", "Dave"],
        "score": [88, 72, 95, 60],
        "dept":  ["eng", "mkt", "eng", "mkt"],
    })
    top = df.filter(pl.col("score") >= 80).sort("score", descending=True)
    assert top.shape == (2, 3)
    assert top["name"].to_list() == ["Carol", "Alice"]


def test_polars_expressions():
    df = pl.DataFrame({"price": [10.0, 20.0, 30.0], "qty": [5, 2, 8]})
    result = df.with_columns(
        (pl.col("price") * pl.col("qty")).alias("revenue")
    )
    assert result["revenue"].to_list() == [50.0, 40.0, 240.0]


def test_polars_lazy_groupby():
    df = pl.LazyFrame({
        "dept":   ["eng","eng","mkt","mkt","eng"],
        "salary": [100, 120, 80, 90, 110],
    })
    result = (
        df
        .group_by("dept")
        .agg(pl.col("salary").mean().alias("avg_salary"))
        .sort("avg_salary", descending=True)
        .collect()
    )
    assert result["dept"].to_list()[0] == "eng"


def test_polars_join():
    df_a = pl.DataFrame({"id": [1, 2, 3], "name": ["A","B","C"]})
    df_b = pl.DataFrame({"id": [1, 2, 4], "val":  [10, 20, 40]})
    joined = df_a.join(df_b, on="id", how="inner")
    assert joined.shape == (2, 3)


# ===========================================================================
# PyArrow  —  columnar in-memory format + Parquet I/O
# ===========================================================================

def test_pyarrow_table_creation():
    schema = pa.schema([
        pa.field("id",    pa.int32()),
        pa.field("name",  pa.string()),
        pa.field("score", pa.float64()),
    ])
    table = pa.table(
        {"id": [1, 2, 3], "name": ["Alice","Bob","Carol"], "score": [90.0, 85.0, 92.5]},
        schema=schema,
    )
    assert table.num_rows    == 3
    assert table.num_columns == 3
    assert table.schema      == schema


def test_pyarrow_parquet_round_trip(tmp_path):
    original = pa.table({
        "product": ["Widget","Gadget","Doohickey"],
        "units":   [100, 50, 200],
        "price":   [9.99, 24.99, 4.99],
    })
    path = tmp_path / "data.parquet"
    pq.write_table(original, path)

    loaded = pq.read_table(path)
    assert loaded.num_rows == 3
    assert loaded["product"].to_pylist() == ["Widget","Gadget","Doohickey"]


def test_pyarrow_to_pandas_roundtrip():
    table = pa.table({"x": [1, 2, 3], "y": [4.0, 5.0, 6.0]})
    df    = table.to_pandas()
    back  = pa.Table.from_pandas(df)

    assert isinstance(df, pd.DataFrame)
    assert back.num_rows == 3


def test_pyarrow_column_filter():
    """Read only specific columns from Parquet — columnar advantage."""
    table = pa.table({"a": [1,2,3], "b": [4,5,6], "c": [7,8,9]})
    buf = io.BytesIO()
    pq.write_table(table, buf)
    buf.seek(0)

    partial = pq.read_table(buf, columns=["a", "c"])
    assert partial.column_names == ["a", "c"]
    assert "b" not in partial.column_names


# ===========================================================================
# Matplotlib  —  headless figure creation and export
# ===========================================================================

def test_matplotlib_line_plot():
    fig, ax = plt.subplots()
    x = np.linspace(0, 2 * math.pi, 100)
    ax.plot(x, np.sin(x), label="sin")
    ax.plot(x, np.cos(x), label="cos")
    ax.legend()
    ax.set_title("Trig functions")

    assert len(ax.get_lines()) == 2
    assert ax.get_title() == "Trig functions"
    plt.close(fig)


def test_matplotlib_save_to_buffer():
    fig, ax = plt.subplots()
    ax.bar(["A","B","C"], [3, 1, 4])

    buf = io.BytesIO()
    fig.savefig(buf, format="png")
    buf.seek(0)

    assert buf.read(4) == b"\x89PNG"   # PNG magic bytes
    plt.close(fig)


def test_matplotlib_histogram():
    rng  = np.random.default_rng(0)
    data = rng.normal(0, 1, 1000)

    fig, ax = plt.subplots()
    n, bins, patches = ax.hist(data, bins=20)

    assert len(n) == 20
    assert abs(sum(n) - 1000) < 1    # all values binned
    plt.close(fig)


# ===========================================================================
# scikit-learn  —  Pipeline: StandardScaler + LogisticRegression on Iris
# ===========================================================================

def test_sklearn_pipeline_iris():
    iris    = load_iris()
    X, y    = iris.data, iris.target
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    pipe = Pipeline([
        ("scaler", StandardScaler()),
        ("clf",    LogisticRegression(max_iter=200, random_state=42)),
    ])
    pipe.fit(X_train, y_train)
    y_pred  = pipe.predict(X_test)
    acc     = accuracy_score(y_test, y_pred)
    assert acc >= 0.93   # Iris is easy; expect >93 % accuracy


def test_sklearn_cross_validation():
    iris = load_iris()
    pipe = Pipeline([
        ("scaler", StandardScaler()),
        ("clf",    LogisticRegression(max_iter=200, random_state=0)),
    ])
    scores = cross_val_score(pipe, iris.data, iris.target, cv=5)
    assert scores.mean() >= 0.95
    assert scores.min()  >= 0.90


def test_sklearn_pipeline_step_access():
    pipe = Pipeline([
        ("scaler", StandardScaler()),
        ("clf",    LogisticRegression()),
    ])
    assert isinstance(pipe.named_steps["scaler"], StandardScaler)
    assert isinstance(pipe.named_steps["clf"],    LogisticRegression)
    # Access params through the pipeline
    pipe.set_params(clf__max_iter=500)
    assert pipe.named_steps["clf"].max_iter == 500

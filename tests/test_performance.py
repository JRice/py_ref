"""
Performance patterns and measurement.

Covers: timeit, cProfile + pstats, __slots__ memory savings,
generators vs materialised lists (sys.getsizeof), NumPy vectorisation,
functools.cached_property (see also test_data_model.py), lru_cache stats.

Big-O commentary is in docstrings — it can't be "tested" but belongs here.
"""

from __future__ import annotations

import cProfile
import functools
import io
import math
import pstats
import sys
import timeit

import numpy as np
import pytest


# ===========================================================================
# timeit  —  micro-benchmarks
# ===========================================================================

def test_timeit_list_vs_generator_build():
    """
    List comprehension vs generator expression — which is faster to BUILD?
    Answer: list comprehension is faster (less overhead per item).
    Generator is faster/cheaper when you don't need all values.
    """
    list_time = timeit.timeit(
        "[x*x for x in range(1000)]",
        number=1000,
    )
    gen_time = timeit.timeit(
        "list(x*x for x in range(1000))",   # force full evaluation
        number=1000,
    )
    # Both complete in reasonable time (no infinite loop)
    assert list_time < 5.0
    assert gen_time  < 5.0


def test_timeit_dict_lookup_vs_list_scan():
    """
    dict/set O(1) lookup vs list O(n) scan — the gap grows with n.
    """
    setup = "data = list(range(10_000)); s = set(data)"

    list_time = timeit.timeit("9999 in data", setup=setup, number=10_000)
    set_time  = timeit.timeit("9999 in s",    setup=setup, number=10_000)

    assert set_time < list_time   # set lookup is faster


def test_timeit_string_concat_vs_join():
    """
    += string concatenation is O(n²) due to immutability.
    ''.join() is O(n) — always prefer join for building strings.
    """
    concat_time = timeit.timeit(
        "s = ''; [s := s + c for c in 'abcdefghij' * 50]",
        number=1000,
    )
    join_time = timeit.timeit(
        "''.join('abcdefghij' * 50)",
        number=1000,
    )
    assert join_time < concat_time


# ===========================================================================
# cProfile  —  find where time is actually spent
# ===========================================================================

def slow_func(n: int) -> int:
    """Intentionally slow: Python-loop sum of squares."""
    total = 0
    for i in range(n):
        total += i * i
    return total


def test_cprofile_runs_and_captures():
    pr = cProfile.Profile()
    pr.enable()
    result = slow_func(10_000)
    pr.disable()

    assert result == sum(i*i for i in range(10_000))

    # Parse stats — check that slow_func shows up as a called function
    stream = io.StringIO()
    ps = pstats.Stats(pr, stream=stream).sort_stats("cumulative")
    ps.print_stats(10)

    output = stream.getvalue()
    assert "slow_func" in output


def test_cprofile_context_manager_style():
    """Using cProfile as a context manager (Python 3.8+)."""
    with cProfile.Profile() as pr:
        [x**2 for x in range(5000)]

    stats = pstats.Stats(pr)
    # If no exception, profile captured successfully
    assert stats.total_calls > 0


# ===========================================================================
# __slots__  —  reduce per-instance memory overhead
# ===========================================================================
# Without __slots__: each instance has a __dict__ (~200 bytes overhead).
# With    __slots__: instance dict eliminated; fixed attribute layout.

class WithDict:
    def __init__(self, x, y, z):
        self.x = x
        self.y = y
        self.z = z


class WithSlots:
    __slots__ = ("x", "y", "z")

    def __init__(self, x, y, z):
        self.x = x
        self.y = y
        self.z = z


def test_slots_no_instance_dict():
    obj = WithSlots(1, 2, 3)
    assert not hasattr(obj, "__dict__")   # no dict overhead

    obj2 = WithDict(1, 2, 3)
    assert hasattr(obj2, "__dict__")


def test_slots_no_dict_overhead():
    """
    sys.getsizeof only measures the object head — the separately-allocated
    __dict__ isn't included, making size comparisons unreliable across Python
    versions (CPython 3.12+ stores dict compactly inline).

    The real saving is the absent __dict__: slots instances never allocate one.
    """
    with_slots = WithSlots(1, 2, 3)
    with_dict  = WithDict(1, 2, 3)

    assert not hasattr(with_slots, "__dict__")   # no per-instance dict
    assert hasattr(with_dict,      "__dict__")   # has a dict

    # The per-instance dict itself costs memory:
    dict_overhead = sys.getsizeof(with_dict.__dict__)
    assert dict_overhead > 0


def test_slots_cannot_set_undeclared_attr():
    obj = WithSlots(1, 2, 3)
    with pytest.raises(AttributeError):
        obj.w = 99   # 'w' not in __slots__


# ===========================================================================
# Generators vs materialised lists  —  memory
# ===========================================================================

def test_generator_constant_memory():
    """
    A generator expression uses O(1) memory regardless of how many items
    it *would* produce — it computes values on demand.
    """
    # Generator object itself is tiny
    gen  = (x * x for x in range(1_000_000))
    gen_size = sys.getsizeof(gen)

    # A list of the same values is proportional to n
    lst  = [x * x for x in range(1_000)]
    list_size = sys.getsizeof(lst)

    # Generator is much smaller than even a 1000-element list
    assert gen_size < list_size


def test_generator_lazy_evaluation():
    """Generators compute values on demand — useful for infinite sequences."""
    def naturals():
        n = 0
        while True:
            yield n
            n += 1

    gen   = naturals()
    first = [next(gen) for _ in range(5)]
    assert first == [0, 1, 2, 3, 4]
    # The generator hasn't computed anything beyond what was requested


def test_itertools_islice_avoids_materialising():
    import itertools
    # Take first 5 items from an infinite sequence without building a list
    gen    = itertools.count(0)
    result = list(itertools.islice(gen, 5))
    assert result == [0, 1, 2, 3, 4]


# ===========================================================================
# NumPy vectorisation  —  avoid Python loops on numerical data
# ===========================================================================

def python_sum_squares(arr):
    return sum(x * x for x in arr)


def numpy_sum_squares(arr):
    return float(np.sum(arr ** 2))


def test_numpy_vectorisation_correctness():
    data = list(range(1000))
    np_data = np.array(data, dtype=float)
    assert python_sum_squares(data) == numpy_sum_squares(np_data)


def test_numpy_vectorisation_speed():
    """NumPy should be significantly faster than a Python loop."""
    n = 100_000
    py_data = list(range(n))
    np_data = np.arange(n, dtype=float)

    py_time = timeit.timeit(lambda: python_sum_squares(py_data), number=10)
    np_time = timeit.timeit(lambda: numpy_sum_squares(np_data),  number=10)

    assert np_time < py_time   # NumPy wins on numerical work


def test_numpy_avoid_python_loop_with_where():
    """np.where is a vectorised if/else — avoids a Python loop entirely."""
    prices     = np.array([10.0, 25.0, 5.0, 15.0, 30.0])
    thresholds = 15.0

    # Python loop version
    labels_py = ["expensive" if p >= thresholds else "cheap" for p in prices]

    # NumPy version
    labels_np = np.where(prices >= thresholds, "expensive", "cheap")

    assert list(labels_np) == labels_py


# ===========================================================================
# lru_cache statistics
# ===========================================================================

def test_lru_cache_stats():
    @functools.lru_cache(maxsize=4)
    def fib(n: int) -> int:
        if n < 2:
            return n
        return fib(n - 1) + fib(n - 2)

    fib(10)
    info = fib.cache_info()
    assert info.hits   > 0      # many sub-problems reused
    assert info.misses > 0
    assert info.currsize <= 4   # never exceeds maxsize

    fib.cache_clear()
    assert fib.cache_info().currsize == 0


# ===========================================================================
# Big-O reference (commentary — verified by logic, not timing)
# ===========================================================================

def test_bigo_comments_as_documentation():
    """
    Container       Operation               Average     Worst
    ─────────────── ─────────────────────── ─────────── ────────
    list            index (lst[i])          O(1)        O(1)
    list            append                  O(1)*       O(n)  amortised
    list            insert(0, x)            O(n)        O(n)
    list            x in lst                O(n)        O(n)
    list            sort                    O(n log n)  O(n log n)

    dict / set      get / __contains__      O(1)        O(n)  rare collision
    dict / set      insert / delete         O(1)        O(n)

    deque           appendleft / popleft    O(1)        O(1)
    deque           index (d[i])            O(n)        O(n)  ← use list!

    heapq           heappush / heappop      O(log n)    O(log n)
    heapq           heapify                 O(n)        O(n)

    str             s + t (concat)          O(n+m)      O(n+m)
    str             ''.join(lst)            O(n)        O(n)  ← prefer this
    """
    # Verify the most commonly surprising ones:

    # list `in` is O(n) — use set for repeated membership tests
    big_list = list(range(10_000))
    big_set  = set(big_list)

    list_time = timeit.timeit(lambda: 9999 in big_list, number=5000)
    set_time  = timeit.timeit(lambda: 9999 in big_set,  number=5000)
    assert set_time < list_time

    # deque O(1) appendleft vs list O(n) insert(0, x)
    from collections import deque
    d = deque(range(1000))
    lst = list(range(1000))

    dq_time   = timeit.timeit(lambda: d.appendleft(0),   number=10_000)
    lst_time  = timeit.timeit(lambda: lst.insert(0, 0),  number=10_000)
    assert dq_time < lst_time

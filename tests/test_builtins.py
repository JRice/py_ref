"""
Common Python builtins and itertools — quick-reference examples.
"""

import itertools
import functools


# ---------------------------------------------------------------------------
# min / max / sum / abs
# ---------------------------------------------------------------------------

def test_min_max_with_key():
    products = [
        {"name": "Widget", "price": 9.99},
        {"name": "Gadget", "price": 24.99},
        {"name": "Doohickey", "price": 4.99},
    ]
    cheapest = min(products, key=lambda p: p["price"])
    priciest = max(products, key=lambda p: p["price"])
    assert cheapest["name"] == "Doohickey"
    assert priciest["name"] == "Gadget"


def test_sum_abs():
    deltas = [3, -7, 2, -1, 5]
    assert sum(deltas) == 2
    assert sum(abs(d) for d in deltas) == 18


def test_min_max_default():
    # Avoid ValueError on empty sequence
    assert min([], default=0) == 0
    assert max([], default=-1) == -1


# ---------------------------------------------------------------------------
# range / enumerate
# ---------------------------------------------------------------------------

def test_range_variants():
    assert list(range(5)) == [0, 1, 2, 3, 4]
    assert list(range(2, 8, 2)) == [2, 4, 6]
    assert list(range(5, 0, -1)) == [5, 4, 3, 2, 1]


def test_enumerate():
    colors = ["red", "green", "blue"]
    result = [(i, c) for i, c in enumerate(colors, start=1)]
    assert result == [(1, "red"), (2, "green"), (3, "blue")]


# ---------------------------------------------------------------------------
# zip / zip_longest
# ---------------------------------------------------------------------------

def test_zip_pairs_columns():
    names = ["Alice", "Bob", "Carol"]
    scores = [88, 92, 79]
    pairs = list(zip(names, scores))
    assert pairs == [("Alice", 88), ("Bob", 92), ("Carol", 79)]

    # zip stops at shortest — use zip_longest to pad
    from itertools import zip_longest
    a = [1, 2, 3]
    b = [10, 20]
    assert list(zip_longest(a, b, fillvalue=0)) == [(1,10),(2,20),(3,0)]


def test_zip_transpose():
    matrix = [[1, 2, 3], [4, 5, 6]]
    transposed = list(zip(*matrix))
    assert transposed == [(1, 4), (2, 5), (3, 6)]


# ---------------------------------------------------------------------------
# sorted / reversed
# ---------------------------------------------------------------------------

def test_sorted_preserves_original():
    nums = [3, 1, 4, 1, 5]
    s = sorted(nums)
    assert s == [1, 1, 3, 4, 5]
    assert nums == [3, 1, 4, 1, 5]   # original unchanged


def test_reversed():
    result = list(reversed([1, 2, 3, 4]))
    assert result == [4, 3, 2, 1]

    # strings don't support reversed() directly — convert first
    assert "".join(reversed("hello")) == "olleh"


# ---------------------------------------------------------------------------
# any / all
# ---------------------------------------------------------------------------

def test_any_all():
    statuses = [True, True, False, True]
    assert any(statuses) is True
    assert all(statuses) is False

    nums = [2, 4, 6, 8]
    assert all(n % 2 == 0 for n in nums)   # generator, short-circuits
    assert not any(n > 10 for n in nums)


# ---------------------------------------------------------------------------
# map / filter
# ---------------------------------------------------------------------------

def test_map_filter():
    nums = range(10)

    evens = list(filter(lambda n: n % 2 == 0, nums))
    assert evens == [0, 2, 4, 6, 8]

    squared = list(map(lambda n: n ** 2, evens))
    assert squared == [0, 4, 16, 36, 64]

    # Idiomatic Python prefers comprehensions over map/filter
    assert [n**2 for n in range(10) if n % 2 == 0] == squared


# ---------------------------------------------------------------------------
# functools
# ---------------------------------------------------------------------------

def test_reduce():
    nums = [1, 2, 3, 4, 5]
    product = functools.reduce(lambda acc, n: acc * n, nums)
    assert product == 120


def test_partial():
    def power(base, exp):
        return base ** exp

    square = functools.partial(power, exp=2)
    cube   = functools.partial(power, exp=3)
    assert square(5) == 25
    assert cube(3) == 27


def test_lru_cache():
    call_count = 0

    @functools.lru_cache(maxsize=128)
    def slow_compute(n: int) -> int:
        nonlocal call_count
        call_count += 1
        return n * n

    slow_compute(5)
    slow_compute(5)   # cache hit
    slow_compute(6)
    assert call_count == 2   # only 2 unique calls
    assert slow_compute.cache_info().hits == 1


# ---------------------------------------------------------------------------
# itertools
# ---------------------------------------------------------------------------

def test_itertools_chain():
    combined = list(itertools.chain([1, 2], [3, 4], [5]))
    assert combined == [1, 2, 3, 4, 5]


def test_itertools_combinations_permutations():
    letters = ["A", "B", "C"]
    assert len(list(itertools.combinations(letters, 2))) == 3    # (A,B),(A,C),(B,C)
    assert len(list(itertools.permutations(letters, 2))) == 6    # ordered pairs


def test_itertools_product():
    # Cartesian product — e.g., all (size, color) combinations
    sizes = ["S", "M"]
    colors = ["red", "blue"]
    variants = list(itertools.product(sizes, colors))
    assert len(variants) == 4
    assert ("M", "blue") in variants


def test_itertools_groupby():
    # groupby requires the input to be sorted by the key first
    data = [
        {"dept": "eng", "name": "Alice"},
        {"dept": "eng", "name": "Bob"},
        {"dept": "mkt", "name": "Carol"},
    ]
    data.sort(key=lambda d: d["dept"])
    groups = {k: [e["name"] for e in v]
              for k, v in itertools.groupby(data, key=lambda d: d["dept"])}
    assert groups["eng"] == ["Alice", "Bob"]
    assert groups["mkt"] == ["Carol"]


def test_itertools_islice():
    # Lazy slice of any iterable — no materializing the whole thing
    naturals = itertools.count(1)    # infinite iterator
    first_five = list(itertools.islice(naturals, 5))
    assert first_five == [1, 2, 3, 4, 5]


def test_itertools_accumulate():
    # Running totals (like Ruby's inject with history)
    daily_sales = [100, 150, 90, 200, 130]
    running = list(itertools.accumulate(daily_sales))
    assert running == [100, 250, 340, 540, 670]

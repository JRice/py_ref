"""
Python loop idioms — comprehensions, generators, walrus, unpacking, etc.
"""


# ---------------------------------------------------------------------------
# Comprehensions
# ---------------------------------------------------------------------------

def test_list_comprehension_with_condition():
    # Filter + transform in one expression
    words = ["apple", "fig", "banana", "kiwi", "cherry"]
    long_upper = [word.upper() for word in words if len(word) > 4]
    assert long_upper == ["APPLE", "BANANA", "CHERRY"]


def test_nested_comprehension_flatten():
    matrix = [[1, 2, 3], [4, 5, 6], [7, 8, 9]]
    flat = [number for row in matrix for number in row]
    assert flat == [1, 2, 3, 4, 5, 6, 7, 8, 9]


def test_dict_comprehension():
    words = ["hello", "world", "python"]
    length_map = {word: len(word) for word in words}
    assert length_map == {"hello": 5, "world": 5, "python": 6}


def test_set_comprehension():
    nums = [1, 2, 2, 3, 3, 3]
    squares = {number**2 for number in nums}
    assert squares == {1, 4, 9}


def test_generator_expression_is_lazy():
    # Generator doesn't materialise the list — useful for large datasets
    gen = (number**2 for number in range(10))
    assert next(gen) == 0
    assert next(gen) == 1
    assert sum(gen) == 4+9+16+25+36+49+64+81   # rest of the sequence (0,1 already consumed)


# ---------------------------------------------------------------------------
# enumerate / zip in loops
# ---------------------------------------------------------------------------

def test_enumerate_loop():
    fruits = ["apple", "banana", "cherry"]
    indexed = {}
    for i, fruit in enumerate(fruits):
        indexed[fruit] = i
    assert indexed == {"apple": 0, "banana": 1, "cherry": 2}


def test_zip_loop_parallel_iteration():
    keys   = ["a", "b", "c"]
    values = [1, 2, 3]
    result = {}
    for key, value in zip(keys, values):
        result[key] = value
    assert result == {"a": 1, "b": 2, "c": 3}


# ---------------------------------------------------------------------------
# Walrus operator (:=) — assign-and-test in one expression
# ---------------------------------------------------------------------------

def test_walrus_in_while():
    data = iter([10, 20, 0, 30])   # 0 signals "stop"
    results = []
    while (value := next(data, None)) is not None:
        if value == 0:
            break
        results.append(value)
    assert results == [10, 20]


def test_walrus_in_comprehension():
    # Avoid calling an expensive function twice
    import math
    nums = [1, 4, 9, 16, 25]
    # Only include if sqrt is an integer
    perfect = [root for number in nums if (root := int(math.sqrt(number)))**2 == number]
    assert perfect == [1, 2, 3, 4, 5]


# ---------------------------------------------------------------------------
# Starred unpacking
# ---------------------------------------------------------------------------

def test_starred_unpacking_in_loop():
    rows = [(1, "Alice", 30), (2, "Bob", 25)]
    ids = []
    for user_id, *_ in rows:
        ids.append(user_id)
    assert ids == [1, 2]


def test_starred_swap_and_rotate():
    a, b, c = 1, 2, 3
    a, b = b, a          # swap — no temp variable
    assert (a, b) == (2, 1)

    lst = [1, 2, 3, 4]
    first, *rest = lst
    rotated = rest + [first]
    assert rotated == [2, 3, 4, 1]


# ---------------------------------------------------------------------------
# for / else  — the "nobreak" pattern (not in Ruby!)
# ---------------------------------------------------------------------------

def test_for_else_no_match():
    primes = [2, 3, 5, 7, 11]
    target = 6
    found = False
    for prime in primes:
        if prime == target:
            found = True
            break
    else:
        # else runs only if loop completed without break
        found = False
    assert not found


def test_for_else_match():
    items = ["a", "b", "needle", "c"]
    result = None
    for item in items:
        if item == "needle":
            result = item
            break
    else:
        result = "not found"
    assert result == "needle"


# ---------------------------------------------------------------------------
# Generator functions — yield
# ---------------------------------------------------------------------------

def fibonacci():
    a, b = 0, 1
    while True:
        yield a
        a, b = b, a + b


def test_generator_function():
    gen = fibonacci()
    first_8 = [next(gen) for _ in range(8)]
    assert first_8 == [0, 1, 1, 2, 3, 5, 8, 13]


def test_generator_pipeline():
    """Chain generators for memory-efficient data processing."""
    def read_numbers(n):
        yield from range(n)

    def only_even(numbers):
        for number in numbers:
            if number % 2 == 0:
                yield number

    def squared(numbers):
        for number in numbers:
            yield number * number

    pipeline = squared(only_even(read_numbers(10)))
    assert list(pipeline) == [0, 4, 16, 36, 64]


# ---------------------------------------------------------------------------
# itertools.pairwise (Python 3.10+) and sliding via zip
# ---------------------------------------------------------------------------

def test_pairwise_consecutive_diffs():
    import itertools
    prices = [100, 110, 105, 120, 115]
    diffs = [second - first for first, second in itertools.pairwise(prices)]
    assert diffs == [10, -5, 15, -5]


# ---------------------------------------------------------------------------
# dict iteration patterns
# ---------------------------------------------------------------------------

def test_dict_iteration():
    d = {"x": 1, "y": 2, "z": 3}

    keys   = list(d.keys())
    values = list(d.values())
    items  = list(d.items())

    assert keys == ["x", "y", "z"]
    assert values == [1, 2, 3]
    assert items == [("x", 1), ("y", 2), ("z", 3)]

    # Modify during iteration — iterate a copy
    for key in list(d):
        if d[key] < 2:
            del d[key]
    assert d == {"y": 2, "z": 3}

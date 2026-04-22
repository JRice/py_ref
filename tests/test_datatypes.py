"""
Common Python data types — 201-level reference.
Each test is a self-contained example of a real-world use case.
"""

from collections import Counter, defaultdict, deque
from enum import Enum, StrEnum, auto
import heapq
from dataclasses import dataclass, field


# ---------------------------------------------------------------------------
# list
# ---------------------------------------------------------------------------

def test_list_comprehension_and_slicing():
    # Fahrenheit -> Celsius for a week of temps
    temps_f = [32, 68, 86, 77, 95, 104, 59]
    temps_c = [(f - 32) * 5 / 9 for f in temps_f]
    assert temps_c[0] == 0.0
    assert round(temps_c[1], 1) == 20.0

    # Slicing: last 3 days, every-other day
    assert temps_f[-3:] == [95, 104, 59]
    assert temps_f[::2] == [32, 86, 95, 59]


def test_list_unpacking():
    first, *middle, last = [10, 20, 30, 40, 50]
    assert first == 10
    assert middle == [20, 30, 40]
    assert last == 50


def test_list_as_stack_and_queue():
    stack = []
    stack.append("task_a")
    stack.append("task_b")
    assert stack.pop() == "task_b"   # LIFO

    from collections import deque
    queue = deque(["task_a", "task_b"])
    queue.append("task_c")
    assert queue.popleft() == "task_a"  # FIFO


# ---------------------------------------------------------------------------
# set
# ---------------------------------------------------------------------------

def test_set_operations():
    # Which users are in both the beta program and the newsletter?
    beta_users = {"alice", "bob", "carol"}
    newsletter = {"bob", "carol", "dave"}

    both = beta_users & newsletter           # intersection
    either = beta_users | newsletter         # union
    only_beta = beta_users - newsletter      # difference
    exclusive = beta_users ^ newsletter      # symmetric difference

    assert both == {"bob", "carol"}
    assert "dave" in either
    assert only_beta == {"alice"}
    assert exclusive == {"alice", "dave"}


def test_set_dedup_preserving_order():
    # sets are unordered; use dict.fromkeys to dedup while keeping insertion order
    tags = ["python", "web", "python", "api", "web"]
    unique = list(dict.fromkeys(tags))
    assert unique == ["python", "web", "api"]


# ---------------------------------------------------------------------------
# dict
# ---------------------------------------------------------------------------

def test_dict_comprehension_and_merge():
    prices = {"apple": 1.2, "banana": 0.5, "cherry": 3.0}

    # Apply 10% discount
    discounted = {k: round(v * 0.9, 2) for k, v in prices.items()}
    assert discounted["apple"] == 1.08

    # Merge two dicts (later dict wins on conflict) — Python 3.9+ | operator
    overrides = {"cherry": 2.5, "date": 4.0}
    merged = prices | overrides
    assert merged["cherry"] == 2.5
    assert "date" in merged


def test_dict_get_and_setdefault():
    config = {"host": "localhost"}
    assert config.get("port", 5432) == 5432       # safe get with default
    config.setdefault("port", 5432)               # set only if missing
    assert config["port"] == 5432


# ---------------------------------------------------------------------------
# Enum / StrEnum
# ---------------------------------------------------------------------------

class Color(Enum):
    RED = auto()
    GREEN = auto()
    BLUE = auto()


class Status(StrEnum):
    PENDING = auto()    # value == "pending" (lowercased name)
    ACTIVE = auto()
    CLOSED = auto()


def test_enum():
    assert Color.RED != Color.GREEN
    assert list(Color) == [Color.RED, Color.GREEN, Color.BLUE]
    assert Color["RED"] is Color.RED         # lookup by name
    assert Color(Color.RED.value) is Color.RED


def test_str_enum():
    # StrEnum values are strings — great for DB columns / JSON fields
    assert Status.PENDING == "pending"
    assert f"Order is {Status.ACTIVE}" == "Order is active"
    assert Status("closed") is Status.CLOSED


# ---------------------------------------------------------------------------
# Counter
# ---------------------------------------------------------------------------

def test_counter_word_frequency():
    words = "the quick brown fox jumps over the lazy dog the".split()
    counts = Counter(words)

    assert counts["the"] == 3
    assert counts["missing"] == 0           # no KeyError — returns 0

    top2 = counts.most_common(2)
    assert top2[0] == ("the", 3)

    # Counters support arithmetic
    more = Counter({"the": 1, "fox": 2})
    combined = counts + more
    assert combined["the"] == 4


# ---------------------------------------------------------------------------
# defaultdict
# ---------------------------------------------------------------------------

def test_defaultdict_grouping():
    # Group employees by department without checking key existence
    records = [
        ("engineering", "alice"),
        ("marketing", "bob"),
        ("engineering", "carol"),
        ("marketing", "dave"),
    ]
    by_dept: defaultdict[str, list] = defaultdict(list)
    for dept, name in records:
        by_dept[dept].append(name)

    assert by_dept["engineering"] == ["alice", "carol"]
    assert by_dept["unknown"] == []          # missing key auto-creates empty list


def test_defaultdict_counting():
    text = "abracadabra"
    freq: defaultdict[str, int] = defaultdict(int)
    for ch in text:
        freq[ch] += 1
    assert freq["a"] == 5


# ---------------------------------------------------------------------------
# deque
# ---------------------------------------------------------------------------

def test_deque_sliding_window_log():
    # Keep only the last N log entries (bounded buffer)
    log: deque[str] = deque(maxlen=3)
    for msg in ["boot", "connect", "login", "query", "logout"]:
        log.append(msg)

    assert list(log) == ["login", "query", "logout"]


def test_deque_rotate():
    # Rotate a job queue — move next job to front
    jobs = deque(["a", "b", "c", "d"])
    jobs.rotate(-1)                # shift left: "a" goes to end
    assert list(jobs) == ["b", "c", "d", "a"]


# ---------------------------------------------------------------------------
# heapq  (min-heap by default)
# ---------------------------------------------------------------------------

def test_heapq_task_priority_queue():
    # Process tasks in priority order (lower number = higher priority)
    import heapq
    tasks = []
    heapq.heappush(tasks, (3, "send email"))
    heapq.heappush(tasks, (1, "fix outage"))
    heapq.heappush(tasks, (2, "deploy patch"))

    assert heapq.heappop(tasks) == (1, "fix outage")
    assert heapq.heappop(tasks) == (2, "deploy patch")


def test_heapq_top_k():
    scores = [42, 7, 99, 55, 31, 88, 12]
    top3 = heapq.nlargest(3, scores)
    assert top3 == [99, 88, 55]

    bottom3 = heapq.nsmallest(3, scores)
    assert bottom3 == [7, 12, 31]


def test_heapq_max_heap_trick():
    # Python only has min-heap; negate values to simulate max-heap
    nums = [3, 1, 4, 1, 5, 9, 2, 6]
    heap = [-n for n in nums]
    heapq.heapify(heap)
    assert -heapq.heappop(heap) == 9    # largest first


# ---------------------------------------------------------------------------
# dataclass
# ---------------------------------------------------------------------------

@dataclass(order=True)
class Point:
    x: float
    y: float
    label: str = field(default="", compare=False)  # excluded from ordering


def test_dataclass():
    p1 = Point(1.0, 2.0, label="A")
    p2 = Point(1.0, 3.0, label="B")
    assert p1 < p2                       # ordered by (x, y)
    assert p1 == Point(1.0, 2.0)        # label ignored in comparison

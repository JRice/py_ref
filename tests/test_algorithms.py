"""
Classic algorithm patterns — concise reference implementations.
Each solves a concrete problem; the pattern name is in the test name.
"""

import heapq
from collections import defaultdict, deque
from functools import cache, lru_cache


# ---------------------------------------------------------------------------
# DFS — depth-first search
# ---------------------------------------------------------------------------

def test_dfs_connected_components():
    """Count islands (connected groups of 1s) in a grid."""
    grid = [
        [1, 1, 0, 0],
        [1, 0, 0, 1],
        [0, 0, 0, 1],
        [0, 1, 1, 0],
    ]
    rows, cols = len(grid), len(grid[0])
    visited = set()

    def dfs(row, col):
        if row < 0 or row >= rows or col < 0 or col >= cols:
            return
        if (row, col) in visited or grid[row][col] == 0:
            return
        visited.add((row, col))
        for delta_row, delta_col in [(-1,0),(1,0),(0,-1),(0,1)]:
            dfs(row + delta_row, col + delta_col)

    islands = 0
    for row in range(rows):
        for col in range(cols):
            if grid[row][col] == 1 and (row, col) not in visited:
                dfs(row, col)
                islands += 1

    assert islands == 3


# ---------------------------------------------------------------------------
# BFS — breadth-first search / shortest path
# ---------------------------------------------------------------------------

def test_bfs_shortest_path():
    """Shortest path (hops) in an unweighted graph."""
    graph = {
        "A": ["B", "C"],
        "B": ["A", "D"],
        "C": ["A", "D", "E"],
        "D": ["B", "C", "F"],
        "E": ["C"],
        "F": ["D"],
    }

    def bfs(start, end):
        queue = deque([(start, [start])])
        visited = {start}
        while queue:
            node, path = queue.popleft()
            if node == end:
                return path
            for neighbor in graph[node]:
                if neighbor not in visited:
                    visited.add(neighbor)
                    queue.append((neighbor, path + [neighbor]))
        return None

    path = bfs("A", "F")
    assert path == ["A", "B", "D", "F"]   # length 3 hops


# ---------------------------------------------------------------------------
# Backtracking — subsets / power set
# ---------------------------------------------------------------------------

def test_backtracking_subsets():
    """Generate all subsets of [1, 2, 3]."""
    def subsets(nums):
        result = []
        def bt(start, current):
            result.append(list(current))
            for i in range(start, len(nums)):
                current.append(nums[i])
                bt(i + 1, current)
                current.pop()
        bt(0, [])
        return result

    result = subsets([1, 2, 3])
    assert len(result) == 8   # 2^3
    assert [] in result
    assert [1, 2, 3] in result


def test_backtracking_combinations():
    """All combinations of k items from n — e.g. pick 2 toppings from 4."""
    from itertools import combinations
    toppings = ["mushroom", "olive", "pepper", "onion"]
    picks = list(combinations(toppings, 2))
    assert len(picks) == 6
    assert ("mushroom", "olive") in picks


# ---------------------------------------------------------------------------
# Top-K with heapq
# ---------------------------------------------------------------------------

def test_top_k_frequent_words():
    """Return the k most frequent words."""
    words = "the cat sat on the mat the cat sat".split()
    counts = defaultdict(int)
    for word in words:
        counts[word] += 1

    # min-heap of size k — cheapest to maintain
    k = 2
    heap = []
    for word, freq in counts.items():
        heapq.heappush(heap, (freq, word))
        if len(heap) > k:
            heapq.heappop(heap)   # evict least frequent

    top_k = sorted(heap, reverse=True)
    assert top_k[0][1] == "the"   # freq=3


# ---------------------------------------------------------------------------
# Sorting
# ---------------------------------------------------------------------------

def test_sorting_with_key():
    people = [
        {"name": "Alice", "age": 30},
        {"name": "Bob",   "age": 25},
        {"name": "Carol", "age": 30},
    ]
    # Sort by age asc, then name asc
    people.sort(key=lambda person: (person["age"], person["name"]))
    assert people[0]["name"] == "Bob"
    assert people[1]["name"] == "Alice"   # age=30, A < C


def test_sort_stability_and_reverse():
    nums = [3, 1, 4, 1, 5, 9, 2, 6]
    assert sorted(nums) == [1, 1, 2, 3, 4, 5, 6, 9]
    assert sorted(nums, reverse=True)[0] == 9


# ---------------------------------------------------------------------------
# Sliding Window
# ---------------------------------------------------------------------------

def test_sliding_window_max_sum():
    """Maximum sum of any contiguous subarray of length k."""
    nums = [2, 1, 5, 1, 3, 2]
    k = 3

    window_sum = sum(nums[:k])
    max_sum = window_sum
    for i in range(k, len(nums)):
        window_sum += nums[i] - nums[i - k]
        max_sum = max(max_sum, window_sum)

    assert max_sum == 9   # [5, 1, 3]


def test_sliding_window_longest_no_repeat():
    """Longest substring without repeating characters."""
    s = "abcabcbb"
    left = 0
    seen = {}
    best = 0
    for right, ch in enumerate(s):
        if ch in seen and seen[ch] >= left:
            left = seen[ch] + 1
        seen[ch] = right
        best = max(best, right - left + 1)

    assert best == 3   # "abc"


# ---------------------------------------------------------------------------
# Two Pointers
# ---------------------------------------------------------------------------

def test_two_pointers_pair_sum():
    """Find pair that sums to target in sorted array."""
    nums = [1, 2, 4, 6, 8, 9, 14, 15]
    target = 13
    lower, upper = 0, len(nums) - 1
    found = None
    while lower < upper:
        pair_sum = nums[lower] + nums[upper]
        if pair_sum == target:
            found = (nums[lower], nums[upper])
            break
        elif pair_sum < target:
            lower += 1
        else:
            upper -= 1

    assert found == (4, 9)


def test_two_pointers_remove_duplicates():
    """Remove duplicates in-place from sorted array; return new length."""
    nums = [1, 1, 2, 3, 3, 4]
    write = 1
    for read in range(1, len(nums)):
        if nums[read] != nums[read - 1]:
            nums[write] = nums[read]
            write += 1
    assert write == 4
    assert nums[:write] == [1, 2, 3, 4]

# ---------------------------------------------------------------------------
# Fast / Slow Pointers — Floyd's cycle detection
# ---------------------------------------------------------------------------

def test_fast_slow_pointers_cycle():
    """Detect a cycle in a linked list."""
    class Node:
        def __init__(self, val):
            self.val = val
            self.next = None

    # Build: 1 -> 2 -> 3 -> 4 -> 2  (cycle at node 2)
    n1, n2, n3, n4 = Node(1), Node(2), Node(3), Node(4)
    n1.next = n2; n2.next = n3; n3.next = n4; n4.next = n2

    slow = fast = n1
    has_cycle = False
    while fast and fast.next:
        slow = slow.next
        fast = fast.next.next
        if slow is fast:
            has_cycle = True
            break

    assert has_cycle


# ---------------------------------------------------------------------------
# Topological Sort — Kahn's algorithm (BFS)
# ---------------------------------------------------------------------------

def test_topological_sort_task_order():
    """Find a valid order to complete tasks given prerequisites."""
    # (prerequisite, task) edges
    edges = [(0,1), (0,2), (1,3), (2,3), (3,4)]
    node_count = 5

    # "in-degree" is a graph-theory term for "number of input edges". It's used here to keep track of the number of
    # pre-requisites required and is decremented as those tasks are added to the task-order: once it's zero, all pre-
    # reques have been fulfilled and we can add the task itself.
    in_degree = [0] * node_count
    adjacency_list = defaultdict(list)
    for source, destination in edges:
        adjacency_list[source].append(destination)
        in_degree[destination] += 1

    queue = deque(i for i in range(node_count) if in_degree[i] == 0)
    order = []
    while queue:
        node = queue.popleft()
        order.append(node)
        for neighbor in adjacency_list[node]:
            in_degree[neighbor] -= 1
            if in_degree[neighbor] == 0:
                queue.append(neighbor)

    assert len(order) == node_count          # all nodes visited — no cycle
    assert order.index(0) < order.index(3)
    assert order.index(3) < order.index(4)

# ---------------------------------------------------------------------------
# Merge Intervals
# ---------------------------------------------------------------------------

def test_merge_intervals():
    """Merge overlapping calendar events."""
    intervals = [[1,3],[2,6],[8,10],[15,18]]
    intervals.sort(key=lambda interval: interval[0])

    merged = [intervals[0]]
    for start, end in intervals[1:]:
        if start <= merged[-1][1]:
            merged[-1][1] = max(merged[-1][1], end)
        else:
            merged.append([start, end])

    assert merged == [[1,6],[8,10],[15,18]]


# ---------------------------------------------------------------------------
# Memoization — top-down DP
# ---------------------------------------------------------------------------

@cache    # functools.cache — unbounded LRU, thread-safe in 3.9+
def fib(n: int) -> int:
    if n < 2:
        return n
    return fib(n - 1) + fib(n - 2)


def test_memoization_fibonacci():
    assert fib(0) == 0
    assert fib(10) == 55
    assert fib(50) == 12586269025   # instant with memo; blows up without


# ---------------------------------------------------------------------------
# Tabulation — bottom-up Dynamic Programming (DP)
# ---------------------------------------------------------------------------

def test_tabulation_coin_change():
    """Minimum coins to make amount (classic DP)."""
    coins = [1, 5, 10, 25]
    amount = 36

    dp = [float("inf")] * (amount + 1)
    dp[0] = 0
    for target_amount in range(1, amount + 1):
        for coin in coins:
            if coin <= target_amount:
                dp[target_amount] = min(dp[target_amount], dp[target_amount - coin] + 1)

    assert dp[amount] == 3   # 25 + 10 + 1


# ---------------------------------------------------------------------------
# Monotonic Stack — next greater element
# ---------------------------------------------------------------------------

def test_monotonic_stack_next_greater():
    """For each element, find the next greater element to its right."""
    nums = [2, 1, 4, 3, 6, 5]
    result = [-1] * len(nums)
    stack = []   # stores indices, maintains decreasing values

    for i, number in enumerate(nums):
        while stack and nums[stack[-1]] < number:
            result[stack.pop()] = number
        stack.append(i)

    assert result == [4, 4, 6, 6, -1, -1]


# ---------------------------------------------------------------------------
# Monotonic Deque — sliding window maximum
# ---------------------------------------------------------------------------

def test_monotonic_deque_window_max():
    """Maximum value in every window of size k."""
    nums = [1, 3, -1, -3, 5, 3, 6, 7]
    k = 3
    queue: deque[int] = deque()   # stores indices; front = max of current window
    result = []

    for i, number in enumerate(nums):
        # Remove indices outside window
        while queue and queue[0] < i - k + 1:
            queue.popleft()
        # Maintain decreasing order
        while queue and nums[queue[-1]] < number:
            queue.pop()
        queue.append(i)
        if i >= k - 1:
            result.append(nums[queue[0]])

    assert result == [3, 3, 5, 5, 6, 7]

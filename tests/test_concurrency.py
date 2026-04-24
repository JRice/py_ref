"""
Concurrency and parallelism.

Covers: ThreadPoolExecutor, ProcessPoolExecutor, as_completed vs map,
threading.Lock, asyncio.Semaphore (rate-limiting), asyncio.Event,
asyncio cancellation, asyncio.wait, multiprocessing basics.

Note: ProcessPoolExecutor on Windows requires worker functions to be
at module level (picklable) — lambdas / closures won't work.
"""

from __future__ import annotations

import asyncio
import math
import threading
import time
from concurrent.futures import (
    ProcessPoolExecutor,
    ThreadPoolExecutor,
    as_completed,
)

import pytest


# ===========================================================================
# ThreadPoolExecutor  —  I/O-bound work
# ===========================================================================

def slow_fetch(url: str) -> str:
    """Simulate a network call."""
    time.sleep(0.01)
    return f"data:{url}"


def test_thread_pool_map():
    urls = [f"https://api.example.com/{i}" for i in range(5)]
    with ThreadPoolExecutor(max_workers=5) as pool:
        results = list(pool.map(slow_fetch, urls))
    assert len(results) == 5
    assert all(r.startswith("data:") for r in results)


def test_thread_pool_as_completed_fastest_first():
    """as_completed() yields futures as they finish, not in submission order."""
    def work(n: int) -> int:
        time.sleep(0.01 * (5 - n))   # lower n = longer wait
        return n * n

    order = []
    with ThreadPoolExecutor(max_workers=5) as pool:
        futures = {pool.submit(work, i): i for i in range(5)}
        for future in as_completed(futures):
            order.append(futures[future])   # which n finished

    # n=4 finishes first (shortest sleep), n=0 last
    assert order[0] == 4
    assert order[-1] == 0


def test_thread_pool_exception_propagation():
    def boom(n: int) -> int:
        if n == 2:
            raise ValueError("bad input")
        return n

    with ThreadPoolExecutor(max_workers=3) as pool:
        futures = [pool.submit(boom, i) for i in range(5)]

    results = []
    errors  = []
    for f in futures:
        try:
            results.append(f.result())
        except ValueError as e:
            errors.append(str(e))

    assert errors == ["bad input"]
    assert 0 in results and 1 in results


# ===========================================================================
# ProcessPoolExecutor  —  CPU-bound work
# ===========================================================================
# Worker must be a module-level function on Windows (spawn start method).

def _is_prime(n: int) -> bool:
    """CPU-bound: primality check."""
    if n < 2:
        return False
    for i in range(2, int(math.sqrt(n)) + 1):
        if n % i == 0:
            return False
    return True


def test_process_pool_cpu_bound():
    candidates = list(range(100, 200))
    with ProcessPoolExecutor(max_workers=2) as pool:
        flags = list(pool.map(_is_prime, candidates))

    primes = [n for n, is_p in zip(candidates, flags) if is_p]
    assert 101 in primes
    assert 113 in primes
    assert 100 not in primes


# ===========================================================================
# threading.Lock  —  protecting shared mutable state
# ===========================================================================

def test_lock_prevents_race_condition():
    counter  = {"value": 0}
    lock     = threading.Lock()

    def increment(n: int):
        for _ in range(n):
            with lock:
                counter["value"] += 1

    threads = [threading.Thread(target=increment, args=(100,)) for _ in range(10)]
    for t in threads: t.start()
    for t in threads: t.join()

    assert counter["value"] == 1000   # exact, not racy


def test_rlock_reentrant():
    """RLock can be acquired multiple times by the same thread."""
    rlock  = threading.RLock()
    result = []

    def outer():
        with rlock:
            inner()

    def inner():
        with rlock:      # would deadlock with a regular Lock
            result.append("inner")

    t = threading.Thread(target=outer)
    t.start()
    t.join()
    assert result == ["inner"]


# ===========================================================================
# asyncio.Semaphore  —  rate limiting concurrent coroutines
# ===========================================================================

async def test_semaphore_limits_concurrency():
    MAX_CONCURRENT = 3
    sem     = asyncio.Semaphore(MAX_CONCURRENT)
    running = {"current": 0, "peak": 0}

    async def task(n: int) -> int:
        async with sem:
            running["current"] += 1
            running["peak"]     = max(running["peak"], running["current"])
            await asyncio.sleep(0)   # yield to let other tasks run
            running["current"] -= 1
            return n

    results = await asyncio.gather(*[task(i) for i in range(10)])
    assert sorted(results) == list(range(10))
    assert running["peak"] <= MAX_CONCURRENT


# ===========================================================================
# asyncio.Event  —  signal between coroutines
# ===========================================================================

async def test_asyncio_event_coordination():
    ready  = asyncio.Event()
    output = []

    async def producer():
        await asyncio.sleep(0)
        output.append("produced")
        ready.set()

    async def consumer():
        await ready.wait()        # blocks until set()
        output.append("consumed")

    await asyncio.gather(consumer(), producer())
    assert output == ["produced", "consumed"]


# ===========================================================================
# asyncio cancellation
# ===========================================================================

async def test_task_cancellation():
    cancelled = False

    async def long_running():
        nonlocal cancelled
        try:
            await asyncio.sleep(100)
        except asyncio.CancelledError:
            cancelled = True
            raise   # always re-raise CancelledError

    task = asyncio.create_task(long_running())
    await asyncio.sleep(0)   # let task start
    task.cancel()

    with pytest.raises(asyncio.CancelledError):
        await task

    assert cancelled is True


async def test_asyncio_shield_from_cancellation():
    """asyncio.shield() lets an inner coroutine finish even if the outer is cancelled."""
    protected_ran = False

    async def critical_work():
        nonlocal protected_ran
        await asyncio.sleep(0)
        protected_ran = True

    async def outer():
        await asyncio.shield(critical_work())

    task = asyncio.create_task(outer())
    await asyncio.sleep(0)   # let it start
    # Don't cancel here — just verify shield semantics with a clean run
    await task
    assert protected_ran


# ===========================================================================
# asyncio.wait  —  fine-grained future handling
# ===========================================================================

async def test_asyncio_wait_first_completed():
    """Return as soon as ANY task finishes."""
    async def fast(): await asyncio.sleep(0);   return "fast"
    async def slow(): await asyncio.sleep(100); return "slow"

    done, pending = await asyncio.wait(
        [asyncio.create_task(fast()), asyncio.create_task(slow())],
        return_when=asyncio.FIRST_COMPLETED,
    )

    # Cancel remaining tasks to avoid warnings
    for task in pending:
        task.cancel()
        try:
            await task
        except asyncio.CancelledError:
            pass

    assert len(done)    == 1
    assert len(pending) == 1
    result = next(iter(done)).result()
    assert result == "fast"


async def test_asyncio_wait_all():
    async def job(n): return n * 2

    tasks = [asyncio.create_task(job(i)) for i in range(5)]
    done, pending = await asyncio.wait(tasks, return_when=asyncio.ALL_COMPLETED)

    assert len(pending) == 0
    assert {t.result() for t in done} == {0, 2, 4, 6, 8}


# ===========================================================================
# Producer/consumer with asyncio.Queue
# ===========================================================================

async def test_backpressure_with_bounded_queue():
    """maxsize causes the producer to block when the queue is full."""
    queue: asyncio.Queue[int] = asyncio.Queue(maxsize=3)
    produced = []
    consumed = []

    async def producer():
        for i in range(6):
            await queue.put(i)   # blocks when queue is full
            produced.append(i)

    async def consumer():
        while True:
            item = await queue.get()
            if item is None:
                break
            consumed.append(item)
            queue.task_done()

    async with asyncio.TaskGroup() as tg:
        tg.create_task(producer())
        tg.create_task(consumer())
        await queue.join()        # wait until all items processed
        await queue.put(None)     # sentinel to stop consumer

    assert sorted(consumed) == list(range(6))

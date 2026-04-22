"""
asyncio patterns and HTTP fetching with httpx.
All network calls are mocked so tests run offline and deterministically.
"""

import asyncio
import pytest
import httpx
import respx
from unittest.mock import AsyncMock, patch


# ---------------------------------------------------------------------------
# asyncio basics — gather
# ---------------------------------------------------------------------------

async def fetch_price(product_id: int) -> dict:
    """Simulates an async I/O-bound call."""
    await asyncio.sleep(0)   # yield control — simulates awaiting I/O
    return {"id": product_id, "price": product_id * 10.0}


async def test_asyncio_gather():
    """Run multiple coroutines concurrently; collect all results."""
    results = await asyncio.gather(
        fetch_price(1),
        fetch_price(2),
        fetch_price(3),
    )
    assert len(results) == 3
    assert results[0] == {"id": 1, "price": 10.0}
    assert results[2]["price"] == 30.0


async def test_asyncio_gather_error_handling():
    """return_exceptions=True prevents one failure from cancelling others."""
    async def may_fail(n):
        if n == 2:
            raise ValueError("bad input")
        return n * 10

    results = await asyncio.gather(
        may_fail(1), may_fail(2), may_fail(3),
        return_exceptions=True,
    )
    assert results[0] == 10
    assert isinstance(results[1], ValueError)
    assert results[2] == 30


# ---------------------------------------------------------------------------
# asyncio.TaskGroup (Python 3.11+) — structured concurrency
# ---------------------------------------------------------------------------

async def test_task_group_all_succeed():
    """TaskGroup cancels ALL sibling tasks if any raises — fail fast."""
    collected = []

    async def worker(name: str, delay: float):
        await asyncio.sleep(delay)
        collected.append(name)

    async with asyncio.TaskGroup() as tg:
        tg.create_task(worker("alpha", 0.0))
        tg.create_task(worker("beta",  0.0))
        tg.create_task(worker("gamma", 0.0))

    assert set(collected) == {"alpha", "beta", "gamma"}


async def test_task_group_propagates_exception():
    async def boom():
        raise RuntimeError("task failed")

    with pytest.raises(ExceptionGroup) as exc_info:
        async with asyncio.TaskGroup() as tg:
            tg.create_task(boom())

    assert any(isinstance(e, RuntimeError) for e in exc_info.value.exceptions)


# ---------------------------------------------------------------------------
# asyncio.timeout (Python 3.11+)
# ---------------------------------------------------------------------------

async def test_asyncio_timeout_fires():
    with pytest.raises(TimeoutError):
        async with asyncio.timeout(0.01):
            await asyncio.sleep(10)   # will be cancelled


async def test_asyncio_timeout_passes():
    async with asyncio.timeout(1.0):
        await asyncio.sleep(0)   # completes immediately


# ---------------------------------------------------------------------------
# Async generators
# ---------------------------------------------------------------------------

async def paginated_results(total: int, page_size: int):
    """Simulate paginated API — yields one page at a time."""
    for offset in range(0, total, page_size):
        await asyncio.sleep(0)   # simulate network hop
        yield list(range(offset, min(offset + page_size, total)))


async def test_async_generator():
    pages = []
    async for page in paginated_results(total=7, page_size=3):
        pages.append(page)

    assert pages == [[0,1,2], [3,4,5], [6]]


# ---------------------------------------------------------------------------
# httpx async client — mocked with respx
# ---------------------------------------------------------------------------

@respx.mock
async def test_httpx_get_json():
    respx.get("https://api.example.com/users/1").mock(
        return_value=httpx.Response(200, json={"id": 1, "name": "Alice"})
    )

    async with httpx.AsyncClient() as client:
        resp = await client.get("https://api.example.com/users/1")

    assert resp.status_code == 200
    assert resp.json() == {"id": 1, "name": "Alice"}


@respx.mock
async def test_httpx_concurrent_requests():
    for uid in range(1, 4):
        respx.get(f"https://api.example.com/users/{uid}").mock(
            return_value=httpx.Response(200, json={"id": uid})
        )

    async with httpx.AsyncClient() as client:
        results = await asyncio.gather(
            client.get("https://api.example.com/users/1"),
            client.get("https://api.example.com/users/2"),
            client.get("https://api.example.com/users/3"),
        )

    assert [r.json()["id"] for r in results] == [1, 2, 3]


@respx.mock
async def test_httpx_post_and_error_handling():
    respx.post("https://api.example.com/items").mock(
        return_value=httpx.Response(201, json={"created": True})
    )
    respx.get("https://api.example.com/missing").mock(
        return_value=httpx.Response(404, json={"error": "not found"})
    )

    async with httpx.AsyncClient() as client:
        created = await client.post("https://api.example.com/items", json={"name": "Widget"})
        missing = await client.get("https://api.example.com/missing")

    assert created.status_code == 201
    assert missing.status_code == 404


# ---------------------------------------------------------------------------
# Realistic scraping pattern — parse HTML with stdlib html.parser
# (mocked so no real network; shows the structure you'd use in production)
# ---------------------------------------------------------------------------

from html.parser import HTMLParser


class TitleParser(HTMLParser):
    """Extract the first <title> tag from an HTML document."""
    def __init__(self):
        super().__init__()
        self._in_title = False
        self.title: str | None = None

    def handle_starttag(self, tag, attrs):
        if tag == "title":
            self._in_title = True

    def handle_data(self, data):
        if self._in_title and self.title is None:
            self.title = data.strip()

    def handle_endtag(self, tag):
        if tag == "title":
            self._in_title = False


@respx.mock
async def test_scrape_page_title():
    html = b"""<!DOCTYPE html>
    <html><head><title>Python Is Great</title></head><body></body></html>"""

    respx.get("https://example.com").mock(
        return_value=httpx.Response(200, content=html, headers={"content-type": "text/html"})
    )

    async with httpx.AsyncClient() as client:
        resp = await client.get("https://example.com")

    parser = TitleParser()
    parser.feed(resp.text)
    assert parser.title == "Python Is Great"


# ---------------------------------------------------------------------------
# asyncio.Queue — producer / consumer pattern
# ---------------------------------------------------------------------------

async def test_asyncio_queue_producer_consumer():
    queue: asyncio.Queue[int] = asyncio.Queue()
    results: list[int] = []

    async def producer():
        for i in range(5):
            await queue.put(i)
        await queue.put(None)   # sentinel

    async def consumer():
        while True:
            item = await queue.get()
            if item is None:
                break
            results.append(item * 2)
            queue.task_done()

    await asyncio.gather(producer(), consumer())
    assert results == [0, 2, 4, 6, 8]

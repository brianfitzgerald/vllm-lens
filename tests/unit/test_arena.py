"""The shared-memory capture arena on a fake vLLM ring buffer: seal, read, release and
error markers. No GPU."""

import json

import pytest
import torch

from vllm_lens._helpers._arena import (
    HEADER_BYTES,
    capture_path,
    load_capture,
    read_capture,
    release_capture,
    write_marker,
)


@pytest.mark.parametrize(
    "positions", [None, [2, 5, 9]], ids=["every-position", "selected"]
)
def test_a_sealed_capture_reads_back_and_releases(arena, tmp_path, positions):
    """Rows written after the chunk header read back as ``[layers, rows, hidden]``, with
    the positions of a selector; release sets the released byte and removes the marker."""
    rows = torch.randn(3, 2, 4).to(torch.bfloat16)
    nbytes = HEADER_BYTES + rows.numel() * 2 + 8 * 3
    address, ident, offset = arena.allocate(nbytes)
    start = offset + HEADER_BYTES
    arena.bytes[start : start + rows.numel() * 2] = rows.view(torch.uint8).flatten()
    path = capture_path(str(tmp_path), "key", "req-1")
    arena.seal(
        path,
        address,
        ident,
        offset,
        {
            "dtype": "bfloat16",
            "rows": 3,
            "layers": [1, 4],
            "hidden": 4,
            "positions": None if positions is None else len(positions),
            "positions_offset": start + rows.numel() * 2,
        },
        positions,
        start + rows.numel() * 2,
    )
    capture = read_capture(path)
    assert torch.equal(capture["residual_stream"], rows.permute(1, 0, 2))
    assert capture.get("residual_stream_positions") == positions
    loaded = load_capture(path)
    assert torch.equal(loaded["residual_stream"], rows.permute(1, 0, 2))
    assert int(arena.bytes[offset + 1]) == 1
    assert not list(tmp_path.iterdir())


def test_a_reused_chunk_is_not_read(arena, tmp_path):
    """A marker whose chunk now holds another monotonic id fails instead of reading."""
    address, ident, offset = arena.allocate(HEADER_BYTES + 8)
    path = capture_path(str(tmp_path), "key", "req-2")
    header = {"dtype": "bfloat16", "rows": 0, "layers": [0], "hidden": 4}
    arena.seal(path, address, ident, offset, header, None, 0)
    with open(path) as file:
        marker = json.load(file)
    marker["id"] += 1
    write_marker(path, marker)
    with pytest.raises(RuntimeError, match="released or overwritten"):
        read_capture(path)
    release_capture(path)
    assert not list(tmp_path.iterdir())


def test_an_error_marker_raises(tmp_path):
    """A capture the worker could not hold reaches the reader as its error."""
    path = capture_path(str(tmp_path), "key", "req-3")
    write_marker(path, {"error": "needs 9 bytes"})
    with pytest.raises(RuntimeError, match="needs 9 bytes"):
        load_capture(path)
    assert not list(tmp_path.iterdir())


async def test_one_watcher_wakes_every_waiting_request(tmp_path):
    """Requests waiting for markers share one scan task, and each wakes when its own marker
    appears; a missing marker times out."""
    import asyncio

    from vllm_lens._helpers._arena import _WATCHERS, wait_capture_async

    paths = [capture_path(str(tmp_path), "key", f"req-{i}") for i in range(3)]
    waits = [asyncio.ensure_future(wait_capture_async(p, 5.0)) for p in paths]
    await asyncio.sleep(0.01)
    assert len(_WATCHERS) == 1 and not any(w.done() for w in waits)
    for path in paths[:2]:
        write_marker(path, {"error": "x"})
    await asyncio.wait_for(asyncio.gather(*waits[:2]), 1.0)
    assert not waits[2].done()
    waits[2].cancel()
    with pytest.raises(TimeoutError, match="was not written"):
        await wait_capture_async(str(tmp_path / "missing.lens"), 0.05)


def test_a_full_arena_of_running_captures_does_not_wait(arena):
    """When the oldest chunk belongs to a running request, waiting would stop it from
    finishing, so allocation returns None at once."""
    assert arena.allocate(arena.size // 2) is not None
    assert arena.allocate(arena.size // 2) is None

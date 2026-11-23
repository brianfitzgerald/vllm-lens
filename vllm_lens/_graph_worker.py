"""The vLLM worker of vllm-lens, on vLLM's V2 model runner with CUDA graphs. Ops inside the
graph steer each armed layer from a table indexed per token and write each captured token's
stream to a GPU ring; after each step a kernel copies the ring rows into a shared-memory
arena, one chunk per request."""

from __future__ import annotations

import collections
import logging
import os
import pickle
import queue
import threading
from typing import Any, NamedTuple

import numpy as np
import torch
from vllm import envs
from vllm.triton_utils import tl, triton
from vllm.utils.torch_utils import direct_register_custom_op
from vllm.v1.worker.gpu_worker import Worker

from vllm_lens._cudagraph import LensGraphConfig
from vllm_lens._helpers._arena import (
    CAPTURE_DIR_ENV,
    HEADER_BYTES,
    MAX_CHUNK_BYTES,
    Arena,
    capture_path,
    write_marker,
)
from vllm_lens._helpers._positions import LAST_PROMPT_TOKEN, POSITIONS_KEY
from vllm_lens._helpers._serialize import deserialize_tensor
from vllm_lens._helpers.types import SteeringVector
from vllm_lens._layers import _discover_layer_modules
from vllm_lens._route import COUNT, RING, SKIP, START, STEER, route_kernel

logger = logging.getLogger("vllm.lens")

# The ops live as long as this library, which is why it is held at module level.
_LIBRARY = torch.library.Library("vllm_lens", "FRAGMENT")


@triton.jit
def _capture_kernel(
    hidden,
    residual,
    ring,
    dest,
    sentinel,
    width,
    hidden_stride,
    residual_stride,
    ring_stride,
    HAS_RESIDUAL: tl.constexpr,
    BLOCK: tl.constexpr,
):
    """Write token ``t``'s stream to ring row ``dest[t]``, unless that row is the sentinel."""
    token = tl.program_id(0)
    row = tl.load(dest + token).to(tl.int64)
    cols = tl.program_id(1) * BLOCK + tl.arange(0, BLOCK)
    mask = (cols < width) & (row != sentinel)
    stream = tl.load(hidden + token * hidden_stride + cols, mask=mask)
    if HAS_RESIDUAL:
        added = stream.to(tl.float32) + tl.load(
            residual + token * residual_stride + cols, mask=mask
        ).to(tl.float32)
        stream = added.to(stream.dtype)
    tl.store(ring + row * ring_stride + cols, stream, mask=mask)


def _capture_op(
    hidden_states: torch.Tensor,
    residual: torch.Tensor | None,
    ring: torch.Tensor,
    dest: torch.Tensor,
) -> None:
    """Write each token's stream (``hidden + residual``) to its ring row in ``dest``; rows
    at the sentinel (the last ring row) are not read or written."""
    n, width = hidden_states.shape
    other = hidden_states if residual is None else residual
    if n == 0:
        return
    if (
        not hidden_states.is_cuda
        or hidden_states.stride(1) != 1
        or other.stride(1) != 1
    ):
        stream = hidden_states if residual is None else hidden_states + residual
        ring.index_copy_(0, dest[:n], stream)
        return
    block = 1024
    _capture_kernel[(n, triton.cdiv(width, block))](  # type: ignore[reportIndexIssue]
        hidden_states,
        other,
        ring,
        dest,
        ring.shape[0] - 1,
        width,
        hidden_states.stride(0),
        other.stride(0),
        ring.stride(0),
        HAS_RESIDUAL=residual is not None,
        BLOCK=block,
    )


direct_register_custom_op(
    "capture", _capture_op, mutates_args=["ring"], target_lib=_LIBRARY
)


def _steer_hook(module: Any, args: Any, output: Any) -> Any:
    """Add each token's table row to the layer output with plain torch ops, which Inductor
    fuses; row 0 is zero. A norm table row is scaled by the unsteered stream norm."""
    hidden, residual = output if isinstance(output, tuple) else (output, None)
    ids = module._lens_ids[: hidden.shape[0]]
    delta = module._lens_add.index_select(0, ids)
    if module._lens_norm is not None:
        stream = hidden if residual is None else hidden + residual
        norm = torch.linalg.vector_norm(
            stream, dim=-1, keepdim=True, dtype=torch.float32
        )
        delta = delta + module._lens_norm.index_select(0, ids) * norm
    if residual is None:
        return hidden + delta.to(hidden.dtype)
    # The same stream; Inductor fuses an add to the residual into the next layer's norm.
    return hidden, residual + delta.to(residual.dtype)


def _capture_hook(module: Any, args: Any, output: Any) -> None:
    """Call the capture op on the layer output; Dynamo traces this body into the graph."""
    hidden, residual = output if isinstance(output, tuple) else (output, None)
    torch.ops.vllm_lens.capture(hidden, residual, module._lens_ring, module._lens_dest)


class _Step(NamedTuple):
    """One forward step: each request's id, runner index, ``extra_args``, positions computed
    before the step and prompt length; request i has batch rows ``bounds[i]:bounds[i + 1]``."""

    req_ids: list[str]
    req_idx: np.ndarray
    extras: list[dict[str, Any] | None]
    computed: np.ndarray
    prompt_len: np.ndarray
    bounds: np.ndarray


class _Prepared(NamedTuple):
    """One request's steering as ``[steer layers, hidden]`` float32 rows: at every token and,
    with a position list, at those positions (which include the every-token rows)."""

    add: torch.Tensor
    norm: torch.Tensor | None
    pos_add: torch.Tensor | None
    pos_norm: torch.Tensor | None
    positions: np.ndarray | None


class _Capture:
    """One request's capture: its arena chunk and marker, its capture-layer indices, and its
    mode: every position, a sorted position list, or the mean over the prompt."""

    def __init__(
        self, marker: str, keep: list[int], extra: dict[str, Any], prompt_len: int
    ) -> None:
        self.marker = marker
        self.keep = keep
        self.prompt_len = prompt_len
        self.mean = extra.get("output_residual_stream_pool") == "mean"
        selector = extra.get(POSITIONS_KEY)
        self.selector: np.ndarray | None = None
        if selector is not None:
            self.selector = np.asarray(
                [prompt_len - 1] if selector == LAST_PROMPT_TOKEN else selector,
                dtype=np.int64,
            )
        # One past the last chunk row written so far, and the rows the chunk holds.
        self.rows = 0
        self.rows_max = 0
        # The chunk's address, monotonic id and data offset in the arena.
        self.chunk: tuple[int, int, int] = (0, 0, 0)
        # Why the capture has no chunk, if it has none.
        self.error: str | None = None
        # Whether every row this capture selects is written.
        self.done = False


class _Run(NamedTuple):
    """Ring rows ``[start, start + rows)`` that go to chunk rows from ``file_row``."""

    capture: _Capture
    start: int
    rows: int
    file_row: int


# The counts change every step; a specialization per value class would compile again.
@triton.jit(do_not_specialize=["n_runs", "n_tiles"])
def _gather_kernel(
    ring,
    arena,
    runs,
    keep,
    n_runs,
    n_tiles,
    ring_stride,
    width,
    max_layers,
    BLOCK: tl.constexpr,
):
    """Copy the step's ring rows to their chunks in the arena, one ``BLOCK`` of one layer of
    one row per tile; a few programs loop over the tiles, since the copy is bound by the
    link to host memory, and the other SMs stay with the forward pass. A run is
    ``(ring start, first step row, chunk element offset, layers kept)``."""
    blocks = tl.cdiv(width, BLOCK)
    for tile in range(tl.program_id(0), n_tiles, tl.num_programs(0)):
        row = tile // (max_layers * blocks)
        slot = (tile // blocks) % max_layers
        cols = (tile % blocks) * BLOCK + tl.arange(0, BLOCK)
        low = 0
        high = n_runs - 1
        while low < high:
            middle = (low + high + 1) // 2
            if tl.load(runs + middle * 4 + 1) <= row:
                low = middle
            else:
                high = middle - 1
        ring_start = tl.load(runs + low * 4)
        first = tl.load(runs + low * 4 + 1)
        dest = tl.load(runs + low * 4 + 2)
        kept = tl.load(runs + low * 4 + 3)
        if slot < kept:
            mask = cols < width
            layer = tl.load(keep + low * max_layers + slot)
            offset = row - first
            values = tl.load(
                ring + (ring_start + offset) * ring_stride + layer * width + cols,
                mask=mask,
            )
            tl.store(
                arena + dest + (offset * kept + slot) * width + cols, values, mask=mask
            )


# Programs of the gather kernel, which copies captures of a subset of the layers.
_GATHER_PROGRAMS = int(os.environ.get("VLLM_LENS_GATHER_PROGRAMS", "132"))


class _Drain:
    """The GPU ring and the arena: after each step, one kernel on a side stream copies the
    step's ring rows into the requests' arena chunks; a thread seals finished captures. The
    engine takes ring rows and waits only when the ring is full."""

    def __init__(
        self,
        ring: torch.Tensor,
        layers: list[int],
        hidden: int,
        max_reqs: int,
        arena_gib: float,
        timeout: float,
    ) -> None:
        self.ring = ring
        self.size = ring.shape[0] - 1
        self.layers = layers
        self.hidden = hidden
        self.arena = Arena(arena_gib, timeout)
        # The arena as elements of the model dtype, which the copies write.
        self.elements = self.arena.bytes.view(ring.dtype)
        if not self.arena.bytes.is_pinned():
            raise RuntimeError(
                "The registered capture arena is not pinned memory to torch, so copies "
                "into it would block the engine loop."
            )
        self.stream = torch.cuda.Stream(ring.device)
        # Each live ring segment: its rows and the event after which it is free.
        self.live: collections.deque[list[Any]] = collections.deque()
        self.head = 0
        # Two pinned copies of the run table, so a step never writes the one in use.
        self.host_runs = [
            torch.zeros((max_reqs, 4), dtype=torch.int64).pin_memory() for _ in range(2)
        ]
        self.host_keep = [
            torch.zeros((max_reqs, len(layers)), dtype=torch.int32).pin_memory()
            for _ in range(2)
        ]
        self.runs = torch.zeros((max_reqs, 4), dtype=torch.int64, device=ring.device)
        self.keep = torch.zeros(
            (max_reqs, len(layers)), dtype=torch.int32, device=ring.device
        )
        self.events: list[torch.cuda.Event | None] = [None, None]
        self.slot = 0
        self.seals: queue.SimpleQueue[Any] = queue.SimpleQueue()
        self.error: BaseException | None = None
        self.thread = threading.Thread(
            target=self._run, name="vllm-lens-seal", daemon=True
        )
        self.thread.start()

    def take(self, rows: int) -> int:
        """The first of ``rows`` free ring rows; waits for the oldest copies when the ring
        is full."""
        if rows > self.size:
            raise RuntimeError(
                f"A step captures {rows} rows, more than the {self.size} of "
                "capture_ring_tokens."
            )
        while True:
            while self.live and self.live[0][2] is not None and self.live[0][2].query():
                self.live.popleft()
            start = self.head if self.head + rows <= self.size else 0
            if all(end <= start or low >= start + rows for low, end, _ in self.live):
                break
            oldest = self.live[0][2]
            if oldest is None:
                raise RuntimeError("The capture ring holds an unfinished step")
            oldest.synchronize()
        self.live.append([start, start + rows, None])
        self.head = start + rows
        return start

    def drop(self) -> None:
        """Free the segment of a step that failed."""
        self.live.pop()

    def launch(self, after: torch.cuda.Event, runs: list[_Run]) -> None:
        """Copy the runs of the last segment to their chunks once ``after`` completes. A
        run of every capture layer is one copy on the GPU's copy engines, which leaves the
        SMs to the forward pass; runs of fewer layers share one gather kernel."""
        segment = self.live[-1]
        if not runs:
            segment[2] = after
            return
        every = len(self.layers)
        row_elements = every * self.hidden
        element = self.ring.element_size()
        partial = [run for run in runs if len(run.capture.keep) < every]
        with torch.cuda.stream(self.stream):
            self.stream.wait_event(after)
            for run in runs:
                if len(run.capture.keep) < every:
                    continue
                start = (run.capture.chunk[2] + HEADER_BYTES) // element
                start += run.file_row * row_elements
                count = run.rows * row_elements
                self.elements[start : start + count].copy_(
                    self.ring[run.start : run.start + run.rows].view(-1),
                    non_blocking=True,
                )
            if partial:
                self._gather(partial)
            done = torch.cuda.Event()
            done.record(self.stream)
        segment[2] = done

    def _gather(self, runs: list[_Run]) -> None:
        """Copy runs of fewer than every capture layer with one kernel, on the side stream."""
        slot = self.slot = self.slot ^ 1
        previous = self.events[slot]
        if previous is not None:
            previous.synchronize()
        table = self.host_runs[slot].numpy()
        keep = self.host_keep[slot].numpy()
        first = 0
        for i, run in enumerate(runs):
            kept = len(run.capture.keep)
            data = run.capture.chunk[2] + HEADER_BYTES
            dest = data // self.ring.element_size() + run.file_row * kept * self.hidden
            table[i] = (run.start, first, dest, kept)
            keep[i, :kept] = run.capture.keep
            first += run.rows
        n = len(runs)
        block = 1024
        self.runs[:n].copy_(self.host_runs[slot][:n], non_blocking=True)
        self.keep[:n].copy_(self.host_keep[slot][:n], non_blocking=True)
        tiles = first * len(self.layers) * triton.cdiv(self.hidden, block)
        _gather_kernel[(min(_GATHER_PROGRAMS, tiles),)](  # type: ignore[reportIndexIssue]
            self.ring,
            self.elements,
            self.runs,
            self.keep,
            n,
            tiles,
            self.ring.stride(0),
            self.hidden,
            len(self.layers),
            BLOCK=block,
        )
        done = torch.cuda.Event()
        done.record(self.stream)
        self.events[slot] = done

    def seal(
        self, capture: _Capture, rows: int, after: torch.cuda.Event | None
    ) -> None:
        """Queue the seal of a capture after its copies (and ``after``, for a mean row)."""
        with torch.cuda.stream(self.stream):
            if after is not None:
                self.stream.wait_event(after)
            done = torch.cuda.Event()
            done.record(self.stream)
        self.seals.put((capture, rows, done))

    def _seal(self, capture: _Capture, rows: int, done: torch.cuda.Event) -> None:
        """Write the header and the marker of one capture once its rows are copied."""
        if capture.error is not None:
            write_marker(capture.marker, {"error": capture.error})
            return
        done.synchronize()
        address, ident, offset = capture.chunk
        kept = len(capture.keep)
        row_bytes = kept * self.hidden * self.ring.element_size()
        positions = (
            None if capture.selector is None else capture.selector[:rows].tolist()
        )
        positions_offset = offset + HEADER_BYTES + capture.rows_max * row_bytes
        self.arena.seal(
            capture.marker,
            address,
            ident,
            offset,
            {
                "dtype": str(self.ring.dtype).removeprefix("torch."),
                "rows": rows,
                "layers": [self.layers[k] for k in capture.keep],
                "hidden": self.hidden,
                "positions": None if positions is None else len(positions),
                "positions_offset": positions_offset,
            },
            positions,
            positions_offset,
        )

    def _run(self) -> None:
        """Seal the finished captures in order."""
        torch.cuda.set_device(self.ring.device)
        try:
            while True:
                item = self.seals.get()
                if item is None:
                    return
                self._seal(*item)
        except BaseException as error:
            logger.exception("The vllm-lens seal thread failed")
            self.error = error


def _prepare(
    vectors: list[SteeringVector],
    tables: dict[tuple[str, int], torch.Tensor],
    steer_layers: list[int],
    hidden: int,
    device: torch.device,
    dtype: torch.dtype,
) -> _Prepared:
    """Sum the vectors of one request into float32 rows per steer layer; inline values are
    rounded to the model dtype first, as the model would hold them."""
    position = {layer: k for k, layer in enumerate(steer_layers)}
    rows = {
        name: torch.zeros(
            (len(steer_layers), hidden), dtype=torch.float32, device=device
        )
        for name in ("add", "norm", "pos_add", "pos_norm")
    }
    matched = False
    positions: list[int] | None = None
    for sv in vectors:
        if sv.activations is None:
            table = tables.get((sv.table or "", sv.version))
            if table is None or sv.row is None or sv.row >= table.shape[0]:
                raise ValueError(
                    f"Steering table {sv.table!r} row {sv.row} is not available"
                )
            values = table[sv.row : sv.row + 1].expand(len(sv.layer_indices), -1)
        else:
            values = sv.activations
            if device.type == "cuda" and not values.is_cuda:
                # A pinned copy does not wait for the work queued on the GPU.
                values = values.pin_memory().to(device, non_blocking=True)
            values = values.to(device=device, dtype=dtype).float()
        if values.dim() != 2 or values.shape[-1] != hidden:
            raise ValueError(
                f"vector shape {tuple(values.shape)}, model width {hidden}"
            )
        if sv.norm_match:
            values = values / (values.norm(dim=-1, keepdim=True) + 1e-6)
            matched = True
        values = values * sv.scale
        if sv.position_indices is not None:
            if positions is not None and positions != sv.position_indices:
                raise ValueError(
                    "The vectors of one request must share position_indices"
                )
            positions = sv.position_indices
        name = ("pos_" if sv.position_indices is not None else "") + (
            "norm" if sv.norm_match else "add"
        )
        for layer, index in sv.layer_index_map.items():
            rows[name][position[layer]] += values[index]
    if positions is None:
        return _Prepared(
            rows["add"], rows["norm"] if matched else None, None, None, None
        )
    return _Prepared(
        rows["add"],
        rows["norm"] if matched else None,
        rows["pos_add"] + rows["add"],
        rows["pos_norm"] + rows["norm"] if matched else None,
        np.asarray(sorted(set(positions)), dtype=np.int64),
    )


class GraphWorker(Worker):
    """The vLLM worker of an engine with armed layers: wraps V2's runner so each step plans
    its steering ids and capture rows before the forward and hands the rows to the drain."""

    def load_model(self, *, load_dummy_weights: bool = False) -> None:
        """Load the model, then set up the layers this engine's config names."""
        super().load_model(load_dummy_weights=load_dummy_weights)
        config = LensGraphConfig.from_vllm_config(self.vllm_config)
        runner: Any = self.model_runner
        compilation = self.vllm_config.compilation_config
        parallel = self.parallel_config
        # Each of these splits or reorders a forward pass in a way the step view does not follow.
        refused = {
            "the V1 model runner": getattr(runner, "req_states", None) is None,
            "pipeline parallelism": parallel.pipeline_parallel_size > 1,
            "micro-batching (DBO)": parallel.use_ubatching,
            "prefill context parallelism": getattr(
                parallel, "prefill_context_parallel_size", 1
            )
            > 1,
            "adaptive draft verification": getattr(
                runner, "adaptive_verification", None
            )
            is not None,
            "sequence parallelism": bool(compilation.pass_config.enable_sp),
            "Inductor graph partition": compilation.use_inductor_graph_partition,
            "breakable CUDA graphs": envs.VLLM_USE_BREAKABLE_CUDAGRAPH,
        }
        for name, used in refused.items():
            if used:
                raise RuntimeError(f"vllm-lens does not support {name}.")
        layer_map = _discover_layer_modules(self)
        missing = sorted(
            {*config.capture_layers, *config.steer_layers} - layer_map.keys()
        )
        if missing:
            raise ValueError(
                f"additional_config['vllm_lens'] names layer(s) {missing}, but the "
                f"model has layers 0..{len(layer_map) - 1}."
            )
        max_tokens = runner.scheduler_config.max_num_batched_tokens
        max_reqs = runner.max_num_reqs
        hidden = runner.model_config.get_hidden_size()
        dtype = runner.model_config.dtype
        device = self.device
        tp = parallel.tensor_parallel_size
        # The stream is the same on every tensor-parallel rank, so rank 0 alone captures.
        capture_layers = list(config.capture_layers) if self.rank % tp == 0 else []
        steer_layers = list(config.steer_layers)
        self._lens_dtype = dtype
        self._lens_hidden = hidden
        self._steer_layers = steer_layers
        self._capture_layers = capture_layers
        self._steering_data: dict[str, _Prepared] = {}
        self._steering_tables: dict[tuple[str, int], torch.Tensor] = {}
        # The prepared rows of each distinct ``_steer_rows`` value.
        self._row_cache: dict[str, _Prepared] = {}
        self._captures: dict[str, _Capture] = {}
        self._capture_dir = os.environ.get(CAPTURE_DIR_ENV, "")
        if capture_layers and not os.path.isdir(self._capture_dir):
            raise RuntimeError(
                f"{CAPTURE_DIR_ENV}={self._capture_dir!r} is not a directory"
            )
        # Rows 2i + 1 and 2i + 2 steer runner index i at every token and at its positions.
        self._add = torch.zeros(
            (2 * max_reqs + 1, len(steer_layers), hidden),
            dtype=torch.float32 if config.steer_norm_match else dtype,
            device=device,
        )
        self._norm = (
            torch.zeros_like(self._add, dtype=torch.float32)
            if config.steer_norm_match and steer_layers
            else None
        )
        self._every = np.zeros(max_reqs, dtype=np.int64)
        self._positional: dict[int, np.ndarray] = {}
        self._mean = torch.zeros(
            (max_reqs if capture_layers else 0, len(capture_layers), hidden),
            dtype=torch.float32,
            device=device,
        )
        ring_rows = config.capture_ring_tokens or 2 * max_tokens
        ring = torch.zeros(
            (ring_rows + 1 if capture_layers else 1, len(capture_layers), hidden),
            dtype=dtype,
            device=device,
        )
        self._drain = (
            _Drain(
                ring,
                capture_layers,
                hidden,
                max_reqs,
                config.capture_arena_gib,
                float(os.environ.get("VLLM_LENS_CAPTURE_TIMEOUT", "120")),
            )
            if capture_layers
            else None
        )
        self._max_model_len = runner.model_config.max_model_len
        self._sentinel = ring.shape[0] - 1
        # Row 0 holds each token's table row, row 1 its ring row.
        self._meta = torch.tensor(
            [[0], [self._sentinel]], dtype=torch.int64, device=device
        ).repeat(1, max_tokens)
        # Each step's plan, one column per request (see _route), and its patches: tokens
        # and values for position-list steering rows and ring rows. Two pinned copies, so a
        # step never writes the one the last copy reads.
        self._plan_table = torch.zeros((5, max_reqs), dtype=torch.int64, device=device)
        self._host_plan = [
            torch.zeros((5, max_reqs), dtype=torch.int64).pin_memory() for _ in range(2)
        ]
        self._patches = torch.zeros(
            (2, 2, max_tokens), dtype=torch.int64, device=device
        )
        self._host_patches = [
            torch.zeros((2, 2, max_tokens), dtype=torch.int64).pin_memory()
            for _ in range(2)
        ]
        self._meta_events = [torch.cuda.Event() for _ in range(2)]
        self._meta_slot = 0
        self._meta_length = 0
        self._meta_signature: tuple | None = None

        extras: dict[str, dict[str, Any]] = {}
        pending: dict[str, Any] = {"runs": None, "pooled": []}
        add_requests = runner.add_requests
        finish_requests = runner.finish_requests
        prepare_inputs = runner.prepare_inputs
        execute_model = runner.execute_model

        def _finish_requests(scheduler_output: Any) -> Any:
            """Seal the captures of finished requests and drop their state; a preempted
            request keeps its capture and gets its rows again when it is added back."""
            preempted = scheduler_output.preempted_req_ids or set()
            for req_id in scheduler_output.finished_req_ids | preempted:
                index = runner.req_states.req_id_to_index.get(req_id)
                if index is not None:
                    self._every[index] = 0
                    self._positional.pop(index, None)
                if req_id in preempted:
                    continue
                extras.pop(req_id, None)
                capture = self._captures.pop(req_id, None)
                if capture is not None:
                    self._seal(capture, index)
            return finish_requests(scheduler_output)

        def _add_requests(scheduler_output: Any) -> Any:
            """Keep each new request's extra_args, which V2 drops, and write its table rows."""
            out = add_requests(scheduler_output)
            for new in scheduler_output.scheduled_new_reqs:
                extra = getattr(new.sampling_params, "extra_args", None)
                if not extra:
                    continue
                extras[new.req_id] = extra
                index = runner.req_states.req_id_to_index[new.req_id]
                self._admit(
                    new.req_id,
                    index,
                    extra,
                    new.prompt_len,
                    getattr(new.sampling_params, "max_tokens", None),
                )
            return out

        def _prepare_inputs(*args: Any, **kwargs: Any) -> Any:
            """Build the step from V2's input batch and plan its ids and ring rows."""
            input_batch = prepare_inputs(*args, **kwargs)
            n = int(input_batch.num_reqs)
            req_ids = list(input_batch.req_ids[:n])
            req_idx = input_batch.idx_mapping_np[:n]
            pending["runs"], pending["pooled"] = self._plan(
                _Step(
                    req_ids,
                    req_idx,
                    [extras.get(req_id) for req_id in req_ids],
                    input_batch.num_computed_tokens_np[:n],
                    runner.req_states.prompt_len.np[req_idx],
                    input_batch.query_start_loc_np[: n + 1],
                )
            )
            return input_batch

        def _execute_model(*args: Any, **kwargs: Any) -> Any:
            """Run V2's step, then add its prompt rows to the means and copy its ring rows
            to the arena."""
            pending["runs"], pending["pooled"] = None, []
            try:
                out = execute_model(*args, **kwargs)
            except BaseException:
                if pending["runs"] is not None:
                    assert self._drain is not None
                    self._drain.drop()
                raise
            runs = pending["runs"]
            if runs is not None:
                assert self._drain is not None
                ring = self._drain.ring
                for index, start, count, reset in pending["pooled"]:
                    if reset:
                        self._mean[index].zero_()
                    self._mean[index] += ring[start : start + count].float().sum(0)
                # A dummy forward pass before the next plan writes no ring rows.
                self._meta[1, : self._meta_length].fill_(self._sentinel)
                event = torch.cuda.Event()
                event.record()
                self._drain.launch(event, runs)
            return out

        runner.add_requests = _add_requests
        runner.finish_requests = _finish_requests
        runner.prepare_inputs = _prepare_inputs
        runner.execute_model = _execute_model
        from vllm.model_executor.layers.layernorm import RMSNorm

        import vllm_lens._fused_norm  # noqa: F401

        last = max(layer_map)
        final_norm = None
        for module in runner.model.modules():
            layers: Any = getattr(module, "layers", None)
            if isinstance(layers, torch.nn.ModuleList) and any(
                member is layer_map[last] for member in layers
            ):
                final_norm = getattr(module, "norm", None)
        fused: list[int] = []
        dummy = torch.zeros((1, hidden), dtype=dtype, device=device)
        for layer_idx in sorted({*steer_layers, *capture_layers}):
            following: Any = (
                final_norm
                if layer_idx == last
                else getattr(layer_map.get(layer_idx + 1), "input_layernorm", None)
            )
            if (
                os.environ.get("VLLM_LENS_FUSED_NORM", "1") == "0"
                or type(following) is not RMSNorm
                or following.variance_size_override is not None
                or not following.has_weight
            ):
                continue
            steer = steer_layers.index(layer_idx) if layer_idx in steer_layers else None
            capture = (
                capture_layers.index(layer_idx) if layer_idx in capture_layers else None
            )
            following.register_buffer("_lens_ids", self._meta[0], persistent=False)
            following.register_buffer("_lens_dest", self._meta[1], persistent=False)
            following.register_buffer(
                "_lens_add",
                None if steer is None else self._add[:, steer],
                persistent=False,
            )
            following.register_buffer(
                "_lens_norm",
                None if steer is None or self._norm is None else self._norm[:, steer],
                persistent=False,
            )
            following.register_buffer(
                "_lens_ring",
                dummy if capture is None else ring[:, capture],
                persistent=False,
            )
            following._lens_capture = capture is not None
            original = following.forward

            def forward(
                x: torch.Tensor,
                residual: torch.Tensor | None = None,
                _module: Any = following,
                _original: Any = original,
            ) -> Any:
                """The next norm, with the armed layer's steering and capture fused in."""
                if residual is None:
                    return _original(x)
                return torch.ops.vllm_lens.lens_norm(
                    x,
                    residual,
                    _module.weight,
                    _module.variance_epsilon,
                    _module._lens_add,
                    _module._lens_norm,
                    _module._lens_ids,
                    _module._lens_ring,
                    _module._lens_dest,
                    _module._lens_capture,
                )

            following.forward = forward
            fused.append(layer_idx)
        for k, layer_idx in enumerate(steer_layers):
            if layer_idx in fused:
                continue
            layer = layer_map[layer_idx]
            layer.register_buffer("_lens_ids", self._meta[0], persistent=False)
            layer.register_buffer("_lens_add", self._add[:, k], persistent=False)
            norm = None if self._norm is None else self._norm[:, k]
            layer.register_buffer("_lens_norm", norm, persistent=False)
            # Registered first, so a layer steers before it captures.
            layer.register_forward_hook(_steer_hook)
        for k, layer_idx in enumerate(capture_layers):
            if layer_idx in fused:
                continue
            layer = layer_map[layer_idx]
            layer.register_buffer("_lens_ring", ring[:, k], persistent=False)
            layer.register_buffer("_lens_dest", self._meta[1], persistent=False)
            layer.register_forward_hook(_capture_hook)
        logger.info(
            "vllm-lens: steer layers %s (norm_match %s), capture layers %s, ring %d rows, "
            "fused into the next norm: %s",
            steer_layers,
            config.steer_norm_match,
            capture_layers,
            self._sentinel,
            fused,
        )

    def _admit(
        self,
        req_id: str,
        index: int,
        extra: dict[str, Any],
        prompt_len: int,
        max_tokens: int | None = None,
    ) -> None:
        """Write a new request's table rows, and allocate its capture chunk."""
        prepared = None
        key = extra.get("_steering_id")
        if key is not None:
            prepared = self._steering_data.get(key)
            if prepared is None:
                logger.error("Steering data of request %s is not available", req_id)
        elif extra.get("_steer_rows"):
            rows_key = repr(extra["_steer_rows"])
            prepared = self._row_cache.get(rows_key)
            if prepared is None:
                try:
                    prepared = self._prepare_vectors(
                        [
                            SteeringVector(
                                table=name,
                                version=version,
                                row=row,
                                scale=scale,
                                layer_indices=layers,
                                norm_match=norm_match,
                            )
                            for name, version, row, scale, layers, norm_match in extra[
                                "_steer_rows"
                            ]
                        ]
                    )
                except ValueError:
                    logger.exception("Steering of request %s is not available", req_id)
                else:
                    if len(self._row_cache) >= 1024:
                        self._row_cache.clear()
                    self._row_cache[rows_key] = prepared
        if prepared is not None:
            # These copies run on the compute stream after the steps of the request that
            # held this index before, so they never change a step in flight.
            for offset, add, norm in (
                (1, prepared.add, prepared.norm),
                (2, prepared.pos_add, prepared.pos_norm),
            ):
                if add is None:
                    continue
                self._add[2 * index + offset].copy_(add)
                if self._norm is not None:
                    if norm is None:
                        self._norm[2 * index + offset].zero_()
                    else:
                        self._norm[2 * index + offset].copy_(norm)
            self._every[index] = 2 * index + 1
            if prepared.positions is not None:
                self._positional[index] = prepared.positions
        wanted = extra.get("output_residual_stream")
        key = extra.get("_capture_key")
        if wanted is None or key is None or self._drain is None:
            return
        if req_id in self._captures:
            # A resumed request computes its positions again.
            capture = self._captures[req_id]
            capture.done = capture.error is not None
        else:
            keep = [
                k
                for k, layer in enumerate(self._capture_layers)
                if not isinstance(wanted, list) or layer in wanted
            ]
            # vLLM's internal id is the external id and an 8-character suffix.
            capture = self._captures[req_id] = _Capture(
                capture_path(self._capture_dir, key, req_id[:-9]),
                keep,
                extra,
                prompt_len,
            )
            if capture.mean:
                capture.rows_max = 1
            elif capture.selector is not None:
                capture.rows_max = len(capture.selector)
            else:
                generated = self._max_model_len if max_tokens is None else max_tokens
                capture.rows_max = min(prompt_len + generated, self._max_model_len) - 1
            row_bytes = len(keep) * self._lens_hidden * self._lens_dtype.itemsize
            nbytes = HEADER_BYTES + capture.rows_max * row_bytes
            if capture.selector is not None:
                nbytes += 8 * len(capture.selector)
            limit = min(MAX_CHUNK_BYTES, self._drain.arena.size // 4)
            if nbytes > limit:
                capture.error = (
                    f"The capture of request {req_id[:-9]} needs {nbytes} bytes, more than "
                    f"the {limit} one chunk may hold: lower max_tokens, capture fewer "
                    "layers or raise capture_arena_gib."
                )
                capture.done = True
            else:
                chunk = self._drain.arena.allocate(nbytes)
                if chunk is None:
                    capture.error = (
                        f"The capture of request {req_id[:-9]} does not fit the arena, "
                        "which holds the captures of running requests: raise "
                        "capture_arena_gib or capture fewer requests at once."
                    )
                    capture.done = True
                    logger.warning(capture.error)
                else:
                    capture.chunk = chunk
        self._mean[index].zero_()

    def _plan(
        self, step: _Step
    ) -> tuple[list[_Run] | None, list[tuple[int, int, int, bool]]]:
        """Fill this step's per-token table and ring rows and copy them to the device; return
        the ring runs to write (None without capture rows) and each mean's prompt rows."""
        bounds = step.bounds
        n = int(bounds[-1])
        captured = [
            (i, self._captures[req_id])
            for i, req_id in enumerate(step.req_ids)
            if req_id in self._captures and not self._captures[req_id].done
        ]
        positional = [
            i for i, index in enumerate(step.req_idx) if int(index) in self._positional
        ]
        signature = (tuple(step.req_ids), bounds.tobytes())
        if not captured and not positional and signature == self._meta_signature:
            return None, []
        self._meta_signature = None if captured or positional else signature
        counts = np.diff(bounds)
        n_reqs = len(step.req_ids)
        slot = self._meta_slot = self._meta_slot ^ 1
        self._meta_events[slot].synchronize()
        plan = self._host_plan[slot].numpy()
        patches = self._host_patches[slot].numpy()
        plan[STEER, :n_reqs] = self._every[step.req_idx]
        plan[RING, :n_reqs] = -1
        plan[SKIP, :n_reqs] = 0
        plan[COUNT, :n_reqs] = 0
        plan[START, :n_reqs] = bounds[:-1]
        # The number of steering-row and ring-row patches.
        patched = [0, 0]
        for i in positional:
            index = int(step.req_idx[i])
            first = int(step.computed[i])
            positions = self._positional[index]
            inside = positions[(positions >= first) & (positions < first + counts[i])]
            k = patched[0]
            patches[0, 0, k : k + len(inside)] = bounds[i] + inside - first
            patches[0, 1, k : k + len(inside)] = 2 * index + 2
            patched[0] += len(inside)
        runs: list[_Run] | None = None
        pooled: list[tuple[int, int, int, bool]] = []
        # Each capture's batch row, its batch rows (None for a run from the row's start, or
        # explicit rows), count, file row (-1 for a mean), runner index and whether its mean
        # starts again.
        wanted: list[tuple[_Capture, int, np.ndarray | None, int, int, int, bool]] = []
        for i, capture in captured:
            first = int(step.computed[i])
            count = int(counts[i])
            if capture.mean:
                high = min(first + count, capture.prompt_len)
                capture.done = high >= capture.prompt_len
                if high > first:
                    index = int(step.req_idx[i])
                    wanted.append(
                        (capture, i, None, high - first, -1, index, first == 0)
                    )
            elif capture.selector is None:
                count = min(count, capture.rows_max - first)
                if count > 0:
                    wanted.append((capture, i, None, count, first, 0, False))
                    capture.rows = max(capture.rows, first + count)
            else:
                low = int(np.searchsorted(capture.selector, first))
                high = int(np.searchsorted(capture.selector, first + count))
                capture.done = high == len(capture.selector)
                if high > low:
                    rows = int(bounds[i]) + capture.selector[low:high] - first
                    wanted.append((capture, i, rows, high - low, low, 0, False))
                    capture.rows = max(capture.rows, high)
        total = sum(item[3] for item in wanted)
        if total:
            assert self._drain is not None
            offset = self._drain.take(total)
            runs = []
            for capture, i, rows, count, file_row, index, reset in wanted:
                if rows is None or count == 1:
                    plan[RING, i] = offset
                    plan[SKIP, i] = 0 if rows is None else rows[0] - bounds[i]
                    plan[COUNT, i] = count
                else:
                    k = patched[1]
                    patches[1, 0, k : k + count] = rows
                    patches[1, 1, k : k + count] = np.arange(offset, offset + count)
                    patched[1] += count
                if file_row < 0:
                    pooled.append((index, offset, count, reset))
                else:
                    runs.append(_Run(capture, offset, count, file_row))
                offset += count
        length = max(n, self._meta_length)
        self._plan_table.copy_(self._host_plan[slot], non_blocking=True)
        route_kernel[(triton.cdiv(length, 256),)](  # type: ignore[reportIndexIssue]
            self._plan_table,
            self._meta,
            self._plan_table.stride(0),
            self._meta.stride(0),
            n_reqs,
            n,
            length,
            self._sentinel,
            STEPS=max(1, self._plan_table.shape[1].bit_length()),
            BLOCK=256,
        )
        for kind, count in enumerate(patched):
            if count:
                self._patches[kind, :, :count].copy_(
                    self._host_patches[slot][kind, :, :count], non_blocking=True
                )
                self._meta[kind].index_put_(
                    (self._patches[kind, 0, :count],), self._patches[kind, 1, :count]
                )
        self._meta_events[slot].record()
        self._meta_length = n
        return runs, pooled

    def _seal(self, capture: _Capture, index: int | None) -> None:
        """Queue the seal of a finished request's capture after its rows; a mean is divided
        and copied into its chunk here, on the compute stream."""
        assert self._drain is not None
        rows = capture.rows
        after = None
        if capture.mean and index is not None and capture.error is None:
            values = (self._mean[index, capture.keep] / capture.prompt_len).to(
                self._lens_dtype
            )
            start = (capture.chunk[2] + HEADER_BYTES) // self._lens_dtype.itemsize
            target = self._drain.elements[start : start + values.numel()]
            target.copy_(values.flatten(), non_blocking=True)
            after = torch.cuda.Event()
            after.record()
            rows = 1
        self._drain.seal(capture, rows, after)

    def _prepare_vectors(self, vectors: list[SteeringVector]) -> _Prepared:
        """Prepare one request's vectors for this worker's steer layers."""
        return _prepare(
            vectors,
            self._steering_tables,
            self._steer_layers,
            self._lens_hidden,
            self.device,  # type: ignore
            self._lens_dtype,
        )

    def set_steering_data(self, key: str, pickled_data: bytes) -> None:
        """Prepare the inline steering vectors of one request."""
        self._steering_data[key] = self._prepare_vectors(pickle.loads(pickled_data))

    def clear_steering_data(self, key: str) -> None:
        """Drop the prepared vectors of a completed request."""
        self._steering_data.pop(key, None)

    def set_steering_table(
        self, name: str, payload: dict[str, Any], version: int = 0
    ) -> None:
        """Hold the ``(N, hidden)`` rows of a ``serialize_tensor`` payload as version
        ``version`` of table ``name``, rounded to the model dtype as a vector is."""
        self._steering_tables[(name, version)] = (
            deserialize_tensor(payload)
            .to(device=self.device, dtype=self._lens_dtype)
            .float()
        )

    def clear_steering_table(
        self, name: str | None, version: int | None = None
    ) -> None:
        """Drop one version of table ``name``, every version of it, or every table."""
        self._row_cache.clear()
        self._steering_tables = {
            key: table
            for key, table in self._steering_tables.items()
            if name is not None
            and (key[0] != name or (version is not None and key[1] != version))
        }

    def _debug_layer_discovery(self) -> list[int]:
        """Global indices of the decoder layers found by discovery (testing)."""
        return sorted(_discover_layer_modules(self))

"""The vLLM worker of vllm-lens, on vLLM's V2 model runner with CUDA graphs. Ops inside the
graph steer each armed layer from a table indexed per token and write each captured token's
stream to a GPU ring; a thread copies the ring rows to one file per request."""

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
from vllm.utils.torch_utils import direct_register_custom_op
from vllm.v1.worker.gpu_worker import Worker

from vllm_lens._cudagraph import LensGraphConfig
from vllm_lens._helpers._capture_file import (
    CAPTURE_DIR_ENV,
    HEADER_BYTES,
    capture_path,
    seal,
)
from vllm_lens._helpers.types import SteeringVector
from vllm_lens._layers import _discover_layer_modules

logger = logging.getLogger("vllm.lens")

# The ops live as long as this library, which is why it is held at module level.
_LIBRARY = torch.library.Library("vllm_lens", "FRAGMENT")


def _capture_op(
    hidden_states: torch.Tensor,
    residual: torch.Tensor | None,
    ring: torch.Tensor,
    dest: torch.Tensor,
) -> None:
    """Write each token's stream (``hidden + residual``) to its ring row in ``dest``."""
    n = hidden_states.shape[0]
    stream = hidden_states if residual is None else hidden_states + residual
    ring.index_copy_(0, dest[:n], stream)


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
    hidden = hidden + delta.to(hidden.dtype)
    return hidden if residual is None else (hidden, residual)


def _capture_hook(module: Any, args: Any, output: Any) -> None:
    """Call the capture op on the layer output; Dynamo traces this body into the graph."""
    hidden, residual = output if isinstance(output, tuple) else (output, None)
    torch.ops.vllm_lens.capture(hidden, residual, module._lens_ring, module._lens_dest)


class _Step(NamedTuple):
    """One forward step: each request's id, runner index, ``extra_args`` and positions
    computed before the step; request i has batch rows ``bounds[i]:bounds[i + 1]``."""

    req_ids: list[str]
    req_idx: np.ndarray
    extras: list[dict[str, Any] | None]
    computed: np.ndarray
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
    """One request's capture: its partial file and its capture-layer indices."""

    def __init__(self, path: str, keep: list[int]) -> None:
        self.path = path
        self.fd = os.open(path, os.O_RDWR | os.O_CREAT | os.O_TRUNC, 0o644)
        self.keep = keep
        # One past the last file row written so far.
        self.rows = 0


class _Run(NamedTuple):
    """Ring rows ``[start, start + rows)`` that go to file rows from ``file_row``."""

    capture: _Capture
    start: int
    rows: int
    file_row: int


class _Drain:
    """The GPU ring and the thread that copies its rows to the request files in step order,
    on its own CUDA stream; the engine takes ring rows and waits when too few are free."""

    def __init__(self, ring: torch.Tensor, layers: list[int], hidden: int) -> None:
        self.ring = ring
        self.size = ring.shape[0] - 1
        self.layers = layers
        self.hidden = hidden
        row_bytes = ring[0].numel() * ring.element_size()
        self.slab_rows = max(1, min(self.size, (64 << 20) // row_bytes))
        self.slabs = [
            torch.empty(
                (self.slab_rows, *ring.shape[1:]), dtype=ring.dtype
            ).pin_memory()
            for _ in range(2)
        ]
        self.events = [torch.cuda.Event() for _ in range(2)]
        self.stream = torch.cuda.Stream(ring.device)
        self.queue: queue.SimpleQueue[Any] = queue.SimpleQueue()
        self.free = threading.Condition()
        self.live: collections.deque[tuple[int, int]] = collections.deque()
        self.head = 0
        self.error: BaseException | None = None
        self.thread = threading.Thread(
            target=self._run, name="vllm-lens-drain", daemon=True
        )
        self.thread.start()

    def take(self, rows: int) -> int:
        """The first of ``rows`` free ring rows; waits for the thread to free rows."""
        if rows > self.size:
            raise RuntimeError(
                f"A step captures {rows} rows, more than the {self.size} of "
                "capture_ring_tokens."
            )
        with self.free:
            while True:
                if self.error is not None:
                    raise RuntimeError(
                        "The vllm-lens capture thread failed"
                    ) from self.error
                start = self.head if self.head + rows <= self.size else 0
                if all(end <= start or low >= start + rows for low, end in self.live):
                    break
                self.free.wait()
            self.live.append((start, start + rows))
            self.head = start + rows
            return start

    def release(self) -> None:
        """Free the oldest ring segment."""
        with self.free:
            self.live.popleft()
            self.free.notify_all()

    def _write(self, slab: torch.Tensor, run: _Run, offset: int) -> None:
        """Write the slab rows of ``run``, from row ``offset`` of the run, to its file."""
        rows = slab.view(torch.uint8).numpy()
        keep = run.capture.keep
        if len(keep) < len(self.layers):
            rows = rows.reshape(rows.shape[0], len(self.layers), -1)[:, keep]
        data = memoryview(np.ascontiguousarray(rows)).cast("B")
        row_bytes = len(keep) * self.hidden * self.ring.element_size()
        position = HEADER_BYTES + (run.file_row + offset) * row_bytes
        while data:
            written = os.pwrite(run.capture.fd, data, position)
            data = data[written:]
            position += written

    def _copy(self, event: torch.cuda.Event, runs: list[_Run]) -> None:
        """Copy the runs of one step to their files through the two pinned slabs, so one
        slab is written while the other is copied."""
        self.stream.wait_event(event)
        pending: tuple[int, int, _Run, int] | None = None
        slot = 0
        for run in runs:
            for offset in range(0, run.rows, self.slab_rows):
                count = min(self.slab_rows, run.rows - offset)
                slot ^= 1
                low = run.start + offset
                with torch.cuda.stream(self.stream):
                    self.slabs[slot][:count].copy_(
                        self.ring[low : low + count], non_blocking=True
                    )
                    self.events[slot].record(self.stream)
                if pending is not None:
                    self._finish(*pending)
                pending = (slot, count, run, offset)
        if pending is not None:
            self._finish(*pending)

    def _finish(self, slot: int, count: int, run: _Run, offset: int) -> None:
        """Wait for one slab copy, then write it."""
        self.events[slot].synchronize()
        self._write(self.slabs[slot][:count], run, offset)

    def _seal(self, capture: _Capture, rows: int) -> None:
        """Seal a file after its rows."""
        try:
            seal(
                capture.fd,
                capture.path,
                dtype=self.ring.dtype,
                rows=rows,
                layers=[self.layers[k] for k in capture.keep],
                hidden=self.hidden,
            )
        except FileNotFoundError:
            # The front end removed the file of an aborted request.
            pass

    def _run(self) -> None:
        """Process the steps and seals in order."""
        torch.cuda.set_device(self.ring.device)
        try:
            while True:
                item = self.queue.get()
                if item is None:
                    return
                if isinstance(item[0], _Capture):
                    self._seal(*item)
                    continue
                event, runs = item
                if runs:
                    self._copy(event, runs)
                else:
                    event.synchronize()
                self.release()
        except BaseException as error:
            logger.exception("The vllm-lens capture thread failed")
            with self.free:
                self.error = error
                self.free.notify_all()


def _prepare(
    vectors: list[SteeringVector],
    steer_layers: list[int],
    hidden: int,
    device: torch.device,
    dtype: torch.dtype,
) -> _Prepared:
    """Sum the vectors of one request into float32 rows per steer layer; the values are
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
        ring_rows = config.capture_ring_tokens or 2 * max_tokens
        ring = torch.zeros(
            (ring_rows + 1 if capture_layers else 1, len(capture_layers), hidden),
            dtype=dtype,
            device=device,
        )
        self._drain = _Drain(ring, capture_layers, hidden) if capture_layers else None
        self._sentinel = ring.shape[0] - 1
        # Row 0 holds each token's table row, row 1 its ring row.
        self._meta = torch.tensor(
            [[0], [self._sentinel]], dtype=torch.int64, device=device
        ).repeat(1, max_tokens)
        # Two pinned copies, so a step never writes the one the last copy reads.
        self._host_meta = [self._meta.cpu().pin_memory() for _ in range(2)]
        self._meta_events = [torch.cuda.Event() for _ in range(2)]
        self._meta_slot = 0
        self._meta_length = 0
        self._meta_signature: tuple | None = None

        extras: dict[str, dict[str, Any]] = {}
        pending: dict[str, Any] = {"runs": None}
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
                    self._seal(capture)
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
                self._admit(new.req_id, index, extra)
            return out

        def _prepare_inputs(*args: Any, **kwargs: Any) -> Any:
            """Build the step from V2's input batch and plan its ids and ring rows."""
            input_batch = prepare_inputs(*args, **kwargs)
            n = int(input_batch.num_reqs)
            req_ids = list(input_batch.req_ids[:n])
            req_idx = input_batch.idx_mapping_np[:n]
            pending["runs"] = self._plan(
                _Step(
                    req_ids,
                    req_idx,
                    [extras.get(req_id) for req_id in req_ids],
                    input_batch.num_computed_tokens_np[:n],
                    input_batch.query_start_loc_np[: n + 1],
                )
            )
            return input_batch

        def _execute_model(*args: Any, **kwargs: Any) -> Any:
            """Run V2's step, then hand its ring rows to the drain."""
            pending["runs"] = None
            try:
                out = execute_model(*args, **kwargs)
            except BaseException:
                if pending["runs"] is not None:
                    assert self._drain is not None
                    self._drain.release()
                raise
            runs = pending["runs"]
            if runs is not None:
                assert self._drain is not None
                # A dummy forward pass before the next plan writes no ring rows.
                self._meta[1, : self._meta_length].fill_(self._sentinel)
                event = torch.cuda.Event()
                event.record()
                self._drain.queue.put((event, runs))
            return out

        runner.add_requests = _add_requests
        runner.finish_requests = _finish_requests
        runner.prepare_inputs = _prepare_inputs
        runner.execute_model = _execute_model
        for k, layer_idx in enumerate(steer_layers):
            layer = layer_map[layer_idx]
            layer.register_buffer("_lens_ids", self._meta[0], persistent=False)
            layer.register_buffer("_lens_add", self._add[:, k], persistent=False)
            norm = None if self._norm is None else self._norm[:, k]
            layer.register_buffer("_lens_norm", norm, persistent=False)
            # Registered first, so a layer steers before it captures.
            layer.register_forward_hook(_steer_hook)
        for k, layer_idx in enumerate(capture_layers):
            layer = layer_map[layer_idx]
            layer.register_buffer("_lens_ring", ring[:, k], persistent=False)
            layer.register_buffer("_lens_dest", self._meta[1], persistent=False)
            layer.register_forward_hook(_capture_hook)
        logger.info(
            "vllm-lens: steer layers %s (norm_match %s), capture layers %s, ring %d rows",
            steer_layers,
            config.steer_norm_match,
            capture_layers,
            self._sentinel,
        )

    def _admit(self, req_id: str, index: int, extra: dict[str, Any]) -> None:
        """Write a new request's table rows, and open its capture file."""
        prepared = None
        key = extra.get("_steering_id")
        if key is not None:
            prepared = self._steering_data.get(key)
            if prepared is None:
                logger.error("Steering data of request %s is not available", req_id)
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
        if req_id not in self._captures:
            keep = [
                k
                for k, layer in enumerate(self._capture_layers)
                if not isinstance(wanted, list) or layer in wanted
            ]
            # vLLM's internal id is the external id and an 8-character suffix.
            self._captures[req_id] = _Capture(
                capture_path(self._capture_dir, key, req_id[:-9], partial=True), keep
            )

    def _plan(self, step: _Step) -> list[_Run] | None:
        """Fill this step's per-token table and ring rows and copy them to the device; return
        the ring runs to write, or None without capture rows."""
        bounds = step.bounds
        n = int(bounds[-1])
        captured = [
            (i, self._captures[req_id])
            for i, req_id in enumerate(step.req_ids)
            if req_id in self._captures
        ]
        positional = [
            i for i, index in enumerate(step.req_idx) if int(index) in self._positional
        ]
        signature = (tuple(step.req_ids), bounds.tobytes())
        if not captured and not positional and signature == self._meta_signature:
            return None
        self._meta_signature = None if captured or positional else signature
        counts = np.diff(bounds)
        slot = self._meta_slot = self._meta_slot ^ 1
        self._meta_events[slot].synchronize()
        host = self._host_meta[slot].numpy()
        length = max(n, self._meta_length)
        host[0, :n] = np.repeat(self._every[step.req_idx], counts)
        host[0, n:length] = 0
        host[1, :length] = self._sentinel
        for i in positional:
            index = int(step.req_idx[i])
            first = int(step.computed[i])
            positions = self._positional[index]
            inside = positions[(positions >= first) & (positions < first + counts[i])]
            host[0, bounds[i] + inside - first] = 2 * index + 2
        runs: list[_Run] | None = None
        total = int(sum(counts[i] for i, _ in captured))
        if total:
            assert self._drain is not None
            offset = self._drain.take(total)
            runs = []
            for i, capture in captured:
                first = int(step.computed[i])
                count = int(counts[i])
                start = int(bounds[i])
                host[1, start : start + count] = np.arange(offset, offset + count)
                runs.append(_Run(capture, offset, count, first))
                capture.rows = max(capture.rows, first + count)
                offset += count
        self._meta[:, :length].copy_(
            self._host_meta[slot][:, :length], non_blocking=True
        )
        self._meta_events[slot].record()
        self._meta_length = n
        return runs

    def _seal(self, capture: _Capture) -> None:
        """Queue the seal of a finished request's capture after its rows."""
        assert self._drain is not None
        self._drain.queue.put((capture, capture.rows))

    def set_steering_data(self, key: str, pickled_data: bytes) -> None:
        """Prepare the steering vectors of one request."""
        self._steering_data[key] = _prepare(
            pickle.loads(pickled_data),
            self._steer_layers,
            self._lens_hidden,
            self.device,  # type: ignore
            self._lens_dtype,
        )

    def clear_steering_data(self, key: str) -> None:
        """Drop the prepared vectors of a completed request."""
        self._steering_data.pop(key, None)

    def _debug_layer_discovery(self) -> list[int]:
        """Global indices of the decoder layers found by discovery (testing)."""
        return sorted(_discover_layer_modules(self))

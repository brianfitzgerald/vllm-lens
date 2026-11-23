"""The norm after an armed layer, fused with its steering and capture: one kernel adds the
layer output to the residual, adds each token's steering row, writes the stream to its ring
row and applies the next RMSNorm, with the math of vLLM's ``fused_add_rms_norm``."""

from __future__ import annotations

import torch
from vllm.triton_utils import tl, triton
from vllm.utils.torch_utils import direct_register_custom_op

# The ops live as long as this library, which is why it is held at module level.
_LIBRARY = torch.library.Library("vllm_lens", "FRAGMENT")


@triton.jit
def _lens_norm_kernel(
    x,
    residual,
    weight,
    out,
    residual_out,
    add,
    norm_table,
    ids,
    ring,
    dest,
    sentinel,
    width,
    x_stride,
    residual_stride,
    table_stride,
    norm_stride,
    ring_stride,
    eps,
    STEER: tl.constexpr,
    NORM: tl.constexpr,
    CAPTURE: tl.constexpr,
    BLOCK: tl.constexpr,
):
    """One token: the stream ``x + residual`` (plus its steering row) as the new residual,
    written to its ring row unless that is the sentinel, and its weighted RMSNorm."""
    token = tl.program_id(0).to(tl.int64)
    cols = tl.arange(0, BLOCK)
    mask = cols < width
    stream = tl.load(x + token * x_stride + cols, mask=mask, other=0.0).to(
        tl.float32
    ) + tl.load(residual + token * residual_stride + cols, mask=mask, other=0.0).to(
        tl.float32
    )
    if STEER:
        row = tl.load(ids + token).to(tl.int64)
        delta = tl.load(add + row * table_stride + cols, mask=mask, other=0.0).to(
            tl.float32
        )
        if NORM:
            magnitude = tl.sqrt(tl.sum(stream * stream, axis=0))
            delta += (
                tl.load(norm_table + row * norm_stride + cols, mask=mask, other=0.0)
                * magnitude
            )
        stream += delta
    rounded = stream.to(residual_out.dtype.element_ty)
    tl.store(residual_out + token * width + cols, rounded, mask=mask)
    if CAPTURE:
        target = tl.load(dest + token).to(tl.int64)
        tl.store(
            ring + target * ring_stride + cols,
            rounded,
            mask=mask & (target != sentinel),
        )
    variance = tl.sum(stream * stream, axis=0) / width
    normed = stream * tl.rsqrt(variance + eps)
    scale = tl.load(weight + cols, mask=mask, other=0.0)
    normed = normed.to(weight.dtype.element_ty) * scale
    tl.store(out + token * width + cols, normed.to(out.dtype.element_ty), mask=mask)


def _lens_norm(
    x: torch.Tensor,
    residual: torch.Tensor,
    weight: torch.Tensor,
    eps: float,
    add: torch.Tensor | None,
    norm_table: torch.Tensor | None,
    ids: torch.Tensor,
    ring: torch.Tensor,
    dest: torch.Tensor,
    capture: bool,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return the RMSNorm of the steered stream and the stream; with ``capture``, write the
    stream to its ring rows."""
    n, width = x.shape
    out = torch.empty((n, width), dtype=x.dtype, device=x.device)
    residual_out = torch.empty((n, width), dtype=x.dtype, device=x.device)
    if n == 0:
        return out, residual_out
    _lens_norm_kernel[(n,)](  # type: ignore[reportIndexIssue]
        x,
        residual,
        weight,
        out,
        residual_out,
        x if add is None else add,
        x if norm_table is None else norm_table,
        ids,
        ring,
        dest,
        ring.shape[0] - 1,
        width,
        x.stride(0),
        residual.stride(0),
        0 if add is None else add.stride(0),
        0 if norm_table is None else norm_table.stride(0),
        ring.stride(0),
        eps,
        STEER=add is not None,
        NORM=norm_table is not None,
        CAPTURE=capture,
        BLOCK=triton.next_power_of_2(width),
        num_warps=8 if width >= 4096 else 4,
    )
    return out, residual_out


def _lens_norm_fake(
    x: torch.Tensor,
    residual: torch.Tensor,
    weight: torch.Tensor,
    eps: float,
    add: torch.Tensor | None,
    norm_table: torch.Tensor | None,
    ids: torch.Tensor,
    ring: torch.Tensor,
    dest: torch.Tensor,
    capture: bool,
) -> tuple[torch.Tensor, torch.Tensor]:
    """The output shapes of ``lens_norm``, for tracing."""
    return torch.empty_like(x), torch.empty_like(x)


direct_register_custom_op(
    "lens_norm",
    _lens_norm,
    mutates_args=["ring"],
    fake_impl=_lens_norm_fake,
    target_lib=_LIBRARY,
)

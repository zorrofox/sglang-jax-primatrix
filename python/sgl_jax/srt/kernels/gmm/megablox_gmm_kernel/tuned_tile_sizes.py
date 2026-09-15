"""Shape-specific GMM v2 tile sizes measured on supported TPU generations."""

from __future__ import annotations

import jax.numpy as jnp

from sgl_jax.srt.utils.jax_utils import get_device_name

# Key: (lhs dtype, rhs dtype, groups, M, K, N).
# Values are (tile_m, tile_k, tile_n).
# Wildcard-m entries (size_m == -1) apply to calls with at least this many rows.
LARGE_M_FLOOR = 4096

TUNED_TILE_SIZES_GMM_V2 = {
    "TPU v7": {
        # DeepSeek V4-Flash EPMoE, tp8/ep1 (256 replicated experts, inter sharded to
        # 256), bf16 activations x fp8 e4m3 weights, prefill rows (8K chunk: 49152).
        # Swept 2026-09-15 on v7x: 0.524 -> 0.366 ms and 0.483 -> 0.371 ms per call.
        ("bfloat16", "float8_e4m3fn", 256, -1, 4096, 256): (256, 4096, 256),
        ("bfloat16", "float8_e4m3fn", 256, -1, 256, 4096): (256, 256, 4096),
        # Ling-3.0-tiny replicated EPMoE, decode BS=1 hot wi shape.
        # Measured kernel latency: 0.555ms -> 0.382ms (31.1% lower).
        ("bfloat16", "bfloat16", 128, 32, 1536, 512): (32, 768, 512),
        # Ling-3.0-tiny replicated EPMoE, 2K balanced prefill hot shapes.
        ("bfloat16", "bfloat16", 128, 2048, 1536, 512): (32, 1536, 512),
        ("bfloat16", "bfloat16", 128, 2048, 512, 1536): (32, 512, 1536),
    },
}


def get_tuned_gmm_v2_tile_sizes(
    *,
    lhs_dtype: jnp.dtype,
    rhs_dtype: jnp.dtype,
    num_groups: int,
    size_m: int,
    size_k: int,
    size_n: int,
    device_name: str | None = None,
) -> tuple[int, int, int] | None:
    if device_name is None:
        device_name = get_device_name()
    table = TUNED_TILE_SIZES_GMM_V2.get(device_name)
    if table is None:
        return None
    lhs, rhs = jnp.dtype(lhs_dtype).name, jnp.dtype(rhs_dtype).name
    exact = table.get((lhs, rhs, int(num_groups), int(size_m), int(size_k), int(size_n)))
    if exact is not None:
        return exact
    # Prefill rows per call depend on routing (EP) and chunk size, so large-m
    # entries may use size_m == -1 ("any m >= LARGE_M_FLOOR").
    if int(size_m) >= LARGE_M_FLOOR:
        return table.get((lhs, rhs, int(num_groups), -1, int(size_k), int(size_n)))
    return None

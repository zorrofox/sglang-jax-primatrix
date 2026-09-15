"""Page-run DMA writer for flat ``[rows, D]`` KV caches (Pallas TPU).

Prefill writes thousands of rows whose destinations are contiguous within each
allocator page; XLA's scatter handles them one row at a time (about 7 GB/s on v7x:
1.2 ms per layer for an 8K chunk). This kernel splits the write into fixed
``run``-row segments, DMAs a whole segment when its destinations are one
tile-aligned contiguous range, and falls back to a read-modify-write of the
16-row tile for rows of segments that are not contiguous, so any layout stays
correct. The cache is input/output aliased; untouched rows are never copied.
"""

from __future__ import annotations

import functools

import jax
import jax.numpy as jnp
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu

_TILE = 16  # bf16 sublane tile


def _kernel(dst_ref, loc_ref, valid_ref, values_ref, _, cache_hbm_ref, tile_ref, sem, *, run):
    # ``cache_hbm_ref`` is the aliased output buffer (same HBM as the input cache).
    seg = pl.program_id(0)
    dst = dst_ref[seg]

    @pl.when(dst >= 0)
    def _contiguous():
        start = pl.multiple_of(dst, _TILE)  # the wrapper only marks tile-aligned runs
        copy = pltpu.make_async_copy(values_ref, cache_hbm_ref.at[pl.ds(start, run)], sem)
        copy.start()
        copy.wait()

    @pl.when(dst < 0)
    def _rows():
        for r in range(run):
            token = seg * run + r

            @pl.when(valid_ref[token] != 0)
            def _one(r=r, token=token):
                loc = loc_ref[token]
                tile_start = pl.multiple_of((loc // _TILE) * _TILE, _TILE)
                load = pltpu.make_async_copy(
                    cache_hbm_ref.at[pl.ds(tile_start, _TILE)], tile_ref, sem
                )
                load.start()
                load.wait()
                tile_ref[pl.ds(loc - tile_start, 1), :] = values_ref[pl.ds(r, 1), :]
                store = pltpu.make_async_copy(
                    tile_ref, cache_hbm_ref.at[pl.ds(tile_start, _TILE)], sem
                )
                store.start()
                store.wait()


def paged_row_write(cache, values, loc, valid, *, run: int = 128, interpret: bool = False):
    """``cache.at[loc].set(values)`` for valid rows, by page-run DMA.

    ``cache`` [R, D] (bf16), ``values`` [T, D], ``loc`` [T] int32 destinations,
    ``valid`` [T] bool. Rows with ``valid`` False or out-of-range ``loc`` are dropped.
    """
    cache = jnp.asarray(cache)
    rows, dim = cache.shape
    values = jnp.asarray(values, cache.dtype)
    loc = jnp.asarray(loc, jnp.int32)
    valid = jnp.asarray(valid, bool) & (loc >= 0) & (loc < rows)
    n = values.shape[0]
    if run % _TILE:
        raise ValueError("run must be a multiple of the 16-row tile")
    n_pad = -(-n // run) * run
    if n_pad != n:
        values = jnp.pad(values, ((0, n_pad - n), (0, 0)))
        loc = jnp.pad(loc, (0, n_pad - n))
        valid = jnp.pad(valid, (0, n_pad - n))
    seg = n_pad // run
    l2 = loc.reshape(seg, run)
    v2 = valid.reshape(seg, run)
    base = l2[:, 0]
    contiguous = (
        jnp.all(v2, axis=1)
        & jnp.all(l2 == base[:, None] + jnp.arange(run, dtype=jnp.int32)[None, :], axis=1)
        & (base % _TILE == 0)
        & (base + run <= rows)
    )
    dst = jnp.where(contiguous, base, -1).astype(jnp.int32)
    return pl.pallas_call(
        functools.partial(_kernel, run=run),
        grid_spec=pltpu.PrefetchScalarGridSpec(
            num_scalar_prefetch=3,
            grid=(seg,),
            in_specs=(
                pl.BlockSpec((run, dim), lambda i, *_: (i, 0)),
                pl.BlockSpec(memory_space=pltpu.HBM),
            ),
            out_specs=pl.BlockSpec(memory_space=pltpu.HBM),
            scratch_shapes=(pltpu.VMEM((_TILE, dim), cache.dtype), pltpu.SemaphoreType.DMA),
        ),
        out_shape=jax.ShapeDtypeStruct(cache.shape, cache.dtype),
        input_output_aliases={4: 0},
        compiler_params=pltpu.CompilerParams(
            dimension_semantics=("arbitrary",), disable_bounds_checks=True
        ),
        interpret=interpret,
        name=f"dsv4-paged-row-write-r{run}-d{dim}",
    )(dst, loc, valid.astype(jnp.int32), values, cache)

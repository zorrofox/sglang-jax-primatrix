"""Precompile context ladder for DeepSeek-V4 read-table capacity buckets (CPU-only)."""

import pytest

from sgl_jax.srt.layers.attention.deepseek_v4_backend import (
    capacity_bucket,
    precompile_capacities,
)
from sgl_jax.srt.model_executor.compilation_manager import CompilationManager


def test_capacity_bucket_matches_runtime_rule():
    # Same rule the runtime applies to live batches: power of two, minimum 128.
    assert capacity_bucket(0) == 128
    assert capacity_bucket(128) == 128
    assert capacity_bucket(129) == 256
    assert capacity_bucket(8192) == 8192
    assert capacity_bucket(8193) == 16384


def test_precompile_capacities_cover_a_request_of_that_length():
    caps = precompile_capacities(262144)
    assert caps == {4: 65536, 128: 2048}
    caps = precompile_capacities(32768)
    assert caps == {4: 8192, 128: 256}
    # A 1000-token request completes 250 groups at ratio 4 (-> 256) and 7 at ratio 128 (-> 128).
    assert precompile_capacities(1000) == {4: 256, 128: 128}


def test_context_ladder_env(monkeypatch):
    monkeypatch.delenv("SGLANG_JAX_PRECOMPILE_CONTEXT_LADDER", raising=False)
    assert CompilationManager._compute_context_ladder() == [None]
    monkeypatch.setenv("SGLANG_JAX_PRECOMPILE_CONTEXT_LADDER", "262144, 32768,131072,65536")
    assert CompilationManager._compute_context_ladder() == [32768, 65536, 131072, 262144]
    monkeypatch.setenv("SGLANG_JAX_PRECOMPILE_CONTEXT_LADDER", "0,1024")
    with pytest.raises(ValueError):
        CompilationManager._compute_context_ladder()


def test_set_precompile_context_only_touches_backends_that_opt_in():
    class Backend:
        precompile_context_len = None

    class Runner:
        attn_backend = Backend()

    CompilationManager._set_precompile_context(Runner(), 4096)
    assert Runner.attn_backend.precompile_context_len == 4096

    class Other:
        pass

    class Runner2:
        attn_backend = Other()

    CompilationManager._set_precompile_context(Runner2(), 4096)  # no attribute, no error
    assert not hasattr(Runner2.attn_backend, "precompile_context_len")

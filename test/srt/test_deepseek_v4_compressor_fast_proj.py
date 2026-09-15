"""Fast compressor projection == HIGHEST f32 projection to rounding (bf16 activations)."""

import jax.numpy as jnp
import numpy as np

from sgl_jax.srt.layers.attention.dsv4 import compressor as comp


def _run(monkeypatch, fast, x, w):
    monkeypatch.setattr(comp, "_FAST_PROJ", fast)
    return np.asarray(comp._project(jnp.asarray(x, jnp.float32), w), np.float64)


def test_fast_projection_matches_highest(monkeypatch):
    rng = np.random.default_rng(0)
    x = rng.standard_normal((64, 512)).astype(jnp.bfloat16).astype(np.float32)  # bf16-exact
    for w in (
        jnp.asarray(rng.standard_normal((256, 512)) * 0.05, jnp.bfloat16),
        jnp.asarray(rng.standard_normal((256, 512)) * 0.05, jnp.float32),
    ):
        want = _run(monkeypatch, False, x, w)
        got = _run(monkeypatch, True, x, w)
        np.testing.assert_allclose(got, want, rtol=1e-5, atol=1e-5)

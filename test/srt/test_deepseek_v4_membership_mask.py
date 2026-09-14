"""Fused vs scatter membership in dsv4 admissible_mask give the same compressed mask (CPU-only)."""

import numpy as np

from sgl_jax.srt.layers.attention.dsv4 import attention as att


def _case(T, E, K, seed):
    rng = np.random.default_rng(seed)
    qpos = np.sort(rng.integers(0, 4096, size=T)).astype(np.int32)
    qreq = np.zeros(T, np.int32)
    valid = np.ones(T, bool)
    wpos = np.arange(128, dtype=np.int32)
    wreq = np.zeros(128, np.int32)
    entry_ids = np.arange(E, dtype=np.int32)
    creq = np.zeros(E, np.int32)
    selected = np.full((T, K), -1, np.int32)
    for t in range(T):
        k = rng.integers(0, K + 1)
        selected[t, :k] = rng.choice(E, size=k, replace=False)
    return dict(
        query_positions=qpos,
        query_request_ids=qreq,
        valid_token_mask=valid,
        window_positions=wpos,
        window_request_ids=wreq,
        compressed_entry_ids=entry_ids,
        compressed_request_ids=creq,
        window_size=128,
        ratio=4,
        selected_entries=selected,
    )


def test_fused_and_scatter_membership_agree(monkeypatch):
    for T, E, K, seed in ((7, 300, 16, 0), (64, 2048, 64, 1), (33, 1500, 8, 2)):
        kw = _case(T, E, K, seed)
        monkeypatch.setattr(att, "_MEMBERSHIP_FUSED_BUDGET", 1 << 40)
        _, fused = att.admissible_mask(**kw)
        monkeypatch.setattr(att, "_MEMBERSHIP_FUSED_BUDGET", 0)
        _, scatter = att.admissible_mask(**kw)
        np.testing.assert_array_equal(np.asarray(fused), np.asarray(scatter))
        # sanity: every selected, complete entry is admitted and nothing else beyond completeness
        sel = kw["selected_entries"]
        for t in range(T):
            row = np.asarray(fused)[t]
            chosen = set(int(x) for x in sel[t] if x >= 0)
            assert set(np.flatnonzero(row)).issubset(chosen)

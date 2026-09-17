"""Tests for the flywire_gnn pipeline.

Synthetic-data tests run anywhere. The real-data smoke test is skipped unless
a processed cache is present (i.e. after a build on the developer machine).
"""

from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch

from flywire_gnn import FlyWireFAFB, GraphSAGE, MLP
from flywire_gnn.dataset import assemble, make_splits


def tiny_agg():
    pairs = {
        "pre_pt_root_id": np.array([1, 1, 2, 3, 4, 5], dtype=np.int64),
        "post_pt_root_id": np.array([2, 3, 3, 1, 5, 4], dtype=np.int64),
        "syn_count_sum": np.array([7, 3, 9, 2, 6, 11], dtype=np.int64),
        "gaba_w_sum": np.array([0.5, 0.1, 8.0, 0.4, 5.0, 10.0]),
        "ach_w_sum": np.array([6.0, 2.5, 0.5, 1.5, 0.5, 0.5]),
        "glut_w_sum": np.array([0.3, 0.2, 0.2, 0.0, 0.4, 0.3]),
        "oct_w_sum": np.zeros(6),
        "ser_w_sum": np.zeros(6),
        "da_w_sum": np.zeros(6),
    }
    neuro_pre = pd.DataFrame(
        {
            "root_id": np.array([1, 1, 2, 3, 4, 5], dtype=np.int64),
            "neuropil": ["AL_L", "ME_R", "AL_L", "ME_R", "AL_L", "ME_R"],
            "syn": np.array([7, 3, 9, 2, 6, 11], dtype=np.int64),
        }
    )
    neuro_post = pd.DataFrame(
        {
            "root_id": np.array([2, 3, 3, 1, 5, 4], dtype=np.int64),
            "neuropil": ["AL_L", "AL_L", "ME_R", "AL_L", "ME_R", "AL_L"],
            "syn": np.array([7, 3, 9, 2, 6, 11], dtype=np.int64),
        }
    )
    node_out = pd.DataFrame(
        {
            "root_id": np.array([1, 2, 3, 4, 5], dtype=np.int64),
            "syn_total": np.array([10.0, 9.0, 2.0, 6.0, 11.0]),
            "gaba_w_sum": np.array([0.6, 8.0, 0.4, 5.0, 10.0]),
            "ach_w_sum": np.array([8.5, 0.5, 1.5, 0.5, 0.5]),
            "glut_w_sum": np.array([0.5, 0.2, 0.0, 0.4, 0.3]),
            "oct_w_sum": np.zeros(5),
            "ser_w_sum": np.zeros(5),
            "da_w_sum": np.zeros(5),
        }
    )
    node_in = node_out.copy()
    labels_df = pd.DataFrame(
        {
            "root_id": np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10], dtype=np.int64),
            "super_class": ["a", "b", "a", "b", "a", "b", "a", "b", "a", "b"],
        }
    )
    return {
        "pairs": pairs,
        "neuro_pre": neuro_pre,
        "neuro_post": neuro_post,
        "node_out": node_out,
        "node_in": node_in,
        "neuropils": ["AL_L", "ME_R"],
    }, labels_df


def test_assemble_counts():
    agg, labels_df = tiny_agg()
    node_ids = np.arange(1, 11, dtype=np.int64)
    out = assemble(agg, node_ids, labels_df, "super_class", min_synapses=5)
    assert out["meta"]["num_nodes"] == 10
    # pairs with syn >= 5: (1,2,7), (2,3,9), (4,5,6), (5,4,11)
    assert out["meta"]["num_edges"] == 4
    assert out["meta"]["feature_dim"] == 4 + 2 * 2 + 12
    assert out["meta"]["labeled_nodes"] == 10
    assert out["meta"]["dropped_edges_unmapped_root"] == 0
    assert out["x"].shape == (10, 20)
    assert out["edge_index"].dtype == torch.int64


def test_features_finite():
    agg, labels_df = tiny_agg()
    node_ids = np.arange(1, 11, dtype=np.int64)
    out = assemble(agg, node_ids, labels_df, "super_class", min_synapses=5)
    assert torch.isfinite(out["x"]).all()
    assert torch.isfinite(out["edge_attr"]).all()


def test_splits_disjoint_and_deterministic():
    y = torch.tensor([0, 1] * 50, dtype=torch.int64)
    tm, vm, te = make_splits(y, seed=42)
    assert not (tm & vm).any().item()
    assert not (tm & te).any().item()
    assert not (vm & te).any().item()
    assert (tm | vm | te).all().item()
    tm2, vm2, te2 = make_splits(y, seed=42)
    assert (tm == tm2).all().item()
    assert (vm == vm2).all().item()
    assert (te == te2).all().item()


def test_models_forward_small_graph():
    torch.manual_seed(0)
    n, e, f, c = 10, 40, 20, 3
    x = torch.randn(n, f)
    y = torch.randint(0, c, (n,))
    edge_index = torch.randint(0, n, (2, e))
    for model in [MLP(f, hidden=16, out_dim=c), GraphSAGE(f, hidden=16, out_dim=c)]:
        out = model(x, edge_index)
        assert out.shape == (n, c)
        assert torch.isfinite(out).all()
        loss = torch.nn.functional.cross_entropy(out, y)
        loss.backward()
        assert torch.isfinite(loss)


@pytest.mark.skipif(
    not (Path("data") / "processed_v783_super_class_syn5.pt").exists(),
    reason="real FAFB cache not built on this machine",
)
def test_real_cache_counts():
    ds = FlyWireFAFB(root="data", labels="super_class", min_synapses=5, download=False)
    m = ds._cache["meta"]
    assert m["num_nodes"] == 139255
    assert m["num_edges"] == 2700513
    assert m["labeled_nodes"] == 139255
    assert m["feature_dim"] == 174
    assert len(ds._cache["label_names"]) == 9
    tm, vm, te = ds.splits()
    n = m["num_nodes"]
    assert int(tm.sum()) + int(vm.sum()) + int(te.sum()) == n
    assert not (tm & vm).any().item()
    assert not (tm & te).any().item()
    assert not (vm & te).any().item()

"""The FlyWire FAFB (v783) Drosophila connectome as a PyTorch Geometric dataset.

Data sources (all public, no authentication required):
- Connectivity: FlyWire Whole-brain Connectome Connectivity Data v783
  (Zenodo, CC-BY-4.0, doi:10.5281/zenodo.10676866), Dorkenwald et al., Nature 2024
- Cell annotations: Supplementary Data 5 of Schlegel et al., Nature 2024
  (doi:10.1038/s41586-024-07686-5)

Graph layout:
- x:          float32 [N, 174] node features derived from wiring
- edge_index: int64   [2, E] directed edges (pre -> post), pair-level, filtered
              to synapses >= min_synapses (default 5, the published FAFB
              "connection" threshold)
- edge_attr:  float32 [E, 7]: [log1p(syn_count), 6 weighted NT probabilities]
- y:          int64 [N] label index (-1 = unlabeled)
"""

from __future__ import annotations

import hashlib
import os
import time
import urllib.request
from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.feather
import torch

ZENODO_DOI = "10.5281/zenodo.10676866"

CONNECTIONS_URL = (
    "https://zenodo.org/records/10676866/files/"
    "proofread_connections_783.feather?download=1"
)
CONNECTIONS_MD5 = "f48f972d262323a102aed49af1396b8a"
CONNECTIONS_SIZE = 852_022_274

ROOT_IDS_URL = (
    "https://zenodo.org/records/10676866/files/"
    "proofread_root_ids_783.npy?download=1"
)
ROOT_IDS_MD5 = "e0e6c19732fd8c7a4e39a2d170105421"
ROOT_IDS_SIZE = 1_114_168

LABELS_URL = (
    "https://media.springernature.com/original/springer-static/esm/"
    "art%3A10.1038%2Fs41586-024-07686-5/MediaObjects/"
    "41586_2024_7686_MOESM5_ESM.tsv"
)
LABELS_SIZE = 27_015_208

LABEL_COLUMNS = ("super_class", "cell_class", "cell_sub_class", "cell_type")

NT_AVG_COLUMNS = ("gaba_avg", "ach_avg", "glut_avg", "oct_avg", "ser_avg", "da_avg")
NT_NAMES = ("gaba", "ach", "glut", "oct", "ser", "da")

DEFAULT_SEED = 42
DEFAULT_RATIOS = (0.7, 0.15, 0.15)
DEFAULT_MIN_SYNAPSES = 5


def _default_cache_dir() -> Path:
    env = os.environ.get("FLYWIRE_GNN_CACHE")
    if env:
        return Path(env)
    return Path.home() / ".cache" / "flywire_gnn"


def _md5(path: Path, chunk_size: int = 1 << 22) -> str:
    h = hashlib.md5()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(chunk_size), b""):
            h.update(chunk)
    return h.hexdigest()


def _download(url: str, dest: Path, expected_size: Optional[int] = None) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(dest.suffix + ".part")
    for attempt in range(5):
        existing = tmp.stat().st_size if tmp.exists() else 0
        headers = {"Range": f"bytes={existing}-"} if existing else {}
        req = urllib.request.Request(url, headers=headers)
        mode = "ab" if existing else "wb"
        try:
            with urllib.request.urlopen(req, timeout=180) as r, open(tmp, mode) as f:
                while chunk := r.read(1 << 22):
                    f.write(chunk)
            if expected_size is None or tmp.stat().st_size >= expected_size:
                tmp.replace(dest)
                return
        except Exception as e:
            if attempt == 4:
                raise RuntimeError(f"download failed after 5 attempts: {url}") from e
            time.sleep(2 * (attempt + 1))
    raise RuntimeError(f"download failed: {url}")


def aggregate(connections_path: Path) -> Dict[str, object]:
    """Memory-safe aggregation of the pair-by-neuropil connection table.

    Returns numpy/pandas frames:
      pairs:      dict of columns for unique (pre, post) pairs with
                  syn_count and NT weighted sums (all at >=1 synapse)
      neuro_pre / neuro_post: (root_id, neuropil, syn) frames
      node_out / node_in:     per-root syn totals and NT weighted sums
      neuropils:  sorted list of all neuropil names
    """
    tbl = pa.feather.read_table(connections_path, memory_map=True)
    tbl = tbl.filter(
        pc.not_equal(pc.field("pre_pt_root_id"), pc.field("post_pt_root_id"))
    )
    syn_f = tbl.column("syn_count").cast(pa.float64())
    w_cols = []
    for c in NT_AVG_COLUMNS:
        w = c.replace("_avg", "_w")
        tbl = tbl.append_column(w, pc.multiply(tbl.column(c), syn_f))
        w_cols.append(w)

    pairs_t = tbl.group_by(["pre_pt_root_id", "post_pt_root_id"]).aggregate(
        [("syn_count", "sum")] + [(w, "sum") for w in w_cols]
    )
    pairs = {
        c: pairs_t.column(c).combine_chunks().to_numpy(zero_copy_only=False)
        for c in pairs_t.column_names
    }

    def as_frame(t):
        return t.to_pandas()

    neuro_pre = as_frame(
        tbl.group_by(["pre_pt_root_id", "neuropil"]).aggregate([("syn_count", "sum")])
    ).rename(columns={"pre_pt_root_id": "root_id", "syn_count_sum": "syn"})

    neuro_post = as_frame(
        tbl.group_by(["post_pt_root_id", "neuropil"]).aggregate([("syn_count", "sum")])
    ).rename(columns={"post_pt_root_id": "root_id", "syn_count_sum": "syn"})

    node_out = as_frame(
        tbl.group_by("pre_pt_root_id").aggregate(
            [("syn_count", "sum")] + [(w, "sum") for w in w_cols]
        )
    ).rename(columns={"pre_pt_root_id": "root_id", "syn_count_sum": "syn_total"})
    node_in = as_frame(
        tbl.group_by("post_pt_root_id").aggregate(
            [("syn_count", "sum")] + [(w, "sum") for w in w_cols]
        )
    ).rename(columns={"post_pt_root_id": "root_id", "syn_count_sum": "syn_total"})

    neuropils = sorted(
        set(neuro_pre["neuropil"].unique()) | set(neuro_post["neuropil"].unique())
    )

    return {
        "pairs": pairs,
        "neuro_pre": neuro_pre,
        "neuro_post": neuro_post,
        "node_out": node_out,
        "node_in": node_in,
        "neuropils": neuropils,
    }


def assemble(
    agg: Dict[str, object],
    node_ids: np.ndarray,
    labels_df: pd.DataFrame,
    label_col: str,
    min_synapses: int,
) -> Dict[str, object]:
    """Build graph tensors from aggregated frames. Pure numpy/torch, testable."""
    node_ids = np.sort(np.asarray(node_ids, dtype=np.int64))
    n = len(node_ids)

    def map_ids(arr: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        arr = np.asarray(arr, dtype=np.int64)
        pos = np.searchsorted(node_ids, arr)
        pos_c = np.clip(pos, 0, n - 1)
        ok = node_ids[pos_c] == arr
        return pos_c, ok

    pairs = agg["pairs"]
    pre = pairs["pre_pt_root_id"]
    post = pairs["post_pt_root_id"]
    syn = pairs["syn_count_sum"].astype(np.int64)
    wsum_names = [f"{s}_w_sum" for s in NT_NAMES]

    upos, uok = map_ids(pre)
    vpos, vok = map_ids(post)
    valid = uok & vok
    dropped_edges = int((~valid).sum())

    keep = (syn >= min_synapses) & valid
    u = upos[keep]
    v = vpos[keep]
    s = syn[keep]
    wsum = np.stack([pairs[w][keep] for w in wsum_names], axis=1).astype(np.float64)
    nt_edge = (wsum / np.maximum(s, 1)[:, None]).astype(np.float32)

    edge_index = torch.from_numpy(np.stack([u, v])).long()
    edge_attr = torch.from_numpy(
        np.column_stack([np.log1p(s.astype(np.float64)), nt_edge])
    ).float()

    upos_all, _ = map_ids(pre[valid])
    vpos_all, _ = map_ids(post[valid])
    s_all = syn[valid]

    x = np.zeros((n, 4 + 2 * len(agg["neuropils"]) + 12), dtype=np.float32)
    x[:, 0] = np.log1p(np.bincount(upos_all, minlength=n).astype(np.float32))
    x[:, 1] = np.log1p(np.bincount(vpos_all, minlength=n).astype(np.float32))
    x[:, 2] = np.log1p(
        np.bincount(upos_all, weights=s_all, minlength=n).astype(np.float32)
    )
    x[:, 3] = np.log1p(
        np.bincount(vpos_all, weights=s_all, minlength=n).astype(np.float32)
    )

    neuropils = agg["neuropils"]
    n_n = len(neuropils)
    np_name2idx = {name: i for i, name in enumerate(neuropils)}

    def neuro_matrix(df: pd.DataFrame, col0: int) -> None:
        rpos, ok = map_ids(df["root_id"].to_numpy())
        codes = df["neuropil"].map(np_name2idx).to_numpy(dtype=np.int64)
        vals = df["syn"].to_numpy(dtype=np.float64)
        np.add.at(x, (rpos[ok], col0 + codes[ok]), np.log1p(vals[ok]))

    neuro_matrix(agg["neuro_pre"], 4)
    neuro_matrix(agg["neuro_post"], 4 + n_n)

    col_nt = 4 + 2 * n_n

    def nt_profiles(df: pd.DataFrame, col0: int) -> None:
        rpos, ok = map_ids(df["root_id"].to_numpy())
        tot = df["syn_total"].to_numpy(dtype=np.float64)
        for j, name in enumerate(NT_NAMES):
            w = df[f"{name}_w_sum"].to_numpy(dtype=np.float64)
            profile = np.where(tot > 0, w / np.maximum(tot, 1), 0.0)
            x[rpos[ok], col0 + j] = profile[ok].astype(np.float32)

    nt_profiles(agg["node_out"], col_nt)
    nt_profiles(agg["node_in"], col_nt + 6)

    y = np.full(n, -1, dtype=np.int64)
    label_names: list = []
    if labels_df is not None and label_col in labels_df.columns:
        sub = labels_df[["root_id", label_col]].dropna().copy()
        labels = sub[label_col].astype(str)
        label_names = sorted(labels.unique())
        enc = {name: i for i, name in enumerate(label_names)}
        rpos, ok = map_ids(sub["root_id"].to_numpy())
        y[rpos[ok]] = labels.map(enc).to_numpy()[ok]

    return {
        "x": torch.from_numpy(x),
        "edge_index": edge_index,
        "edge_attr": edge_attr,
        "y": torch.from_numpy(y),
        "node_ids": torch.from_numpy(node_ids),
        "neuropils": neuropils,
        "label_col": label_col,
        "label_names": label_names,
        "meta": {
            "num_nodes": n,
            "num_edges": int(edge_index.shape[1]),
            "dropped_edges_unmapped_root": dropped_edges,
            "min_synapses": min_synapses,
            "feature_dim": int(x.shape[1]),
            "labeled_nodes": int((y >= 0).sum()),
        },
    }


def make_splits(
    y: torch.Tensor,
    seed: int = DEFAULT_SEED,
    ratios: Tuple[float, float, float] = DEFAULT_RATIOS,
    stratify: bool = True,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    y_np = y.numpy()
    n = len(y_np)
    labeled = np.where(y_np >= 0)[0]
    labeled.sort()
    if stratify and len(labeled) > 0:
        from sklearn.model_selection import train_test_split

        try:
            train_idx, rest = train_test_split(
                labeled,
                train_size=ratios[0],
                random_state=seed,
                stratify=y_np[labeled],
            )
            rel = ratios[2] / (ratios[1] + ratios[2])
            val_idx, test_idx = train_test_split(
                np.sort(rest), test_size=rel, random_state=seed, stratify=y_np[np.sort(rest)]
            )
        except ValueError:
            rng = np.random.RandomState(seed)
            idx = labeled.copy()
            rng.shuffle(idx)
            k1 = int(len(idx) * ratios[0])
            k2 = int(len(idx) * (ratios[0] + ratios[1]))
            train_idx, val_idx, test_idx = idx[:k1], idx[k1:k2], idx[k2:]
    else:
        rng = np.random.RandomState(seed)
        idx = rng.permutation(n if len(labeled) == 0 else labeled.copy())
        k1 = int(len(idx) * ratios[0])
        k2 = int(len(idx) * (ratios[0] + ratios[1]))
        train_idx, val_idx, test_idx = idx[:k1], idx[k1:k2], idx[k2:]

    def mask(idx) -> torch.Tensor:
        m = np.zeros(n, dtype=bool)
        m[np.asarray(idx, dtype=np.int64)] = True
        return torch.from_numpy(m)

    return mask(train_idx), mask(val_idx), mask(test_idx)


class FlyWireFAFB:
    """FlyWire FAFB v783 connectome as a ready-to-train graph dataset.

    Parameters
    ----------
    root:
        Cache directory (default ~/.cache/flywire_gnn or $FLYWIRE_GNN_CACHE).
    labels:
        Annotation column to use for node classification:
        'super_class' (default), 'cell_class', 'cell_sub_class', 'cell_type'.
    min_synapses:
        Minimum synapse count for a directed edge between two neurons.
        Default 5 (the published FAFB 'connection' threshold). Set 1 for the
        densest graph.
    seed:
        Seed for the train/val/test split (default 42).
    download:
        Download the source files if they are not already in `root`.
    """

    def __init__(
        self,
        root: Optional[str] = None,
        labels: str = "super_class",
        min_synapses: int = DEFAULT_MIN_SYNAPSES,
        seed: int = DEFAULT_SEED,
        download: bool = True,
    ) -> None:
        if labels not in LABEL_COLUMNS:
            raise ValueError(f"labels must be one of {LABEL_COLUMNS}, got {labels!r}")
        self.root = Path(root) if root else _default_cache_dir()
        self.root.mkdir(parents=True, exist_ok=True)
        self.labels_col = labels
        self.min_synapses = int(min_synapses)
        self.seed = seed

        self._conn = self.root / "proofread_connections_783.feather"
        self._ids = self.root / "proofread_root_ids_783.npy"
        self._labels = self.root / "annotations_v783.tsv"
        self._cache_path = (
            self.root / f"processed_v783_{labels}_syn{self.min_synapses}.pt"
        )

        if self._cache_path.exists():
            self._cache = torch.load(self._cache_path, weights_only=False)
        else:
            if not download:
                raise FileNotFoundError(
                    f"no processed cache at {self._cache_path} and download=False"
                )
            self._ensure_raw()
            self._cache = self._build()
            torch.save(self._cache, self._cache_path)

    def _ensure_raw(self) -> None:
        if not self._conn.exists():
            _download(CONNECTIONS_URL, self._conn, CONNECTIONS_SIZE)
        if os.path.getsize(self._conn) != CONNECTIONS_SIZE or (
            _md5(self._conn) != CONNECTIONS_MD5
        ):
            raise RuntimeError("connection file failed integrity check")
        if not self._ids.exists():
            _download(ROOT_IDS_URL, self._ids, ROOT_IDS_SIZE)
        if _md5(self._ids) != ROOT_IDS_MD5:
            raise RuntimeError("root-id file failed integrity check")
        if not self._labels.exists():
            _download(LABELS_URL, self._labels, LABELS_SIZE)

    def _build(self) -> Dict[str, object]:
        agg = aggregate(self._conn)
        node_ids = np.load(self._ids)
        labels_df = pd.read_csv(self._labels, sep="\t", low_memory=False)
        out = assemble(
            agg,
            node_ids=node_ids,
            labels_df=labels_df,
            label_col=self.labels_col,
            min_synapses=self.min_synapses,
        )
        return out

    @property
    def data(self):
        from torch_geometric.data import Data

        c = self._cache
        d = Data(x=c["x"], edge_index=c["edge_index"], edge_attr=c["edge_attr"], y=c["y"])
        d.node_ids = c["node_ids"]
        d.num_classes = len(c["label_names"])
        return d

    def splits(
        self, seed: Optional[int] = None, ratios: Tuple[float, float, float] = DEFAULT_RATIOS
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        return make_splits(self._cache["y"], seed=seed or self.seed, ratios=ratios)

    def summary(self) -> str:
        m = self._cache["meta"]
        return (
            "FlyWire FAFB v783\n"
            f"  nodes:            {m['num_nodes']:,}\n"
            f"  edges:            {m['num_edges']:,} (min_synapses={m['min_synapses']})\n"
            f"  dropped edges:    {m['dropped_edges_unmapped_root']:,}\n"
            f"  feature dim:      {m['feature_dim']}\n"
            f"  labeled nodes:    {m['labeled_nodes']:,} ({m['labeled_nodes']/m['num_nodes']*100:.1f}%)\n"
            f"  task labels:      {self._cache['label_col']} "
            f"({len(self._cache['label_names'])} classes)\n"
            f"  source:           {ZENODO_DOI}"
        )

    def __repr__(self) -> str:
        return (
            f"FlyWireFAFB(labels={self.labels_col!r}, min_synapses={self.min_synapses}, "
            f"nodes={self._cache['meta']['num_nodes']}, "
            f"edges={self._cache['meta']['num_edges']})"
        )

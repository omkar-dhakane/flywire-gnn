import json
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from flywire_gnn import dataset as D

root = Path("data")
out = root / "hf"
out.mkdir(exist_ok=True)

print("aggregating connections (this takes ~40s)...")
agg = D.aggregate(root / "proofread_connections_783.feather")
pairs = agg["pairs"]

n_pairs = len(pairs["pre_pt_root_id"])
print(f"unique directed pairs at >=1 syn: {n_pairs:,}")

tbl_cols = {
    "pre": pa.array(pairs["pre_pt_root_id"], type=pa.int64()),
    "post": pa.array(pairs["post_pt_root_id"], type=pa.int64()),
    "syn_count": pa.array(pairs["syn_count_sum"], type=pa.int32()),
}
for name in D.NT_NAMES:
    w = pairs[f"{name}_w_sum"]
    prob = (w / np.maximum(pairs["syn_count_sum"], 1)).astype(np.float32)
    tbl_cols[name] = pa.array(prob, type=pa.float32())

pairs_tbl = pa.table(tbl_cols)
pq.write_table(pairs_tbl, out / "connections.parquet", compression="zstd", compression_level=9)
print(f"connections.parquet: {(out / 'connections.parquet').stat().st_size / 1e6:.1f} MB")

ann = pd.read_csv(root / "annotations_v783.tsv", sep="\t", low_memory=False)
keep_cols = [
    "root_id", "flow", "super_class", "cell_class", "cell_sub_class", "cell_type",
    "hemibrain_type", "ito_lee_hemilineage", "hartenstein_hemilineage",
    "top_nt", "top_nt_conf", "side", "nerve", "vfb_id", "fbbt_id",
]
ann = ann[keep_cols]
ann.to_parquet(out / "nodes.parquet", compression="zstd", index=False)
print(f"nodes.parquet: {(out / 'nodes.parquet').stat().st_size / 1e6:.1f} MB, {len(ann):,} rows")

class_counts = ann["super_class"].value_counts().to_dict()
meta = {
    "name": "flywire-fafb-connectome",
    "dataset_version": "FAFB_v783",
    "package_version": "0.1.0",
    "license": "CC-BY-4.0",
    "sources": {
        "connectivity": {
            "url": D.CONNECTIONS_URL,
            "md5": D.CONNECTIONS_MD5,
            "size": D.CONNECTIONS_SIZE,
            "doi": D.ZENODO_DOI,
        },
        "labels": {
            "url": D.LABELS_URL,
            "size": D.LABELS_SIZE,
            "doi": "10.1038/s41586-024-07686-5",
        },
    },
    "counts": {
        "nodes_proofread": 139255,
        "unique_directed_pairs_ge1_syn": int(n_pairs),
        "unique_directed_pairs_ge5_syn": 2700513,
        "pair_neuropil_rows_ge1_syn": 16847997,
        "neuropils": 79,
    },
    "features": {
        "dim": 174,
        "layout": [
            "log1p(out_degree)", "log1p(in_degree)",
            "log1p(out_synapses)", "log1p(in_synapses)",
            "log1p(out_synapses_per_neuropil) x79",
            "log1p(in_synapses_per_neuropil) x79",
            "out NT profile (6, synapse-weighted mean prob)",
            "in NT profile (6, synapse-weighted mean prob)",
        ],
    },
    "task": {
        "type": "node_classification",
        "label_column": "super_class",
        "num_classes": 9,
        "class_counts": {str(k): int(v) for k, v in class_counts.items()},
        "split": "70/15/15 train/val/test, stratified, seed=42",
    },
    "edge_threshold_default": 5,
    "edge_attributes": ["log1p(syn_count)", "6 NT probabilities"],
}

(out / "meta.json").write_text(json.dumps(meta, indent=2))
print("meta.json written")
print("done")

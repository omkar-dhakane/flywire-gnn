# flywire-gnn

**The complete fruit-fly brain connectome (FlyWire FAFB v783) as a ready-to-train PyTorch Geometric dataset** — 139,255 proofread neurons, 2.7M directed synaptic connections, cell-type labels on every neuron, deterministic stratified splits, and three reproducible baselines. One `pip install`, one class, no authentication, no left-over data-wrangling.

## Install

```bash
pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install torch_geometric
pip install git+https://github.com/omkar-dhakane/flywire-gnn.git   # or: pip install -e .
```

## Quickstart (the whole thing)

```python
from flywire_gnn import FlyWireFAFB

ds = FlyWireFAFB()                    # downloads 852MB + 1.1MB + 27MB once, caches locally
data = ds.data                        # torch_geometric.data.Data
train_mask, val_mask, test_mask = ds.splits()   # deterministic, stratified, seed 42
print(data)      # Data(x=[139255, 174], edge_index=[2, 2700513], edge_attr=[2700513, 7], y=[139255])
```

Everything is cached in `~/.cache/flywire_gnn` — subsequent loads are instant.

## The dataset

| | |
|---|---|
| Nodes | 139,255 proofread neurons (whole adult female fly brain) |
| Edges | 2,700,513 unique directed pairs at ≥5 synapses (15,091,983 at ≥1) |
| Node features (`x`, 174-dim) | wiring structure: degrees/synapse totals, per-neuropil in/out profiles across 79 brain regions, in/out neurotransmitter profiles |
| Edge attributes (`edge_attr`, 7-dim) | `log1p(syn_count)` + 6 synapse-weighted neurotransmitter probabilities |
| Labels (`y`) | `super_class`: 9 classes covering **100% of nodes** (also available: `cell_class`, `cell_sub_class`, `cell_type`) |
| Task | node classification from wiring structure |
| Splits | 70/15/15 train/val/test, stratified, seed 42 |

Data sources (all public, **no auth needed**):
- Connectivity: FlyWire Whole-brain Connectome Connectivity Data v783 — [Zenodo, CC-BY-4.0](https://doi.org/10.5281/zenodo.10676866)
- Annotation: [Schlegel et al. 2024](https://doi.org/10.1038/s41586-024-07686-5), Supplementary Data 5

## Leaderboard — node classification on `super_class`

Real runs, full-batch, CPU, seed 42, `min_synapses=5`, hidden 128 (see `flywire_gnn/train.py`):

| Model | Test accuracy | Macro-F1 | Best val acc (epoch) | Epochs | Wall time |
|---|---|---|---|---|---|
| MLP (features only) | **0.9851** | 0.7115 | 0.9860 (200) | 200 | 3.5 min |
| GraphSAGE | 0.9812 | **0.7563** | 0.9820 (150) | 150 | 13 min |
| GCN | 0.9166 | 0.4702 | 0.9202 (180) | 200 | 23.5 min |

**What the benchmark shows**: wiring-profile features alone nearly saturate accuracy (98.5%) — most of a cell's coarse class is readable directly from its projection pattern. Mean-aggregated message passing (GraphSAGE) roughly matches features and wins on the rare classes (best macro-F1); GCN's symmetric normalization over-smooths and trails. If your architecture can't beat 0.9851 accuracy *and* 0.7563 macro-F1 on these exact splits, it isn't adding anything over a feature baseline.

### Reproduce

```bash
python -m flywire_gnn.train --models mlp,gcn,sage          # all three
python -m flywire_gnn.train --model sage --epochs 150 --seed 42
python -m flywire_gnn.train --model sage --labels cell_class   # harder multi-class task
python -m flywire_gnn.train --model sage --min-synapses 1       # dense 15M-edge graph
```

## Design choices (and why)

- **`min_synapses=5`** is the published FAFB "connection" threshold (same default as Codex). Pass `min_synapses=1` for the full-density graph.
- **Node features come from the ≥1-synapse wiring** (a neuron's total projection profile); edges are filtered by `min_synapses`. So features are the same regardless of the edge threshold you benchmark.
- **Stratified splits on labeled nodes only**: with `super_class`, all 139,255 nodes are labeled. Deterministic via `seed=42` (NumPy `RandomState` + sklearn stratified split).
- The raw 9.5 GB per-synapse file (`flywire_synapses_783.feather`) is **not** needed: the 852 MB pair×neuropil table is sufficient.

## Not in v1 (deliberately)

Mesh/skeleton loading (use `fafbseg` + meshparty), other connectomes (MANC/MAOL/MCNS/BANC), hosted leaderboard server, spiking neural simulation, per-synapse link prediction. The package is intentionally complete at this scope.

## Tests

```bash
pytest tests/ -v
```

Covers: graph assembly counts on a synthetic graph, feature finiteness, split disjointness/determinism, model forward + backward smoke tests, and (if the real cache is present) the real 139,255-node / 2,700,513-edge invariants.

## License & citation

Code: MIT. Data: **CC-BY-4.0**. Using the dataset means citing:

1. **Dorkenwald et al.** 2024. *Neuronal wiring diagram of an adult brain.* Nature. [doi:10.1038/s41586-024-07558-y](https://doi.org/10.1038/s41586-024-07558-y)
2. **Schlegel et al.** 2024. *Whole-brain annotation and multi-connectome cell typing of Drosophila.* Nature. [doi:10.1038/s41586-024-07686-5](https://doi.org/10.1038/s41586-024-07686-5)

Not affiliated with the FlyWire Consortium. Interactive exploration: [codex.flywire.ai](https://codex.flywire.ai) — analysis in Python: [navis](https://github.com/navis-org/navis) / [fafbseg](https://github.com/navis-org/fafbseg-py).

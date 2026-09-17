---
license: cc-by-4.0
tags:
  - connectome
  - neuroscience
  - drosophila
  - graph-neural-network
  - gnn
  - benchmark
task_categories:
  - graph-machine-learning
pretty_name: FlyWire FAFB v783 Connectome (GNN-ready)
---

# FlyWire FAFB v783 Connectome — GNN-ready package

The complete proofread wiring diagram of an adult female *Drosophila melanogaster* brain — **139,255 neurons and their synaptic connections** — repackaged as a ready-to-train graph dataset. Companion to the [`flywire-gnn`](https://github.com/) Python package.

This is a **dataset packaging** of two public, no-auth sources:

| File | Contents | Source |
|---|---|---|
| `connections.parquet` (474 MB) | **15,091,983 unique directed neuron→neuron pairs at ≥1 synapse**: `pre`, `post`, `syn_count`, and 6 synapse-weighted neurotransmitter probabilities (`gaba`, `ach`, `glut`, `oct`, `ser`, `da`) | FlyWire v783 release (Zenodo, CC-BY-4.0): [doi:10.5281/zenodo.10676866](https://doi.org/10.5281/zenodo.10676866) |
| `nodes.parquet` (2.9 MB) | **139,255 proofread neurons**: `root_id` + annotations `flow`, `super_class`, `cell_class`, `cell_sub_class`, `cell_type`, `hemibrain_type`, hemilineages, `top_nt`, `top_nt_conf`, `side`, `nerve`, `vfb_id`, `fbbt_id` | Supplementary Data 5 of Schlegel et al. 2024: [doi:10.1038/s41586-024-07686-5](https://doi.org/10.1038/s41586-024-07686-5) |
| `meta.json` | dataset statistics, feature schema, task definition | this package |

## Benchmark task: node classification on `super_class`

9 coarse classes (optic / central / sensory / visual_projection / ascending / descending / visual_centrifugal / motor / endocrine). **Every neuron is labeled.** The default benchmark graph uses directed edges at **≥5 synapses** (the published FAFB "connection" threshold) → **2,700,513 unique directed edges**.

- **Node features (174-dim, wiring-derived)**: `log1p(out_degree)`, `log1p(in_degree)`, `log1p(out/in synapses)`, `log1p` of out/in synapse counts across 79 neuropils, synapse-weighted mean NT profiles (in & out, 6 each).
- **Edge attributes**: `log1p(syn_count)` + 6 weighted NT probabilities.
- **Splits**: 70/15/15 train/val/test, stratified by label, seed 42.

## Baseline results (real runs, CPU, seed 42)

| Model | Test accuracy | Macro-F1 | Best val acc (epoch) | Epochs | Wall time |
|---|---|---|---|---|---|
| MLP (feat. only) | **0.9851** | 0.7115 | 0.9860 (200) | 200 | 212 s |
| GCN | 0.9166 | 0.4702 | 0.9202 (180) | 200 | 1414 s |
| GraphSAGE | 0.9812 | **0.7563** | 0.9820 (150) | 150 | 783 s |

**What the benchmark shows**: wiring-profile features alone nearly saturate accuracy (98.5%); message passing with mean aggregation (GraphSAGE) matches features and adds robustness on the rare classes (best macro-F1); GCN's normalized averaging over-smooths and trails. Report your model's accuracy and macro-F1 to compare on the same splits.

## License & citation

Connectivity data is **CC-BY-4.0**. If you use this dataset, cite:

1. Dorkenwald et al., *Neuronal wiring diagram of an adult brain*, Nature 2024. [doi:10.1038/s41586-024-07558-y](https://doi.org/10.1038/s41586-024-07558-y)
2. Schlegel et al., *Whole-brain annotation and multi-connectome cell typing of Drosophila*, Nature 2024. [doi:10.1038/s41586-024-07686-5](https://doi.org/10.1038/s41586-024-07686-5)

Annotations redistributed here are from the Schlegel et al. 2024 supplementary data; the connectivity graph is derived from the FlyWire v783 public release. This package is not affiliated with the FlyWire Consortium. Explore interactively at [codex.flywire.ai](https://codex.flywire.ai).

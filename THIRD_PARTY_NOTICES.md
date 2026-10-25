# Third-party code and data

This review artifact does not relicense third-party material. No blanket MIT or
other license is assigned to the package. Original rights and applicable terms
remain with the respective authors; confirm redistribution permissions before
public release. Submission-author identities are withheld for double-blind review.

## DnD-Net

The model, split/noise-generation helpers, finite-PPR preprocessing,
neighborhood-alignment loss and early stopping are extracted from the released
DnD-Net implementation, with attribution retained:

- Repository: https://github.com/kaize0409/DND-NET
- Paper: *Divide and Denoise: Empowering Simple Models for Robust
  Semi-Supervised Node Classification against Label Noise* (KDD 2024),
  https://doi.org/10.1145/3637528.3671798
- Files: `expose/official_ops.py`, `expose/early_stop.py`, `expose/dndnet.py`.

Only required definitions are included. The EXPOSE fitter and paired DnD-Net
wrapper separate test evaluation from training. The original losses and
early-stopping tie behavior are retained, including nonstandard operations
inside the released DnD-Net consistency/uncertainty helpers; they are not
silently corrected here. PubMed baseline parameters are explicitly marked as
an executable validation reconstruction rather than released official values.

## Benchmark graphs

`data/*.npz` contains numerical copies of the exact graph tensors used in the
experiments, not Python pickle files or precomputed representations:

- Cora-ML and CiteSeer: processed graphs distributed by DnD-Net.
- PubMed: PyTorch Geometric Planetoid dataset.
- Coauthor-CS: PyTorch Geometric Coauthor dataset.

Dataset authors retain all rights. Upstream sources and original benchmark
citations should be acknowledged when reusing these datasets. PyG dataset
documentation: https://pytorch-geometric.readthedocs.io/en/2.6.1/modules/datasets.html

This package supplies training code for EXPOSE and the paired DnD-Net baseline.
It does not include stored benchmark scores or implementations of the other
baseline methods cited in the paper.

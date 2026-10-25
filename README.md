# EXPOSE

PyTorch implementation of **Rethinking Label Noise in Graph Learning: A Propagation Exposure Perspective**.

## Installation

Tested with Python 3.10 and PyTorch 2.5.1 (CUDA 12.1).

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install torch==2.5.1 --index-url https://download.pytorch.org/whl/cu121
pip install -r requirements.txt
```

For PubMed and Coauthor-CS, a GPU with at least 24 GB memory and 32 GB system RAM are recommended. Preprocessing also requires several GB of disk space.

## Files and datasets

```text
expose/             Model, preprocessing and training code
data/               Cora-ML, CiteSeer, PubMed and Coauthor-CS
configs/table5.json Hyperparameters for each dataset and noise setting
run.py              Training entry point
aggregate.py        Calculate mean accuracy and standard deviation
requirements.txt    Python dependencies
```

Datasets are included as numerical `.npz` files containing features, edges and labels. No separate download is required. This release covers the four-graph experiments in Table 5, with runnable EXPOSE and DnD-Net implementations.

## Usage

Run commands from the repository root. Start with one run on Cora-ML with 40% uniform noise:

```bash
CUDA_VISIBLE_DEVICES=0 python run.py --cell cora_ml_u40 --seeds 1
```

Run all noise settings on each dataset (20 seeds per setting):

```bash
# Cora-ML
CUDA_VISIBLE_DEVICES=0 python run.py --cell cora_ml_clean cora_ml_u20 cora_ml_u40 cora_ml_u60 cora_ml_p20 cora_ml_p30 cora_ml_p40

# CiteSeer
CUDA_VISIBLE_DEVICES=0 python run.py --cell citeseer_clean citeseer_u20 citeseer_u40 citeseer_u60 citeseer_p20 citeseer_p30 citeseer_p40

# PubMed
CUDA_VISIBLE_DEVICES=0 python run.py --cell pubmed_clean pubmed_u20 pubmed_u40 pubmed_u60 pubmed_p20 pubmed_p30 pubmed_p40

# Coauthor-CS
CUDA_VISIBLE_DEVICES=0 python run.py --cell cs_clean cs_u20 cs_u40 cs_u60 cs_p20 cs_p30 cs_p40
```

`u20/u40/u60` denote 20/40/60% uniform noise; `p20/p30/p40` denote 20/30/40% pair noise. Parameters are loaded from `configs/table5.json`. Run seeds default to 1–20, with noise-generation seed fixed at 10.

Alternatively, run all four datasets at once:

```bash
CUDA_VISIBLE_DEVICES=0 python run.py
```

Add `--method dnd` for DnD-Net, or `--method both` for both methods:

```bash
CUDA_VISIBLE_DEVICES=0 python run.py --method both
```

Summarize completed 20-seed runs:

```bash
python aggregate.py --results outputs/table5 --out outputs/summary --require-all
```

Omit `--require-all` to summarize a subset of settings; each still requires 20 seeds. New results go to `outputs/`, and preprocessing is cached in `.cache/`. Completed runs are skipped. Use `--out` and `--cache` to change these directories; use separate caches for concurrent GPU jobs.

Training uses up to 200 epochs and validation-based checkpoint selection. Fresh GPU preprocessing can cause numerical differences from the original cached runs, so per-seed results need not be identical. PubMed DnD-Net parameters were reconstructed using validation data, not an official released configuration.

## Acknowledgments

The backbone, preprocessing, split/noise helpers and neighborhood consistency code build on [DnD-Net](https://github.com/kaize0409/DND-NET), *Divide and Denoise: Empowering Simple Models for Robust Semi-Supervised Node Classification against Label Noise* (KDD 2024). Cora-ML and CiteSeer use its processed graphs; PubMed and Coauthor-CS use the [PyTorch Geometric datasets](https://pytorch-geometric.readthedocs.io/en/2.6.1/modules/datasets.html). Third-party code and data retain their original rights and applicable terms; this repository does not relicense them.

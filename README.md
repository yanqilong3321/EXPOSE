# EXPOSE — Table 5 reproduction

Code, datasets and frozen hyperparameters for the four-graph experiments in
*Rethinking Label Noise in Graph Learning: A Propagation Exposure Perspective*.

This repository contains **no stored accuracy scores, result tables, experiment
logs, checkpoints, cached risk scores or historical training outputs**. All
supervision signals and results are computed by the supplied code when run.

## Contents

```text
expose/                 Method, data protocol, propagation, losses and training
configs/table5.json     Complete frozen hyperparameters for all 28 conditions
data/*.npz              Four benchmark graphs: raw features, edges and labels
run.py                  Recompute evidence and train EXPOSE / paired DnD-Net
aggregate.py            Summarize the new results produced by your runs
tests/                  Regression tests for protocol and source exclusion
tools/check_anonymity.py Source/data-file anonymity check
requirements.txt        Dependencies
THIRD_PARTY_NOTICES.md   Upstream code and dataset attribution
```

The current method keeps observed training labels one-hot; source-label soft
correction is disabled. SRE reliability estimation, source-loss reweighting,
ELP soft pseudo-labels for unlabeled targets and neighborhood consistency are
retained. There are no extra clean training labels or teacher/student models.

Runnable methods are **EXPOSE and paired DnD-Net**. This is not a reimplementation
of every baseline appearing in Table 5. The other baseline implementations and
stored table values are not included. ogbn-arxiv is outside this table's scope.

## 1. Install

The tested environment is Python 3.10.12, PyTorch 2.5.1+cu121, PyTorch Geometric
2.6.1, NumPy 2.2.6 and SciPy 1.15.3, on NVIDIA L40.

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install torch==2.5.1 --index-url https://download.pytorch.org/whl/cu121
python -m pip install -r requirements.txt
```

Use a driver compatible with CUDA 12.1. CPU execution is supported, but the
paper's runs used GPUs. The faithful implementation retains a dense finite-PPR
operator. For PubMed/Coauthor-CS, a 24 GB or larger GPU, 32 GB host RAM and tens
of GB of cache space are recommended. No additional graph downloads are needed:
`data/` includes the numerical snapshots used for the experiments. These `.npz`
files contain arrays only; they do not use Python pickle.

## 2. Check the installation

Run all commands from the repository root:

```bash
python -m unittest discover -s tests -v
python tools/check_anonymity.py
python run.py --help
```

The tests check all 28 configurations and 560 reproducible split/noise
realizations, source exclusion (including class counts), one-hot source targets
and the population-SD aggregation rule. They do not train all models.

## 3. Smoke test: one condition and one seed

```bash
CUDA_VISIBLE_DEVICES=0 python run.py \
  --cell cora_ml_u40 --seeds 1 --method expose \
  --out outputs/smoke --cache .cache/gpu0
```

This recomputes the PPR features/operator, feature-based witness evidence,
source reliability, loss weights and soft pseudo-labels, then trains the MLP.
It does not load archived risk scores or predictions. To also test the paired
DnD-Net baseline, use `--method both`.

## 4. Run the complete Table 5 conditions

Dataset names: `cora_ml`, `citeseer`, `pubmed`, `cs` (Coauthor-CS).
Condition names: `clean`, `u20`, `u40`, `u60`, `p20`, `p30`, `p40`.
For example, `pubmed_p30` means PubMed with 30% pair noise.

```bash
# EXPOSE only: 28 conditions × 20 seeds = 560 fits
CUDA_VISIBLE_DEVICES=0 python run.py --method expose \
  --out outputs/table5 --cache .cache/gpu0

# EXPOSE and paired DnD-Net: 1,120 fits in total
CUDA_VISIBLE_DEVICES=0 python run.py --method both \
  --out outputs/table5 --cache .cache/gpu0
```

Without `--seeds`, runs use **all seeds 1–20**. Without `--cell`, they cover all
28 conditions (24 noisy and four clean). To run only selected cells:

```bash
CUDA_VISIBLE_DEVICES=0 python run.py --method both \
  --cell cora_ml_u20 cora_ml_u40 cora_ml_u60 --seeds 1 2 3 \
  --out outputs/subset --cache .cache/gpu0
```

For two GPUs, launch the following in separate terminals. The work is disjoint;
separate caches avoid concurrent writes during graph preprocessing.

```bash
# Terminal 1
CUDA_VISIBLE_DEVICES=0 python run.py --method both \
  --cell cora_ml_clean cora_ml_u20 cora_ml_u40 cora_ml_u60 cora_ml_p20 cora_ml_p30 cora_ml_p40 \
         pubmed_clean pubmed_u20 pubmed_u40 pubmed_u60 pubmed_p20 pubmed_p30 pubmed_p40 \
  --out outputs/table5 --cache .cache/gpu0

# Terminal 2
CUDA_VISIBLE_DEVICES=1 python run.py --method both \
  --cell citeseer_clean citeseer_u20 citeseer_u40 citeseer_u60 citeseer_p20 citeseer_p30 citeseer_p40 \
         cs_clean cs_u20 cs_u40 cs_u60 cs_p20 cs_p30 cs_p40 \
  --out outputs/table5 --cache .cache/gpu1
```

## 5. Summarize your new runs

```bash
python aggregate.py --results outputs/table5 --out outputs/aggregate --require-all
```

This writes `outputs/aggregate/summary.csv`, with mean test accuracy (%) and
population standard deviation (`ddof=0`). Each condition/method must contain
exactly seeds 1–20, with no duplicates. `--require-all` additionally requires
all 28 cells for each method present. Omit it to summarize a subset of cells,
each of which still needs its complete 20 runs. The single-seed smoke test
cannot be used as the paper's 20-run result.

Training creates the following **new local outputs**, not shipped in the repo:

```text
outputs/<run>/<cell>/s<seed>/<method>/
    result.json         Accuracy and checkpoint-selection metadata
    predictions.npz     Predictions and training history
    model.pt            Selected model weights
    evidence.npz        Fresh source evidence (EXPOSE only)
.cache/                 Recomputed label-free graph preprocessing
```

Completed runs are skipped when resumed. Do not launch duplicate owners of the
same cell/seed/method. Partial runs with predictions but no completion record
are not overwritten: inspect them or use a new output directory. `.gitignore`
excludes all generated output/cache directories from version control.

## Protocol and hyperparameters

- Seeds 1–20; **noise seed fixed at 10** for every run.
- Maximum 200 epochs; patience 100; checkpoint selected on validation labels.
- DnD-Net's split rounding is preserved: `int(0.05*n/classes+1)` training nodes
  **per class**, `int(0.20*n+1)` validation nodes, and the remaining nodes for test.
- Pair noise flips class `c` to `(c-1) % classes`. Uniform noise samples from the
  corresponding class transition matrix. The requested rate is a probability,
  not a forced exact count of corrupted nodes.
- Source scoring uses observed training labels, raw features and graph
  relations only. The assessed source is excluded from all votes and counts.
- Validation labels do not enter evidence construction or training losses.
  Test labels are evaluated only after the model and checkpoint are fixed.
- `configs/table5.json` specifies every condition's optimizer, backbone, PPR,
  evidence gate, weighting, attenuation and auxiliary-loss schedule. No search
  is performed by `run.py` and no test scores are needed to resolve parameters.
- CiteSeer P-30 includes its selected feature-length normalization, voting
  exponents, KL consistency/confidence filter and validation-loss tie break.
  PubMed P-20/P-30 use their selected training settings. PubMed DnD-Net baseline
  parameters are an executable validation reconstruction, **not official
  released PubMed parameters**.

The exact split counts produced by this code are:

| Dataset | Nodes | Train | Validation | Test |
|---|---:|---:|---:|---:|
| Cora-ML | 2,810 | 147 | 563 | 2,100 |
| CiteSeer | 2,110 | 108 | 423 | 1,579 |
| PubMed | 19,717 | 987 | 3,944 | 14,786 |
| Coauthor-CS | 18,333 | 930 | 3,667 | 13,736 |

## Numerical reproducibility

The original experiments cached GPU-computed propagation inputs. Fresh CUDA
sparse reductions can differ by around 1e-7, which may be amplified by training
and validation checkpoint selection. New per-seed results are therefore **not
guaranteed to be bitwise identical** to the original runs. The compact release
does not include the original multi-GB propagation caches or any result data.
During packaging, the refactored trainers were checked on one condition per
graph for both methods using the original fixed inputs, with all eight checked
checkpoints and predictions matching exactly. This was not a new full 560-run
evaluation of the refactored code.

## Attribution and anonymity

See `THIRD_PARTY_NOTICES.md`. Third-party authors and upstream repository links
are retained as required attribution. The source/data content is checked for
submission-author information, personal paths and credentials. Repository owner,
account details and Git history must be considered separately; a clean code
archive does not make a personally owned GitHub URL anonymous.

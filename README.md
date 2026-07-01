# VOID Cell

Standalone VOID-style Perturb-seq trainer for virtual cell perturbation. This is
not a wrapper around `scDFM`: it trains directly from AnnData expression matrices
using gene columns, perturbation IDs, a signed coexpression graph, and a spectral
3D/4D gene manifold.

The current best path is:

- raw expression data, no gene tokenizer
- delta flow target: predict the perturbation residual and add it to control
- signed/weighted gene neighborhoods with separate positive and negative channels
- Muon for hidden 2D matrices, AdamW for embeddings/norms/biases/output heads
- CFM + MMD + reconstruction + bulk losses

## Codebase Structure

The codebase is split into 5 core modules:

- `run.py`: The main entry point. Contains argument parsing, the training loop, and the custom optimizers (`HybridMuonAdamW`).
- `model.py`: Contains the `VoidCellModel` architecture and the spectral coexpression geometry builders.
- `blocks.py`: Houses the foundational PyTorch modules, layers (like `ValueEncoder`), and the `VoidGeneBlock` neighborhood message-passing layers.
- `data.py`: Handles dataset parsing, PyTorch dataloaders, and splitting strategies for AnnData files.
- `eval.py`: Holds all evaluation generation logic, distance metrics, distribution-matching metrics, and fast evaluation paths.

## Data Layout

Put the downloaded datasets here:

```text
data/norman.h5ad
data/combosciplex.h5ad
```

Official dataset pages:

- Norman: https://figshare.com/articles/dataset/Norman_et_al_2019_Science_labeled_Perturb-seq_data/24688110
- ComboSciPlex subset: https://figshare.com/articles/dataset/combosciplex/25062230?file=44229635

## Running the Model

Here is the recommended command to run the model using the stable recipe. Note the usage of `run.py` instead of the older monolithic script:

```bash
%env PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

%cd /content/protein

!python run.py \
  --dataset combosciplex \
  --data-path data/combosciplex.h5ad \
  --output-dir runs/combosciplex_void_world_upgrade \
  --steps 2000 \
  --batch-size 64 \
  --lr 5e-5 \
  --optimizer muon \
  --muon-lr 0.005 \
  --warmup-steps 200 \
  --grad-clip 0.5 \
  --n-top-genes 3000 \
  --infer-top-genes 1000 \
  --store-dtype float16 \
  --graph-cells 8192 \
  --dim 192 \
  --hidden 512 \
  --encode-blocks 4 \
  --think-steps 4 \
  --graph-k 30 \
  --manifold-dim 4 \
  --source-memory static \
  --action-field drug_manifold \
  --dynamic-edge-gate \
  --flow-target delta \
  --delta-noise-start 0.25 \
  --delta-noise-end 1.0 \
  --delta-noise-warmup 2000 \
  --gamma 0.25 \
  --recon-weight 0.5 \
  --bulk-loss-weight 2.0 \
  --action-aux-weight 0.01 \
  --eval-every 500 \
  --eval-top-genes 1000 \
  --eval-gene-selection hvg_test \
  --eval-cells 128 \
  --eval-batch-size 32 \
  --ode-method euler \
  --no-cell-eval \
  --precision fp16 \
  --log-every 25 \
  --use-mmd \
  --num-workers 2 \
  --compile
```

## Reading The Logs

Training lines look like this:

```text
[train] step 500/5000 | loss ... | lr muon=...,adamw=... | flow ... | rec ... | bulk ... | mmd ...
```

Fast eval lines look like this:

```text
[eval] step 500 | fast | L2 ... | MSE ... | MAE ... | DE-Spear ... | PearsonDelta ... | DS n/a | DM n/a | PearsonHat ... | PearsonHat20 ...
```

Metric priority for model usefulness:

- `DE-Spear` and `PearsonDelta`: most important for perturbation direction and
  average drug effect.
- `L2`, `MSE`, `MAE`: useful sanity checks for expression-level distance, but
  lower is not automatically more biologically useful.
- `PearsonHat` and `PearsonHat20`: residual single-cell heterogeneity metrics.
  They are harder and noisier because Perturb-seq is unpaired.
- `DS` and `DM`: official `cell_eval` metrics. They are `n/a` in fast eval.

## Implementation Notes

- The graph is built from training/control expression only, not test rows.
- Gene neighborhoods are signed and weighted: positive and negative edges are
  pooled separately.
- The spectral gene manifold uses `--manifold-dim 4` by default.
- The model processes raw gene expression fields and direct gene IDs. There is
  no scGPT/Geneformer-style tokenization.
- `--flow-target delta` trains on `target - source`; generated expression is
  `source + predicted_delta`.
- MMD is still the main distribution-level loss. NCE is not used.

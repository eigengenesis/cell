# VOID Cell

Standalone Perturb-seq training path for a VOID-style gene world model.

This folder intentionally does not import `scDFM`. It trains from raw AnnData
expression matrices and direct gene-column indices:

- expression values come from `adata.X` after dataset preprocessing
- gene identity is `torch.arange(n_genes)`, not a tokenizer
- perturbations are dataset condition IDs plus optional direct gene-column IDs
- the gene graph is built from train expression coexpression
- the loss is CFM plus optional MMD/reconstruction/bulk terms; NCE is not used

## Data

Place the datasets here:

```text
data/norman.h5ad
data/combosciplex.h5ad
```

Official sources:

- Norman: https://figshare.com/articles/dataset/Norman_et_al_2019_Science_labeled_Perturb-seq_data/24688110
- ComboSciPlex subset: https://figshare.com/articles/dataset/combosciplex/25062230?file=44229635

## Smoke Run

```bash
python void_cell/train_void_cell.py \
  --dataset norman \
  --data-path data/norman.h5ad \
  --output-dir runs/norman_void_cell_smoke \
  --steps 100 \
  --eval-every 100 \
  --no-cell-eval
```

## Benchmark Run

```bash
python void_cell/train_void_cell.py \
  --dataset norman \
  --data-path data/norman.h5ad \
  --output-dir runs/norman_void_cell_additive \
  --split additive \
  --fold 1 \
  --steps 200000 \
  --batch-size 32 \
  --lr 1e-4 \
  --optimizer adamw \
  --n-top-genes 5000 \
  --infer-top-genes 1000 \
  --manifold-dim 4 \
  --recon-weight 0.5 \
  --bulk-loss-weight 2.0 \
  --eval-top-genes 1000 \
  --use-mmd
```

Install `cell-eval==0.5.42` to enable the same official evaluator used by
scDFM. The script writes checkpoints, official per-eval `results.csv` /
`agg_results.csv`, and `metrics.csv` with the benchmark columns:
`Setting, Model, L2, MSE, MAE, DE-Spearman, Pearson_Delta, DS,
Pearson_Delta_Hat, Pearson_Delta_Hat20`.

The gene field uses a signed weighted coexpression graph plus a spectral
3D/4D gene manifold. Positive and negative edges are contracted separately,
so anticorrelated genes are not averaged into the same message as correlated
genes.

Two training variants are exposed for ablation:

- `--flow-target cell` learns the absolute perturbed expression field.
- `--flow-target delta` learns the perturbation residual and adds it back to
  the control cell at inference.

`--optimizer muon` uses Muon only for hidden 2D weight matrices and keeps
embeddings, biases, normalization parameters, and output heads on AdamW.

## Hyperparameter Sweep

Use the sweep harness before longer runs:

```bash
python void_cell/sweep_void_cell.py \
  --dataset norman \
  --data-path data/norman.h5ad \
  --base-output-dir runs/sweep_norman_micro \
  --profile micro \
  --steps 130 \
  --eval-every 100 \
  --dim 128 \
  --hidden 384 \
  --encode-blocks 2 \
  --think-steps 4 \
  --n-top-genes 2000 \
  --infer-top-genes 700 \
  --eval-top-genes 500 \
  --eval-cells 24 \
  --resume
```

The `micro` profile screens absolute flow, delta flow, and Muon-on-delta.
After Muon/delta wins, use `--profile muon` to sweep the narrower optimizer,
bulk-loss, and MMD-gamma neighborhood around that winner.
It writes `sweep_plan.json`, `sweep_results.csv`, and
`best_screen_command.sh`. Add `--promote-top-k 2 --official-steps 700
--official-eval-every 500` to rerun the best configs with official
`cell-eval`.

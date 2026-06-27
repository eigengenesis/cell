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

## Data Layout

Put the downloaded datasets here:

```text
data/norman.h5ad
data/combosciplex.h5ad
```

Official dataset pages:

- Norman: https://figshare.com/articles/dataset/Norman_et_al_2019_Science_labeled_Perturb-seq_data/24688110
- ComboSciPlex subset: https://figshare.com/articles/dataset/combosciplex/25062230?file=44229635

## Best ComboSciPlex Trial

This is the stable recipe that recovered the strong ComboSciPlex fast-eval
numbers. Keep this as the baseline before testing new ideas.

Important: do not add `--eval-seed`, `--eval-delta-noise-scale`, or
`--neighbor-gate` when comparing against the current best run. The old strong
run used the default stochastic eval path.

```bash
%cd /content/protein
!python void_cell/train_void_cell.py \
  --dataset combosciplex \
  --data-path data/combosciplex.h5ad \
  --output-dir runs/combosciplex_void_oldrng_stable_retry \
  --steps 10000 \
  --max-hours 3 \
  --save-every 500 \
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
  --neighbor-chunk 1024 \
  --flow-target delta \
  --gamma 0.25 \
  --recon-weight 0.5 \
  --bulk-loss-weight 2.0 \
  --hetero-weight 0.0 \
  --delta-noise-scale 1.0 \
  --eval-every 500 \
  --eval-top-genes 1000 \
  --eval-cells 64 \
  --eval-batch-size 32 \
  --no-cell-eval \
  --precision fp16 \
  --log-every 25 \
  --use-mmd \
  --num-workers 2 \
  --compile
```

Expected quick signal: `genes=3000 perturbations=18 params=6814338`. If the
parameter count jumps to about `10426498`, `--neighbor-gate` is on and this is
not the baseline architecture.

## Best Norman Trial

Use this for the Norman additive split trial. Norman is heavier than
ComboSciPlex, so `--checkpoint-blocks` is usually worth keeping on Colab.

```bash
%cd /content/protein
!python void_cell/train_void_cell.py \
  --dataset norman \
  --data-path data/norman.h5ad \
  --output-dir runs/norman_void_muon_delta_gamma025_2k \
  --split additive \
  --fold 1 \
  --steps 2000 \
  --save-every 500 \
  --batch-size 96 \
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
  --think-steps 8 \
  --graph-k 30 \
  --manifold-dim 4 \
  --neighbor-chunk 1024 \
  --flow-target delta \
  --gamma 0.25 \
  --recon-weight 0.5 \
  --bulk-loss-weight 2.0 \
  --hetero-weight 0.0 \
  --delta-noise-scale 1.0 \
  --eval-every 500 \
  --eval-top-genes 1000 \
  --eval-cells 64 \
  --eval-batch-size 32 \
  --no-cell-eval \
  --precision fp16 \
  --log-every 25 \
  --use-mmd \
  --num-workers 2 \
  --checkpoint-blocks
```

For a longer Norman run, keep the same recipe and increase `--steps` or add a
wall-clock cap with `--max-hours`.

## Smoke Test

Use this only to check that the data path, dependencies, and CUDA setup work.

```bash
python void_cell/train_void_cell.py \
  --dataset norman \
  --data-path data/norman.h5ad \
  --output-dir runs/norman_void_smoke \
  --steps 100 \
  --eval-every 50 \
  --batch-size 16 \
  --dim 96 \
  --hidden 256 \
  --encode-blocks 2 \
  --think-steps 4 \
  --n-top-genes 1200 \
  --infer-top-genes 600 \
  --no-cell-eval
```

## Reading The Logs

Training lines look like this:

```text
[train] step 500/10000 | loss ... | lr muon=...,adamw=... | flow ... | rec ... | bulk ... | mmd ...
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

## Official Cell Eval

Fast eval is for iteration. Official metrics require `cell_eval` and can be slow
or dependency-sensitive on Colab.

To run official eval at checkpoints, remove `--no-cell-eval` or pass
`--cell-eval`. The script writes:

```text
runs/.../eval_step_XXXXXXX/results.csv
runs/.../eval_step_XXXXXXX/agg_results.csv
runs/.../metrics.csv
```

By default, `de_direction_match` is skipped for speed. To compute `DM`, pass a
custom `--cell-eval-skip-metrics` list that excludes `de_direction_match`.

## Experimental Flags

These are real ablation knobs, not default best-run settings:

- `--neighbor-gate`: enables drug-gated neighbor messages. This increases the
  ComboSciPlex 192-dim model from about `6.8M` to `10.4M` parameters. Treat it
  as a separate architecture ablation.
- `--eval-delta-noise-scale 0.0`: deterministic eval noise for delta flow. It
  can change every generated metric, not only `PearsonHat`.
- `--eval-seed`: fixed eval RNG. Useful for controlled comparisons, but do not
  use it when comparing directly to the old stochastic-eval baseline.
- `--hetero-weight`: experimental control-residual preservation loss. It is off
  in the stable recipe because `0.1` hurt the ComboSciPlex trial.
- `--dir-weight`: directional cosine loss. Still experimental.

## Hyperparameter Sweeps

Use the sweep harness for short screens before spending hours:

```bash
python void_cell/sweep_void_cell.py \
  --dataset norman \
  --data-path data/norman.h5ad \
  --base-output-dir runs/sweep_norman_muon \
  --profile muon \
  --steps 350 \
  --eval-every 300 \
  --batch-size 96 \
  --precision fp16 \
  --resume
```

The sweep writes:

```text
runs/.../sweep_plan.json
runs/.../sweep_results.csv
runs/.../best_screen_command.sh
```

Use `--promote-top-k` or `--promote-trials` only after the short screen has a
clear winner.

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

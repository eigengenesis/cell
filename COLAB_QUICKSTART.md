# VOID Cell Colab Quickstart

## 1. Install Runtime Dependencies

```python
!pip install -q scanpy anndata scipy pandas scikit-learn h5py gdown cell-eval==0.5.42 pdex==0.1.21
```

Torch is already installed on Colab GPU runtimes. `cell-eval==0.5.42`
and `pdex==0.1.21` are the evaluator versions pinned by the scDFM environment.
The trainer uses raw AnnData expression matrices and direct gene-column IDs;
there is no gene tokenizer and no NCE loss path.

## 2. Put Code and Data in Place

Upload or clone the project so this path exists:

```text
/content/protein/void_cell/train_void_cell.py
```

Create the data folder:

```python
!mkdir -p /content/protein/data
```

Download from the official scDFM Google Drive folder:

```python
!gdown --folder "https://drive.google.com/drive/folders/1cNpYAt9jVWZN82miNZtkP10YeSo7hufL?usp=sharing" -O /content/scdfm_data
!find /content/scdfm_data -name "*.h5ad" -print
```

Then place the files as:

```text
/content/protein/data/norman.h5ad
/content/protein/data/combosciplex.h5ad
```

If `gdown --folder` hits a quota or permission issue, download the `.h5ad`
files manually from the same folder and upload them to `/content/protein/data`.

## 3. Fast Norman Smoke Test

```python
%cd /content/protein
!python void_cell/train_void_cell.py \
  --dataset norman \
  --data-path data/norman.h5ad \
  --output-dir runs/norman_void_smoke \
  --steps 200 \
  --batch-size 16 \
  --n-top-genes 1000 \
  --infer-top-genes 500 \
  --store-dtype float16 \
  --graph-cells 2048 \
  --dim 96 \
  --hidden 256 \
  --encode-blocks 2 \
  --think-steps 4 \
  --eval-every 0 \
  --eval-cells 64 \
  --eval-batch-size 32 \
  --no-cell-eval \
  --use-mmd
```

## 4. Main Norman Additive Run

```python
%cd /content/protein
!python void_cell/train_void_cell.py \
  --dataset norman \
  --data-path data/norman.h5ad \
  --output-dir runs/norman_void_additive \
  --split additive \
  --fold 1 \
  --steps 200000 \
  --batch-size 32 \
  --lr 1e-4 \
  --warmup-steps 100 \
  --n-top-genes 5000 \
  --infer-top-genes 1000 \
  --store-dtype float16 \
  --graph-cells 8192 \
  --dim 192 \
  --hidden 512 \
  --encode-blocks 4 \
  --think-steps 8 \
  --graph-k 30 \
  --manifold-dim 4 \
  --neighbor-chunk 128 \
  --checkpoint-blocks \
  --recon-weight 0.5 \
  --bulk-loss-weight 2.0 \
  --eval-every 5000 \
  --eval-top-genes 1000 \
  --eval-cells 128 \
  --eval-batch-size 64 \
  --cell-eval-threads 2 \
  --use-mmd
```

## 4a. Faster T4 Utilization Test

After the low-RAM smoke works, use this to push VRAM/throughput before starting
a long run:

```python
%cd /content/protein
!python void_cell/train_void_cell.py \
  --dataset norman \
  --data-path data/norman.h5ad \
  --output-dir runs/norman_void_t4_test \
  --steps 1000 \
  --batch-size 64 \
  --n-top-genes 2000 \
  --infer-top-genes 1000 \
  --store-dtype float16 \
  --graph-cells 4096 \
  --dim 192 \
  --hidden 512 \
  --encode-blocks 4 \
  --think-steps 8 \
  --graph-k 30 \
  --neighbor-chunk 128 \
  --checkpoint-blocks \
  --eval-every 0 \
  --no-cell-eval \
  --precision fp16 \
  --log-every 25 \
  --use-mmd
```

If VRAM is still low and system RAM is stable, increase `--batch-size` to 96
or 128. If system RAM rises too high, keep `--n-top-genes 2000` and increase
only batch size first.

## 4b. Time-Boxed 2-Hour Trial

Use this only after the short sweep points to a winner. It trains until either
`--steps` is hit or two hours pass, then saves `last.pt` and `final.pt`.
Replace the optimizer / flow-target flags with the best screen config if the
sweep disagrees with this candidate.

```python
%cd /content/protein
!python void_cell/train_void_cell.py \
  --dataset norman \
  --data-path data/norman.h5ad \
  --output-dir runs/norman_void_additive_trial2h \
  --split additive \
  --fold 1 \
  --steps 200000 \
  --max-hours 2 \
  --save-every 1000 \
  --batch-size 96 \
  --lr 1e-4 \
  --optimizer muon \
  --muon-lr 0.02 \
  --warmup-steps 100 \
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
  --neighbor-chunk 128 \
  --checkpoint-blocks \
  --flow-target delta \
  --recon-weight 0.5 \
  --bulk-loss-weight 2.0 \
  --eval-every 500 \
  --eval-top-genes 1000 \
  --eval-cells 64 \
  --eval-batch-size 32 \
  --cell-eval-threads 2 \
  --precision fp16 \
  --log-every 25 \
  --use-mmd
```

## 4c. Short Optimizer / Flow / Loss Sweep

Screen several short proxy runs first. This pass uses a smaller model/gene set
so it can reject bad configs before spending Colab-hours on full trials. The
`micro` profile compares absolute-cell flow, residual delta-flow, and
Muon-on-delta while keeping MMD enabled.

```python
%cd /content/protein
!python void_cell/sweep_void_cell.py \
  --dataset norman \
  --data-path data/norman.h5ad \
  --base-output-dir runs/sweep_norman_micro \
  --profile micro \
  --steps 130 \
  --eval-every 100 \
  --batch-size 64 \
  --n-top-genes 2000 \
  --infer-top-genes 700 \
  --store-dtype float16 \
  --graph-cells 4096 \
  --dim 128 \
  --hidden 384 \
  --encode-blocks 2 \
  --think-steps 4 \
  --graph-k 30 \
  --manifold-dim 4 \
  --neighbor-chunk 1024 \
  --eval-top-genes 500 \
  --eval-cells 24 \
  --eval-batch-size 24 \
  --warmup-steps 40 \
  --num-workers 0 \
  --precision fp16 \
  --resume
```

Then confirm the top configs with the full model. Completed screen runs are
skipped, and only the top configs are rerun longer with official `cell-eval`.

```python
%cd /content/protein
!python void_cell/sweep_void_cell.py \
  --dataset norman \
  --data-path data/norman.h5ad \
  --base-output-dir runs/sweep_norman_micro \
  --profile micro \
  --steps 130 \
  --eval-every 100 \
  --batch-size 64 \
  --n-top-genes 2000 \
  --infer-top-genes 700 \
  --store-dtype float16 \
  --graph-cells 4096 \
  --dim 128 \
  --hidden 384 \
  --encode-blocks 2 \
  --think-steps 4 \
  --graph-k 30 \
  --manifold-dim 4 \
  --neighbor-chunk 1024 \
  --eval-top-genes 500 \
  --eval-cells 24 \
  --eval-batch-size 24 \
  --warmup-steps 40 \
  --num-workers 0 \
  --precision fp16 \
  --resume \
  --promote-top-k 2 \
  --official-steps 500 \
  --official-eval-every 400 \
  --official-batch-size 96 \
  --official-n-top-genes 3000 \
  --official-infer-top-genes 1000 \
  --official-graph-cells 8192 \
  --official-dim 192 \
  --official-hidden 512 \
  --official-encode-blocks 4 \
  --official-think-steps 8 \
  --official-eval-top-genes 1000 \
  --official-eval-cells 48 \
  --official-eval-batch-size 32 \
  --official-num-workers 2 \
  --official-warmup-steps 100
```

If Muon/delta wins the promotion, run the focused Muon neighborhood next:

```python
%cd /content/protein
!python void_cell/sweep_void_cell.py \
  --dataset norman \
  --data-path data/norman.h5ad \
  --base-output-dir runs/sweep_norman_muon \
  --profile muon \
  --steps 250 \
  --eval-every 200 \
  --batch-size 64 \
  --n-top-genes 2000 \
  --infer-top-genes 700 \
  --store-dtype float16 \
  --graph-cells 4096 \
  --dim 128 \
  --hidden 384 \
  --encode-blocks 2 \
  --think-steps 4 \
  --graph-k 30 \
  --manifold-dim 4 \
  --neighbor-chunk 1024 \
  --eval-top-genes 500 \
  --eval-cells 24 \
  --eval-batch-size 24 \
  --warmup-steps 60 \
  --num-workers 0 \
  --precision fp16 \
  --resume
```

## 5. ComboSciPlex Run

```python
%cd /content/protein
!python void_cell/train_void_cell.py \
  --dataset combosciplex \
  --data-path data/combosciplex.h5ad \
  --output-dir runs/combosciplex_void \
  --steps 200000 \
  --batch-size 32 \
  --lr 1e-4 \
  --warmup-steps 100 \
  --n-top-genes 5000 \
  --infer-top-genes 1000 \
  --store-dtype float16 \
  --graph-cells 8192 \
  --dim 192 \
  --hidden 512 \
  --encode-blocks 4 \
  --think-steps 8 \
  --graph-k 30 \
  --manifold-dim 4 \
  --recon-weight 0.5 \
  --bulk-loss-weight 2.0 \
  --eval-every 5000 \
  --eval-top-genes 1000 \
  --eval-cells 128 \
  --eval-batch-size 64 \
  --cell-eval-threads 2 \
  --use-mmd
```

## 6. Outputs

Each run writes:

```text
runs/<run_name>/config.json
runs/<run_name>/geometry_top30_m4.pt
runs/<run_name>/last.pt
runs/<run_name>/final.pt
runs/<run_name>/metrics.csv
runs/<run_name>/eval_step_<step>/results.csv
runs/<run_name>/eval_step_<step>/agg_results.csv
```

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

Upload or clone the project so these files exist in `/content/protein/cell`:
- `run.py`: Entry point and training loop.
- `model.py`: Model architecture.
- `blocks.py`: Neural network layers.
- `data.py`: Dataset logic.
- `eval.py`: Evaluation metrics.

Create the data folder:

```python
!mkdir -p /content/protein/cell/data
```

Download from the official scDFM Google Drive folder:

```python
!gdown --folder "https://drive.google.com/drive/folders/1cNpYAt9jVWZN82miNZtkP10YeSo7hufL?usp=sharing" -O /content/scdfm_data
!find /content/scdfm_data -name "*.h5ad" -print
```

Then place the files as:

```text
/content/protein/cell/data/norman.h5ad
/content/protein/cell/data/combosciplex.h5ad
```

If `gdown --folder` hits a quota or permission issue, download the `.h5ad`
files manually from the same folder and upload them to `/content/protein/cell/data`.

## 3. Fast Norman Smoke Test

```python
%cd /content/protein/cell
!python run.py \
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
%cd /content/protein/cell
!python run.py \
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

## 5. ComboSciPlex Run

```python
%env PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

%cd /content/protein/cell

!python run.py \
  --dataset combosciplex \
  --data-path data/combosciplex.h5ad \
  --output-dir runs/combosciplex_ab_directional_residual_gate_hvg \
  --steps 5000 \
  --max-hours 5 \
  --save-every 1000 \
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
  --directional-shifts \
  --directional-residual-gate \
  --directional-gate-init -4.0 \
  --shift-dims 3 \
  --shift-stencil cube \
  --shift-temperature 4.0 \
  --shift-code-strength 1.0 \
  --neighbor-chunk 512 \
  --flow-target delta \
  --gamma 0.25 \
  --recon-weight 0.5 \
  --bulk-loss-weight 2.0 \
  --hetero-weight 0.0 \
  --delta-noise-scale 1.0 \
  --eval-every 500 \
  --eval-top-genes 1000 \
  --eval-gene-selection hvg_test \
  --eval-cells 128 \
  --eval-seed 123 \
  --eval-batch-size 16 \
  --no-cell-eval \
  --precision fp16 \
  --log-every 25 \
  --use-mmd \
  --num-workers 2 \
  --compile \
  --compile-mode default
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

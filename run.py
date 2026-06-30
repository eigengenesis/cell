#!/usr/bin/env python3
"""
Standalone VOID Cell trainer.

No scDFM imports and no gene tokenizer. Genes are direct AnnData column indices.
%env PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

%cd /content/protein
!python void_cell/latest.py \
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
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import pickle
import random
import time
from dataclasses import asdict, dataclass
from pathlib import Path

from model import VoidCellModel, build_coexpression_geometry
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint
from torch.utils.data import DataLoader, Dataset


COMBOSCIPLEX_DEFAULT_TEST = [
    "Panobinostat+Crizotinib",
    "Panobinostat+Curcumin",
    "Panobinostat+SRT1720",
    "Panobinostat+Sorafenib",
    "SRT2104+Alvespimycin",
    "control+Alvespimycin",
    "control+Dacinostat",
]

DEFAULT_CELL_EVAL_SKIP = ",".join(
    [
        "mse_delta",
        "mae_delta",
        "discrimination_score_l2",
        "discrimination_score_cosine",
        "pearson_edistance",
        "overlap_at_N",
        "overlap_at_50",
        "overlap_at_100",
        "overlap_at_200",
        "overlap_at_500",
        "precision_at_N",
        "precision_at_50",
        "precision_at_100",
        "precision_at_200",
        "precision_at_500",
        "de_spearman_sig",
        "de_direction_match",
        "de_sig_genes_recall",
        "de_nsig_counts",
        "pr_auc",
        "roc_auc",
        "clustering_agreement",
    ]
)


def require_anndata_stack():
    try:
        import anndata as ad
        import scanpy as sc
        from scipy import sparse, stats
    except Exception as exc:
        raise RuntimeError(
            "VOID Cell needs anndata, scanpy, scipy, and their normal scientific "
            "stack. Install the scDFM environment or `pip install anndata scanpy scipy`."
        ) from exc
    return ad, sc, sparse, stats


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def dense_array(x):
    if hasattr(x, "toarray"):
        return x.toarray()
    return np.asarray(x)


def storage_dtype(name: str):
    if name == "float16":
        return np.float16
    if name == "float32":
        return np.float32
    raise ValueError(f"Unsupported storage dtype: {name}")


def normalize_condition(condition: str) -> str:
    condition = str(condition).replace("ctrl", "control")
    if condition == "control":
        return "control+control"
    return condition


def split_condition(condition: str) -> tuple[str, str]:
    parts = normalize_condition(condition).split("+")
    if len(parts) == 1:
        return parts[0], "control"
    return parts[0], parts[-1]


@dataclass
class PreparedData:
    x: np.ndarray
    gene_names: list[str]
    conditions: np.ndarray
    modes: np.ndarray
    is_control: np.ndarray
    perturbation_ids: np.ndarray
    perturbation_gene_ids: np.ndarray
    perturbation_names: list[str]
    train_conditions: list[str]
    test_conditions: list[str]


def highly_variable_or_all(adata, sc, n_top_genes: int):
    if n_top_genes <= 0 or n_top_genes >= adata.n_vars:
        return adata
    sc.pp.highly_variable_genes(adata, inplace=True, n_top_genes=n_top_genes)
    return adata[:, adata.var["highly_variable"].to_numpy()].copy()


def load_scdfm_norman_split(path: Path, split: str, fold: int) -> set[str] | None:
    split_name = "split_results_unseen.pkl" if split == "unseen" else "split_results.pkl"
    split_path = path.parent / "norman" / split_name
    if not split_path.exists():
        return None
    with split_path.open("rb") as f:
        split_results = pickle.load(f)
    test = list(split_results[int(fold)]["test"])
    if split == "combinations":
        test = test[:15]
        held_genes = {g for cond in test for g in normalize_condition(cond).split("+") if g != "control"}
        test.extend([f"{g}+control" for g in held_genes])
    return {normalize_condition(cond) for cond in test}


def prepare_norman(path: Path, n_top_genes: int, split: str, fold: int, seed: int, store_dtype: str) -> PreparedData:
    _, sc, _, _ = require_anndata_stack()
    adata = sc.read_h5ad(path)
    adata.obs["condition"] = adata.obs["condition"].map(normalize_condition)

    # Match the benchmark spirit: HVGs plus perturbed genes forced into the field.
    sc.pp.highly_variable_genes(adata, inplace=True, n_top_genes=n_top_genes)
    pert_names = sorted({p for c in adata.obs["condition"].unique() for p in c.split("+")})
    for pert in pert_names:
        if pert in adata.var_names:
            adata.var.loc[pert, "highly_variable"] = True
    adata = adata[:, adata.var["highly_variable"].to_numpy()].copy()

    conditions = adata.obs["condition"].to_numpy()
    is_control = np.array([c == "control+control" for c in conditions])
    all_conditions = np.array(sorted(set(conditions)))
    double_conditions = np.array([c for c in all_conditions if "control" not in c])
    cached_test = load_scdfm_norman_split(path, split, fold)
    if cached_test is not None:
        test = cached_test
    elif split in ("additive", "combinations"):
        shuffled = double_conditions.copy()
        # scDFM writes split_results.pkl with np.random.seed(42 + fold) and
        # np.random.shuffle. RandomState keeps our fold identities aligned.
        rng = np.random.RandomState(seed + fold)
        rng.shuffle(shuffled)
        test = set(shuffled[: int(len(shuffled) * 0.3)].tolist())
        if split == "combinations":
            test = set(list(test)[:15])
            held_genes = {g for cond in test for g in cond.split("+")}
            test.update({f"{g}+control" for g in held_genes})
    elif split == "unseen":
        single_genes = sorted({g for c in double_conditions for g in c.split("+")})
        rng = np.random.RandomState(seed + fold)
        rng.shuffle(single_genes)
        held_genes = set(single_genes[:12])
        test = {c for c in double_conditions if any(g in held_genes for g in c.split("+"))}
        test.update({f"{g}+control" for g in held_genes})
    else:
        raise ValueError(f"Unsupported Norman split: {split}")

    modes = np.array(["test" if c in test else "train" for c in conditions])
    modes[is_control] = "control"
    return build_prepared(adata, conditions, modes, is_control, store_dtype)


def prepare_combosciplex(path: Path, n_top_genes: int, test_conditions: list[str] | None, store_dtype: str) -> PreparedData:
    _, sc, _, _ = require_anndata_stack()
    adata = sc.read_h5ad(path)
    if "counts" in adata.layers:
        adata.X = adata.layers["counts"].copy()
    sc.pp.normalize_total(adata)
    sc.pp.log1p(adata)
    adata = highly_variable_or_all(adata, sc, n_top_genes)

    conditions = np.array([normalize_condition(c) for c in adata.obs["condition"].to_numpy()])
    test_set = set(test_conditions or COMBOSCIPLEX_DEFAULT_TEST)
    test_set = {normalize_condition(c) for c in test_set}
    is_control = np.array([c == "control+control" for c in conditions])
    modes = np.array(["test" if c in test_set else "train" for c in conditions])
    modes[is_control] = "control"
    return build_prepared(adata, conditions, modes, is_control, store_dtype)


def build_prepared(adata, conditions, modes, is_control, store_dtype: str) -> PreparedData:
    x = dense_array(adata.X).astype(storage_dtype(store_dtype), copy=False)
    gene_names = list(map(str, adata.var_names))
    gene_to_idx = {g: i for i, g in enumerate(gene_names)}
    perturbation_names = sorted({p for c in conditions for p in c.split("+")})
    pert_to_idx = {p: i for i, p in enumerate(perturbation_names)}

    perturbation_ids = []
    perturbation_gene_ids = []
    for condition in conditions:
        p1, p2 = split_condition(condition)
        ids = [pert_to_idx[p1], pert_to_idx[p2]]
        gene_ids = [gene_to_idx.get(p1, -1), gene_to_idx.get(p2, -1)]
        perturbation_ids.append(ids)
        perturbation_gene_ids.append(gene_ids)

    train_conditions = sorted({c for c, m in zip(conditions, modes) if m == "train" and c != "control+control"})
    test_conditions = sorted({c for c, m in zip(conditions, modes) if m == "test"})

    return PreparedData(
        x=x,
        gene_names=gene_names,
        conditions=conditions,
        modes=modes,
        is_control=is_control,
        perturbation_ids=np.asarray(perturbation_ids, dtype=np.int64),
        perturbation_gene_ids=np.asarray(perturbation_gene_ids, dtype=np.int64),
        perturbation_names=perturbation_names,
        train_conditions=train_conditions,
        test_conditions=test_conditions,
    )


class PerturbationBatchDataset(Dataset):
    def __init__(self, data: PreparedData, batch_size: int, repeats: int = 1000, mixed_conditions: bool = False):
        self.data = data
        self.batch_size = int(batch_size)
        self.mixed_conditions = bool(mixed_conditions)
        self.conditions = data.train_conditions
        self.repeats = int(repeats)
        self.control_idx = np.where(data.is_control)[0]
        self.by_condition = {
            cond: np.where((data.conditions == cond) & (data.modes == "train"))[0]
            for cond in self.conditions
        }
        self.by_condition = {k: v for k, v in self.by_condition.items() if len(v) > 0}
        self.conditions = sorted(self.by_condition)
        if len(self.conditions) == 0:
            raise ValueError("No train perturbation conditions found.")

    def __len__(self):
        return len(self.conditions) * self.repeats

    def __getitem__(self, _idx):
        src_idx = np.random.choice(self.control_idx, self.batch_size, replace=True)
        if self.mixed_conditions:
            target_indices = []
            for _ in range(self.batch_size):
                cond = random.choice(self.conditions)
                target_indices.append(np.random.choice(self.by_condition[cond]))
            tgt_idx = np.asarray(target_indices, dtype=np.int64)
        else:
            cond = random.choice(self.conditions)
            tgt_idx = np.random.choice(self.by_condition[cond], self.batch_size, replace=True)
        return {
            "source": torch.from_numpy(self.data.x[src_idx]),
            "target": torch.from_numpy(self.data.x[tgt_idx]),
            "perturbation_id": torch.from_numpy(self.data.perturbation_ids[tgt_idx]),
            "perturbation_gene_id": torch.from_numpy(self.data.perturbation_gene_ids[tgt_idx]),
        }


def make_lognorm_poisson_noise(control_log, alpha=0.8, total=1e4, eps=1e-8):
    base = torch.expm1(control_log.clamp_min(0))
    scale = total / (base.sum(dim=1, keepdim=True) + eps)
    lam = (alpha * base * scale).clamp_min(1e-8)
    return torch.log1p(torch.poisson(lam))


def pairwise_sq_dists(x, y):
    return torch.cdist(x, y, p=2).square()


@torch.no_grad()
def median_sigmas(x, scales=(0.5, 1.0, 2.0, 4.0)):
    d2 = pairwise_sq_dists(x, x)
    tri = d2[~torch.eye(d2.size(0), dtype=torch.bool, device=d2.device)]
    med = torch.median(tri).clamp_min(1e-12)
    return [float(torch.sqrt(med * s).item()) for s in scales]


def mmd_multi_sigma(x, y, sigmas):
    dxx = pairwise_sq_dists(x, x)
    dyy = pairwise_sq_dists(y, y)
    dxy = pairwise_sq_dists(x, y)
    vals = []
    for sigma in sigmas:
        beta = 1.0 / (2.0 * sigma * sigma + 1e-12)
        kxx = torch.exp(-beta * dxx)
        kyy = torch.exp(-beta * dyy)
        kxy = torch.exp(-beta * dxy)
        m, n = x.size(0), y.size(0)
        vals.append(
            (kxx.sum() - kxx.diag().sum()) / (m * (m - 1) + 1e-12)
            + (kyy.sum() - kyy.diag().sum()) / (n * (n - 1) + 1e-12)
            - 2.0 * kxy.mean()
        )
    return torch.stack(vals).mean()


def make_flow_noise(source, args, eval_mode: bool = False):
    if args.flow_target == "delta":
        scale = args.delta_noise_scale
        if eval_mode and args.eval_delta_noise_scale is not None:
            scale = args.eval_delta_noise_scale
        return torch.randn_like(source) * float(scale)
    if args.noise == "poisson":
        return make_lognorm_poisson_noise(source)
    return torch.randn_like(source)


def train_step(model, batch, args, device):
    source = batch["source"].to(device=device, dtype=torch.float32)
    target = batch["target"].to(device=device, dtype=torch.float32)
    pert = batch["perturbation_id"].to(device)
    pert_gene = batch["perturbation_gene_id"].to(device)
    b, g = source.shape
    sample_genes = torch.randperm(g, device=device)[: args.infer_top_genes]
    t = torch.rand(b, device=device)
    target_state = target - source if args.flow_target == "delta" else target
    noise = make_flow_noise(source, args)
    x_t = (1.0 - t[:, None]) * noise + t[:, None] * target_state
    dx = target_state - noise
    pred = model(x_t, source, t, pert, pert_gene)
    flow_loss = F.mse_loss(pred[:, sample_genes], dx[:, sample_genes])
    loss = flow_loss
    state_hat = x_t + pred * (1.0 - t[:, None])
    if args.flow_target == "delta":
        x1_hat = source + state_hat
        dir_pred = state_hat
        dir_true = target_state
    else:
        x1_hat = state_hat
        dir_pred = x1_hat - source
        dir_true = target - source
    rec_loss = F.mse_loss(x1_hat[:, sample_genes], target[:, sample_genes])
    bulk_loss = F.mse_loss(
        x1_hat[:, sample_genes].mean(dim=0),
        target[:, sample_genes].mean(dim=0),
    )
    if args.hetero_weight > 0:
        pred_resid = x1_hat[:, sample_genes] - x1_hat[:, sample_genes].mean(dim=0, keepdim=True)
        source_resid = source[:, sample_genes] - source[:, sample_genes].mean(dim=0, keepdim=True)
        hetero_loss = F.mse_loss(pred_resid, source_resid)
        loss = loss + args.hetero_weight * hetero_loss
    else:
        hetero_loss = loss.new_tensor(0.0)
    if args.dir_weight > 0:
        dir_loss = 1.0 - F.cosine_similarity(
            dir_pred[:, sample_genes],
            dir_true[:, sample_genes],
            dim=-1,
            eps=1e-8,
        ).mean()
        loss = loss + args.dir_weight * dir_loss
    else:
        dir_loss = loss.new_tensor(0.0)
    if args.recon_weight > 0:
        loss = loss + args.recon_weight * rec_loss
    if args.bulk_loss_weight > 0:
        loss = loss + args.bulk_loss_weight * bulk_loss
    if args.use_mmd:
        sigmas = median_sigmas(target[:, sample_genes])
        mmd = mmd_multi_sigma(x1_hat[:, sample_genes], target[:, sample_genes], sigmas)
        loss = loss + args.gamma * mmd
    else:
        mmd = loss.new_tensor(0.0)
    parts = {
        "flow": float(flow_loss.detach().item()),
        "dir": float(dir_loss.detach().item()),
        "hetero": float(hetero_loss.detach().item()),
        "rec": float(rec_loss.detach().item()),
        "bulk": float(bulk_loss.detach().item()),
        "mmd": float(mmd.detach().item()),
    }
    return loss, parts


@torch.inference_mode()
def generate(model, source, pert, pert_gene, steps, args, device):
    source = source.to(device=device, dtype=torch.float32)
    pert = pert.to(device)
    pert_gene = pert_gene.to(device)
    x = make_flow_noise(source, args, eval_mode=True)
    dt = 1.0 / float(steps)
    for i in range(steps):
        t = torch.full((source.size(0),), i / float(steps), device=device)
        x = x + dt * model(x, source, t, pert, pert_gene)
    if args.flow_target == "delta":
        return (source + x).clamp_min(0)
    return x.clamp_min(0)


def pearson(a, b):
    a = np.asarray(a).reshape(-1)
    b = np.asarray(b).reshape(-1)
    if np.std(a) < 1e-8 or np.std(b) < 1e-8:
        return float("nan")
    return float(np.corrcoef(a, b)[0, 1])


def rankdata(x):
    order = np.argsort(x)
    ranks = np.empty_like(order, dtype=np.float64)
    ranks[order] = np.arange(len(x), dtype=np.float64)
    return ranks


def format_duration(seconds: float) -> str:
    seconds = max(0, int(seconds))
    days, seconds = divmod(seconds, 86400)
    hours, seconds = divmod(seconds, 3600)
    minutes, seconds = divmod(seconds, 60)
    if days:
        return f"{days}d {hours:02d}h"
    if hours:
        return f"{hours:02d}:{minutes:02d}:{seconds:02d}"
    return f"{minutes:02d}:{seconds:02d}"


def format_metric(value, digits: int = 4) -> str:
    if value is None:
        return "n/a"
    try:
        value = float(value)
    except (TypeError, ValueError):
        return str(value)
    if not np.isfinite(value):
        return "n/a"
    if abs(value) < 1e-3 and value != 0:
        return f"{value:.2e}"
    return f"{value:.{digits}f}"


def parse_skip_metrics(value: str | None) -> list[str] | None:
    if value is None:
        return None
    value = value.strip()
    if not value or value.lower() in {"none", "false", "0"}:
        return None
    return [part.strip() for part in value.split(",") if part.strip()]


def select_eval_gene_indices(data: PreparedData, args) -> np.ndarray:
    n_genes = data.x.shape[1]
    n_eval = int(args.eval_top_genes)
    if n_eval <= 0 or n_eval >= n_genes:
        return np.arange(n_genes, dtype=np.int64)

    method = getattr(args, "eval_gene_selection", "mean")
    if method == "hvg_test":
        ad, sc, _, _ = require_anndata_stack()
        eval_mask = (data.modes == "test") | data.is_control
        if eval_mask.any():
            x_eval = data.x[eval_mask].astype(np.float32, copy=False)
            adata_eval = ad.AnnData(X=x_eval)
            adata_eval.var_names = [str(g) for g in data.gene_names]
            sc.pp.highly_variable_genes(adata_eval, inplace=True, n_top_genes=n_eval)
            gene_idx = np.where(adata_eval.var["highly_variable"].to_numpy())[0]
            if len(gene_idx) > 0:
                return np.sort(gene_idx.astype(np.int64))

    if method == "hvg_train":
        ad, sc, _, _ = require_anndata_stack()
        train_mask = (data.modes == "train") | data.is_control
        if train_mask.any():
            x_train = data.x[train_mask].astype(np.float32, copy=False)
            adata_train = ad.AnnData(X=x_train)
            adata_train.var_names = [str(g) for g in data.gene_names]
            sc.pp.highly_variable_genes(adata_train, inplace=True, n_top_genes=n_eval)
            gene_idx = np.where(adata_train.var["highly_variable"].to_numpy())[0]
            if len(gene_idx) > 0:
                return np.sort(gene_idx.astype(np.int64))

    if method != "mean":
        raise ValueError(f"Unsupported eval gene selection: {method}")
    train_mask = (data.modes == "train") | data.is_control
    mean_expr = data.x[train_mask].astype(np.float32, copy=False).mean(axis=0)
    gene_idx = np.argsort(mean_expr)[-n_eval:]
    return np.sort(gene_idx.astype(np.int64))


def perturbation_kind(condition: str) -> str:
    p1, p2 = split_condition(condition)
    if p1 == "control" and p2 == "control":
        return "Control"
    if p1 == "control" or p2 == "control":
        return "Single"
    return "Double"


def train_reference_mean(data: PreparedData, condition: str, gene_idx: np.ndarray, control_mean: np.ndarray) -> np.ndarray:
    train_idx = np.where((data.conditions == condition) & (data.modes == "train"))[0]
    if len(train_idx) > 0:
        return data.x[train_idx][:, gene_idx].astype(np.float32, copy=False).mean(axis=0)

    p1, p2 = split_condition(condition)
    if p1 != "control" and p2 != "control":
        single_refs = []
        for pert in (p1, p2):
            single = f"{pert}+control"
            idx = np.where((data.conditions == single) & (data.modes == "train"))[0]
            if len(idx) > 0:
                single_refs.append(data.x[idx][:, gene_idx].astype(np.float32, copy=False).mean(axis=0))
        if len(single_refs) == 2:
            return single_refs[0] + single_refs[1] - control_mean

    return control_mean


def pearson_residual_rows(pred: np.ndarray, real: np.ndarray) -> float:
    vals = [pearson(p, r) for p, r in zip(pred, real)]
    return float(np.nanmean(vals)) if vals else float("nan")


def build_eval_predictions(model, data: PreparedData, args, device, eval_seed: int | None = None):
    model.eval()
    eval_t0 = time.time()
    rng = np.random.default_rng(args.seed if eval_seed is None else eval_seed)
    control_idx = np.where(data.is_control)[0]
    gene_idx = select_eval_gene_indices(data, args)
    gene_names = [data.gene_names[i] for i in gene_idx]

    if args.eval_control_cells > 0 and len(control_idx) > args.eval_control_cells:
        eval_control_idx = rng.choice(control_idx, args.eval_control_cells, replace=False)
    else:
        eval_control_idx = control_idx

    control_eval = data.x[eval_control_idx][:, gene_idx].astype(np.float32, copy=False)
    control_mean = control_eval.mean(axis=0)
    pred_parts = [control_eval]
    real_parts = [control_eval]
    pred_obs = ["control"] * len(eval_control_idx)
    real_obs = ["control"] * len(eval_control_idx)
    rows = []

    eval_items = []
    for cond in data.test_conditions:
        target_idx = np.where((data.conditions == cond) & (data.modes == "test"))[0]
        if len(target_idx) == 0:
            continue
        eval_items.append((cond, target_idx))

    if args.eval_progress:
        n_pred_preview = min(args.eval_cells, len(control_idx))
        n_batches_preview = math.ceil(n_pred_preview / max(1, args.eval_batch_size))
        print(
            "[eval] fast eval start | "
            f"conditions {len(eval_items)} | cells/condition {n_pred_preview} | "
            f"batch {args.eval_batch_size} | ode_steps {args.ode_steps} | "
            f"forwards/condition {n_batches_preview * args.ode_steps} | "
            f"genes {len(gene_idx)} {getattr(args, 'eval_gene_selection', 'mean')}",
            flush=True,
        )

    if args.eval_max_conditions > 0 and len(eval_items) > args.eval_max_conditions:
        pick = rng.choice(len(eval_items), args.eval_max_conditions, replace=False)
        eval_items = [eval_items[i] for i in sorted(pick.tolist())]
        if args.eval_progress:
            print(f"[eval] sampled {len(eval_items)} conditions for fast training eval", flush=True)

    for cond_i, (cond, target_idx) in enumerate(eval_items, start=1):
        cond_t0 = time.time()
        n_pred = min(args.eval_cells, len(control_idx))
        n_batches = math.ceil(n_pred / max(1, args.eval_batch_size))
        if args.eval_progress:
            print(
                f"[eval] condition {cond_i}/{len(eval_items)} {cond} | "
                f"cells {n_pred} | batches {n_batches} | forwards {n_batches * args.ode_steps}",
                flush=True,
            )
        src_idx = rng.choice(control_idx, n_pred, replace=len(control_idx) < n_pred)
        pert = torch.from_numpy(data.perturbation_ids[target_idx[:1]].repeat(n_pred, axis=0))
        pert_gene = torch.from_numpy(data.perturbation_gene_ids[target_idx[:1]].repeat(n_pred, axis=0))
        preds = []
        for start in range(0, n_pred, args.eval_batch_size):
            end = min(start + args.eval_batch_size, n_pred)
            src = torch.from_numpy(data.x[src_idx[start:end]])
            pred = generate(model, src, pert[start:end], pert_gene[start:end], args.ode_steps, args, device)
            preds.append(pred[:, gene_idx].cpu().numpy().astype(np.float32, copy=False))
        pred = np.concatenate(preds, axis=0)

        target = data.x[target_idx][:, gene_idx].astype(np.float32, copy=False)
        pred_mean = pred.mean(axis=0)
        target_mean = target.mean(axis=0)

        delta_pred = pred_mean - control_mean
        src_all = data.x[src_idx][:, gene_idx].astype(np.float32, copy=False)
        src_mean = src_all.mean(axis=0)
        delta_pred_source = pred_mean - src_mean

        delta_true = target_mean - control_mean
        top20_delta = np.argsort(np.abs(delta_true))[-min(20, len(delta_true)) :]

        n_pair = min(len(pred), len(target))
        pair_idx = rng.choice(len(target), n_pair, replace=len(target) < n_pair)
        pair_target = target[pair_idx]
        ref = train_reference_mean(data, cond, gene_idx, control_mean)
        residual_pred = pred[:n_pair] - ref
        residual_true = pair_target - ref
        variance = residual_true.var(axis=0)
        top20_var = np.argsort(variance)[-min(20, len(variance)) :]

        rows.append(
            {
                "condition": cond,
                "kind": perturbation_kind(cond),
                "L2": float(np.linalg.norm(pred_mean - target_mean)),
                "MSE": float(np.mean((pred_mean - target_mean) ** 2)),
                "MAE": float(np.mean(np.abs(pred_mean - target_mean))),
                "Pearson_Delta": pearson(delta_pred, delta_true),
                "Pearson_Delta_Source": pearson(delta_pred_source, delta_true),
                "Pearson_Delta_Hat": pearson_residual_rows(residual_pred, residual_true),
                "Pearson_Delta_Hat20": pearson_residual_rows(residual_pred[:, top20_var], residual_true[:, top20_var]),
                "DE-Spearman_fast_top20": pearson(rankdata(delta_pred[top20_delta]), rankdata(delta_true[top20_delta])),
            }
        )

        pred_parts.append(pred)
        real_parts.append(target)
        pred_obs.extend([cond] * len(pred))
        real_obs.extend([cond] * len(target))
        if args.eval_progress:
            print(
                f"[eval] condition {cond_i}/{len(eval_items)} done | "
                f"condition_elapsed {format_duration(time.time() - cond_t0)} | "
                f"eval_elapsed {format_duration(time.time() - eval_t0)}",
                flush=True,
            )

    return {
        "rows": rows,
        "pred_x": np.concatenate(pred_parts, axis=0),
        "real_x": np.concatenate(real_parts, axis=0),
        "pred_obs": pred_obs,
        "real_obs": real_obs,
        "gene_names": gene_names,
        "gene_count": len(gene_idx),
        "gene_selection": getattr(args, "eval_gene_selection", "mean"),
    }


def aggregate_metric_rows(rows: list[dict], args, kind: str | None = None) -> dict:
    selected = [r for r in rows if kind is None or r["kind"] == kind]
    if not selected:
        return {}
    agg = {
        "Setting": kind if kind is not None else args.split,
        "Model": "VOID-Cell",
        "Eval": "fast",
    }
    for key in [
        "L2",
        "MSE",
        "MAE",
        "Pearson_Delta",
        "Pearson_Delta_Source",
        "Pearson_Delta_Hat",
        "Pearson_Delta_Hat20",
        "DE-Spearman_fast_top20",
    ]:
        agg[key] = float(np.nanmean([r[key] for r in selected]))
    return agg


def agg_value(agg_results, column: str):
    if agg_results is None or column not in agg_results.columns:
        return float("nan")
    if "statistic" in agg_results.columns:
        row = agg_results.filter(agg_results["statistic"] == "mean")
        if row.height == 0:
            return float("nan")
        return float(row[column][0])
    frame = agg_results.to_pandas()
    if "statistic" in frame.columns:
        row = frame[frame["statistic"] == "mean"]
        if row.empty:
            return float("nan")
        return float(row[column].iloc[0])
    return float("nan")


def run_cell_eval(eval_data: dict, out_dir: Path, args, step: int):
    try:
        import anndata as ad
        import pandas as pd
        from cell_eval import MetricsEvaluator
    except Exception as exc:
        print(f"[eval] official cell_eval unavailable: {exc}")
        print("[eval] install with: pip install cell-eval==0.5.42 pdex==0.1.21")
        return None, None

    eval_dir = out_dir / f"eval_step_{step:07d}"
    obs_pred = pd.DataFrame({"perturbation": eval_data["pred_obs"]})
    obs_real = pd.DataFrame({"perturbation": eval_data["real_obs"]})
    var = pd.DataFrame(index=eval_data["gene_names"])
    pred = ad.AnnData(X=eval_data["pred_x"], obs=obs_pred, var=var.copy())
    real = ad.AnnData(X=eval_data["real_x"], obs=obs_real, var=var.copy())

    evaluator = MetricsEvaluator(
        adata_pred=pred,
        adata_real=real,
        control_pert="control",
        pert_col="perturbation",
        num_threads=args.cell_eval_threads,
        batch_size=args.cell_eval_batch_size,
        outdir=str(eval_dir),
    )
    results, agg_results = evaluator.compute(
        profile=args.cell_eval_profile,
        skip_metrics=parse_skip_metrics(args.cell_eval_skip_metrics),
        write_csv=False,
    )
    eval_dir.mkdir(parents=True, exist_ok=True)
    results.write_csv(eval_dir / "results.csv")
    agg_results.write_csv(eval_dir / "agg_results.csv")
    if args.save_eval_anndata:
        pred.write_h5ad(eval_dir / "pred.h5ad")
        real.write_h5ad(eval_dir / "real.h5ad")
    return results, agg_results


def evaluate(model, data: PreparedData, args, device, out_dir: Path | None = None, step: int = 0):
    was_training = model.training
    try:
        if args.eval_seed is None:
            eval_data = build_eval_predictions(model, data, args, device)
        else:
            fork_devices = [device] if device.type == "cuda" else []
            with torch.random.fork_rng(devices=fork_devices):
                torch.manual_seed(args.eval_seed)
                if device.type == "cuda":
                    torch.cuda.manual_seed_all(args.eval_seed)
                eval_data = build_eval_predictions(model, data, args, device, eval_seed=args.eval_seed)
        rows = eval_data["rows"]
        if not rows:
            return {}

        metrics = aggregate_metric_rows(rows, args)
        metrics["EvalGenes"] = eval_data.get("gene_count", float("nan"))
        metrics["EvalGeneSelection"] = eval_data.get("gene_selection", "")
        metrics["DE-Spearman"] = metrics.get("DE-Spearman_fast_top20", float("nan"))
        metrics["DS"] = float("nan")
        metrics["DM"] = float("nan")
        if args.cell_eval:
            _, agg_results = run_cell_eval(eval_data, out_dir or Path(args.output_dir), args, step)
            if agg_results is not None:
                metrics["Eval"] = "official"
                metrics["DE-Spearman"] = agg_value(agg_results, "de_spearman_lfc_sig")
                metrics["DS"] = agg_value(agg_results, "discrimination_score_l1")
                metrics["DM"] = agg_value(agg_results, "de_direction_match")
                metrics["Pearson_Delta"] = agg_value(agg_results, "pearson_delta")
                metrics["MSE"] = agg_value(agg_results, "mse")
                metrics["MAE"] = agg_value(agg_results, "mae")
            else:
                metrics["Eval"] = "fast_missing_cell_eval"
        return metrics
    finally:
        if was_training:
            model.train()


def write_metrics(path: Path, metrics: dict):
    columns = [
        "Step",
        "Elapsed",
        "Setting",
        "Model",
        "Eval",
        "EvalGenes",
        "EvalGeneSelection",
        "L2",
        "MSE",
        "MAE",
        "DE-Spearman",
        "Pearson_Delta",
        "Pearson_Delta_Source",
        "DS",
        "DM",
        "Pearson_Delta_Hat",
        "Pearson_Delta_Hat20",
    ]
    exists = path.exists()
    with path.open("a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=columns)
        if not exists:
            writer.writeheader()
        writer.writerow({c: metrics.get(c, "") for c in columns})


def metrics_log_line(metrics: dict) -> str:
    parts = [
        f"step {metrics.get('Step', 'n/a')}",
        str(metrics.get("Eval", "fast")),
        f"Genes {metrics.get('EvalGenes', 'n/a')}",
        f"L2 {format_metric(metrics.get('L2'))}",
        f"MSE {format_metric(metrics.get('MSE'), 5)}",
        f"MAE {format_metric(metrics.get('MAE'), 5)}",
        f"DE-Spear {format_metric(metrics.get('DE-Spearman'))}",
        f"PearsonDelta {format_metric(metrics.get('Pearson_Delta'))}",
        f"PearsonDeltaSrc {format_metric(metrics.get('Pearson_Delta_Source'))}",
        f"DS {format_metric(metrics.get('DS'))}",
        f"DM {format_metric(metrics.get('DM'))}",
        f"PearsonHat {format_metric(metrics.get('Pearson_Delta_Hat'))}",
        f"PearsonHat20 {format_metric(metrics.get('Pearson_Delta_Hat20'))}",
    ]
    return "[eval] " + " | ".join(parts)


def zeropower_via_newtonschulz5(g: torch.Tensor, steps: int = 5, eps: float = 1e-7) -> torch.Tensor:
    if g.numel() == 0:
        return g
    original_dtype = g.dtype
    use_bf16 = g.is_cuda and torch.cuda.is_bf16_supported()
    x = g.bfloat16() if use_bf16 else g.float()
    norm = x.norm()
    if float(norm) < eps:
        return torch.zeros_like(g)
    x = x / (norm + eps)
    transposed = x.size(0) > x.size(1)
    if transposed:
        x = x.T

    a, b, c = 3.4445, -4.7750, 2.0315
    for _ in range(max(int(steps), 1)):
        aa = x @ x.T
        bb = b * aa + c * (aa @ aa)
        x = a * x + bb @ x

    if transposed:
        x = x.T
    return x.to(dtype=original_dtype)


class HybridMuonAdamW(torch.optim.Optimizer):
    def __init__(self, param_groups):
        super().__init__(param_groups, {})

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            if group.get("use_muon", False):
                self._muon_step(group)
            else:
                self._adamw_step(group)
        return loss

    def _muon_step(self, group):
        lr = group["lr"]
        momentum = group["momentum"]
        weight_decay = group["weight_decay"]
        ns_steps = group["ns_steps"]
        for p in group["params"]:
            if p.grad is None:
                continue
            if p.grad.is_sparse:
                raise RuntimeError("Muon does not support sparse gradients.")
            if weight_decay != 0:
                p.mul_(1.0 - lr * weight_decay)
            grad = p.grad
            state = self.state[p]
            if "momentum_buffer" not in state:
                state["momentum_buffer"] = torch.zeros_like(p)
            buf = state["momentum_buffer"]
            buf.mul_(momentum).add_(grad)

            matrix = buf.reshape(buf.size(0), -1)
            update = zeropower_via_newtonschulz5(matrix, steps=ns_steps)
            rows, cols = matrix.shape
            scale = math.sqrt(max(1.0, rows / max(cols, 1)))
            p.add_(update.reshape_as(p), alpha=-lr * scale)

    def _adamw_step(self, group):
        lr = group["lr"]
        beta1, beta2 = group["betas"]
        eps = group["eps"]
        weight_decay = group["weight_decay"]
        for p in group["params"]:
            if p.grad is None:
                continue
            if p.grad.is_sparse:
                raise RuntimeError("AdamW does not support sparse gradients.")
            if weight_decay != 0:
                p.mul_(1.0 - lr * weight_decay)
            grad = p.grad
            state = self.state[p]
            if len(state) == 0:
                state["step"] = 0
                state["exp_avg"] = torch.zeros_like(p)
                state["exp_avg_sq"] = torch.zeros_like(p)
            exp_avg = state["exp_avg"]
            exp_avg_sq = state["exp_avg_sq"]
            state["step"] += 1
            step = state["step"]
            exp_avg.mul_(beta1).add_(grad, alpha=1.0 - beta1)
            exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=1.0 - beta2)
            bias_correction1 = 1.0 - beta1**step
            bias_correction2 = 1.0 - beta2**step
            denom = exp_avg_sq.sqrt().div_(math.sqrt(bias_correction2)).add_(eps)
            p.addcdiv_(exp_avg, denom, value=-lr / bias_correction1)


def use_muon_for_parameter(name: str, param: torch.nn.Parameter) -> bool:
    if param.ndim < 2:
        return False
    lowered = name.lower()
    adamw_only = ("embedding", "velocity", "out_norm", "norm", "bias")
    return not any(token in lowered for token in adamw_only)


def build_optimizer(model, args):
    betas = (args.adam_beta1, args.adam_beta2)
    if args.optimizer == "adamw":
        return torch.optim.AdamW(
            model.parameters(),
            lr=args.lr,
            betas=betas,
            eps=args.adam_eps,
            weight_decay=args.weight_decay,
        )
    if args.optimizer == "adamw_fused":
        try:
            return torch.optim.AdamW(
                model.parameters(),
                lr=args.lr,
                betas=betas,
                eps=args.adam_eps,
                weight_decay=args.weight_decay,
                fused=torch.cuda.is_available(),
            )
        except TypeError:
            return torch.optim.AdamW(
                model.parameters(),
                lr=args.lr,
                betas=betas,
                eps=args.adam_eps,
                weight_decay=args.weight_decay,
            )
    if args.optimizer != "muon":
        raise ValueError(f"Unsupported optimizer: {args.optimizer}")

    muon_params = []
    adamw_params = []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if use_muon_for_parameter(name, param):
            muon_params.append(param)
        else:
            adamw_params.append(param)

    groups = []
    if muon_params:
        groups.append(
            {
                "params": muon_params,
                "lr": args.muon_lr,
                "momentum": args.muon_momentum,
                "weight_decay": args.weight_decay,
                "ns_steps": args.muon_ns_steps,
                "use_muon": True,
                "name": "muon",
            }
        )
    if adamw_params:
        groups.append(
            {
                "params": adamw_params,
                "lr": args.lr,
                "betas": betas,
                "eps": args.adam_eps,
                "weight_decay": args.weight_decay,
                "use_muon": False,
                "name": "adamw",
            }
        )
    return HybridMuonAdamW(groups)


def format_optimizer_lrs(optimizer) -> str:
    if len(optimizer.param_groups) == 1:
        return f"{optimizer.param_groups[0]['lr']:.2e}"
    return ",".join(
        f"{group.get('name', i)}={group['lr']:.2e}"
        for i, group in enumerate(optimizer.param_groups)
    )


def train_log_line(step: int, loss_value: float, elapsed: float, args, optimizer, loss_parts: dict | None = None) -> str:
    speed = (step + 1) / max(elapsed, 1e-9)
    remaining_steps = max(args.steps - step - 1, 0)
    eta_steps = remaining_steps / speed if speed > 0 else 0.0
    parts = [
        f"step {step}/{args.steps}",
        f"loss {loss_value:.6f}",
        f"speed {speed:.3f} step/s",
        f"elapsed {format_duration(elapsed)}",
        f"eta {format_duration(eta_steps)}",
        f"lr {format_optimizer_lrs(optimizer)}",
    ]
    if loss_parts:
        parts.extend(
            [
                f"flow {loss_parts.get('flow', 0.0):.4f}",
                f"dir {loss_parts.get('dir', 0.0):.4f}",
                f"het {loss_parts.get('hetero', 0.0):.4f}",
                f"rec {loss_parts.get('rec', 0.0):.4f}",
                f"bulk {loss_parts.get('bulk', 0.0):.4f}",
            ]
        )
        if args.use_mmd:
            parts.append(f"mmd {loss_parts.get('mmd', 0.0):.4f}")
    if args.max_hours > 0:
        limit = args.max_hours * 3600.0
        parts.append(f"wall_left {format_duration(max(limit - elapsed, 0.0))}")
    return "[train] " + " | ".join(parts)


def lr_multiplier(step: int, args) -> float:
    if args.lr_schedule == "none":
        return 1.0
    warmup = max(int(args.warmup_steps), 0)
    if warmup > 0 and step < warmup:
        return max((step + 1) / float(warmup), 1e-3)
    denom = max(args.steps - warmup, 1)
    progress = min(max((step - warmup) / float(denom), 0.0), 1.0)
    cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
    return args.min_lr_ratio + (1.0 - args.min_lr_ratio) * cosine


def model_state_dict(model):
    base = model._orig_mod if hasattr(model, "_orig_mod") else model
    return base.state_dict()


def save_checkpoint(path: Path, model, step: int, args) -> None:
    torch.save({"model": model_state_dict(model), "step": step, "args": vars(args)}, path)


def load_data(args):
    path = Path(args.data_path)
    if args.dataset == "norman":
        return prepare_norman(path, args.n_top_genes, args.split, args.fold, args.seed, args.store_dtype)
    if args.dataset == "combosciplex":
        return prepare_combosciplex(path, args.n_top_genes, args.combosciplex_test_conditions, args.store_dtype)
    raise ValueError(args.dataset)


def train(args):
    seed_everything(args.seed)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    with (out_dir / "config.json").open("w") as f:
        json.dump(vars(args), f, indent=2)

    data = load_data(args)
    train_mask = (data.modes == "train") | data.is_control
    neighbor_cache = out_dir / f"geometry_top{args.graph_k}_m{args.manifold_dim}.pt"
    train_rows = np.where(train_mask)[0]
    geometry = build_coexpression_geometry(
        data.x,
        args.graph_k,
        args.manifold_dim,
        neighbor_cache,
        row_idx=train_rows,
        max_cells=args.graph_cells,
        seed=args.seed,
    )

    device = torch.device(args.device if args.device else ("cuda" if torch.cuda.is_available() else "cpu"))
    model = VoidCellModel(
        n_genes=data.x.shape[1],
        n_perturbations=len(data.perturbation_names),
        neighbors=geometry["neighbors"],
        edge_weights=geometry["weights"],
        manifold_coords=geometry["coords"],
        dim=args.dim,
        hidden=args.hidden,
        encode_blocks=args.encode_blocks,
        think_steps=args.think_steps,
        dropout=args.dropout,
        residual_scale=args.residual_scale,
        neighbor_chunk=args.neighbor_chunk,
        neighbor_gate=args.neighbor_gate,
        directional_shifts=args.directional_shifts,
        directional_residual_gate=args.directional_residual_gate,
        directional_gate_init=args.directional_gate_init,
        shift_dims=args.shift_dims,
        shift_stencil=args.shift_stencil,
        shift_temperature=args.shift_temperature,
        shift_code_strength=args.shift_code_strength,
        spatial_grid_shifts=args.spatial_grid_shifts,
        spatial_grid_dims=args.spatial_grid_dims,
        spatial_grid_side=args.spatial_grid_side,
        spatial_shift_stencil=args.spatial_shift_stencil,
        spatial_shift_code_strength=args.spatial_shift_code_strength,
        graph_message_weight=args.graph_message_weight,
        spatial_message_weight=args.spatial_message_weight,
        self_weight=args.self_weight,
        neighbor_weight=args.neighbor_weight,
        global_weight=args.global_weight,
        checkpoint_blocks=args.checkpoint_blocks,
    ).to(device)
    eval_model = model
    if args.compile:
        model = torch.compile(model, mode=args.compile_mode)
        if args.compile_eval:
            eval_model = model
    base_model = eval_model._orig_mod if hasattr(eval_model, "_orig_mod") else eval_model
    extra = ""
    if getattr(base_model, "spatial_grid_shifts", False):
        extra = f" spatial_grid={base_model.grid_shape} spatial_dirs={base_model.n_spatial_dirs}"
    if getattr(base_model, "directional_residual_gate", False):
        gate = torch.sigmoid(base_model.directional_gate_logit.detach()).cpu().item()
        extra += f" directional_gate={gate:.4f}"
    print(
        f"genes={data.x.shape[1]} perturbations={len(data.perturbation_names)} "
        f"params={sum(p.numel() for p in model.parameters())}{extra}"
    )

    ds = PerturbationBatchDataset(data, args.batch_size, mixed_conditions=args.mixed_condition_batch)
    loader = DataLoader(
        ds,
        batch_size=1,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
        persistent_workers=args.num_workers > 0,
    )
    optimizer = build_optimizer(model, args)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda s: lr_multiplier(s, args))
    use_amp = device.type == "cuda" and args.precision != "fp32"
    amp_dtype = torch.float16 if args.precision == "fp16" else torch.bfloat16
    try:
        scaler = torch.amp.GradScaler("cuda", enabled=use_amp and args.precision == "fp16")
    except TypeError:
        scaler = torch.cuda.amp.GradScaler(enabled=use_amp and args.precision == "fp16")

    step = 0
    t0 = time.time()
    while step < args.steps:
        for batch in loader:
            batch = {k: v.squeeze(0) for k, v in batch.items()}
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type=device.type, dtype=amp_dtype, enabled=use_amp):
                loss, loss_parts = train_step(model, batch, args, device)
            if scaler.is_enabled():
                old_scale = scaler.get_scale()
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
                scaler.step(optimizer)
                scaler.update()
                optimizer_stepped = scaler.get_scale() >= old_scale
            else:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
                optimizer.step()
                optimizer_stepped = True
            if optimizer_stepped:
                scheduler.step()
            if step % args.log_every == 0:
                elapsed = max(time.time() - t0, 1e-9)
                print(train_log_line(step, loss.item(), elapsed, args, optimizer, loss_parts), flush=True)
            if args.eval_every > 0 and step > 0 and step % args.eval_every == 0:
                save_checkpoint(out_dir / "last.pt", model, step, args)
                try:
                    metrics = evaluate(eval_model, data, args, device, out_dir=out_dir, step=step)
                except RuntimeError as exc:
                    if "out of memory" not in str(exc).lower():
                        raise
                    print(f"[eval] skipped at step {step}: CUDA out of memory during eval", flush=True)
                    if device.type == "cuda":
                        torch.cuda.empty_cache()
                    metrics = None
                if metrics:
                    metrics["Step"] = step
                    metrics["Elapsed"] = format_duration(time.time() - t0)
                    write_metrics(out_dir / "metrics.csv", metrics)
                    print(metrics_log_line(metrics), flush=True)
                save_checkpoint(out_dir / "last.pt", model, step, args)
            step += 1
            if args.save_every > 0 and step > 0 and step % args.save_every == 0:
                save_checkpoint(out_dir / "last.pt", model, step, args)
            if args.max_hours > 0 and (time.time() - t0) >= args.max_hours * 3600.0:
                save_checkpoint(out_dir / "last.pt", model, step, args)
                save_checkpoint(out_dir / "final.pt", model, step, args)
                print(f"Reached max_hours={args.max_hours:.2f}; saved checkpoint at step={step}.")
                return
            if step >= args.steps:
                break
    save_checkpoint(out_dir / "final.pt", model, step, args)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", choices=["norman", "combosciplex"], default="norman")
    p.add_argument("--data-path", required=True)
    p.add_argument("--output-dir", required=True)
    p.add_argument("--split", choices=["additive", "combinations", "unseen"], default="additive")
    p.add_argument("--fold", type=int, default=1)
    p.add_argument("--n-top-genes", type=int, default=5000)
    p.add_argument("--infer-top-genes", type=int, default=1000)
    p.add_argument("--store-dtype", choices=["float32", "float16"], default="float32")
    p.add_argument("--combosciplex-test-conditions", nargs="*", default=None)
    p.add_argument("--steps", type=int, default=200000)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--optimizer", choices=["adamw", "adamw_fused", "muon"], default="adamw")
    p.add_argument("--adam-beta1", type=float, default=0.9)
    p.add_argument("--adam-beta2", type=float, default=0.999)
    p.add_argument("--adam-eps", type=float, default=1e-8)
    p.add_argument("--muon-lr", type=float, default=0.02)
    p.add_argument("--muon-momentum", type=float, default=0.95)
    p.add_argument("--muon-ns-steps", type=int, default=5)
    p.add_argument("--lr-schedule", choices=["none", "cosine"], default="cosine")
    p.add_argument("--warmup-steps", type=int, default=100)
    p.add_argument("--min-lr-ratio", type=float, default=0.1)
    p.add_argument("--weight-decay", type=float, default=1e-3)
    p.add_argument("--grad-clip", type=float, default=1.0)
    p.add_argument("--dim", type=int, default=192)
    p.add_argument("--hidden", type=int, default=512)
    p.add_argument("--encode-blocks", type=int, default=4)
    p.add_argument("--think-steps", type=int, default=8)
    p.add_argument("--graph-k", type=int, default=30)
    p.add_argument("--graph-cells", type=int, default=8192)
    p.add_argument("--manifold-dim", type=int, default=4)
    p.add_argument("--neighbor-chunk", type=int, default=256)
    p.add_argument("--neighbor-gate", action="store_true")
    p.add_argument("--directional-shifts", action="store_true")
    p.add_argument("--directional-residual-gate", action="store_true")
    p.add_argument("--directional-gate-init", type=float, default=-4.0)
    p.add_argument("--shift-dims", type=int, default=0)
    p.add_argument("--shift-stencil", choices=["axis", "cube"], default="axis")
    p.add_argument("--shift-temperature", type=float, default=4.0)
    p.add_argument("--shift-code-strength", type=float, default=1.0)
    p.add_argument("--spatial-grid-shifts", action="store_true")
    p.add_argument("--spatial-grid-dims", type=int, default=3)
    p.add_argument("--spatial-grid-side", type=int, default=0)
    p.add_argument("--spatial-shift-stencil", choices=["axis", "cube"], default="cube")
    p.add_argument("--spatial-shift-code-strength", type=float, default=1.0)
    p.add_argument("--graph-message-weight", type=float, default=1.0)
    p.add_argument("--spatial-message-weight", type=float, default=1.0)
    p.add_argument("--self-weight", type=float, default=1.0)
    p.add_argument("--neighbor-weight", type=float, default=1.0)
    p.add_argument("--global-weight", type=float, default=1.0)
    p.add_argument("--checkpoint-blocks", action="store_true")
    p.add_argument("--dropout", type=float, default=0.05)
    p.add_argument("--residual-scale", type=float, default=0.1)
    p.add_argument("--noise", choices=["gaussian", "poisson"], default="gaussian")
    p.add_argument("--flow-target", choices=["cell", "delta"], default="cell")
    p.add_argument("--delta-noise-scale", type=float, default=1.0)
    p.add_argument("--eval-delta-noise-scale", type=float, default=None)
    p.add_argument("--use-mmd", action="store_true")
    p.add_argument("--gamma", type=float, default=0.5)
    p.add_argument("--dir-weight", type=float, default=0.0)
    p.add_argument("--hetero-weight", type=float, default=0.0)
    p.add_argument("--recon-weight", type=float, default=0.5)
    p.add_argument("--bulk-loss-weight", type=float, default=2.0)
    p.add_argument("--mixed-condition-batch", action="store_true")
    p.add_argument("--ode-steps", type=int, default=20)
    p.add_argument("--eval-top-genes", type=int, default=1000)
    p.add_argument("--eval-gene-selection", choices=["mean", "hvg_test", "hvg_train"], default="mean")
    p.add_argument("--eval-cells", type=int, default=128)
    p.add_argument("--eval-seed", type=int, default=None)
    p.add_argument("--eval-control-cells", type=int, default=0)
    p.add_argument("--eval-batch-size", type=int, default=64)
    p.add_argument("--eval-every", type=int, default=5000)
    p.add_argument("--eval-progress", action="store_true")
    p.add_argument("--eval-max-conditions", type=int, default=0)
    p.add_argument("--cell-eval", dest="cell_eval", action="store_true", default=True)
    p.add_argument("--no-cell-eval", dest="cell_eval", action="store_false")
    p.add_argument("--cell-eval-threads", type=int, default=4)
    p.add_argument("--cell-eval-batch-size", type=int, default=100)
    p.add_argument("--cell-eval-profile", choices=["full", "minimal", "de", "anndata", "vcc"], default="full")
    p.add_argument("--cell-eval-skip-metrics", default=DEFAULT_CELL_EVAL_SKIP)
    p.add_argument("--save-eval-anndata", action="store_true")
    p.add_argument("--save-every", type=int, default=5000)
    p.add_argument("--max-hours", type=float, default=0.0)
    p.add_argument("--log-every", type=int, default=25)
    p.add_argument("--num-workers", type=int, default=0)
    p.add_argument("--precision", choices=["fp32", "fp16", "bf16"], default="fp32")
    p.add_argument("--compile", action="store_true")
    p.add_argument("--compile-eval", action="store_true")
    p.add_argument("--compile-mode", default="default")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--device", default=None)
    return p.parse_args()


if __name__ == "__main__":
    train(parse_args())

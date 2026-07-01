from __future__ import annotations
import csv
import math
import time
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
from data import PreparedData, normalize_condition, split_condition, require_anndata_stack, DEFAULT_CELL_EVAL_SKIP


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

def make_flow_noise(source, args, eval_mode: bool = False, delta_noise_scale: float | None = None):
    if args.flow_target == "delta":
        scale = args.delta_noise_scale if delta_noise_scale is None else delta_noise_scale
        if eval_mode and args.eval_delta_noise_scale is not None:
            scale = args.eval_delta_noise_scale
        return torch.randn_like(source) * float(scale)
    if args.noise == "poisson":
        return make_lognorm_poisson_noise(source)
    return torch.randn_like(source)

@torch.inference_mode()
def generate(model, source, pert, pert_gene, steps, args, device):
    source = source.to(device=device, dtype=torch.float32)
    pert = pert.to(device)
    pert_gene = pert_gene.to(device)
    x = make_flow_noise(source, args, eval_mode=True)
    dt = 1.0 / float(steps)
    method = getattr(args, "ode_method", "euler")
    for i in range(steps):
        t = torch.full((source.size(0),), i / float(steps), device=device)
        if method == "euler":
            x = x + dt * model(x, source, t, pert, pert_gene)
        elif method == "heun":
            k1 = model(x, source, t, pert, pert_gene)
            t_next = torch.full((source.size(0),), (i + 1) / float(steps), device=device)
            k2 = model(x + dt * k1, source, t_next, pert, pert_gene)
            x = x + 0.5 * dt * (k1 + k2)
        elif method == "rk4":
            half_t = torch.full((source.size(0),), (i + 0.5) / float(steps), device=device)
            next_t = torch.full((source.size(0),), (i + 1) / float(steps), device=device)
            k1 = model(x, source, t, pert, pert_gene)
            k2 = model(x + 0.5 * dt * k1, source, half_t, pert, pert_gene)
            k3 = model(x + 0.5 * dt * k2, source, half_t, pert, pert_gene)
            k4 = model(x + dt * k3, source, next_t, pert, pert_gene)
            x = x + (dt / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)
        else:
            raise ValueError(f"Unsupported ODE method: {method}")
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
            f"ode_method {getattr(args, 'ode_method', 'euler')} | "
            f"eval_delta_noise {getattr(args, 'eval_delta_noise_scale', None)} | "
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
        skip_metrics=None if getattr(args, "official_eval_no_skip", False) else parse_skip_metrics(args.cell_eval_skip_metrics),
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
        metrics["EvalOdeMethod"] = getattr(args, "ode_method", "euler")
        metrics["EvalOdeSteps"] = getattr(args, "ode_steps", float("nan"))
        eval_noise = getattr(args, "eval_delta_noise_scale", None)
        if eval_noise is None:
            eval_noise = getattr(args, "delta_noise_scale", float("nan"))
        metrics["EvalDeltaNoiseScale"] = eval_noise
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
        "EvalOdeMethod",
        "EvalOdeSteps",
        "EvalDeltaNoiseScale",
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
        f"GeneSel {metrics.get('EvalGeneSelection', 'n/a')}",
        f"ODE {metrics.get('EvalOdeMethod', 'euler')}/{metrics.get('EvalOdeSteps', 'n/a')}",
        f"EvalNoise {format_metric(metrics.get('EvalDeltaNoiseScale'))}",
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

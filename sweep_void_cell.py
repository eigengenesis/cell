#!/usr/bin/env python3
"""
Hyperparameter sweep harness for VOID Cell.

Runs short training trials, parses metrics.csv, ranks configs, and optionally
promotes the best configs into a second official-eval pass.
"""

from __future__ import annotations

import argparse
import copy
import csv
import json
import math
import os
import shlex
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any


LOWER_IS_BETTER = {"L2", "MSE", "MAE"}


@dataclass
class Trial:
    name: str
    overrides: dict[str, Any]


def preset_trials(profile: str) -> list[Trial]:
    base = {"lr": 1e-4, "recon_weight": 0.5, "bulk_loss_weight": 2.0, "gamma": 0.5}
    if profile == "micro":
        rows = [
            ("baseline_flow_mmd", {"recon_weight": 0.0, "bulk_loss_weight": 0.0}),
            ("adamw_cell_bulk2", {}),
            ("adamw_delta_bulk2", {"flow_target": "delta"}),
            ("muon_delta_bulk2", {"flow_target": "delta", "optimizer": "muon", "muon_lr": 0.02}),
        ]
    elif profile == "tiny":
        rows = [
            ("adamw_cell_bulk2", {}),
            ("adamw_delta_bulk2", {"flow_target": "delta"}),
            ("adamw_delta_lr2e4", {"flow_target": "delta", "lr": 2e-4}),
            ("adamw_delta_bulk4", {"flow_target": "delta", "bulk_loss_weight": 4.0}),
            ("muon_delta_bulk2", {"flow_target": "delta", "optimizer": "muon", "muon_lr": 0.02}),
            ("muon_delta_lr1e2", {"flow_target": "delta", "optimizer": "muon", "muon_lr": 0.01}),
        ]
    elif profile == "l2":
        rows = [
            ("baseline_flow_mmd", {"recon_weight": 0.0, "bulk_loss_weight": 0.0}),
            ("adamw_cell_bulk2", {}),
            ("adamw_delta_bulk1", {"flow_target": "delta", "bulk_loss_weight": 1.0}),
            ("adamw_delta_bulk2", {"flow_target": "delta"}),
            ("adamw_delta_bulk4", {"flow_target": "delta", "bulk_loss_weight": 4.0}),
            ("adamw_delta_lr2e4", {"flow_target": "delta", "lr": 2e-4}),
            ("adamw_delta_rec1", {"flow_target": "delta", "recon_weight": 1.0}),
            ("adamw_delta_gamma1", {"flow_target": "delta", "gamma": 1.0}),
            ("muon_delta_bulk2", {"flow_target": "delta", "optimizer": "muon", "muon_lr": 0.02}),
            ("muon_delta_lr1e2", {"flow_target": "delta", "optimizer": "muon", "muon_lr": 0.01}),
        ]
    elif profile == "muon":
        rows = [
            ("adamw_delta_control", {"flow_target": "delta"}),
            ("muon_delta_base", {"flow_target": "delta", "optimizer": "muon", "muon_lr": 0.02}),
            ("muon_delta_muonlr1e2", {"flow_target": "delta", "optimizer": "muon", "muon_lr": 0.01}),
            ("muon_delta_muonlr3e2", {"flow_target": "delta", "optimizer": "muon", "muon_lr": 0.03}),
            ("muon_delta_adamlr2e4", {"flow_target": "delta", "optimizer": "muon", "lr": 2e-4, "muon_lr": 0.02}),
            ("muon_delta_bulk1", {"flow_target": "delta", "optimizer": "muon", "muon_lr": 0.02, "bulk_loss_weight": 1.0}),
            ("muon_delta_bulk4", {"flow_target": "delta", "optimizer": "muon", "muon_lr": 0.02, "bulk_loss_weight": 4.0}),
            ("muon_delta_gamma025", {"flow_target": "delta", "optimizer": "muon", "muon_lr": 0.02, "gamma": 0.25}),
            ("muon_delta_gamma1", {"flow_target": "delta", "optimizer": "muon", "muon_lr": 0.02, "gamma": 1.0}),
            ("muon_delta_dir01", {"flow_target": "delta", "optimizer": "muon", "muon_lr": 0.005, "lr": 5e-5, "gamma": 0.25, "dir_weight": 0.1}),
            ("muon_delta_dir025", {"flow_target": "delta", "optimizer": "muon", "muon_lr": 0.005, "lr": 5e-5, "gamma": 0.25, "dir_weight": 0.25}),
        ]
    elif profile == "broad":
        rows = [
            ("baseline_flow_mmd", {"recon_weight": 0.0, "bulk_loss_weight": 0.0}),
            ("adamw_cell_bulk2", {}),
            ("adamw_fused_cell_bulk2", {"optimizer": "adamw_fused"}),
            ("adamw_delta_bulk1", {"flow_target": "delta", "bulk_loss_weight": 1.0}),
            ("adamw_delta_bulk2", {"flow_target": "delta"}),
            ("adamw_delta_bulk4", {"flow_target": "delta", "bulk_loss_weight": 4.0}),
            ("adamw_delta_lr2e4", {"flow_target": "delta", "lr": 2e-4}),
            ("adamw_delta_lr3e4", {"flow_target": "delta", "lr": 3e-4}),
            ("adamw_delta_rec1", {"flow_target": "delta", "recon_weight": 1.0}),
            ("adamw_delta_gamma1", {"flow_target": "delta", "gamma": 1.0}),
            ("muon_cell_bulk2", {"optimizer": "muon", "muon_lr": 0.02}),
            ("muon_delta_bulk2", {"flow_target": "delta", "optimizer": "muon", "muon_lr": 0.02}),
            ("muon_delta_lr1e2", {"flow_target": "delta", "optimizer": "muon", "muon_lr": 0.01}),
            ("m3_delta_bulk2", {"flow_target": "delta", "manifold_dim": 3}),
            ("m4_k40_delta_bulk2", {"flow_target": "delta", "graph_k": 40}),
        ]
    else:
        raise ValueError(f"Unknown profile: {profile}")

    trials = []
    for name, overrides in rows:
        config = dict(base)
        config.update(overrides)
        trials.append(Trial(name=name, overrides=config))
    return trials


def append_flag(cmd: list[str], name: str, value: Any) -> None:
    flag = "--" + name.replace("_", "-")
    if isinstance(value, bool):
        if value:
            cmd.append(flag)
        return
    if value is None:
        return
    if isinstance(value, (list, tuple)):
        if not value:
            return
        cmd.append(flag)
        cmd.extend(map(str, value))
        return
    cmd.extend([flag, str(value)])


def build_train_command(args, trial: Trial, output_dir: Path, official_eval: bool, steps: int, eval_every: int) -> list[str]:
    train_script = Path(__file__).with_name("train_void_cell.py")
    optimizer_keys = {
        "optimizer",
        "adam_beta1",
        "adam_beta2",
        "adam_eps",
        "muon_lr",
        "muon_momentum",
        "muon_ns_steps",
        "flow_target",
        "delta_noise_scale",
        "eval_delta_noise_scale",
        "dir_weight",
        "hetero_weight",
    }
    optimizer_values = {key: trial.overrides.get(key, getattr(args, key)) for key in optimizer_keys}
    cmd = [
        sys.executable,
        str(train_script),
        "--dataset",
        args.dataset,
        "--data-path",
        args.data_path,
        "--output-dir",
        str(output_dir),
        "--steps",
        str(steps),
        "--eval-every",
        str(eval_every),
        "--batch-size",
        str(args.batch_size),
        "--n-top-genes",
        str(args.n_top_genes),
        "--infer-top-genes",
        str(args.infer_top_genes),
        "--store-dtype",
        args.store_dtype,
        "--graph-cells",
        str(args.graph_cells),
        "--dim",
        str(args.dim),
        "--hidden",
        str(args.hidden),
        "--encode-blocks",
        str(args.encode_blocks),
        "--think-steps",
        str(args.think_steps),
        "--graph-k",
        str(args.graph_k),
        "--manifold-dim",
        str(args.manifold_dim),
        "--neighbor-chunk",
        str(args.neighbor_chunk),
        "--eval-top-genes",
        str(args.eval_top_genes),
        "--eval-cells",
        str(args.eval_cells),
        "--eval-batch-size",
        str(args.eval_batch_size),
        "--cell-eval-threads",
        str(args.cell_eval_threads),
        "--precision",
        args.precision,
        "--log-every",
        str(args.log_every),
        "--save-every",
        str(args.save_every),
        "--warmup-steps",
        str(args.warmup_steps),
        "--min-lr-ratio",
        str(args.min_lr_ratio),
        "--lr-schedule",
        args.lr_schedule,
        "--dir-weight",
        str(optimizer_values["dir_weight"]),
        "--hetero-weight",
        str(optimizer_values["hetero_weight"]),
        "--optimizer",
        optimizer_values["optimizer"],
        "--adam-beta1",
        str(optimizer_values["adam_beta1"]),
        "--adam-beta2",
        str(optimizer_values["adam_beta2"]),
        "--adam-eps",
        str(optimizer_values["adam_eps"]),
        "--muon-lr",
        str(optimizer_values["muon_lr"]),
        "--muon-momentum",
        str(optimizer_values["muon_momentum"]),
        "--muon-ns-steps",
        str(optimizer_values["muon_ns_steps"]),
        "--flow-target",
        optimizer_values["flow_target"],
        "--delta-noise-scale",
        str(optimizer_values["delta_noise_scale"]),
        "--seed",
        str(args.seed),
    ]
    append_flag(cmd, "eval_delta_noise_scale", optimizer_values["eval_delta_noise_scale"])
    if args.neighbor_gate:
        cmd.append("--neighbor-gate")
    if args.dataset == "norman":
        cmd.extend(["--split", args.split, "--fold", str(args.fold)])
    if args.max_hours_per_trial > 0:
        cmd.extend(["--max-hours", str(args.max_hours_per_trial)])
    if args.num_workers > 0:
        cmd.extend(["--num-workers", str(args.num_workers)])
    if args.checkpoint_blocks:
        cmd.append("--checkpoint-blocks")
    if args.use_mmd:
        cmd.append("--use-mmd")
    if official_eval:
        cmd.append("--cell-eval")
    else:
        cmd.append("--no-cell-eval")

    for key, value in trial.overrides.items():
        if key in optimizer_keys:
            continue
        append_flag(cmd, key, value)
    return cmd


def command_text(cmd: list[str]) -> str:
    return " ".join(shlex.quote(part) for part in cmd)


def read_last_metrics(run_dir: Path) -> dict[str, str] | None:
    path = run_dir / "metrics.csv"
    if not path.exists():
        return None
    with path.open(newline="") as f:
        rows = list(csv.DictReader(f))
    return rows[-1] if rows else None


def parse_float(value: Any) -> float:
    try:
        x = float(value)
    except (TypeError, ValueError):
        return math.nan
    return x


def rank_key(row: dict[str, Any], metric: str):
    value = parse_float(row.get(metric))
    if not math.isfinite(value):
        value = math.inf if metric in LOWER_IS_BETTER else -math.inf
    primary = value if metric in LOWER_IS_BETTER else -value
    mse = parse_float(row.get("MSE"))
    pearson = parse_float(row.get("Pearson_Delta"))
    return (
        primary,
        mse if math.isfinite(mse) else math.inf,
        -pearson if math.isfinite(pearson) else math.inf,
    )


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    columns = []
    for row in rows:
        for key in row:
            if key not in columns:
                columns.append(key)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)


def print_ranking(rows: list[dict[str, Any]], metric: str, limit: int = 12) -> None:
    print("\n=== Sweep ranking ===", flush=True)
    print(f"{'rank':>4} {'trial':<24} {'Eval':<22} {'L2':>10} {'MSE':>10} {'MAE':>10} {'Pearson':>10} {'DS':>8}", flush=True)
    for i, row in enumerate(rows[:limit], 1):
        print(
            f"{i:>4} {row.get('trial',''):<24} {row.get('Eval',''):<22} "
            f"{row.get('L2',''):>10} {row.get('MSE',''):>10} {row.get('MAE',''):>10} "
            f"{row.get('Pearson_Delta',''):>10} {row.get('DS',''):>8}",
            flush=True,
        )
    print(f"ranked_by={metric}", flush=True)


def run_trial(cmd: list[str], log_path: Path, dry_run: bool) -> int:
    print("\n$ " + command_text(cmd), flush=True)
    if dry_run:
        return 0
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("w") as log:
        proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
        assert proc.stdout is not None
        for line in proc.stdout:
            print(line, end="", flush=True)
            log.write(line)
            log.flush()
        return proc.wait()


def build_plan(args, official_eval: bool, steps: int, eval_every: int, stage_name: str, trials: list[Trial]) -> list[dict[str, Any]]:
    plan = []
    for idx, trial in enumerate(trials, 1):
        out = Path(args.base_output_dir) / stage_name / f"{idx:02d}_{trial.name}"
        cmd = build_train_command(args, trial, out, official_eval=official_eval, steps=steps, eval_every=eval_every)
        plan.append(
            {
                "trial": trial.name,
                "stage": stage_name,
                "output_dir": str(out),
                "overrides": trial.overrides,
                "command": cmd,
            }
        )
    return plan


def collect_result(item: dict[str, Any], returncode: int) -> dict[str, Any]:
    row = {
        "trial": item["trial"],
        "stage": item["stage"],
        "returncode": returncode,
        "output_dir": item["output_dir"],
    }
    metrics = read_last_metrics(Path(item["output_dir"]))
    if metrics:
        row.update(metrics)
    for key, value in item["overrides"].items():
        row[f"hp_{key}"] = value
    return row


def run_plan(args, plan: list[dict[str, Any]]) -> list[dict[str, Any]]:
    results = []
    for item in plan:
        out = Path(item["output_dir"])
        metrics = read_last_metrics(out)
        if metrics and args.resume and not args.overwrite:
            print(f"\n[skip] {item['trial']} already has metrics at {out / 'metrics.csv'}", flush=True)
            results.append(collect_result(item, 0))
            continue
        if out.exists() and args.overwrite and not args.dry_run:
            import shutil

            shutil.rmtree(out)
        returncode = run_trial(item["command"], out / "train.log", args.dry_run)
        results.append(collect_result(item, returncode))
        if returncode != 0 and args.stop_on_failure:
            raise RuntimeError(f"Trial failed with returncode={returncode}: {item['trial']}")
    return results


def promote_trials(args, ranked: list[dict[str, Any]], original_trials: list[Trial]) -> list[Trial]:
    by_name = {t.name: t for t in original_trials}
    promoted = []
    if args.promote_trials:
        for name in args.promote_trials:
            trial = by_name.get(name)
            if trial is None:
                choices = ", ".join(sorted(by_name))
                raise ValueError(f"Unknown --promote-trials entry '{name}'. Choices: {choices}")
            promoted.append(Trial(name=f"pick{len(promoted)+1:02d}_{trial.name}", overrides=dict(trial.overrides)))
        return promoted
    for row in ranked[: args.promote_top_k]:
        trial = by_name.get(row["trial"])
        if trial is None:
            continue
        promoted.append(Trial(name=f"rank{len(promoted)+1:02d}_{trial.name}", overrides=dict(trial.overrides)))
    return promoted


def official_args(args):
    promoted = copy.copy(args)
    for name in [
        "batch_size",
        "n_top_genes",
        "infer_top_genes",
        "graph_cells",
        "dim",
        "hidden",
        "encode_blocks",
        "think_steps",
        "graph_k",
        "manifold_dim",
        "neighbor_chunk",
        "eval_top_genes",
        "eval_cells",
        "eval_batch_size",
        "num_workers",
        "warmup_steps",
        "log_every",
    ]:
        value = getattr(args, f"official_{name}", None)
        if value is not None:
            setattr(promoted, name, value)
    return promoted


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", choices=["norman", "combosciplex"], default="norman")
    p.add_argument("--data-path", required=True)
    p.add_argument("--base-output-dir", default="runs/void_sweeps")
    p.add_argument("--profile", choices=["micro", "tiny", "l2", "muon", "broad"], default="l2")
    p.add_argument("--rank-metric", default="L2")
    p.add_argument("--steps", type=int, default=350)
    p.add_argument("--eval-every", type=int, default=300)
    p.add_argument("--official-eval", action="store_true")
    p.add_argument("--promote-top-k", type=int, default=0)
    p.add_argument("--promote-trials", nargs="*", default=None)
    p.add_argument("--official-steps", type=int, default=700)
    p.add_argument("--official-eval-every", type=int, default=500)
    p.add_argument("--max-trials", type=int, default=0)
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--resume", action="store_true")
    p.add_argument("--overwrite", action="store_true")
    p.add_argument("--stop-on-failure", action="store_true")

    p.add_argument("--split", choices=["additive", "combinations", "unseen"], default="additive")
    p.add_argument("--fold", type=int, default=1)
    p.add_argument("--batch-size", type=int, default=96)
    p.add_argument("--n-top-genes", type=int, default=3000)
    p.add_argument("--infer-top-genes", type=int, default=1000)
    p.add_argument("--store-dtype", choices=["float32", "float16"], default="float16")
    p.add_argument("--graph-cells", type=int, default=8192)
    p.add_argument("--dim", type=int, default=192)
    p.add_argument("--hidden", type=int, default=512)
    p.add_argument("--encode-blocks", type=int, default=4)
    p.add_argument("--think-steps", type=int, default=8)
    p.add_argument("--graph-k", type=int, default=30)
    p.add_argument("--manifold-dim", type=int, default=4)
    p.add_argument("--neighbor-chunk", type=int, default=1024)
    p.add_argument("--neighbor-gate", action="store_true")
    p.add_argument("--checkpoint-blocks", action="store_true", default=True)
    p.add_argument("--no-checkpoint-blocks", dest="checkpoint_blocks", action="store_false")
    p.add_argument("--use-mmd", action="store_true", default=True)
    p.add_argument("--no-mmd", dest="use_mmd", action="store_false")
    p.add_argument("--dir-weight", type=float, default=0.0)
    p.add_argument("--hetero-weight", type=float, default=0.0)
    p.add_argument("--optimizer", choices=["adamw", "adamw_fused", "muon"], default="adamw")
    p.add_argument("--adam-beta1", type=float, default=0.9)
    p.add_argument("--adam-beta2", type=float, default=0.999)
    p.add_argument("--adam-eps", type=float, default=1e-8)
    p.add_argument("--muon-lr", type=float, default=0.02)
    p.add_argument("--muon-momentum", type=float, default=0.95)
    p.add_argument("--muon-ns-steps", type=int, default=5)
    p.add_argument("--flow-target", choices=["cell", "delta"], default="cell")
    p.add_argument("--delta-noise-scale", type=float, default=1.0)
    p.add_argument("--eval-delta-noise-scale", type=float, default=None)
    p.add_argument("--warmup-steps", type=int, default=100)
    p.add_argument("--lr-schedule", choices=["none", "cosine"], default="cosine")
    p.add_argument("--min-lr-ratio", type=float, default=0.1)
    p.add_argument("--eval-top-genes", type=int, default=1000)
    p.add_argument("--eval-cells", type=int, default=48)
    p.add_argument("--eval-batch-size", type=int, default=32)
    p.add_argument("--cell-eval-threads", type=int, default=2)
    p.add_argument("--precision", choices=["fp32", "fp16", "bf16"], default="fp16")
    p.add_argument("--log-every", type=int, default=50)
    p.add_argument("--save-every", type=int, default=1000)
    p.add_argument("--max-hours-per-trial", type=float, default=0.0)
    p.add_argument("--num-workers", type=int, default=2)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--official-batch-size", type=int, default=None)
    p.add_argument("--official-n-top-genes", type=int, default=None)
    p.add_argument("--official-infer-top-genes", type=int, default=None)
    p.add_argument("--official-graph-cells", type=int, default=None)
    p.add_argument("--official-dim", type=int, default=None)
    p.add_argument("--official-hidden", type=int, default=None)
    p.add_argument("--official-encode-blocks", type=int, default=None)
    p.add_argument("--official-think-steps", type=int, default=None)
    p.add_argument("--official-graph-k", type=int, default=None)
    p.add_argument("--official-manifold-dim", type=int, default=None)
    p.add_argument("--official-neighbor-chunk", type=int, default=None)
    p.add_argument("--official-eval-top-genes", type=int, default=None)
    p.add_argument("--official-eval-cells", type=int, default=None)
    p.add_argument("--official-eval-batch-size", type=int, default=None)
    p.add_argument("--official-num-workers", type=int, default=None)
    p.add_argument("--official-warmup-steps", type=int, default=None)
    p.add_argument("--official-log-every", type=int, default=None)
    return p.parse_args()


def main():
    args = parse_args()
    if args.eval_every >= args.steps:
        args.eval_every = max(1, args.steps - 1)
        print(f"[sweep] adjusted eval_every to {args.eval_every} so the screen eval fires", flush=True)
    if args.official_eval_every >= args.official_steps:
        args.official_eval_every = max(1, args.official_steps - 1)
        print(f"[sweep] adjusted official_eval_every to {args.official_eval_every} so official eval fires", flush=True)
    out_root = Path(args.base_output_dir)
    out_root.mkdir(parents=True, exist_ok=True)
    trials = preset_trials(args.profile)
    if args.max_trials > 0:
        trials = trials[: args.max_trials]

    screen_plan = build_plan(
        args,
        official_eval=args.official_eval,
        steps=args.steps,
        eval_every=args.eval_every,
        stage_name="screen",
        trials=trials,
    )
    with (out_root / "sweep_plan.json").open("w") as f:
        json.dump(
            {
                "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
                "profile": args.profile,
                "rank_metric": args.rank_metric,
                "screen": [
                    {k: v for k, v in item.items() if k != "command"} | {"command": command_text(item["command"])}
                    for item in screen_plan
                ],
            },
            f,
            indent=2,
        )

    results = run_plan(args, screen_plan)
    ranked = sorted(results, key=lambda row: rank_key(row, args.rank_metric))
    write_csv(out_root / "sweep_results.csv", ranked)
    print_ranking(ranked, args.rank_metric)

    if ranked:
        best_cmd = screen_plan[[item["trial"] for item in screen_plan].index(ranked[0]["trial"])]["command"]
        (out_root / "best_screen_command.sh").write_text(command_text(best_cmd) + "\n")

    if args.promote_top_k > 0 or args.promote_trials:
        promoted = promote_trials(args, ranked, trials)
        if promoted:
            print(f"\n=== Promoting top {len(promoted)} configs to official eval ===", flush=True)
            official_run_args = official_args(args)
            official_plan = build_plan(
                official_run_args,
                official_eval=True,
                steps=official_run_args.official_steps,
                eval_every=official_run_args.official_eval_every,
                stage_name="official",
                trials=promoted,
            )
            official_results = run_plan(args, official_plan)
            official_ranked = sorted(official_results, key=lambda row: rank_key(row, args.rank_metric))
            write_csv(out_root / "official_results.csv", official_ranked)
            print_ranking(official_ranked, args.rank_metric)
            if official_ranked:
                best_cmd = official_plan[[item["trial"] for item in official_plan].index(official_ranked[0]["trial"])]["command"]
                (out_root / "best_official_command.sh").write_text(command_text(best_cmd) + "\n")


if __name__ == "__main__":
    main()

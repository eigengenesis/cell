from __future__ import annotations
import pickle
import random
from dataclasses import dataclass
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import Dataset


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


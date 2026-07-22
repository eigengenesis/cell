from __future__ import annotations

import torch


def build_gene_column_lookup(gene_token_ids: torch.Tensor, vocab_size: int) -> torch.Tensor:
    lookup = torch.full((int(vocab_size),), -1, dtype=torch.long)
    lookup[gene_token_ids.cpu().long()] = torch.arange(gene_token_ids.numel(), dtype=torch.long)
    return lookup


def build_neighbor_column_table(
    mask_path: str,
    gene_token_ids: torch.Tensor,
    vocab_size: int,
    topk: int,
    corr_path: str | None = None,
) -> torch.Tensor:
    mask = torch.load(mask_path, map_location="cpu").bool()
    corr = torch.load(corr_path, map_location="cpu").float() if corr_path else None
    gene_tokens = gene_token_ids.cpu().long()
    table = torch.full((int(vocab_size), int(topk)), -1, dtype=torch.long)

    for token in gene_tokens.tolist():
        allowed = ~mask[token, gene_tokens]
        allowed &= gene_tokens.ne(token)
        candidate_columns = torch.nonzero(allowed, as_tuple=False).flatten()
        if candidate_columns.numel() == 0:
            continue
        if corr is not None:
            scores = corr[token, gene_tokens[candidate_columns]].abs()
            keep = torch.topk(scores, k=min(int(topk), scores.numel())).indices
            candidate_columns = candidate_columns[keep]
        else:
            candidate_columns = candidate_columns[: int(topk)]
        table[token, : candidate_columns.numel()] = candidate_columns
    return table


def action_aware_gene_sample(
    n_genes: int,
    n_select: int,
    perturbation_ids: torch.Tensor,
    gene_column_by_token: torch.Tensor,
    neighbor_columns: torch.Tensor,
    device: torch.device,
) -> tuple[torch.Tensor, int, int, float]:
    n_select = min(int(n_select), int(n_genes))
    tokens = torch.unique(perturbation_ids.detach().reshape(-1).long()).cpu()
    tokens = tokens[(tokens >= 0) & (tokens < gene_column_by_token.numel())]

    all_target_columns = gene_column_by_token[tokens]
    all_target_columns = torch.unique(all_target_columns[all_target_columns >= 0])

    target_columns = all_target_columns
    if target_columns.numel() > n_select:
        perm = torch.randperm(target_columns.numel())
        target_columns = target_columns[perm[:n_select]]

    neighbors = neighbor_columns[tokens].reshape(-1) if tokens.numel() else torch.empty(0, dtype=torch.long)
    neighbors = torch.unique(neighbors[neighbors >= 0])
    if target_columns.numel():
        neighbors = neighbors[~torch.isin(neighbors, target_columns)]
    neighbor_budget = max(n_select - target_columns.numel(), 0)
    if neighbors.numel() > neighbor_budget:
        perm = torch.randperm(neighbors.numel())
        neighbors = neighbors[perm[:neighbor_budget]]

    mandatory = torch.cat([target_columns, neighbors]).to(device=device)
    available = torch.ones(int(n_genes), dtype=torch.bool, device=device)
    available[mandatory] = False
    random_columns = torch.randperm(int(n_genes), device=device)
    random_budget = max(n_select - mandatory.numel(), 0)
    random_columns = random_columns[available[random_columns]][:random_budget]
    selected = torch.cat([mandatory, random_columns])
    selected = selected[torch.randperm(selected.numel(), device=device)]

    all_target_columns = all_target_columns.to(device=device)
    covered = (
        torch.isin(all_target_columns, selected).float().mean().item()
        if all_target_columns.numel() else 1.0
    )
    return selected, int(all_target_columns.numel()), int(mandatory.numel()), float(covered)
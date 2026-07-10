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
    tokens = torch.unique(perturbation_ids.detach().reshape(-1).long())
    tokens = tokens[(tokens >= 0) & (tokens < gene_column_by_token.numel())]

    target_columns = gene_column_by_token[tokens]
    target_columns = target_columns[target_columns >= 0]
    neighbors = neighbor_columns[tokens].reshape(-1) if tokens.numel() else torch.empty(0, dtype=torch.long)
    neighbors = neighbors[neighbors >= 0]
    target_columns = torch.unique(target_columns)
    neighbors = torch.unique(neighbors)
    if target_columns.numel():
        neighbors = neighbors[~torch.isin(neighbors, target_columns)]
    neighbor_budget = max(n_select - target_columns.numel(), 0)
    mandatory = torch.cat([target_columns, neighbors[:neighbor_budget]]).to(device=device)

    available = torch.ones(int(n_genes), dtype=torch.bool, device=device)
    available[mandatory] = False
    random_columns = torch.randperm(int(n_genes), device=device)
    random_columns = random_columns[available[random_columns]][: n_select - mandatory.numel()]
    selected = torch.cat([mandatory, random_columns])
    selected = selected[torch.randperm(selected.numel(), device=device)]
    target_columns = target_columns.to(device=device)
    if target_columns.numel():
        covered = torch.isin(target_columns, selected).float().mean().item()
    else:
        covered = 1.0
    return selected, int(target_columns.numel()), int(mandatory.numel()), float(covered)

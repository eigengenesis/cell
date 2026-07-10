from pathlib import Path

import torch

from model import model
from sampling import action_aware_gene_sample, build_gene_column_lookup


def make_model(tmp_path: Path):
    vocab_size = 12
    mask = torch.ones(vocab_size, vocab_size, dtype=torch.bool)
    for token in range(4, vocab_size):
        for offset in (-1, 1):
            neighbor = 4 + ((token - 4 + offset) % (vocab_size - 4))
            mask[token, neighbor] = False
    mask_path = tmp_path / "mask.pt"
    torch.save(mask, mask_path)
    wire_path = tmp_path / "wire.pt"
    torch.save(torch.randn(vocab_size, 4), wire_path)
    return model(
        ntoken=vocab_size,
        d_model=16,
        d_hid=32,
        nlayers=2,
        dropout=0.0,
        mask_path=str(mask_path),
        wire_path=str(wire_path),
        use_wire=True,
        think_steps=2,
        control_token_id=3,
    )


def test_adaln_stays_zero_after_model_initialization(tmp_path):
    net = make_model(tmp_path)
    for block in [*net.encode, net.ghost]:
        assert torch.count_nonzero(block.ada.net[-1].weight) == 0
        assert torch.count_nonzero(block.ada.net[-1].bias) == 0


def test_composition_and_local_action_are_order_invariant(tmp_path):
    net = make_model(tmp_path)
    perturbations = torch.tensor([[4, 5], [5, 4]])
    global_action, components, valid = net.compose_perturbation(perturbations)
    assert torch.allclose(global_action[0], global_action[1], atol=1e-6)

    gene_row = torch.arange(4, 12)
    field = net.local_action_field(
        perturbations, components, valid, gene_row,
        b=2, g=gene_row.numel(), dtype=components.dtype,
    )
    assert torch.allclose(field[0], field[1], atol=1e-6)

    gene_ids = gene_row.repeat(2, 1)
    current = torch.rand(1, gene_row.numel()).repeat(2, 1)
    source = torch.rand(1, gene_row.numel()).repeat(2, 1)
    prediction = net(gene_ids, current, torch.full((2,), 0.5), source, perturbations)
    assert torch.allclose(prediction[0], prediction[1], atol=1e-6)


def test_control_contributes_zero_action(tmp_path):
    net = make_model(tmp_path)
    torch.nn.init.constant_(net.action_rho[-1].bias, 2.0)
    perturbations = torch.tensor([[3, 3]])
    global_action, components, valid = net.compose_perturbation(perturbations)
    field = net.local_action_field(
        perturbations, components, valid, torch.arange(4, 12),
        b=1, g=8, dtype=components.dtype,
    )
    assert torch.count_nonzero(global_action) == 0
    assert torch.count_nonzero(field) == 0


def test_noop_ghost_update_does_not_amplify_state():
    x = torch.randn(2, 7, 8)
    global_state = torch.randn(2, 8)
    x_init = torch.randn_like(x)
    global_init = torch.randn_like(global_state)
    ghost_x = x + x_init
    global_input = global_state + global_init
    out_x, out_global = model.apply_ghost_update(
        x, global_state, ghost_x, global_input,
        cand_x=ghost_x, cand_global=global_input,
        ghost_gate=torch.tensor(0.1),
    )
    assert torch.equal(out_x, x)
    assert torch.equal(out_global, global_state)


def test_action_aware_sampler_keeps_targets_and_neighbors():
    gene_tokens = torch.arange(4, 14)
    lookup = build_gene_column_lookup(gene_tokens, vocab_size=14)
    neighbors = torch.full((14, 3), -1, dtype=torch.long)
    neighbors[5] = torch.tensor([0, 2, 3])
    neighbors[8] = torch.tensor([4, 6, 7])
    selected, target_count, mandatory_count, coverage = action_aware_gene_sample(
        n_genes=10,
        n_select=8,
        perturbation_ids=torch.tensor([[5, 8], [5, 8]]),
        gene_column_by_token=lookup,
        neighbor_columns=neighbors,
        device=torch.device("cpu"),
    )
    expected = {0, 1, 2, 3, 4, 6, 7}
    assert target_count == 2
    assert mandatory_count == len(expected)
    assert coverage == 1.0
    assert expected.issubset(set(selected.tolist()))


def test_checkpoint_roundtrip_reproduces_output(tmp_path):
    net = make_model(tmp_path).eval()
    gene_ids = torch.arange(4, 12).repeat(2, 1)
    current = torch.rand(2, 8)
    source = torch.rand(2, 8)
    t = torch.tensor([0.25, 0.75])
    perturbations = torch.tensor([[4, 5], [6, 7]])
    with torch.no_grad():
        expected = net(gene_ids, current, t, source, perturbations)

    checkpoint = tmp_path / "checkpoint.pt"
    torch.save(net.state_dict(), checkpoint)
    restored = make_model(tmp_path).eval()
    restored.load_state_dict(torch.load(checkpoint, weights_only=True))
    with torch.no_grad():
        actual = restored(gene_ids, current, t, source, perturbations)
    assert torch.equal(actual, expected)

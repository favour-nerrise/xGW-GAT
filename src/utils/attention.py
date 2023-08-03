import networkx as nx
import torch


def save_attn_scores(attn, data):
    """
    Save aggregated attention scores to file
    """
    edge_idx = attn[0]
    alpha = attn[1]
    alpha = alpha[: edge_idx.size(1)]
    if alpha.dim() == 2:
        alpha_max = getattr(torch, "max")(alpha, dim=-1)
        if isinstance(alpha_max, tuple):
            alpha_max = alpha_max[0]

    G = nx.Graph()
    edge_idx_cpu = edge_idx.cpu().detach().numpy()
    edge_w_cpu = alpha_max.cpu().detach().numpy()
    for (u, v), w in zip(edge_idx_cpu.T, edge_w_cpu):
        G.add_edge(u, v, weight=w)

    # Save attention graph
    nx.write_graphml_lxml(
        G, f"results/explanations/graphs/attn_graphs_{int(data.s_ID.item())}.graphml"
    )

    # Save attention coefficients
    attn_agg = (edge_idx, alpha_max)
    torch.save(
        attn_agg,
        f"results/explanations/attn_scores/attn_out_{int(data.s_ID.item())}.pt",
    )

import networkx as nx
import numpy as np
import pandas as pd
import torch
from torch_geometric.explain import AttentionExplainer, Explainer

from src.utils.data_utils import edge_index_to_adj_matrix
from src.utils.visualization import to_brainnet, visualize_graph


def get_top_k(mask, k=10):
    """
    Threshold explanation edge masks to top-k attention weights/edges
    """
    mask = torch.Tensor(mask)
    _, index = torch.topk(
        mask.flatten(),
        k=k,
    )

    out = torch.zeros_like(mask.flatten())
    out[index] = 1.0
    return out.view(mask.size())


def explain(model, dataloader, args):
    """
    Generate and save explanation masks for each sample and per class
    """
    datasets = {"0": [], "1": [], "2": [], "3": []}
    for data in dataloader:
        if str(data.y.detach().cpu().item()) in datasets:
            datasets[str(data.y.detach().cpu().item())].append(data)

    # Train explainer mask
    explainer_args = {
        "mode": "binary_classification"
        if args.num_classes == 2
        else "multiclass_classification"
    }
    explainer = Explainer(
        model=model,
        algorithm=AttentionExplainer(),
        explanation_type="model",
        edge_mask_type="object",
        model_config=dict(
            mode=explainer_args["mode"],
            task_level="graph",
            return_type="probs",
        ),
    )

    explanation_adjs = []
    explanation_edges = []

    for i in range(args.num_classes):
        adjs = []
        edges_all = []
        for data in list(datasets.values())[i]:
            data = data.to(args.device)
            exp = explainer(
                data, data.edge_index, edge_attr=data.edge_attr, batch=data.batch
            )
            G, edges, _ = visualize_graph(
                exp.edge_index,
                edge_attr=exp.edge_attr,
                x=exp.x,
                y=exp.y,
                threshold_num=10,
            )

            nx.write_graphml_lxml(
                G,
                f"outputs/explanations/graphs/xGWGAT_exp_graph_{int(exp.x.s_ID.item())}.graphml",
            )
            subgraph = exp.get_explanation_subgraph()

            mask = subgraph.edge_mask
            edges_all.append(edges)
            adj = edge_index_to_adj_matrix(
                subgraph.edge_index, subgraph.edge_attr, args.num_nodes
            )
            adjs.append(adj)

        avg_edges = np.array(edges_all).mean(axis=0)
        explanation_edges.append(avg_edges)

        avg_adjs = np.array(adjs).mean(axis=0)
        masked_adj = get_top_k(avg_adjs, k=10)
        explanation_adjs.append(masked_adj.numpy())

    explanation_adjs = np.array(explanation_adjs, dtype="object")
    explanation_edges = np.array(explanation_edges, dtype="object")

    roi_xyz = pd.read_csv(
        "community_networks/roi_coords.csv", index_col=False, header=None, skiprows=1
    )
    roi_xyz = roi_xyz.drop(roi_xyz.columns[0], axis=1).T
    roi_labels = pd.read_csv("community_networks/roi_names_mod.csv")["ROI"]
    roi_colors = pd.read_csv("community_networks/roi_names_mod.csv")["Color"].to_numpy()

    # Generate .node and .edge files for visualization in BrainNet Viewer
    to_brainnet(edges, roi_xyz, roi_labels, C=roi_colors, prefix="top10")
    for i in range(len(explanation_edges)):
        to_brainnet(
            explanation_edges[i], roi_xyz, roi_labels, C=roi_colors, prefix=f"top10_{i}"
        )

    return

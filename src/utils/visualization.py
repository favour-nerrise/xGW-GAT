from typing import Optional

import networkx as nx
import numpy as np
import torch
from torch import Tensor
from torch_geometric.data import Data
from torch_geometric.utils import to_networkx

from src.utils.data_utils import *


def visualize_graph(
    edge_index,
    edge_attr: Optional[Tensor],
    node_atts=None,
    x=None,
    y: Optional[torch.FloatTensor] = None,
    threshold_num=None,
):
    # Adapted from https://github.com/HennyJie/IBGNN/
    r"""Visualizes the graph given an edge mask
    :attr:`edge_mask`.
    Args:
        edge_index (LongTensor): The edge indices.
        edge_mask (Tensor): The edge mask.
        y (Tensor, optional): The ground-truth node-prediction labels used
            as node colorings. (default: :obj:`None`)
        threshold (float, optional): Sets a threshold for visualizing
            important edges. If set to :obj:`None`, will visualize all
            edges with transparancy indicating the importance of edges.
            (default: :obj:`None`)
        **kwargs (optional): Additional arguments passed to
            :func:`nx.draw`.
    :rtype: :class:`matplotlib.axes.Axes`, :class:`networkx.DiGraph`
    """
    if edge_attr is not None:
        assert edge_attr.size(0) == edge_index.size(1)

    subset = torch.arange(edge_index.max().item() + 1, device=edge_index.device)

    if edge_attr is None:
        edge_attr = torch.ones(edge_index.size(1), device=edge_index.device)

    if y is None:
        y = torch.zeros(edge_index.max().item() + 1, device=edge_index.device)
    else:
        y = y.cpu()
        y = y[subset].to(torch.float) / y.max().item()

    data = Data(edge_index=edge_index, att=edge_attr, x=x, y=y, num_nodes=y.size(0)).to(
        "cpu"
    )
    G = to_networkx(data, node_attrs=["y"], edge_attrs=["att"])
    mapping = {k: i for k, i in enumerate(subset.tolist())}
    G = nx.relabel_nodes(G, mapping)

    att_array = np.array([data["att"] for _, _, data in G.edges(data=True)])
    min_att, max_att = np.amin(att_array), np.amax(att_array)
    # reward = (max_att - min_att) / 10
    # att_array = self.reward_edge_postprocessing(att_array, edge_index, reward)
    # range_att = max_att - min_att
    # if range_att == 0:
    #     range_att = max_att
    graph_nodes = G.nodes

    edges = edge_index_to_adj_matrix(edge_index, edge_attr, y.shape[0])

    unfiltered_edges = edges.copy()
    if threshold_num is not None:
        edges = denoise_graph(edges, 0, threshold_num=threshold_num)

    return G, edges, unfiltered_edges


def denoise_graph(
    adj,
    node_idx,
    feat=None,
    label=None,
    threshold=None,
    threshold_num=None,
    max_component=True,
):
    """Cleaning a graph by thresholding its node values.
    Args:
        - adj               :  Adjacency matrix.
        - node_idx          :  Index of node to highlight (TODO What is this used for??)
        - feat              :  An array of node features.
        - label             :  A list of node labels.
        - threshold         :  The weight threshold.
        - theshold_num      :  The maximum number of nodes to threshold.
        - max_component     :  TODO  Looks like this has already been implemented
    """
    num_nodes = adj.shape[-1]
    G = nx.Graph()
    G.add_nodes_from(range(num_nodes))
    G.nodes[node_idx]["self"] = 1
    if feat is not None:
        for node in G.nodes():
            G.nodes[node]["feat"] = feat[node]
    if label is not None:
        for node in G.nodes():
            G.nodes[node]["label"] = label[node]

    if threshold_num is not None:
        # this is for symmetric graphs: edges are repeated twice in adj
        adj_threshold_num = threshold_num * 2
        # adj += np.random.rand(adj.shape[0], adj.shape[1]) * 1e-4
        neigh_size = len(adj[adj > 0])
        threshold_num = min(neigh_size, adj_threshold_num)
        threshold = np.sort(adj[adj > 0])[-threshold_num]

    if threshold is not None:
        weighted_edge_list = [
            (i, j, adj[i, j] if adj[i, j] >= threshold else 0)
            for i in range(num_nodes)
            for j in range(num_nodes)
        ]
    else:
        weighted_edge_list = [
            (i, j, adj[i, j])
            for i in range(num_nodes)
            for j in range(num_nodes)
            if adj[i, j] > 1e-6
        ]
    G.add_weighted_edges_from(weighted_edge_list)

    for i in range(num_nodes):
        for j in range(num_nodes):
            adj[i][j] = weighted_edge_list[i * num_nodes + j][2]
    return adj


def to_brainnet(
    edges,
    roi_xyz,
    roi_labels,
    C=None,
    S=None,
    path="outputs/explanations/bnv",
    prefix="bnv",
):
    """Export data to plaintext file(s) for use with BrainNet Viewer
    [1]. For details regarding .node and .edge file construction, the
    user is directed to the BrainNet Viewer User Manual.
    This code was quality tested using BrainNet version 1.61 released on
    2017-10-31 with MATLAB 9.3.0.713579 (R2017b).
    Parameters:
    -----------
    edges : numpy array
        N x N matrix containing edge weights.
    roi_xyz : pandas dataframe
        N x 3 dataframe containing the (x, y, z) MNI coordinates of each
        brain ROI.
    roi_names : pandas series
        Names of each ROI as string.
    C : pandas series
        Node color value (defaults to same color). For modular color,
        use integers; for continuous data use floats.
    S : pandas series
        Node size value (defaults to same size).
    path : string
        Path to output directory (default is current directory). Note:
        do not include trailing '/' at end.
    prefix : string
        Filename prefix for output files.
    Returns
    -------
    <prefix>.node, <prefix>.edge : files
        Plaintext output files for input to BrainNet.
    References
    ----------
    [1] Xia M, Wang J, He Y (2013) BrainNet Viewer: A Network
        Visualization Tool for Human Brain Connectomics. PLoS ONE 8:
        e68910.
    """

    N = len(roi_xyz)  # number of nodes

    if C is None:
        C = np.ones(N)

    if S is None:
        S = np.ones(N)

    # BrainNet does not recognize node labels with white space, replace
    # spaces with underscore
    labels = roi_labels.str.replace(" ", "_").to_list()

    # Build .node dataframe
    df = roi_xyz.copy()
    df = df.assign(C=C).assign(S=S).assign(labels=labels)

    # Output .node file
    df.to_csv(f"{path}/{prefix}.node", sep="\t", header=False, index=False)
    print(f"Saved {path}/{prefix}.node")

    # Output .edge file
    np.savetxt(f"{path}/{prefix}.edge", edges, delimiter="\t")
    print(f"Saved {path}/{prefix}.edge")

"""
Utility functions for data processing.
"""
import copy
import csv
import math
import os
from typing import List

import networkx as nx
import numpy as np
import pandas as pd
import scipy
import torch
import torch_geometric
from scipy.sparse import coo_matrix
from torch import Tensor
from torch_geometric.data import Data
from torch_geometric.utils import degree
from tqdm import tqdm

from src.utils import SPD


def get_x(dataset: List[Data]):
    """
    Get the y values from a list of Data objects.
    """
    x = []
    for d in dataset:
        x.append(d.x)
    return torch.stack(x)


def get_y(dataset: List[Data]):
    """
    Get the y values from a list of Data objects.
    """
    y = []
    for d in dataset:
        y.append(d.y.item())
    return np.array(y)


def calculate_bin_edges(dataset: List[Data], num_bins: int = 10) -> Tensor:
    """
    Calculate the bin edges for a given edge attribute tensor.
    :param dataset: The dataset to calculate the bin edges for.
    :param num_bins: The number of bins.
    :return: The bin edges.
    """
    all_edges = np.concatenate([data.edge_attr.numpy() for data in dataset])
    _, bin_edges = np.histogram(all_edges, bins=num_bins)
    bin_edges = np.sort(bin_edges)
    return Tensor(bin_edges)


def create_features(data, score, args, method="eigen"):
    """Given data matrix and score vector, creates and saves
    the dictionaries for pairwise similarity features.
    Possible values for method:
        'abs': absolute differences
        'geo': geometric distance
        'tan': vectorized tangent matrix
        'node': node degree centrality
        'eigen': eigenvector centrality
        'close': closeness centrality
    """

    data_dict = {}
    score_dict = {}
    spd = SPD.SPD(args.num_nodes)
    print("Starting topological feature extraction...")
    for i in tqdm(range(data.shape[2])):
        for j in range(i + 1, data.shape[2]):
            if method == "abs":
                dist = np.abs(data[:, :, i] - data[:, :, j])
                dist = dist[np.triu_indices_from(dist, k=1)]
            if method == "geo":
                dist = spd.dist(data[:, :, i], data[:, :, j])
            if method in (
                "tan",
                "node",
                "eigen",
                "close",
                "concat_orig",
                "concat_scale",
            ):
                dist = spd.transp(
                    data[:, :, 0] + np.eye(args.num_nodes) * 1e-10,
                    data[:, :, i] + np.eye(args.num_nodes) * 1e-10,
                    spd.log(
                        data[:, :, i] + np.eye(args.num_nodes) * 1e-10,
                        data[:, :, j] + np.eye(args.num_nodes) * 1e-10,
                    ),
                )
                if method == "tan":
                    dist = dist[np.triu_indices_from(dist)]
                if method == "node":
                    dist = nx.convert_matrix.from_numpy_matrix(np.abs(dist))
                    dist = np.array(list(dict(dist.degree(weight="weight")).values()))
                if method == "eigen":
                    dist = nx.convert_matrix.from_numpy_matrix(np.abs(dist))
                    dist = np.array(
                        list(
                            dict(
                                nx.eigenvector_centrality(dist, weight="weight")
                            ).values()
                        )
                    )
                if method == "close":
                    dist = nx.convert_matrix.from_numpy_matrix(np.abs(dist))
                    dist = np.array(
                        list(
                            dict(
                                nx.closeness_centrality(dist, distance="weight")
                            ).values()
                        )
                    )
                if method in ("concat_orig", "concat_scale"):
                    dist = nx.convert_matrix.from_numpy_matrix(np.abs(dist))
                    dist_a = np.array(list(dict(dist.degree(weight="weight")).values()))
                    dist_b = np.array(
                        list(
                            dict(
                                nx.eigenvector_centrality(dist, weight="weight")
                            ).values()
                        )
                    )
                    dist_c = np.array(
                        list(
                            dict(
                                nx.closeness_centrality(dist, distance="weight")
                            ).values()
                        )
                    )
                    dist = (dist_a, dist_b, dist_c)
                data_dict[(i, j)] = data_dict[(j, i)] = dist
                score_dict[(i, j)] = score_dict[(j, i)] = np.abs(score[i] - score[j])
    if method == "concat_orig":
        dicts = data_dict
        data_dict = {}
        for key in dicts.keys():
            data_dict[key] = np.concatenate(dicts[key])
    if method == "concat_scale":
        dicts = data_dict
        lists = np.array([[x[0], x[1], x[2]] for x in dicts.values()])
        list_a = lists[:, 0]
        list_b = lists[:, 1]
        list_c = lists[:, 2]
        max_a, min_a = np.max(list_a, axis=0), np.min(list_a, axis=0)
        max_b, min_b = np.max(list_b, axis=0), np.min(list_b, axis=0)
        max_c, min_c = np.max(list_c, axis=0), np.min(list_c, axis=0)
        diff_a = max_a - min_a
        diff_b = max_b - min_b
        diff_c = max_c - min_c
        data_dict = {}
        for key in dicts.keys():
            a, b, c = dicts[key]
            data_dict[key] = np.concatenate(
                ((a - min_a) / diff_a, (b - min_b) / diff_b, (c - min_c) / diff_c)
            )

    return data_dict, score_dict


def load_scores(n_classes=4):
    # Handle gait scores loading for multi/binary classification
    if n_classes == 4:
        print("---------------------------------")
        print("Experiment: Multi-classification")
        print("---------------------------------")
        scores = torch.load(f"{Config.DATA_FOLDER}scores_multi.npy")
    elif n_classes == 2:
        # Binary classification
        print("---------------------------------")
        print("Experiment: Binary classification")
        print("---------------------------------")
        scores = torch.load(f"{Config.DATA_FOLDER}scores_binary.npy")
    else:
        raise Exception(
            f"Invalid number of classes, can't load ratings. Expected n_classes=2 or 4, but got {n_classes}."
        )
    return scores


def load_dataset_pytorch(n_classes=4):
    """Loads the data for the given population into a list of Pytorch Geometric
    Data objects, which then can be used to create DataLoaders.
    """
    connectomes = torch.load(f"{Config.DATA_FOLDER}connectomes.npy")
    scores = load_scores(n_classes)

    # Filter out empty connectomes
    connectomes[connectomes < 0] = 0

    pyg_data = []
    for subject in range(scores.shape[0]):
        sparse_mat = to_sparse(connectomes[:, :, subject])
        pyg_data.append(
            torch_geometric.data.Data(
                x=torch.eye(Config.ROI, dtype=torch.float),
                y=scores[subject].float(),
                edge_index=sparse_mat._indices(),
                edge_attr=sparse_mat._values().float(),
            )
        )
    # edge_index, edge_attr = get_graph(connectomes[:, :, subject])
    #   pyg_data.append(torch_geometric.data.Data(x=torch.eye(Config.ROI, dtype=torch.float),
    #                                           y=scores[subject].float(), edge_index=edge_index.float(),
    #                                           edge_attr=edge_attr.float()))
    return pyg_data


def get_graph(mat):
    """
    Get edge_index and edge_attribute from a graph
    :param x: (T, C)
    :param weighted: True or False
    :return: edge_index: (2, num_edges)
             edge_weight:(num_edges, 1)
    """
    G = nx.from_numpy_matrix(mat.numpy(), create_using=nx.Graph)
    edge_index = torch.tensor(list(G.edges))
    # print(edge_index.shape)
    edge_attribute = []
    for node1, node2, data in G.edges(data=True):
        edge_attribute.append(data["weight"])
    edge_attribute = torch.tensor(edge_attribute)
    # print(edge_attribute.shape)
    return edge_index, edge_attribute


def to_sparse(mat):
    """Transforms a square matrix to torch.sparse tensor

    Methods ._indices() and ._values() can be used to access to
    edge_index and edge_attr while generating Data objects
    """
    coo = coo_matrix(mat, dtype="float64")
    row = torch.from_numpy(coo.row.astype(np.int64))
    col = torch.from_numpy(coo.col.astype(np.int64))
    coo_index = torch.stack([row, col], dim=0)
    coo_values = torch.from_numpy(coo.data.astype(np.float64).reshape(-1, 1)).reshape(
        -1
    )
    sparse_mat = torch.sparse.LongTensor(coo_index, coo_values)
    return sparse_mat


def load_dataset_cpm(n_classes=4):
    """Loads the data for given population in the upper triangular matrix form
    as required by CPM functions.
    """
    connectomes = np.array(torch.load(f"{Config.DATA_FOLDER}connectomes.npy"))
    scores = load_scores(n_classes)

    fc_data = {}
    behav_data = {}
    for subject in range(scores.shape[0]):  # take upper triangular part of each matrix
        fc_data[subject] = connectomes[:, :, subject][
            np.triu_indices_from(connectomes[:, :, subject], k=1)
        ]
        behav_data[subject] = {"score": scores[subject].item()}
    return pd.DataFrame.from_dict(fc_data, orient="index"), pd.DataFrame.from_dict(
        behav_data, orient="index"
    )


def get_loaders(train, test, batch_size=1):
    """Returns data loaders for given data lists"""
    train_loader = torch_geometric.data.DataLoader(train, batch_size=batch_size)
    test_loader = torch_geometric.data.DataLoader(test, batch_size=batch_size)
    return train_loader, test_loader


def to_dense(data):
    """Returns a copy of the data object in Dense form."""
    denser = torch_geometric.transforms.ToDense()
    copy_data = denser(copy.deepcopy(data))
    return copy_data


def counts_from_cm(cm):
    """
    Returns TP, FN FP, and TN for each class in the confusion matrix
    """
    tp_all, fn_all, fp_all, tn_all = 0.0, 0.0, 0.0, 0.0

    for i in range(cm.shape[0]):
        tp = cm[0][i]

        fn_mask = np.zeros(cm.shape)
        fn_mask[i, :] = 1
        fn_mask[i, i] = 0
        fn = np.sum(np.multiply(cm, fn_mask))

        fp_mask = np.zeros(cm.shape)
        fp_mask[:, i] = 1
        fp_mask[i, i] = 0
        fp = np.sum(np.multiply(cm, fp_mask))

        tn_mask = 1 - (fn_mask + fp_mask)
        tn_mask[i, i] = 0
        tn = np.sum(np.multiply(cm, tn_mask))

        tp_all += tp
        fn_all += fn
        fp_all += fp
        tn_all += tn
    return tp_all, fn_all, fp_all, tn_all


def save_edges_to_mat(edges: np.ndarray, filename):
    edge_dict = {"edges": edges}
    scipy.io.savemat(filename, edge_dict)


def edge_index_to_adj_matrix(
    edge_index: Tensor, edge_attr: Tensor, num_node: int
) -> np.ndarray:
    adj = np.zeros((num_node, num_node))
    for i in range(edge_index.shape[1]):
        source = edge_index[0, i].item()
        target = edge_index[1, i].item()
        adj[source, target] = edge_attr[i].item()
    return adj


def edges2adj(edges: Tensor, num_nodes: int = 0) -> np.ndarray:
    if num_nodes == 0:
        num_nodes = int(math.sqrt(edges.shape[0]))
    adj = np.zeros((num_nodes, num_nodes))
    for index, edge in enumerate(edges):
        adj[index % num_nodes, int(index / num_nodes)] = edge
    return adj


def count_degrees(train_dataset):
    max_degree = -1
    for data in train_dataset:
        if isinstance(data, list):
            g = data[0]
        else:
            g = data
        d = degree(g.edge_index[1], num_nodes=g.num_nodes, dtype=torch.long)
        max_degree = max(max_degree, int(d.max()))

    # Compute the in-degree histogram tensor
    deg = torch.zeros(max_degree + 1, dtype=torch.long)
    print("Computing degrees for PNA...")
    for data in train_dataset:
        if isinstance(data, list):
            g = data[0]
        else:
            g = data
        d = degree(g.edge_index[1], num_nodes=g.num_nodes, dtype=torch.long)
        deg += torch.bincount(d, minlength=deg.numel())

    return deg


def generate_full_edges(num_nodes) -> Tensor:
    full_edge_index = np.zeros((2, num_nodes * num_nodes), dtype=np.long)

    for source in range(0, num_nodes):
        for target in range(0, num_nodes):
            row = source * num_nodes + target
            full_edge_index[0, row] = source
            full_edge_index[1, row] = target

    full_edge_index_tensor = torch.LongTensor(full_edge_index)
    return full_edge_index_tensor


def map_edges_attr(attr: Tensor, edge_index: Tensor, num_nodes: int) -> Tensor:
    new_edge_attrs = np.zeros((num_nodes * num_nodes,))
    for i in range(attr.shape[0]):
        source = edge_index[0, i]
        target = edge_index[1, i]

        # maps edge attr to new index
        new_index = source * num_nodes + target
        new_edge_attrs[new_index] = attr[i].item()

    return Tensor(new_edge_attrs)


def generate_community_labels_for_edges(
    edge_index: Tensor, node_labels: List[int]
) -> Tensor:
    edge_count = edge_index.shape[1]
    edge_community_label = np.zeros(edge_count)
    for row in range(edge_count):
        source: Tensor = edge_index[0, row]
        target: Tensor = edge_index[1, row]
        if node_labels[source.item()] == node_labels[target.item()]:
            # If source.label == target.label,
            # then set the corresponding community label to 1
            edge_community_label[row] = 1

    return Tensor(edge_community_label)

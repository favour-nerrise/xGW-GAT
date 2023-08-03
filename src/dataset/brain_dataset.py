import logging
import os.path as osp
import sys

import numpy as np
import torch
from scipy.io import loadmat
from torch_geometric.data import Data, InMemoryDataset
from torch_geometric.data.dataset import files_exist
from torch_geometric.data.makedirs import makedirs

from .base_transform import BaseTransform
from .private.load_private import load_data_private


def dense_to_ind_val(adj):
    assert adj.dim() >= 2 and adj.dim() <= 3
    assert adj.size(-1) == adj.size(-2)

    index = (torch.isnan(adj) == 0).nonzero(as_tuple=True)
    edge_attr = adj[index]

    return torch.stack(index, dim=0), edge_attr


class BrainDataset(InMemoryDataset):
    def __init__(
        self,
        root,
        name,
        num_classes,
        transform=None,
        pre_transform: BaseTransform = None,
        view=0,
    ):
        self.view: int = view
        self.name = name.upper()
        self.n_classes = num_classes
        self.filename_postfix = (
            str(pre_transform) if pre_transform is not None else None
        )
        print(self.name)
        assert self.name in ["PPMI", "PRIVATE"]
        super(BrainDataset, self).__init__(root, transform, pre_transform)
        self.data, self.slices, self.num_nodes = torch.load(self.processed_paths[0])
        logging.info("Loaded dataset: {}".format(self.name))

    @property
    def raw_dir(self):
        return self.root

    @property
    def processed_dir(self):
        return osp.join(self.root, "processed")

    @property
    def raw_file_names(self):
        return f"{self.name}.mat"

    @property
    def processed_file_names(self):
        name = f"{self.name}_{self.view}_{self.n_classes}"
        if self.filename_postfix is not None:
            name += f"_{self.filename_postfix}"
        return f"{name}.pt"

    def _download(self):
        if files_exist(self.raw_paths) or self.name in ["PRIVATE"]:  # pragma: no cover
            return

        makedirs(self.raw_dir)
        self.download()

    def download(self):
        raise NotImplementedError

    def process(self):
        if self.name == "PRIVATE":
            adj, y, p_IDs, s_IDs = load_data_private(self.raw_dir, self.n_classes)
            y = torch.LongTensor(y)
            adj = torch.Tensor(np.array(adj))
            p_IDs = torch.Tensor(p_IDs)
            s_IDs = torch.Tensor(s_IDs)
            num_graphs = adj.shape[0]
            num_nodes = adj.shape[1]
        else:
            m = loadmat(osp.join(self.raw_dir, self.raw_file_names))
            if self.name == "PPMI":
                if self.view > 2 or self.view < 0:
                    raise ValueError(f"{self.name} only has 3 views")
                raw_data = m["X"]
                num_graphs = raw_data.shape[0]
                num_nodes = raw_data[0][0].shape[0]
                a = np.zeros((num_graphs, num_nodes, num_nodes))
                for i, sample in enumerate(raw_data):
                    a[i, :, :] = sample[0][:, :, self.view]
                adj = torch.Tensor(a)
            else:
                key = "fmri" if self.view == 1 else "dti"
                adj = torch.Tensor(m[key]).transpose(0, 2)
                num_graphs = adj.shape[0]
                num_nodes = adj.shape[1]

            y = torch.Tensor(m["label"]).long().flatten()
            y[y == -1] = 0

        data_list = []
        for i in range(num_graphs):
            edge_index, edge_attr = dense_to_ind_val(adj[i])
            data = Data(
                x=adj[i],
                num_nodes=num_nodes,
                y=y[i],
                edge_index=edge_index,
                edge_attr=edge_attr,
                p_ID=p_IDs[i],
                s_ID=s_IDs[i],
            )
            data_list.append(data)

        if self.pre_filter is not None:
            data_list = [data for data in data_list if self.pre_filter(data)]

        if self.pre_transform is not None:
            data_list = [self.pre_transform(data) for data in data_list]

        data, slices = self.collate(data_list)
        torch.save((data, slices, num_nodes), self.processed_paths[0])

    def _process(self):
        print("Processing...", file=sys.stderr)

        if files_exist(self.processed_paths):  # pragma: no cover
            print("File exists...Done!", file=sys.stderr)
            return

        makedirs(self.processed_dir)
        self.process()

        print("Done!", file=sys.stderr)

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}{self.name}()"

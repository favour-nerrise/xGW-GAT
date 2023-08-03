import torch
import torch.nn.functional as F
from torch import nn
from torch_geometric.nn import (GATv2Conv, Sequential, global_add_pool,
                                global_mean_pool)
from torch_geometric.utils import softmax

from src.utils.attention import save_attn_scores


class MPGATConv(GATv2Conv):
    """
    Adapted from BrainGB
    <https://github.com/HennyJie/BrainGB/>
    """

    def __init__(
        self,
        in_channels,
        out_channels,
        heads,
        dropout=0.0,
        edge_dim=1,
        gat_mp_type: str = "attention_weighted",
    ):
        super().__init__(in_channels, out_channels, heads)
        self.dropout = dropout
        self.gat_mp_type = gat_mp_type
        self.edge_dim = edge_dim

        if edge_dim is not None:
            self.lin_edge = torch.nn.Linear(edge_dim, heads * out_channels, bias=False)

    def message(self, x_j, x_i, edge_attr, index, ptr, size_i):
        x = x_i + x_j

        if edge_attr is not None:
            if edge_attr.dim() == 1:
                edge_attr = edge_attr.view(-1, 1)
            assert self.lin_edge is not None
            edge_attr = self.lin_edge(edge_attr)
            edge_attr = edge_attr.view(-1, self.heads, self.out_channels)
            x = x + edge_attr

        x = F.leaky_relu(x, self.negative_slope)
        alpha = (x * self.att).sum(dim=-1)
        alpha = softmax(alpha, index, ptr, size_i)
        self._alpha = alpha
        alpha = F.dropout(alpha, p=self.dropout, training=self.training)

        attention_score = alpha.unsqueeze(-1)
        edge_weights = torch.abs(edge_attr.view(-1, 1).unsqueeze(-1))

        if self.gat_mp_type == "attention_weighted":
            # (1) att: s^(l+1) = s^l * alpha
            msg = x_j * attention_score
            return msg
        elif self.gat_mp_type == "attention_edge_weighted":
            # (2) e-att: s^(l+1) = s^l * alpha * e
            msg = x_j * attention_score * edge_weights
            return msg
        elif self.gat_mp_type == "sum_attention_edge":
            # (3) m-att-1: s^(l+1) = s^l * (alpha + e), this one may not make sense cause it doesn't used attention score to control all
            msg = x_j * (attention_score + edge_weights)
            return msg
        elif self.gat_mp_type == "edge_node_concat":
            # (4) m-att-2: s^(l+1) = linear(concat(s^l, e) * alpha)
            msg = torch.cat(
                [
                    x_i,
                    x_j * attention_score,
                    edge_attr.view(-1, 1).unsqueeze(-1).expand(-1, self.heads, -1),
                ],
                dim=-1,
            )
            msg = self.lin_edge(msg)
            return msg
        elif self.gat_mp_type == "node_concat":
            # (4) m-att-2: s^(l+1) = linear(concat(s^l, e) * alpha)
            msg = torch.cat([x_i, x_j * attention_score], dim=-1)
            msg = self.lin_edge(msg)
            return msg
        elif self.gat_mp_type == "sum_node_edge_weighted":
            # (5) m-att-3: s^(l+1) = (s^l + e) * alpha
            node_emb_dim = x_j.shape[-1]
            extended_edge = torch.cat([edge_weights] * node_emb_dim, dim=-1)
            sum_node_edge = x_j + extended_edge
            msg = sum_node_edge * attention_score
            return msg
        else:
            raise ValueError(f"Invalid message passing variant {self.gat_mp_type}")


class GATv2(nn.Module):
    """
    The graph attentional operator from the "Graph Attention Networks"
    <https://arxiv.org/abs/1710.10903>
    """

    def __init__(self, in_channels, args):
        super().__init__()
        self.activation = torch.nn.ReLU()
        self.convs = torch.nn.ModuleList()

        self.dropout = args.dropout
        self.edge_dim = args.edge_emb_dim
        self.explain = args.explain
        self.hidden_dim = args.hidden_dim
        self.gat_mp_type = args.gat_mp_type
        self.pooling = args.pooling
        self.num_classes = args.num_classes
        self.num_heads = args.num_heads
        self.num_layers = args.n_GNN_layers
        self.num_nodes = args.num_nodes

        gat_input_dim = in_channels

        for i in range(self.num_layers - 1):
            conv = Sequential(
                "x, edge_index, edge_attr",
                [
                    (
                        MPGATConv(
                            in_channels,
                            self.hidden_dim,
                            self.num_heads,
                            dropout=self.dropout,
                            gat_mp_type=self.gat_mp_type,
                        ),
                        "x, edge_index, edge_attr -> x",
                    ),
                    nn.Linear(self.hidden_dim * self.num_heads, self.hidden_dim),
                    nn.LeakyReLU(negative_slope=0.2),
                    nn.BatchNorm1d(self.hidden_dim),
                ],
            )
            gat_input_dim = self.hidden_dim
            self.convs.append(conv)

        in_channels = 0

        if self.pooling == "concat":
            node_dim = self.hidden_dim
            conv = Sequential(
                "x, edge_index, edge_attr",
                [
                    (
                        MPGATConv(
                            in_channels,
                            self.hidden_dim,
                            self.num_heads,
                            dropout=self.dropout,
                            gat_mp_type=self.gat_mp_type,
                        ),
                        "x, edge_index, edge_attr -> x",
                    ),
                    nn.Linear(self.hidden_dim * self.num_heads, 64),
                    nn.LeakyReLU(negative_slope=0.2),
                    nn.Linear(64, node_dim),
                    nn.LeakyReLU(negative_slope=0.2),
                    nn.BatchNorm1d(node_dim),
                ],
            )
            in_channels = node_dim * self.num_nodes

        elif self.pooling == "sum" or self.pooling == "mean":
            node_dim = self.hidden_dim
            in_channels = node_dim
            conv = Sequential(
                "x, edge_index, edge_attr",
                [
                    (
                        MPGATConv(
                            in_channels,
                            self.hidden_dim,
                            self.num_heads,
                            dropout=self.dropout,
                            gat_mp_type=self.gat_mp_type,
                        ),
                        "x, edge_index, edge_attr -> x",
                    ),
                    nn.Linear(self.hidden_dim * self.num_heads, self.hidden_dim),
                    nn.LeakyReLU(negative_slope=0.2),
                    nn.BatchNorm1d(node_dim),
                ],
            )

        self.convs.append(conv)

        self.fcn = nn.Sequential(
            nn.Linear(in_channels, 256),
            nn.LeakyReLU(negative_slope=0.2),
            nn.Linear(256, 32),
            nn.LeakyReLU(negative_slope=0.2),
            nn.Linear(32, self.num_classes),
        )

    def forward(self, data, edge_index, edge_attr, batch):
        x = data.x
        edge_attr = torch.abs(edge_attr)

        for i, conv in enumerate(self.convs):
            # bz*nodes, hidden
            if self.explain and (
                data.num_nodes == self.num_nodes
            ):  # save attention only when explaining
                if i == self.num_layers - 1:
                    x, attn = conv(
                        x, edge_index, edge_attr, return_attention_weights=True
                    )
                    save_attn_scores(
                        attn, data
                    )  # attn = (edge_index, alpha coefficients)
                else:
                    x = conv(x, edge_index, edge_attr)
            # else:
            x = conv(x, edge_index, edge_attr)

        if self.pooling == "concat":
            x = x.reshape((x.shape[0] // self.num_nodes, -1))
        elif self.pooling == "sum":
            x = global_add_pool(x, batch)  # [N, F]
        elif self.pooling == "mean":
            x = global_mean_pool(x, batch)  # [N, F]

        out = self.fcn(x)
        return out

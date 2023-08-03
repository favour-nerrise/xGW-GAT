import torch
from torch.nn import functional as F


class BrainGNN(torch.nn.Module):
    """Adapted from https://github.com/HennyJie/BrainGB"""

    def __init__(self, gnn, mlp, args, discriminator=lambda x, y: x @ y.t()):
        super(BrainGNN, self).__init__()
        self.gnn = gnn
        self.mlp = mlp
        self.pooling = args.pooling
        self.discriminator = discriminator

    def forward(self, data, edge_index=None, edge_attr=None, batch=None):
        if edge_index is None:
            x, edge_index, edge_attr, batch = (
                data.x,
                data.edge_index,
                data.edge_attr,
                data.batch,
            )
        else:
            x = data.x
        g = self.gnn(data, edge_index, edge_attr, batch)

        if self.pooling == "concat":
            _, g = self.mlp(g)

        log_logits = F.log_softmax(g, dim=-1)

        return log_logits

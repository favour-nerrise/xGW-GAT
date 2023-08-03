import torch

from src.models.BrainGNN import BrainGNN
from src.models.GATv2 import GATv2
from src.models.GCN import GCN
from src.models.MLP import MLP


def build_model(args, num_features):
    """Build a classification model, e.g. GATv2, GCN, MLP

    Args:
        args (_type_): _description_
        num_features (_type_): _description_

    Raises:
        ValueError: if model not found

    Returns:
        nn.module: pyGmodel
    """
    if args.model_name == "gatv2":
        model = BrainGNN(
            GATv2(num_features, args),
            MLP(
                args.num_classes,
                args.hidden_dim,
                args.n_MLP_layers,
                torch.nn.ReLU,
                n_classes=args.num_classes,
            ),
            args,
        )
    elif args.model_name == "gcn":
        model = BrainGNN(
            GCN(num_features, args),
            MLP(
                args.num_classes,
                args.hidden_dim,
                args.n_MLP_layers,
                torch.nn.ReLU,
                n_classes=args.num_classes,
            ),
            args,
        )
    else:
        raise ValueError(f'ERROR: Model name "{args.model_name}" not found!')
    return model

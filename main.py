import argparse
import logging
import os
import os.path as osp
import pickle
import sys
from datetime import datetime
from typing import List

import nni
import numpy as np
import torch
from imblearn.over_sampling import RandomOverSampler
from sklearn.model_selection import KFold
from sklearn.utils import class_weight
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch_geometric.loader import DataLoader
from torch_geometric.logging import init_wandb

from src.dataset import BrainDataset
from src.utils.data_utils import create_features, get_x, get_y
from src.utils.explain import explain
from src.utils.get_transform import get_transform
from src.utils.model_utils import build_model
from src.utils.modified_args import ModifiedArgs
from src.utils.sample_selection import select_samples_per_class
from src.utils.save_model import save_model
from src.utils.train_and_evaluate import test, train_eval

logging.basicConfig()
logger = logging.getLogger()
logger.setLevel(logging.INFO)


class xGW_GAT:
    def main(self):
        parser = argparse.ArgumentParser()
        parser.add_argument("--dataset", type=str, default="PRIVATE")
        parser.add_argument(
            "--model_name", type=str, default="gatv2", choices=["gcn", "gatv2"]
        )
        parser.add_argument("--num_classes", type=int, default=4)
        parser.add_argument(
            "--node_features",
            type=str,
            default="adj",
            choices=[
                "identity",
                "degree",
                "degree_bin",
                "LDP",
                "node2vec",
                "adj",
                "diff_matrix",
                "eigenvector",
                "eigen_norm",
            ],
        )
        parser.add_argument(
            "--centrality_measure",
            type=str,
            default="node",
            choices=[
                "abs",
                "geo",
                "tan",
                "node",
                "eigen",
                "close",
                "concat_orig",
                "concat_scale",
            ],
            help="Chooses the topological measure to be used",
        )
        parser.add_argument("--epochs", type=int, default=200)
        parser.add_argument("--lr", type=float, default=3e-4)
        parser.add_argument("--weight_decay", type=float, default=2e-2)
        parser.add_argument(
            "--gcn_mp_type",
            type=str,
            default="node_concat",
            choices=[
                "weighted_sum",
                "bin_concat",
                "edge_weight_concat",
                "edge_node_concat",
                "node_concat",
            ],
        )
        parser.add_argument(
            "--gat_mp_type",
            type=str,
            default="attention_weighted",
            choices=[
                "attention_weighted",
                "attention_edge_weighted",
                "sum_attention_edge",
                "edge_node_concat",
                "node_concat",
            ],
        )
        parser.add_argument(
            "--pooling", type=str, choices=["sum", "concat", "mean"], default="concat"
        )
        parser.add_argument("--n_GNN_layers", type=int, default=2)
        parser.add_argument("--n_MLP_layers", type=int, default=2)
        parser.add_argument("--num_heads", type=int, default=4)
        parser.add_argument("--hidden_dim", type=int, default=8)
        parser.add_argument("--edge_emb_dim", type=int, default=1)
        parser.add_argument("--bucket_sz", type=float, default=0.05)
        parser.add_argument("--dropout", type=float, default=0.1)
        parser.add_argument("--repeat", type=int, default=1)
        parser.add_argument("--k_fold_splits", type=int, default=4)
        parser.add_argument("--k_list", type=list, default=[4])
        parser.add_argument("--n_select_splits", type=int, default=2)
        parser.add_argument("--test_interval", type=int, default=10)
        parser.add_argument("--train_batch_size", type=int, default=2)
        parser.add_argument("--test_batch_size", type=int, default=1)
        parser.add_argument("--seed", type=int, default=112078)
        parser.add_argument("--diff", type=float, default=0.2)
        parser.add_argument("--mixup", type=int, default=1, choices=[0, 1])
        parser.add_argument("--sample_selection", action="store_true")
        parser.add_argument("--enable_nni", action="store_true")
        parser.add_argument("--explain", action="store_true")
        parser.add_argument("--wandb", action="store_true", help="Track experiment")
        parser.add_argument("--log_result", action="store_true")
        parser.add_argument("--data_folder", type=str, default="datasets/")
        args = parser.parse_args()

        self_dir = os.path.dirname(os.path.realpath(__file__))
        root_dir = osp.join(self_dir, args.data_folder)
        dataset = BrainDataset(
            root=root_dir,
            name=args.dataset,
            pre_transform=get_transform(args.node_features),
            num_classes=args.num_classes,
        )
        args.num_nodes = dataset.num_nodes
        args.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        init_wandb(
            name=f"{args.model_name}-{args.dataset}",
            heads=args.num_heads,
            epochs=args.epochs,
            hidden_channels=args.hidden_dim,
            node_features=args.node_features,
            lr=args.lr,
            weight_decay=args.weight_decay,
            num_classes=args.num_classes,
            device=args.device,
        )

        if args.enable_nni:
            args = ModifiedArgs(args, nni.get_next_parameter())

        # init model
        model_name = str(args.model_name).lower()
        args.model_name = model_name

        y = get_y(dataset)
        connectomes = get_x(dataset).T

        class_weights = class_weight.compute_class_weight(
            class_weight="balanced", classes=np.unique(y), y=y
        )
        class_weights = torch.tensor(class_weights, dtype=torch.float).to(args.device)

        test_accs, test_aucs, preds_all, labels_all = (
            [],
            [],
            [],
            [],
        )

        if args.sample_selection:
            # Check if node centrality features and subject labels exist
            if os.path.exists(
                f"{args.data_folder}data_dict_{args.node_features}_{args.num_classes}.pkl"
            ):
                with open(
                    f"{args.data_folder}data_dict_{args.node_features}_{args.num_classes}.pkl",
                    "rb",
                ) as d_d:
                    data_dict = pickle.load(d_d)
                with open(
                    f"{args.data_folder}score_dict_{args.node_features}_{args.num_classes}.pkl",
                    "rb",
                ) as s_d:
                    score_dict = pickle.load(s_d)
            else:  # Create node centrality features and subject labels
                data_dict, score_dict = create_features(
                    connectomes.numpy(), y, args, args.centrality_measure
                )
                with open(
                    f"{args.data_folder}data_dict_{args.node_features}_{args.num_classes}.pkl",
                    "wb",
                ) as d_d:
                    pickle.dump(data_dict, d_d)
                with open(
                    f"{args.data_folder}score_dict_{args.node_features}_{args.num_classes}.pkl",
                    "wb",
                ) as s_d:
                    pickle.dump(score_dict, s_d)

        fold = -1

        for train_idx, test_idx in KFold(
            args.k_fold_splits,
            shuffle=True,
            random_state=args.seed,
        ).split(dataset):
            fold += 1
            print(f"Cross Validation Fold {fold+1}/{args.k_fold_splits}")

            if args.sample_selection:
                # Select top-k subjects with highest predictive power for labels
                sample_atlas = select_samples_per_class(
                    train_idx,
                    args.n_select_splits,
                    args.k_list,
                    data_dict,
                    score_dict,
                    y,
                    shuffle=True,
                    rs=args.seed,
                )

            for k in args.k_list:
                if args.sample_selection:
                    selected_train_idxs = np.array(
                        [
                            sample_idx
                            for class_samples in sample_atlas.values()
                            for sample_indices in class_samples.values()
                            for sample_idx in sample_indices
                        ]
                    )
                else:
                    selected_train_idxs = np.array(train_idx)

                # Apply RandomOverSampler to balance classes
                train_res_idxs, _ = RandomOverSampler().fit_resample(
                    selected_train_idxs.reshape(-1, 1),
                    [y[i] for i in selected_train_idxs],
                )

                train_set = [dataset[i] for i in train_res_idxs.ravel()]
                test_set = [dataset[i] for i in test_idx]
                train_loader = DataLoader(
                    train_set, batch_size=args.train_batch_size, shuffle=True
                )
                test_loader = DataLoader(
                    test_set, batch_size=args.test_batch_size, shuffle=False
                )

                model = build_model(args, dataset.num_features)
                model = model.to(args.device)
                optimizer = torch.optim.AdamW(
                    model.parameters(), lr=args.lr, weight_decay=args.weight_decay
                )
                scheduler = ReduceLROnPlateau(
                    optimizer, mode="min", factor=0.5, patience=5, verbose=True
                )

                train_acc, train_auc, train_model = train_eval(
                    model,
                    optimizer,
                    scheduler,
                    class_weights,
                    args,
                    train_loader,
                    test_loader,
                )

                save_model(
                    args.epochs, train_model, optimizer, args
                )  # save trained model

                # test the best epoch saved model
                best_model_cp = torch.load(
                    f"model_checkpoints/best_model_{args.model_name}_{args.num_classes}.pth"
                )
                model.load_state_dict(best_model_cp["model_state_dict"])

                test_acc, test_auc, t_preds, t_labels = test(model, test_loader, args)

                logging.info(
                    f"(Performance Last Epoch) | test_acc={(test_acc * 100):.2f}, "
                    + f"test_auc={(test_auc * 100):.2f}"
                )
                test_accs.append(test_acc)
                test_aucs.append(test_auc)
                preds_all.append(t_preds)
                labels_all.append(t_labels)

                if args.explain:
                    explain(model, test_loader, args)

            # Store predictions and targets
            curr_dt = str(datetime.now())
            tag = "multi" if args.num_classes > 2 else "binary"
            saved_results = {}
            saved_results["preds"] = np.array(preds_all, dtype="object")
            saved_results["labels"] = np.array(labels_all, dtype="object")
            np.savez(
                f"./outputs/results/{curr_dt}_{args.model_name}_{args.node_features}_{tag}",
                **saved_results,
            )

            result_str = (
                f"(K Fold Final Result)| avg_acc={(np.mean(test_accs) * 100):.2f} +- {(np.std(test_accs) * 100): .2f}, "
                f"avg_auc={(np.mean(test_aucs) * 100):.2f} +- {np.std(test_aucs) * 100:.2f}\n"
            )
            logging.info(result_str)

            with open(
                f"./outputs/logs/{curr_dt}_{args.model_name}_{args.node_features}_{tag}.log",
                "a",
            ) as f:
                # Write all input arguments to f
                input_arguments: List[str] = sys.argv
                f.write(f"{input_arguments}\n")
                f.write(result_str + "\n")
            if args.enable_nni:
                nni.report_final_result(np.mean(test_aucs))


if __name__ == "__main__":
    xGW_GAT().main()

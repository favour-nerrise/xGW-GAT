import logging

import nni
import numpy as np
import torch
import torch.nn as nn
from sklearn import metrics

from src.utils.metrics import multiclass_roc_auc_score
from src.utils.save_model import SaveBestModel

# Create logger
logger = logging.getLogger("__name__")
level = logging.INFO
logger.setLevel(level)
ch = logging.StreamHandler()
ch.setLevel(level)
logger.addHandler(ch)


def train_eval(
    model, optimizer, scheduler, class_weights, args, train_loader, test_loader=None
):
    """
    Train model
    """
    model.train()
    save_best_model = SaveBestModel()  # initialize SaveBestModel class
    criterion = nn.NLLLoss(weight=class_weights)

    train_preds, train_labels, train_aucs, train_accs = [], [], [], []
    total_correct = 0
    total_samples = 0

    for i in range(args.epochs):
        running_loss = 0  # running loss for logging
        avg_train_losses = []  # average training loss per epoch

        for data in train_loader:
            data = data.to(args.device)
            optimizer.zero_grad()
            out = model(data)
            pred = out.max(dim=1)[1]  # Get predicted labels
            train_preds.append(pred.detach().cpu().tolist())
            train_labels.append(data.y.detach().cpu().tolist())
            total_correct += int((pred == data.y).sum())
            total_samples += data.y.size(0)  # Increment the total number of samples

            loss = criterion(out, data.y)
            loss.backward()
            optimizer.step()

            running_loss += float(loss.item())

        avg_train_loss = running_loss / len(
            train_loader.dataset
        )  # Correctly calculate loss per epoch
        avg_train_losses.append(avg_train_loss)

        train_acc, train_auc, _, _ = test(model, train_loader, args)

        logging.info(
            f"(Train) | Epoch={i+1:03d}/{args.epochs}, loss={avg_train_loss:.4f}, "
            + f"train_acc={(train_acc * 100):.2f}, "
            + f"train_auc={(train_auc * 100):.2f}"
        )

        if (i + 1) % args.test_interval == 0:
            test_acc, test_auc, _, _ = test(model, test_loader, args)
            text = (
                f"(Test) | Epoch {i}), test_acc={(test_acc * 100):.2f}, "
                f"test_auc={(test_auc * 100):.2f}\n"
            )
            logging.info(text)

        if args.enable_nni:
            nni.report_intermediate_result(train_auc)

        if scheduler:
            scheduler.step(avg_train_loss)

        save_best_model(avg_train_loss, i, model, optimizer, criterion, args)

    train_accs, train_aucs = np.array(train_accs), np.array(train_aucs)
    return train_accs, train_aucs, model


@torch.no_grad()
def test(model, loader, args, test_loader=None):
    """
    Test model
    """
    model.eval()

    preds = []
    # preds_prob = []
    labels = []
    test_aucs = []

    for data in loader:
        data = data.to(args.device)
        out = model(data)

        pred = out.max(dim=1)[1]
        # preds_prob.append(torch.exp(out)[:, 1].detach().cpu().tolist())
        preds.append(pred.detach().cpu().numpy().flatten())
        labels.append(data.y.detach().cpu().numpy().flatten())

    labels = np.array(labels).ravel()
    preds = np.array(preds).ravel()

    if args.num_classes > 2:
        try:
            # Compute the ROC AUC score.
            t_auc = multiclass_roc_auc_score(labels, preds)
        except ValueError as err:
            # Handle the exception.
            print(f"Warning: {err}")
            t_auc = 0.5
    else:
        t_auc = metrics.roc_auc_score(labels, preds, average="weighted")

    test_aucs.append(t_auc)

    if test_loader is not None:
        _, test_auc, preds, labels = test(model, test_loader, args)
        test_acc = np.mean(np.array(preds) == np.array(labels))

        return test_auc, test_acc
    else:
        t_acc = np.mean(np.array(preds) == np.array(labels))
        return t_acc, t_auc, preds, labels

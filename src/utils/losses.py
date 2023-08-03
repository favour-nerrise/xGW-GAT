import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class BinaryFocalLoss(nn.Module):
    """
    One possible pytorch implementation of focal loss (https://arxiv.org/abs/1708.02002) for binary classification. The
    idea is that the binary cross entropy is weighted by one minus the estimated probability, so that more confident
    predictions are given less weight in the loss function
    with_logits controls whether the binary cross-entropy is computed with F.binary_cross_entropy_with_logits or
    F.binary_cross_entropy (whether logits are passed in or probabilities)
    reduce = 'none', 'mean', 'sum'
    """

    def __init__(self, alpha=1, gamma=1, with_logits=False, reduce="mean"):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.with_logits = with_logits
        self.reduce = reduce

    def forward(self, inputs, targets):
        if self.with_logits:
            BCE_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduce=False)
        else:
            BCE_loss = F.binary_cross_entropy(inputs, targets, reduce=False)
        pt = torch.exp(-BCE_loss)
        focal_loss = self.alpha * (1 - pt) ** self.gamma * BCE_loss

        if self.reduce == "mean":
            return torch.mean(focal_loss)
        elif self.reduce == "sum":
            return torch.sum(focal_loss)
        else:
            return focal_loss


class FocalLoss(nn.Module):
    """
    One possible pytorch implementation of focal loss (https://arxiv.org/abs/1708.02002), for multiclass classification.
    This module is intended to be easily swappable with nn.CrossEntropyLoss.
    If with_logits is true, then input is expected to be a tensor of raw logits, and a softmax is applied
    If with_logits is false, then input is expected to be a tensor of probabiltiies
    target is expected to be a batch of integer targets (i.e. sparse, not one-hot). This is the same behavior as
    nn.CrossEntropyLoss.
    This loss also ignores contributions where target == ignore_index, in the same way as nn.CrossEntropyLoss
    batch behaviour: reduction = 'none', 'mean', 'sum'
    """

    def __init__(
        self,
        gamma=1,
        eps=1e-7,
        with_logits=True,
        ignore_index=-100,
        reduction="mean",
        smooth_eps=None,
    ):
        super().__init__()

        assert reduction in [
            "none",
            "mean",
            "sum",
        ], "FocalLoss: reduction must be one of ['none', 'mean', 'sum']"

        self.gamma = gamma
        self.eps = eps
        self.with_logits = with_logits
        self.ignore_index = ignore_index
        self.reduction = reduction
        self.smooth_eps = smooth_eps

    def forward(self, input, target):
        return focal_loss(
            input,
            target,
            self.gamma,
            self.eps,
            self.with_logits,
            self.ignore_index,
            self.reduction,
            smooth_eps=self.smooth_eps,
        )


def focal_loss(
    input,
    target,
    gamma=1,
    eps=1e-7,
    with_logits=True,
    ignore_index=-100,
    reduction="mean",
    smooth_eps=None,
):
    """
    A function version of focal loss, meant to be easily swappable with F.cross_entropy. The equation implemented here
    is L_{focal} = - \sum (1 - p_{target})^\gamma p_{target} \log p_{pred}
    If with_logits is true, then input is expected to be a tensor of raw logits, and a softmax is applied
    If with_logits is false, then input is expected to be a tensor of probabiltiies
    target is expected to be a batch of integer targets (i.e. sparse, not one-hot). This is the same behavior as
    nn.CrossEntropyLoss.
    Loss is ignored at indices where the target is equal to ignore_index
    batch behaviour: reduction = 'none', 'mean', 'sum'
    """
    smooth_eps = smooth_eps or 0

    # make target
    y = F.one_hot(target, input.size(-1))

    # apply label smoothing according to target = [eps/K, eps/K, ..., (1-eps) + eps/K, eps/K, eps/K, ...]
    if smooth_eps > 0:
        y = y * (1 - smooth_eps) + smooth_eps / y.size(-1)

    if with_logits:
        pt = F.softmax(input, dim=-1)
    else:
        pt = input

    pt = pt.clamp(
        eps, 1.0 - eps
    )  # a hack-y way to prevent taking the log of a zero, because we might be dealing with
    # probabilities directly.

    loss = -y * torch.log(pt)  # cross entropy
    loss *= (1 - pt) ** gamma  # focal loss factor
    loss = torch.sum(loss, dim=-1)

    # mask the logits so that values at indices which are equal to ignore_index are ignored
    loss = loss[target != ignore_index]

    # batch reduction
    if reduction == "mean":
        return torch.mean(loss, dim=-1)
    elif reduction == "sum":
        return torch.sum(loss, dim=-1)
    else:  # 'none'
        return loss


def _similarity(z1: torch.Tensor, z2: torch.Tensor):
    z1 = F.normalize(z1)
    z2 = F.normalize(z2)
    return z1 @ z2.t()


def nt_xent_loss(
    z1: torch.Tensor, z2: torch.Tensor, batch_size: int, temperature: float
):
    # Space complexity: O(BN) (semi_loss: O(N^2))
    device = z1.device
    num_nodes = z1.size(0)
    num_batches = (num_nodes - 1) // batch_size + 1
    f = lambda x: torch.exp(x / temperature)
    indices = torch.arange(0, num_nodes).to(device)
    losses = []

    for i in range(num_batches):
        batch_mask = indices[i * batch_size : (i + 1) * batch_size]
        intra_similarity = f(_similarity(z1[batch_mask], z1))  # [B, N]
        inter_similarity = f(_similarity(z1[batch_mask], z2))  # [B, N]

        positives = inter_similarity[:, batch_mask].diag()
        negatives = (
            intra_similarity.sum(dim=1)
            + inter_similarity.sum(dim=1)
            - intra_similarity[:, batch_mask].diag()
        )

        losses.append(-torch.log(positives / negatives))

    return torch.cat(losses)


def debiased_nt_xent_loss(
    z1: torch.Tensor, z2: torch.Tensor, tau: float, tau_plus: float
):
    f = lambda x: torch.exp(x / tau)
    intra_similarity = f(_similarity(z1, z1))
    inter_similarity = f(_similarity(z1, z2))

    pos = inter_similarity.diag()
    neg = (
        intra_similarity.sum(dim=1)
        - intra_similarity.diag()
        + inter_similarity.sum(dim=1)
        - inter_similarity.diag()
    )

    num_neg = z1.size()[0] * 2 - 2
    ng = (-num_neg * tau_plus * pos + neg) / (1 - tau_plus)
    ng = torch.clamp(ng, min=num_neg * np.e ** (-1.0 / tau))

    return -torch.log(pos / (pos + ng))


def hardness_nt_xent_loss(
    z1: torch.Tensor, z2: torch.Tensor, tau: float, tau_plus: float, beta: float
):
    f = lambda x: torch.exp(x / tau)
    intra_similarity = f(_similarity(z1, z1))
    inter_similarity = f(_similarity(z1, z2))

    pos = inter_similarity.diag()
    neg = (
        intra_similarity.sum(dim=1)
        - intra_similarity.diag()
        + inter_similarity.sum(dim=1)
        - inter_similarity.diag()
    )

    num_neg = z1.size()[0] * 2 - 2
    imp = (beta * neg.log()).exp()
    reweight_neg = (imp * neg) / neg.mean()
    ng = (-num_neg * tau_plus * pos + reweight_neg) / (1 - tau_plus)
    ng = torch.clamp(ng, min=num_neg * np.e ** (-1.0 / tau))

    return -torch.log(pos / (pos + ng))


def jsd_loss(z1, z2, discriminator, pos_mask, neg_mask=None):
    if neg_mask is None:
        neg_mask = 1 - pos_mask
    num_neg = neg_mask.int().sum()
    num_pos = pos_mask.int().sum()
    similarity = discriminator(z1, z2)

    E_pos = (np.log(2) - F.softplus(-similarity * pos_mask)).sum()
    E_pos /= num_pos
    neg_similarity = similarity * neg_mask
    E_neg = (F.softplus(-neg_similarity) + neg_similarity - np.log(2)).sum()
    E_neg /= num_neg

    return E_neg - E_pos

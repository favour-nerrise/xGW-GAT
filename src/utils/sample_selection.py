"""
Sample selection function.
"""
from collections import defaultdict

import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import KFold


def get_class_id(idx, class_labels):
    """
    Get the class ID for a given sample index.

    Parameters:
        idx (int): The index of the sample for which to get the class ID.
        class_labels (list or array): A list or array containing the class labels of each sample.

    Returns:
        class_id: The class ID of the sample at the specified index 'idx'.
    """
    return class_labels[idx]


def select_samples(
    train_idx, n_splits, k_list, data_dict, score_dict, y=None, shuffle=True, rs=None
):
    """Using the provided data and score dictionaries,
    selects the most important samples
    """
    freq_dict = {k: defaultdict(int) for k in k_list}
    for fold, (train_ids, holdout_ids) in enumerate(
        KFold(n_splits=n_splits, shuffle=shuffle, random_state=rs).split(train_idx)
    ):
        residuals = []
        fiq_errors = []

        for i in range(len(train_ids)):
            for j in range(i + 1, len(train_ids)):
                distance = data_dict[train_idx[train_ids[i]], train_idx[train_ids[j]]]
                fiq_error = score_dict[train_idx[train_ids[i]], train_idx[train_ids[j]]]
                residuals.append(distance)
                fiq_errors.append(fiq_error)

        if np.array(residuals[0]).shape == ():
            residuals = np.array(residuals).reshape(-1, 1)

        selectionNet = LinearRegression()
        selectionNet.fit(residuals, fiq_errors)

        for i in range(holdout_ids.shape[0]):
            R_tst = []
            for j in range(train_ids.shape[0]):
                distance = data_dict[train_idx[holdout_ids[i]], train_idx[train_ids[j]]]
                R_tst.append(distance)
            R_tst = np.stack(R_tst)
            if np.array(R_tst[0]).shape == ():
                R_tst = np.array(R_tst).reshape(-1, 1)
            error_pred = selectionNet.predict(R_tst).ravel()
            for k in k_list:
                for id in train_idx[train_ids[np.argsort(error_pred)[:k]]]:
                    freq_dict[k][id] += 1
    important_samples = {k: [] for k in k_list}
    for k, fd in freq_dict.items():
        ids, freqs = np.array(list(fd.keys())), np.array(list(fd.values()))
        important_samples[k] = ids[np.argsort(freqs)[::-1][:k]]
    return important_samples


from collections import defaultdict


def select_samples_per_class(
    train_idx,
    n_splits,
    k_list,
    data_dict,
    score_dict,
    class_labels,
    shuffle=True,
    rs=None,
):
    """Using the provided data and score dictionaries,
    selects the most important samples per class
    """
    class_dict = defaultdict(lambda: defaultdict(list))
    for idx in train_idx:
        class_id = get_class_id(idx, class_labels)  # Get the class ID of a sample
        class_dict[class_id]["samples"].append(idx)

    important_samples_per_class = defaultdict(lambda: {k: [] for k in k_list})

    for class_id, class_info in class_dict.items():
        class_samples = class_info["samples"]
        n_samples = len(class_samples)

        if n_splits > n_samples:
            n_splits = n_samples

        freq_dict = {k: defaultdict(int) for k in k_list}

        for fold, (train_ids, holdout_ids) in enumerate(
            KFold(n_splits=n_splits, shuffle=shuffle, random_state=rs).split(
                class_samples
            )
        ):
            residuals = []
            score_errors = []

            for i, train_id_i in enumerate(train_ids):
                if (
                    len(train_ids) == 1
                ):  # when only 1 sample per training fold due to few class samples
                    residuals.append(
                        data_dict[
                            class_samples[train_id_i], class_samples[train_id_i] + 1
                        ]
                    )
                    score_errors.append(
                        score_dict[
                            class_samples[train_id_i], class_samples[train_id_i] + 1
                        ]
                    )
                    continue

                for j, train_id_j in enumerate(train_ids[i + 1 :], start=i + 1):
                    distance = data_dict[
                        class_samples[train_id_i], class_samples[train_id_j]
                    ]
                    score_error = score_dict[
                        class_samples[train_id_i], class_samples[train_id_j]
                    ]
                    residuals.append(distance)
                    score_errors.append(score_error)

            if np.array(residuals[0]).shape == ():
                residuals = np.array(residuals).reshape(-1, 1)

            selectionNet = LinearRegression()
            selectionNet.fit(residuals, score_errors)

            for i in range(holdout_ids.shape[0]):
                R_tst = []
                for j in range(train_ids.shape[0]):
                    distance = data_dict[
                        class_samples[holdout_ids[i]], class_samples[train_ids[j]]
                    ]
                    R_tst.append(distance)
                R_tst = np.stack(R_tst)
                if np.array(R_tst[0]).shape == ():
                    R_tst = np.array(R_tst).reshape(-1, 1)
                error_pred = selectionNet.predict(R_tst).ravel()

                for k in k_list:
                    indices_sorted = np.argsort(error_pred)[:k]
                    ids_selected = [
                        class_samples[train_ids[idx]] for idx in indices_sorted
                    ]
                    for id in ids_selected:
                        freq_dict[k][id] += 1

        for k, fd in freq_dict.items():
            ids, freqs = np.array(list(fd.keys())), np.array(list(fd.values()))
            indices_sorted = np.argsort(freqs)[::-1][
                :k
            ]  # Sort in descending order to get top k
            important_samples_per_class[class_id][k] = ids[indices_sorted]

    return important_samples_per_class

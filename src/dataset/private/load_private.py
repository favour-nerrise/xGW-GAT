import numpy as np
import pandas as pd
from sklearn import preprocessing


def load_data_private(root_dir, num_classes):
    if num_classes == 2:
        data = np.load(root_dir + "/private_binary.npy", allow_pickle=True)
    elif num_classes == 4:
        data = np.load(root_dir + "/private_multi.npy", allow_pickle=True)
    else:
        raise Exception(
            f"Invalid number of number of classes, "
            + "expected n_classes=2 or n_classes=4 but got {num_classes}."
        )
    final_pearson = data["corr"]
    final_pearson = [x.astype(np.float32) for x in final_pearson]
    final_pearson = np.stack(
        final_pearson, axis=0
    )  # reshape to (n_subjects, num_nodes, num_nodes)
    labels = data["labels"].astype(np.int32)
    p_ids = data["p_ids"]
    encoder = preprocessing.LabelEncoder()
    p_IDs = encoder.fit_transform(p_ids)
    s_IDs = np.linspace(0, len(p_ids), len(p_ids), False)
    return final_pearson, labels, p_IDs, s_IDs


def process_dataset(fc_data, fc_id, id2gender, id2pearson, label_df):
    final_label, final_pearson = [], []
    for fc, l in zip(fc_data, fc_id):
        if l in id2gender and l in id2pearson:
            if not np.any(np.isnan(id2pearson[l])):
                final_label.append(id2gender[l])
                final_pearson.append(id2pearson[l])
    final_pearson = np.array(final_pearson)
    encoder = preprocessing.LabelEncoder()
    encoder.fit(label_df["sex"])
    labels = encoder.transform(final_label)
    return final_pearson, labels

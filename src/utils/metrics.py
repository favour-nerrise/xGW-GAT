import numpy as np
from sklearn import metrics
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import LabelBinarizer


def multiclass_roc_auc_score(y_test, y_pred, average="weighted"):
    lb = LabelBinarizer()
    lb.fit(y_test)

    y_test = lb.transform(y_test)
    y_pred = lb.transform(y_pred)

    return roc_auc_score(y_test, y_pred, average=average)


def roc_auc_score_multiclass(y_test, y_pred, average="weighted"):
    # creating a set of all the unique classes using the actual class list
    unique_class = set(y_test)
    roc_auc_dict = {}
    for per_class in unique_class:
        # creating a list of all the classes except the current class
        other_class = [x for x in unique_class if x != per_class]

        # marking the current class as 1 and all other classes as 0
        new_actual_class = [0 if x in other_class else 1 for x in y_test]
        new_pred_class = [0 if x in other_class else 1 for x in y_pred]

        # using the sklearn metrics method to calculate the roc_auc_score
        try:
            roc_auc = metrics.roc_auc_score(
                new_actual_class, new_pred_class, average=average
            )
        except ValueError:
            roc_auc = 0.5
        roc_auc_dict[per_class] = roc_auc

        # avg roc auc
        avg_roc_auc = sum(roc_auc_dict.values()) / len(unique_class)
        return (avg_roc_auc, roc_auc_dict)

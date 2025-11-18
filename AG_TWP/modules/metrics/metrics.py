import numpy as np

def accuracy(y_pred, y_true):
    return round(np.sum(y_pred == y_true)*100 / y_pred.size, 2)

def tpr(y_pred, y_true, positive_label=1):
    """recall or true positive rate (TPR) is the proportion of positive cases that were correctly identified by the model."""
    return round(np.sum(y_pred[y_true == positive_label] == positive_label)*100 / np.sum(y_true == positive_label), 2)

def fpr(y_pred, y_true, positive_label=1):
    """False positive rate (FPR) is the proportion of negative cases that were incorrectly identified as positive by the model."""
    return round(np.sum(y_pred[y_true != positive_label] == positive_label)*100 / np.sum(y_true != positive_label), 2)

def precision(y_pred, y_true, positive_label=1):
    return round(np.sum(y_pred[y_true == positive_label] == positive_label)*100 / np.sum(y_pred == positive_label), 2)

def f1_score(y_pred, y_true, positive_label=1):
    """F1 score is the harmonic mean of precision and recall."""
    prec = precision(y_pred, y_true, positive_label)
    rec = tpr(y_pred, y_true, positive_label)
    return round(2 * (prec * rec) / (prec + rec), 2)

def confusion_matrix(y_pred, y_true, positive_label=1):
    tp = np.sum((y_pred == positive_label) & (y_true == positive_label))
    tn = np.sum((y_pred != positive_label) & (y_true != positive_label))
    fp = np.sum((y_pred == positive_label) & (y_true != positive_label))
    fn = np.sum((y_pred != positive_label) & (y_true == positive_label))
    return np.array([[tp, fp], [fn, tn]])


class Metrics:
    def __init__(self):
        self._metrics = {}

    def __getattr__(self, name):
        if name in self._metrics:
            return self._metrics[name]
        raise AttributeError(f"'Metrics' object has no attribute '{name}'")

    def __setattr__(self, name, value):
        if name == "_metrics":
            super().__setattr__(name, value)
        else:
            self._metrics[name] = value  
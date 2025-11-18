from modules.metrics.metrics import Metrics, accuracy, tpr


def evaluate_clf_and_get_metrics(X_test, clf, targets):
    """Function to get accuracy and TPR from a trained classifier
    it must receive the targets to compare with
    Args:
        X_test (array): _description_
        clf (object): trained classifier
        targets (array): expected targets

    Returns:
        tuple: accuracy and TPR
    """
    metrics = Metrics()
    y_pred = clf.predict(X_test)    
    metrics.acc = accuracy(y_pred, targets)
    metrics.tpr = tpr(y_pred, targets) 
    return metrics
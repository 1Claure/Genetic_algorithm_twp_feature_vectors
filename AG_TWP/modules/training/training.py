from sklearn.model_selection import GridSearchCV
from modules.metrics.metrics import Metrics, accuracy, tpr
import numpy as np


def grid_search(X, y, clf, grid):
    grid = GridSearchCV(clf, param_grid=grid, cv=3, scoring='accuracy')
    grid.fit(X,y)
    return grid.best_estimator_


def train_classifier_GS(X, y, clf, grid):
    best_clf = grid_search(X=X, y=y, clf=clf, grid=grid)
    print(best_clf)
    return best_clf

def train_clf_and_get_metrics(X, y, clf, with_gs=False, param_grid=None):
    """Function to train a classifier and obtain accuracy and TPR (True Positive Rate)
    using the training data. It has the option to choose training using grid search, 
    for which 'with_gs' should be set True and receive the parameter grid 'param_grid'.
    Args:
        X (array): data for training
        y (array): targets
        clf (object): classifier to train
        with_gs (bool, optional): Whether or not to use grid search. Defaults to False.
        param_grid (dict, optional): dictionary with parameters for grid search. Defaults to None.

    Returns:
        tuple: classifier, accuracy and TPR
    """
    metrics = Metrics()
    if with_gs:
        if param_grid == None:
            raise ValueError("param_grid must be provided if 'with_gs' is True")
        clf = train_classifier_GS(X, y, clf=clf, grid=param_grid)
    else:
        clf = clf.fit(X, y)

    y_pred = clf.predict(X)
    metrics.acc = accuracy(y_pred, y)
    metrics.tpr = tpr(y_pred, y)
    return clf, metrics



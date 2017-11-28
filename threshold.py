from sklearn.base import BaseEstimator, ClassifierMixin
import numpy as np

def choose_threshold(s, y):
    #return np.amin(s[y == 1])
    #return np.median(s)
    si = np.argsort(s)
    s = s[si]
    y = y[si]
    #sy = sorted(zip(s, y))
    #s = [x for x, _ in sy]
    #y = [x for _, x in sy]

    maxF1 = -np.inf
    bestTh = 0

    for i in range(1, len(y)):
        if y[i] != y[i-1]:
            TP = np.sum(y[i:] == 1)
            FP = np.sum(y[i:] == 0)
            FN = np.sum(y[:i] == 1)
            F1 = (2.*TP)/(2.*TP+FN+FP+1e-10)
            if F1 > maxF1:
                maxF1 = F1
                bestTh = (s[i]+s[i-1])/2.

    return bestTh


class Threshold(BaseEstimator, ClassifierMixin):
    def __init__(self, model):
        self.model = model

    def fit(self, X, y):
        self.model.fit(X, y)
        s = self.model.decision_function(X)
        self.th = choose_threshold(s, y)

    def decision_function(self, X):
        return self.model.decision_function(X)

    def predict(self, X):
        s = self.model.decision_function(X)
        return (s >= self.th).astype(int)

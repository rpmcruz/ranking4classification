from adaboost import AdaBoost, RankBoost
from neuralnet import StochasticRankNet
from ranksvm import RankSVM
from threshold import Threshold
import numpy as np

from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC
from sklearn.metrics import f1_score, roc_auc_score

X, y = load_breast_cancer(True)
K = 10

N0, N1 = np.sum(y == 0), np.sum(y == 1)
IR = min(N0, N1) / max(N0, N1)
print('* dataset: N=%d, p=%d, IR=%.2f' % (*X.shape, IR))
print()

models = [
    ('linear svc', LinearSVC(penalty='l1', tol=1e-3, dual=False)),
    ('balanced linear svc', LinearSVC(class_weight='balanced', penalty='l1', tol=1e-3, dual=False)),
    ('ranksvm', Threshold(RankSVM())),
    ('adaboost', AdaBoost(100)),
    ('rankboost', Threshold(RankBoost(100))),
    ('stochastic ranknet', Threshold(StochasticRankNet(0.7, 10, maxit=10))),
]

f1_scores = np.zeros(len(models))
roc_auc_scores = np.zeros(len(models))

for k, (tr, ts) in enumerate(StratifiedKFold(K, True).split(X, y)):
    print('* Fold %d of %d' % (k+1, K))
    for j, (name, model) in enumerate(models):
        scaler = StandardScaler()
        Xtr = scaler.fit_transform(X[tr])
        Xts = scaler.transform(X[ts])

        model.fit(Xtr, y[tr])
        yp = model.predict(Xts)

        if hasattr(model, 'predict_proba'):
            ss = model.predict_proba(Xts)
        else:
            ss = model.decision_function(Xts)

        f1 = f1_score(y[ts], yp)
        roc_auc = roc_auc_score(y[ts], ss)

        f1_scores[j] = (f1_scores[j]*k + f1)/(k+1)
        roc_auc_scores[j] = (roc_auc_scores[j]*k + roc_auc)/(k+1)

        print('%-20s | F1: %.3f avg: %.3f | ROC AUC: %.3f avg: %.3f' % (
            name, f1, f1_scores[j], roc_auc, roc_auc_scores[j]))

    print()
    print('****************************')
    print()


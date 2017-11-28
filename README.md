# Ranking 4 Classification

Class imbalance is pervasive in certain classification domains. That is, the class distribution is not uniform, in some cases in an extreme fashion. This is a common problem in medicine and health care where there is a wide dispersion of patients suffering from different disease severities; it is inherent in fraud and fault detection where the anomaly is rare; and in many other fields.

These models were produced during the development of the following paper. Please cite the paper if you use them.

* Cruz, R., Fernandes, K., Cardoso, J. S., & Costa, J. F. P. (2016, July). Tackling class imbalance with ranking. In Neural Networks (IJCNN), 2016 International Joint Conference on (pp. 2182-2187). IEEE. [[paper]](http://ieeexplore.ieee.org/abstract/document/7727469/)

These models can be used for traditional ranking problems or be used for classification by using the `Threshold` wrapper. They were only tested in some contexts, they need to be tested more throughly --- let [me know](mailto:ricardo.pdm.cruz@gmail.com) if you have problems.

Usage example:

```python
from sklearn.datasets import load_breast_cancer
X, y = load_breast_cancer(True)  # binary dataset

from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split
from adaboost import RankBoost

Xtr, Xts, ytr, yts = train_test_split(X, y)

model = Threshold(RankBoost(100))
model.fit(Xtr, ytr)
yp = model.predict(Xts)
print(f1_score(yts, yp))
```

"""Проверка логистической регрессии с нуля: сходится и совпадает с sklearn."""
import numpy as np
from sklearn.linear_model import LogisticRegression

from src.models import LogisticRegressionGD


def _data(seed=0, n=600):
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n, 3))
    logits = 2.0 * X[:, 0] - 1.0 * X[:, 1] + 0.5
    y = (rng.uniform(size=n) < 1 / (1 + np.exp(-logits))).astype(int)
    return X, y


def test_loss_decreases():
    X, y = _data()
    model = LogisticRegressionGD(n_iter=300).fit(X, y)
    assert model.loss_history_[-1] < model.loss_history_[0]


def test_agrees_with_sklearn_predictions():
    X, y = _data()
    ours = LogisticRegressionGD(lr=0.5, n_iter=3000, l2=1e-3).fit(X, y)
    theirs = LogisticRegression(C=1000).fit(X, y)
    agreement = np.mean(ours.predict(X) == theirs.predict(X))
    assert agreement > 0.97


def test_balanced_weights_raise_recall_on_rare_class():
    rng = np.random.default_rng(1)
    X = rng.normal(size=(2000, 2))
    y = (X[:, 0] + rng.normal(scale=0.7, size=2000) > 1.6).astype(int)
    plain = LogisticRegressionGD(n_iter=1500).fit(X, y)
    balanced = LogisticRegressionGD(n_iter=1500, class_weight="balanced").fit(X, y)
    recall = lambda m: np.mean(m.predict(X)[y == 1] == 1)
    assert recall(balanced) > recall(plain)

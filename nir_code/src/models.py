"""Модели: логистическая регрессия на градиентном спуске (с нуля) и базовые модели."""
import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin


def sigmoid(z):
    """Логистическая функция 1/(1+exp(-z)); z обрезается, чтобы exp не переполнялся."""
    return 1.0 / (1.0 + np.exp(-np.clip(z, -30, 30)))


class LogisticRegressionGD(BaseEstimator, ClassifierMixin):
    """Логистическая регрессия, обученная полным градиентным спуском.

    Минимизируется средний логарифмический лосс с L2-штрафом (без штрафа на сдвиг):
        J(w) = -mean( s_i * [y ln p + (1-y) ln(1-p)] ) + (l2/2) * |w|^2,
        p = sigmoid(X w + b), s_i: вес объекта.
    Градиент: dJ/dw = X^T (s * (p - y)) / sum(s) + l2 * w.
    """

    def __init__(self, lr=0.1, n_iter=2000, l2=1e-3, class_weight=None):
        self.lr = lr
        self.n_iter = n_iter
        self.l2 = l2
        self.class_weight = class_weight

    def _sample_weights(self, y):
        """Веса объектов: «balanced» уравнивает суммарный вес классов."""
        weights = np.ones(len(y))
        if self.class_weight == "balanced":
            for cls in (0, 1):
                mask = y == cls
                if mask.any():
                    weights[mask] = len(y) / (2.0 * mask.sum())
        return weights

    def fit(self, X, y):
        X, y = np.asarray(X, dtype=float), np.asarray(y, dtype=float)
        weights = self._sample_weights(y)
        total = weights.sum()
        self.coef_ = np.zeros(X.shape[1])
        self.intercept_ = 0.0
        self.loss_history_ = []
        for _ in range(self.n_iter):
            p = sigmoid(X @ self.coef_ + self.intercept_)
            error = weights * (p - y)
            self.coef_ -= self.lr * (X.T @ error / total + self.l2 * self.coef_)
            self.intercept_ -= self.lr * error.sum() / total
            self.loss_history_.append(self._loss(p, y, weights, total))
        self.classes_ = np.array([0, 1])
        return self

    def _loss(self, p, y, weights, total):
        eps = 1e-12
        log_loss = -np.sum(weights * (y * np.log(p + eps) + (1 - y) * np.log(1 - p + eps))) / total
        return log_loss + 0.5 * self.l2 * np.sum(self.coef_ ** 2)

    def predict_proba(self, X):
        p = sigmoid(np.asarray(X, dtype=float) @ self.coef_ + self.intercept_)
        return np.column_stack([1 - p, p])

    def predict(self, X):
        return (self.predict_proba(X)[:, 1] >= 0.5).astype(int)


class MajorityBaseline(BaseEstimator, ClassifierMixin):
    """Всегда предсказывает самый частый класс обучающей выборки."""

    def fit(self, X, y):
        y = np.asarray(y)
        self.majority_ = int(np.bincount(y).argmax())
        self.rate_ = float(np.mean(y))
        self.classes_ = np.array([0, 1])
        return self

    def predict_proba(self, X):
        p = np.full(len(X), float(self.majority_))
        return np.column_stack([1 - p, p])

    def predict(self, X):
        return np.full(len(X), self.majority_)


class PersistenceBaseline(BaseEstimator, ClassifierMixin):
    """Повторяет прошлый семестр: должник сейчас -> должник и в следующем."""

    def fit(self, X, y):
        self.classes_ = np.array([0, 1])
        return self

    def predict_proba(self, X):
        p = (X["n_fail"].to_numpy() > 0).astype(float)
        return np.column_stack([1 - p, p])

    def predict(self, X):
        return (X["n_fail"].to_numpy() > 0).astype(int)


class ConstantRateBaseline(BaseEstimator, ClassifierMixin):
    """Каждому объекту даёт одну и ту же вероятность долга: долю должников в обучении.

    Нужна для прогноза числа должников: сумма таких вероятностей = размер группы x доля.
    """

    def fit(self, X, y):
        self.rate_ = float(np.mean(y))
        self.classes_ = np.array([0, 1])
        return self

    def predict_proba(self, X):
        p = np.full(len(X), self.rate_)
        return np.column_stack([1 - p, p])

    def predict(self, X):
        return (self.predict_proba(X)[:, 1] >= 0.5).astype(int)

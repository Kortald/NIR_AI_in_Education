# Защита от утечки: один студент не может быть одновременно в обучении и в тесте.
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, ClassifierMixin

import config
from src.experiments import oof_probabilities, select_scenario


class SpyModel(BaseEstimator, ClassifierMixin):
    # Запоминает студентов обучения и проверяет, что в тесте их нет.
    #
    # Номер студента подставлен в признак n_disc, чтобы модель его видела.

    def fit(self, X, y):
        self.seen_ = set(X["n_disc"])
        self.classes_ = np.array([0, 1])
        return self

    def predict_proba(self, X):
        assert self.seen_.isdisjoint(set(X["n_disc"])), "студент попал и в обучение, и в тест"
        return np.full((len(X), 2), 0.5)


def _data(n_students=200, seed=0):
    rng = np.random.default_rng(seed)
    student = np.repeat(np.arange(n_students), 3)           # у каждого студента три пары
    frame = pd.DataFrame(0.0, index=range(len(student)), columns=config.FEATURES)
    frame["n_disc"] = student
    frame["n_fail"] = 0
    frame["student"] = student
    frame["y"] = (rng.uniform(size=len(student)) < 0.1).astype(int)
    return frame


def test_cross_validation_never_mixes_students():
    oof_probabilities(_data(), SpyModel(), seed=1)             # упадёт на assert при утечке


def test_threshold_tuning_never_mixes_students():
    oof_probabilities(_data(), SpyModel(), seed=2, tune=True)


def test_new_debtor_scenario_keeps_only_clean_students():
    data = _data()
    data.loc[:5, "n_fail"] = 2
    kept = select_scenario(data, "new")
    assert (kept["n_fail"] == 0).all() and len(kept) < len(data)


def test_calibration_summary_perfect_and_biased():
    from src.experiments import calibration_summary
    rng = np.random.default_rng(0)
    p = rng.uniform(0.02, 0.5, size=6000)
    y = (rng.uniform(size=p.size) < p).astype(int)
    good, table = calibration_summary(y, p)
    assert good["ECE"] < 0.03 and abs(good["наклон"] - 1) < 0.15 and len(table) == 10
    bad, _ = calibration_summary(y, np.clip(p * 0.5, 0, 1))
    assert bad["сумма_прогнозов"] < 0.6 * bad["факт"] and bad["ECE"] > good["ECE"]

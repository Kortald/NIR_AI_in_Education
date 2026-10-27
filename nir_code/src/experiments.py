# Эксперименты: кросс-валидация по студентам, проверка по времени, прогноз числа должников.
#
# Во всех схемах один студент целиком попадает либо в обучение, либо в тест.
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

import config
from src.evaluation import (classification_metrics, qa, qa_verdict, qc, qc_from_confusion,
                            qc_verdict)
from src.models import (ConstantRateBaseline, LogisticRegressionGD, MajorityBaseline,
                        PersistenceBaseline)

BASELINES = ("Константа (мажоритарный класс)", "Повтор прошлого семестра")


def _prepare(model):
    # Пропуски -> медиана обучающей части, затем стандартизация.
    return make_pipeline(SimpleImputer(strategy="median"), StandardScaler(), model)


def classification_models() -> dict:
    # Модели для классификации. Дисбаланс классов компенсируется весами.
    seed = config.SEED
    return {
        BASELINES[0]: MajorityBaseline(),
        BASELINES[1]: PersistenceBaseline(),
        "Логистическая регрессия (с нуля)": _prepare(
            LogisticRegressionGD(lr=0.3, n_iter=1500, l2=1e-3, class_weight="balanced")),
        "Логистическая регрессия (sklearn)": _prepare(
            LogisticRegression(class_weight="balanced", max_iter=2000)),
        "Случайный лес": make_pipeline(SimpleImputer(strategy="median"), RandomForestClassifier(
            n_estimators=300, min_samples_leaf=5, class_weight="balanced_subsample",
            random_state=seed, n_jobs=-1)),
        "Градиентный бустинг": HistGradientBoostingClassifier(
            max_depth=3, learning_rate=0.05, max_iter=200, class_weight="balanced",
            random_state=seed),
    }


def count_models() -> dict:
    # Модели для прогноза числа должников: без весов, чтобы вероятности не искажались.
    seed = config.SEED
    return {
        "Константная доля": ConstantRateBaseline(),
        BASELINES[1]: PersistenceBaseline(),
        "Логистическая регрессия (с нуля)": _prepare(
            LogisticRegressionGD(lr=0.3, n_iter=1500, l2=1e-3)),
        "Логистическая регрессия (sklearn)": _prepare(LogisticRegression(max_iter=2000)),
        "Градиентный бустинг": HistGradientBoostingClassifier(
            max_depth=3, learning_rate=0.05, max_iter=200, random_state=seed),
    }


def select_scenario(pairs: pd.DataFrame, scenario: str) -> pd.DataFrame:
    # «all» - все пары; «new» - только студенты без долга в семестре t (новые должники).
    if scenario == "new":
        return pairs[pairs["n_fail"] == 0].reset_index(drop=True)
    return pairs.reset_index(drop=True)


def oof_probabilities(data: pd.DataFrame, model, seed: int, n_splits: int = config.N_FOLDS,
                      tune: bool = False):
    # Прогноз для каждой пары моделью, не видевшей этого студента (кросс-валидация по студентам).
    #
    # Возвращает (вероятности долга в t+1, метки 0/1). Если tune=True, порог решения для меток
    # подбирается по Qc внутри обучающей части блока, на тест он не смотрит.
    X, y = data[config.FEATURES], data["y"].to_numpy()
    folds = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    proba, labels = np.zeros(len(data)), np.zeros(len(data), dtype=int)
    for train_idx, test_idx in folds.split(X, y, data["student"]):
        train = data.iloc[train_idx]
        fitted = clone(model).fit(X.iloc[train_idx], y[train_idx])
        proba[test_idx] = fitted.predict_proba(X.iloc[test_idx])[:, 1]
        threshold = tune_threshold(model, train, seed) if tune else 0.5
        labels[test_idx] = (proba[test_idx] >= threshold).astype(int)
    return proba, labels


def tune_threshold(model, train: pd.DataFrame, seed: int) -> float:
    # Порог, при котором Qc максимален на внутренней кросс-валидации обучающей части.
    y = train["y"].to_numpy()
    if len(np.unique(y)) < 2:
        return 0.5
    proba, _ = oof_probabilities(train, model, seed, n_splits=config.INNER_FOLDS)
    grid = np.linspace(0.05, 0.95, 91)
    scores = [qc(y, (proba >= t).astype(int)) for t in grid]
    return float(grid[int(np.argmax(scores))])


def cross_validate(pairs: pd.DataFrame, scenario: str):
    # Кросс-валидация по студентам, повторённая N_REPEATS раз.
    #
    # Возвращает (таблица средних метрик, словарь {модель: вероятности первого повтора}, данные).
    data = select_scenario(pairs, scenario)
    y = data["y"].to_numpy()
    rows, first_repeat = [], {}
    for name, model in classification_models().items():
        per_repeat = []
        for repeat in range(config.N_REPEATS):
            proba, labels = oof_probabilities(data, model, config.SEED + repeat,
                                              tune=name not in BASELINES)
            if repeat == 0:
                first_repeat[name] = (proba, labels)
            per_repeat.append(classification_metrics(y, labels, proba))
        rows.append(_summarize(name, per_repeat))
    return pd.DataFrame(rows), first_repeat, data


def _summarize(name: str, per_repeat: list) -> dict:
    # Среднее и разброс метрик по повторам.
    table = pd.DataFrame(per_repeat)
    row = {"модель": name}
    for col in ["Qc", "accuracy", "precision", "recall", "f1", "roc_auc", "pr_auc"]:
        if col in table:
            row[col] = table[col].mean()
            row[col + "_std"] = table[col].std(ddof=0)
    row["Qc_вердикт"] = qc_verdict(row["Qc"], k=2)
    return row


def temporal_check(pairs: pd.DataFrame, scenario: str) -> pd.DataFrame:
    # Обучение на парах семестров 1-2, тест на парах семестра 3 («прошлое -> будущее»).
    #
    # Студенты в обеих частях могут повторяться: признаки теста берутся из семестра 3,
    # а цель из семестра 4, которых обучение не видело.
    data = select_scenario(pairs, scenario)
    train = data[data["semester"].isin([1, 2])]
    test = data[data["semester"] == 3]
    rows = []
    for name, model in classification_models().items():
        fitted = clone(model).fit(train[config.FEATURES], train["y"].to_numpy())
        proba = fitted.predict_proba(test[config.FEATURES])[:, 1]
        threshold = 0.5 if name in BASELINES else tune_threshold(model, train, config.SEED)
        labels = (proba >= threshold).astype(int)
        metrics = classification_metrics(test["y"], labels, proba)
        rows.append({"модель": name, **metrics, "порог": threshold,
                     "n_train": len(train), "n_test": len(test)})
    return pd.DataFrame(rows)


def debtor_counts(pairs: pd.DataFrame):
    # Прогноз числа должников на следующий семестр по группам (учебный план x семестр).
    #
    # Прогноз группы = сумма вероятностей долга её студентов (кросс-валидация по студентам).
    # Возвращает (сводка метрик по моделям, таблица групп, итоги по семестрам).
    data = pairs.reset_index(drop=True)
    groups = {}
    for name, model in count_models().items():
        data[name] = oof_probabilities(data, model, config.SEED)[0]
    key = ["plan", "semester"]
    agg = data.groupby(key).agg(size=("y", "size"), actual=("y", "sum"),
                                **{n: (n, "sum") for n in count_models()}).reset_index()
    big = agg[agg["size"] >= config.MIN_GROUP_SIZE].reset_index(drop=True)
    rows = []
    for name in count_models():
        f, y = big[name].to_numpy(), big["actual"].to_numpy()
        value = qa(f, y)
        rows.append({"модель": name, "групп": len(big), "Qa": value, "Qa_вердикт": qa_verdict(value),
                     "MAE": np.mean(np.abs(f - y)), "RMSE": np.sqrt(np.mean((f - y) ** 2)),
                     "всего_факт": y.sum(), "всего_прогноз": f.sum()})
    f_const = np.full(len(big), big["actual"].mean())
    value = qa(f_const, big["actual"].to_numpy())
    rows.append({"модель": "Справочно: одно и то же число для всех групп", "групп": len(big),
                 "Qa": value, "Qa_вердикт": qa_verdict(value),
                 "MAE": np.mean(np.abs(f_const - big["actual"])),
                 "RMSE": np.sqrt(np.mean((f_const - big["actual"]) ** 2)),
                 "всего_факт": big["actual"].sum(), "всего_прогноз": f_const.sum()})
    by_semester = data.groupby("semester").agg(
        факт=("y", "sum"), **{n: (n, "sum") for n in count_models()}).round(1).reset_index()
    return pd.DataFrame(rows), big, by_semester


CALIBRATION_MODELS = ("Логистическая регрессия (с нуля)", "Логистическая регрессия (sklearn)",
                      "Градиентный бустинг")


def calibration_summary(y, proba, n_bins: int = 10):
    # Калибровка вероятностей: итоговые показатели и таблица по группам одинакового размера.
    #
    # Показатели: Brier, Brier константной доли (для сравнения), ECE (взвешенное среднее
    # |прогноз - доля долгов| по группам), отношение суммы прогнозов к факту, наклон и сдвиг
    # калибровочной прямой (логистическая регрессия метки на logit прогноза; идеал: 1 и 0).
    # Группы строятся по квантилям прогноза, поэтому в каждой около N/n_bins пар.
    y, proba = np.asarray(y, dtype=float), np.asarray(proba, dtype=float)
    clipped = np.clip(proba, 1e-6, 1 - 1e-6)
    logit = np.log(clipped / (1 - clipped)).reshape(-1, 1)
    line = LogisticRegression(C=1e6, max_iter=1000).fit(logit, y)
    bins = pd.qcut(pd.Series(proba).rank(method="first"), n_bins, labels=False)
    table = pd.DataFrame({"bin": bins, "p": proba, "y": y}).groupby("bin").agg(
        пар=("y", "size"), прогноз=("p", "mean"), доля_долгов=("y", "mean"),
        должников=("y", "sum")).reset_index(drop=True)
    ece = float(np.sum(table["пар"] * np.abs(table["прогноз"] - table["доля_долгов"])) / len(y))
    summary = {"Brier": float(np.mean((proba - y) ** 2)),
               "Brier_константа": float(np.mean((y.mean() - y) ** 2)),
               "ECE": ece, "сумма_прогнозов": float(proba.sum()), "факт": float(y.sum()),
               "наклон": float(line.coef_[0, 0]), "сдвиг": float(line.intercept_[0])}
    return summary, table


def calibration_check(pairs: pd.DataFrame):
    # Калибровка вероятностей моделей для прогноза числа должников (те же разбиения, что в
    # debtor_counts). Возвращает (сводка по моделям, таблицы по группам, прогнозы).
    data = pairs.reset_index(drop=True)
    models = count_models()
    rows, tables, probas = [], {}, {}
    for name in CALIBRATION_MODELS:
        proba = oof_probabilities(data, models[name], config.SEED)[0]
        summary, table = calibration_summary(data["y"], proba)
        rows.append({"модель": name, **summary})
        tables[name], probas[name] = table, proba
    return pd.DataFrame(rows), tables, probas


def bootstrap_qc(data: pd.DataFrame, labels: dict, n_boot: int = 1000, seed: int = config.SEED):
    # Доверительные интервалы Qc бутстрэпом по студентам (все пары одного студента вместе).
    #
    # labels: {модель: метки 0/1 первого повтора кросс-валидации}.
    # Для каждой модели также считается разность Qc с «повтором прошлого семестра».
    y = data["y"].to_numpy()
    students, inverse = np.unique(data["student"].to_numpy(), return_inverse=True)
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(students), size=(n_boot, len(students)))

    def boot_values(pred):
        counts = np.zeros((len(students), 4))        # tn, fp, fn, tp на студента
        for column, mask in enumerate([(y == 0) & (pred == 0), (y == 0) & (pred == 1),
                                       (y == 1) & (pred == 0), (y == 1) & (pred == 1)]):
            counts[:, column] = np.bincount(inverse, weights=mask, minlength=len(students))
        sums = counts[idx].sum(axis=1)
        return qc_from_confusion(sums[:, 0], sums[:, 1], sums[:, 2], sums[:, 3])

    reference = boot_values(labels[BASELINES[1]])
    rows = []
    for name, pred in labels.items():
        values = boot_values(pred)
        diff = values - reference
        low, high = np.percentile(values, [2.5, 97.5])
        dlow, dhigh = np.percentile(diff, [2.5, 97.5])
        rows.append({"модель": name, "Qc": qc(y, pred), "ci_low": low, "ci_high": high,
                     "diff_vs_persistence": qc(y, pred) - qc(y, labels[BASELINES[1]]),
                     "diff_ci_low": dlow, "diff_ci_high": dhigh})
    return pd.DataFrame(rows)

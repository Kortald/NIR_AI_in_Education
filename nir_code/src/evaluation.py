# Метрики качества по методике ГОСТ Р 70946-2023 (п. 5.7-5.8) и вспомогательные метрики.
#
# Qc: показатель качества классификации, Qa: показатель качества аппроксимации.
import numpy as np
from sklearn.metrics import (accuracy_score, average_precision_score, f1_score,
                             precision_score, recall_score, roc_auc_score)


def qc(y_true, y_pred) -> float:
    # Qc = 2 - sum_i ( L_i/N_i + F_i/(N_i+F_i) ) по всем классам i.
    #
    # L_i: элементов класса i, ошибочно отнесённых в другой класс.
    # N_i: размер класса i в тестовом наборе.
    # F_i: чужих элементов, ошибочно попавших в класс i.
    # Максимум 2 (все ответы верны).
    y_true, y_pred = np.asarray(y_true), np.asarray(y_pred)
    total = 0.0
    for cls in np.unique(y_true):
        n_i = np.sum(y_true == cls)
        l_i = np.sum((y_true == cls) & (y_pred != cls))
        f_i = np.sum((y_true != cls) & (y_pred == cls))
        total += l_i / n_i + f_i / (n_i + f_i)
    return float(2.0 - total)


def qc_thresholds(k: int) -> tuple[float, float]:
    # Границы таблицы 1: ниже первой несоответствие, от второй высокое соответствие.
    return 2 * (k + 2) / (3 * k), 2 * (2 * k + 1) / (3 * k)


def qc_verdict(value: float, k: int) -> str:
    low, high = qc_thresholds(k)
    if value < low:
        return "несоответствие"
    return "соответствие" if value < high else "высокое соответствие"


def qa(f, y) -> float:
    # Qa = 1 + (f, y) / (|f| |y|): единица плюс косинусное сходство прогноза и факта.
    #
    # Вектор без нулевой нормы обязателен: при нулевых f или y сходство не определено.
    f, y = np.asarray(f, dtype=float), np.asarray(y, dtype=float)
    norm = np.sqrt(np.sum(f ** 2)) * np.sqrt(np.sum(y ** 2))
    if norm == 0:
        return float("nan")
    return float(1.0 + np.sum(f * y) / norm)


def qa_verdict(value: float) -> str:
    # Границы таблицы 2: 1,4 и 1,8.
    if value < 1.4:
        return "несоответствие"
    return "соответствие" if value < 1.8 else "высокое соответствие"


def classification_metrics(y_true, y_pred, y_score=None) -> dict:
    # Набор метрик бинарной классификации, класс 1 = должник.
    y_true = np.asarray(y_true)
    value = qc(y_true, y_pred)
    result = {
        "Qc": value,
        "Qc_вердикт": qc_verdict(value, k=len(np.unique(y_true))),
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "f1": f1_score(y_true, y_pred, zero_division=0),
    }
    if y_score is not None and len(np.unique(y_true)) == 2:
        result["roc_auc"] = roc_auc_score(y_true, y_score)
        result["pr_auc"] = average_precision_score(y_true, y_score)
    return result


def qc_from_confusion(tn, fp, fn, tp):
    # Qc для двух классов по числам матрицы ошибок (работает и с массивами).
    #
    # Класс 0: N0 = tn+fp, L0 = fp, F0 = fn. Класс 1: N1 = tp+fn, L1 = fn, F1 = fp.
    n0, n1 = tn + fp, tp + fn
    return 2.0 - (fp / n0 + fn / (n0 + fn) + fn / n1 + fp / (n1 + fp))

"""Проверка Qc и Qa на примерах, где ответ посчитан вручную."""
import numpy as np
import pytest

from src.evaluation import qa, qa_verdict, qc, qc_from_confusion, qc_thresholds, qc_verdict


def test_qc_perfect_is_two():
    y = [0, 0, 1, 1, 1]
    assert qc(y, y) == pytest.approx(2.0)


def test_qc_hand_example():
    # класс 0: N=2, L=1, F=0 -> 1/2 + 0 = 0.5; класс 1: N=2, L=0, F=1 -> 0 + 1/3
    value = qc([0, 0, 1, 1], [0, 1, 1, 1])
    assert value == pytest.approx(2 - (0.5 + 1 / 3))


def test_qc_majority_predictor_equals_one_minus_minority_share():
    y_true = [0] * 9 + [1]
    assert qc(y_true, [0] * 10) == pytest.approx(0.9)


def test_qc_thresholds_for_two_classes():
    low, high = qc_thresholds(2)
    assert low == pytest.approx(4 / 3)
    assert high == pytest.approx(5 / 3)


def test_qc_verdicts():
    assert qc_verdict(1.0, 2) == "несоответствие"
    assert qc_verdict(1.5, 2) == "соответствие"
    assert qc_verdict(1.9, 2) == "высокое соответствие"


def test_qa_identical_vectors_give_two():
    v = np.array([3.0, 1.0, 4.0])
    assert qa(v, v) == pytest.approx(2.0)


def test_qa_orthogonal_vectors_give_one():
    assert qa([1, 0], [0, 1]) == pytest.approx(1.0)


def test_qa_constant_forecast_is_already_high():
    # Это свойство формулы: для неотрицательных данных даже константа даёт высокий Qa.
    y = np.array([5.0, 20.0, 3.0, 12.0, 8.0])
    assert qa(np.full(5, y.mean()), y) > 1.8   # здесь около 1.85


def test_qa_verdicts():
    assert qa_verdict(1.3) == "несоответствие"
    assert qa_verdict(1.5) == "соответствие"
    assert qa_verdict(1.9) == "высокое соответствие"


def test_qc_from_confusion_matches_qc():
    rng = np.random.default_rng(3)
    y = rng.integers(0, 2, 500)
    pred = np.where(rng.uniform(size=500) < 0.8, y, 1 - y)
    tn, fp = np.sum((y == 0) & (pred == 0)), np.sum((y == 0) & (pred == 1))
    fn, tp = np.sum((y == 1) & (pred == 0)), np.sum((y == 1) & (pred == 1))
    assert qc_from_confusion(tn, fp, fn, tp) == pytest.approx(qc(y, pred))

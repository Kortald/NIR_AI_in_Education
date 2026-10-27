"""Точка входа: один запуск воспроизводит все цифры, таблицы и графики отчёта.

    python3 main.py          # результаты пишутся в results/
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import config
from src import experiments, visualisation
from src.data_loader import load_raw
from src.features import make_pairs
from src.preprocessing import data_facts, prepare_panel

SCENARIOS = {"all": "все студенты", "new": "новые должники (без долга в семестре t)"}


def describe_data(panel, pairs, out):
    """Таблицы описания данных для главы об экспериментах."""
    by_sem = panel.groupby("semester")["debtor"].agg(студенто_семестров="size", должников="sum")
    by_sem["доля"] = by_sem["должников"] / by_sem["студенто_семестров"]
    by_sem.to_csv(out / "data_debtors_by_semester.csv", encoding="utf-8-sig")
    pair_tab = pairs.groupby("semester")["y"].agg(пар="size", должников_в_следующем="sum")
    pair_tab["доля"] = pair_tab["должников_в_следующем"] / pair_tab["пар"]
    pair_tab.to_csv(out / "data_pairs_by_semester.csv", encoding="utf-8-sig")
    trans = pd.crosstab(pairs["debtor_now"], pairs["y"], normalize="index")
    trans.to_csv(out / "data_transition_matrix.csv", encoding="utf-8-sig")
    print("\n== Данные ==")
    print(by_sem.round(3).to_string())
    print(pair_tab.round(3).to_string())
    print("Переходы (строка = долг в t, столбец = долг в t+1):")
    print(trans.round(3).to_string())


def main():
    out = config.RESULTS_DIR
    out.mkdir(exist_ok=True)
    raw = load_raw()
    panel = prepare_panel(raw)
    facts = data_facts(raw)
    (out / "facts.json").write_text(json.dumps(facts, ensure_ascii=False, indent=2),
                                    encoding="utf-8")
    pairs = make_pairs(panel)
    print(f"Панель: {len(panel)} строк, {panel.student.nunique()} студентов. "
          f"Пар t->t+1: {len(pairs)}, студентов в парах: {pairs.student.nunique()}.")
    describe_data(panel, pairs, out)

    for scenario, label in SCENARIOS.items():
        summary, probas, data = experiments.cross_validate(pairs, scenario)
        summary.to_csv(out / f"cv_{scenario}.csv", index=False, encoding="utf-8-sig")
        print(f"\n== Кросс-валидация по студентам: {label} (пар {len(data)}, "
              f"должников {data.y.sum()}) ==")
        cols = ["модель", "Qc", "Qc_std", "Qc_вердикт", "recall", "precision", "f1", "roc_auc"]
        print(summary[cols].round(3).to_string(index=False))
        visualisation.plot_qc(summary, f"Qc по моделям: {label}", out / f"fig_qc_{scenario}.png")
        visualisation.plot_roc(data["y"], {n: v[0] for n, v in probas.items()}, f"ROC: {label}",
                               out / f"fig_roc_{scenario}.png")
        best = summary[~summary["модель"].isin(experiments.BASELINES)].sort_values("Qc").iloc[-1]
        visualisation.plot_confusion(data["y"], probas[best["модель"]][1], best["модель"],
                                     out / f"fig_confusion_{scenario}.png")
        boot = experiments.bootstrap_qc(data, {n: v[1] for n, v in probas.items()})
        boot.to_csv(out / f"bootstrap_{scenario}.csv", index=False, encoding="utf-8-sig")
        print("-- Бутстрэп по студентам (95% интервалы Qc и разность с повтором прошлого семестра):")
        print(boot.round(3).to_string(index=False))
        temporal = experiments.temporal_check(pairs, scenario)
        temporal.to_csv(out / f"temporal_{scenario}.csv", index=False, encoding="utf-8-sig")
        print(f"-- Проверка по времени (обучение t=1,2; тест t=3), {label}:")
        print(temporal[["модель", "Qc", "порог", "recall", "precision", "roc_auc", "n_train", "n_test"]]
              .round(3).to_string(index=False))

    counts, groups, by_semester = experiments.debtor_counts(pairs)
    counts.to_csv(out / "counts_summary.csv", index=False, encoding="utf-8-sig")
    groups.to_csv(out / "counts_groups.csv", index=False, encoding="utf-8-sig")
    by_semester.to_csv(out / "counts_by_semester.csv", index=False, encoding="utf-8-sig")
    print(f"\n== Число должников по группам (план x семестр, групп {len(groups)}) ==")
    print(counts.round(3).to_string(index=False))
    print("Итоги по семестрам:")
    print(by_semester.to_string(index=False))
    visualisation.plot_counts(groups, "Градиентный бустинг", out / "fig_counts.png")

    calib, calib_tables, _ = experiments.calibration_check(pairs)
    calib.to_csv(out / "calibration_summary.csv", index=False, encoding="utf-8-sig")
    pd.concat({n: t for n, t in calib_tables.items()}, names=["модель", "группа"]).to_csv(
        out / "calibration_bins.csv", encoding="utf-8-sig")
    visualisation.plot_calibration(calib_tables, out / "fig_calibration.png")
    print("\n== Калибровка вероятностей (все пары) ==")
    print(calib.round(4).to_string(index=False))

    scaler_model = experiments.classification_models()["Логистическая регрессия (с нуля)"]
    scaler_model.fit(pairs[config.FEATURES], pairs["y"].to_numpy())
    visualisation.plot_loss(scaler_model[-1].loss_history_, out / "fig_loss.png")
    coefs = pd.Series(scaler_model[-1].coef_, index=config.FEATURES).sort_values()
    coefs.to_csv(out / "gd_coefficients.csv", header=["коэффициент"], encoding="utf-8-sig")
    print("\nКоэффициенты логистической регрессии (стандартизованные признаки):")
    print(coefs.round(3).to_string())


if __name__ == "__main__":
    main()

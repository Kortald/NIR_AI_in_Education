"""Признаки и пары «семестр t -> семестр t+1».

Правило против утечки: признаки строки используют только семестры <= t,
цель (долг в t+1) берётся из следующего семестра.
"""
import pandas as pd

import config


def add_history(panel: pd.DataFrame) -> pd.DataFrame:
    """Добавляет накопленную историю студента к семестру t включительно."""
    panel = panel.sort_values(["student", "semester"]).copy()
    panel["cum_debtor_terms"] = panel.groupby("student")["debtor"].cumsum()
    panel["is_part_time"] = (panel["form"] == "Очно-заочная").astype(int)
    return panel


def make_pairs(panel: pd.DataFrame) -> pd.DataFrame:
    """Склеивает семестр t с результатом семестра t+1 того же студента.

    Возвращает признаки (config.FEATURES), цель y (1 = должник в t+1),
    а также служебные столбцы: student, plan, debtor_now, n_disc_next.
    Студенты без семестра t+1 в пары не попадают.
    """
    panel = add_history(panel)
    nxt = panel[["student", "semester", "debtor", "n_disc"]].copy()
    nxt["semester"] -= 1
    nxt = nxt.rename(columns={"debtor": "y", "n_disc": "n_disc_next"})
    pairs = panel.merge(nxt, on=["student", "semester"], how="inner")
    pairs["y"] = pairs["y"].astype(int)
    pairs["debtor_now"] = pairs["debtor"].astype(int)
    columns = ["student", "plan", "debtor_now", "n_disc_next", "y"] + config.FEATURES
    return pairs[columns].reset_index(drop=True)

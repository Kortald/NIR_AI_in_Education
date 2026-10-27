"""Подготовка данных: очистка, свёртка дублей, таблица «студент x семестр».

Определение должника (ГОСТ Р 72391, п. 6.2 а): студент в семестре считается должником,
если хотя бы по одной дисциплине итог «Неудовлетворительно», «Не зачтено» или «Неявка».
"""
import pandas as pd

import config

SCORE_COLUMNS = {
    "ПерваяАттестация": "att1",
    "ВтораяАттестация": "att2",
    "ТретьяАттестация": "att3",
    "Экзамен": "exam",
}


def clean(raw: pd.DataFrame) -> pd.DataFrame:
    """Оставляет семестровые формы обучения, приводит баллы к числам, помечает долги."""
    df = raw[raw["ФормаОбучения"].isin(config.SEMESTER_FORMS)].copy()
    df["semester"] = df["ПериодКонтроля"].map(config.SEMESTER_NUMBER)
    df = df.dropna(subset=["semester", "ИтоговаяОтметка"])
    df = df[df["semester"] <= config.MAX_SEMESTER]
    df["semester"] = df["semester"].astype(int)

    df["fail"] = df["ИтоговаяОтметка"].isin(config.FAIL_MARKS)
    df["noshow"] = df["ИтоговаяОтметка"].eq("Неявка")
    # пересдача и комиссия: важен сам факт, в столбцах бывают и строки («Неявка»)
    df["retake"] = df["Пересдача"].notna()
    df["commission"] = df["Комиссия"].notna()
    for source, target in SCORE_COLUMNS.items():
        df[target] = pd.to_numeric(df[source], errors="coerce")

    df = df.rename(columns={
        "ЗачетнаяКнижка": "student", "УчебныйПлан": "plan",
        "ФормаОбучения": "form", "Дисциплина": "discipline",
    })
    keep = ["student", "plan", "form", "semester", "discipline", "fail", "noshow",
            "retake", "commission", "att1", "att2", "att3", "exam"]
    return df[keep].reset_index(drop=True)


def collapse_duplicates(df: pd.DataFrame) -> pd.DataFrame:
    """Сворачивает несколько строк одной дисциплины (разные преподаватели) в одну.

    Берём худший результат: долг, неявка, пересдача, комиссия считаются, если они есть
    хотя бы в одной строке. Баллы усредняются.
    """
    grouped = df.groupby(["student", "semester", "discipline"], as_index=False)
    return grouped.agg(
        plan=("plan", "first"), form=("form", "first"),
        fail=("fail", "max"), noshow=("noshow", "max"),
        retake=("retake", "max"), commission=("commission", "max"),
        att1=("att1", "mean"), att2=("att2", "mean"),
        att3=("att3", "mean"), exam=("exam", "mean"),
    )


def build_panel(df: pd.DataFrame) -> pd.DataFrame:
    """Строит таблицу «студент x семестр»: счётчики долгов и средние баллы.

    Столбец debtor = True, если в семестре есть хотя бы один долг.
    """
    panel = df.groupby(["student", "semester"], as_index=False).agg(
        plan=("plan", "first"), form=("form", "first"),
        n_disc=("fail", "size"), n_fail=("fail", "sum"),
        n_noshow=("noshow", "sum"), n_retake=("retake", "sum"),
        n_commission=("commission", "sum"),
        mean_att1=("att1", "mean"), mean_att2=("att2", "mean"),
        mean_att3=("att3", "mean"), mean_exam=("exam", "mean"),
    )
    panel["debtor"] = panel["n_fail"] > 0
    panel["fail_share"] = panel["n_fail"] / panel["n_disc"]
    return panel


def prepare_panel(raw: pd.DataFrame) -> pd.DataFrame:
    """Полный путь от сырой таблицы до панели «студент x семестр»."""
    return build_panel(collapse_duplicates(clean(raw)))


def data_facts(raw: pd.DataFrame) -> dict:
    """Числа об исходных данных и о том, что и почему отбрасывается (для отчёта)."""
    from src.features import make_pairs

    forms = raw["ФормаОбучения"].value_counts().to_dict()
    in_forms = raw[raw["ФормаОбучения"].isin(config.SEMESTER_FORMS)]
    mapped = in_forms.assign(semester=in_forms["ПериодКонтроля"].map(config.SEMESTER_NUMBER))
    cleaned = clean(raw)
    panel = build_panel(collapse_duplicates(cleaned))
    pairs = make_pairs(panel)
    new = pairs[pairs["n_fail"] == 0]
    by_student = raw.groupby("ЗачетнаяКнижка")
    debtor_terms = panel[panel["debtor"]]
    all_failed = debtor_terms[debtor_terms["fail_share"] == 1.0]
    last_term = panel.groupby("student")["semester"].transform("max")
    all_failed_last = (all_failed["semester"] == last_term.loc[all_failed.index]).mean()
    return {
        "rows_total": len(raw),
        "students_total": int(raw["ЗачетнаяКнижка"].nunique()),
        "plans_total": int(raw["УчебныйПлан"].nunique()),
        "disciplines_total": int(raw["Дисциплина"].nunique()),
        "rows_by_form": {k: int(v) for k, v in forms.items()},
        "fail_rows_total": int(raw["ИтоговаяОтметка"].isin(config.FAIL_MARKS).sum()),
        "rows_semester_forms": len(in_forms),
        "rows_dropped_no_mark": int(mapped["ИтоговаяОтметка"].isna().sum()),
        "rows_dropped_fifth_semester": int((mapped["semester"] > config.MAX_SEMESTER).sum()),
        "rows_main_sample": len(cleaned),
        "students_main_sample": int(cleaned["student"].nunique()),
        "duplicate_groups": int((cleaned.groupby(["student", "semester", "discipline"]).size() > 1).sum()),
        "rows_after_collapse": len(collapse_duplicates(cleaned)),
        "empty_columns": [c for c in raw.columns if raw[c].isna().all()],
        "retake_for_diploma_filled": int(raw["ПересдачаДляДиплома"].notna().sum()),
        "retake_for_diploma_filled_main": int(in_forms["ПересдачаДляДиплома"].notna().sum()),
        "missing_share": {c: round(float(raw[c].isna().mean()), 4) for c in raw.columns},
        "students_with_several_forms": int((by_student["ФормаОбучения"].nunique() > 1).sum()),
        "students_with_several_plans": int((by_student["УчебныйПлан"].nunique() > 1).sum()),
        "panel_rows": len(panel),
        "debtor_terms": len(debtor_terms),
        "debtor_terms_all_failed_share": float((debtor_terms["fail_share"] == 1.0).mean()),
        "all_failed_terms_last_share": float(all_failed_last),
        "pairs_all": len(pairs), "positives_all": int(pairs["y"].sum()),
        "students_in_pairs": int(pairs["student"].nunique()),
        "pairs_new": len(new), "positives_new": int(new["y"].sum()),
        "forecast_groups_note": "группы: учебный план x семестр, размер не менее config.MIN_GROUP_SIZE",
    }

"""Проверка подготовки данных и построения пар на маленькой таблице."""
import pandas as pd

from src.features import make_pairs
from src.preprocessing import build_panel, clean, collapse_duplicates


def _raw(rows):
    columns = ["ЗачетнаяКнижка", "УчебныйПлан", "ФормаОбучения", "ПериодКонтроля",
               "Дисциплина", "ИтоговаяОтметка", "ПерваяАттестация", "ВтораяАттестация",
               "ТретьяАттестация", "Экзамен", "Пересдача", "Комиссия"]
    return pd.DataFrame(rows, columns=columns)


def test_clean_drops_part_time_and_marks_fails():
    raw = _raw([
        (1, 10, "Очная", "Первый семестр", "A", "Неявка", "Неявка", None, None, None, None, None),
        (1, 10, "Очная", "Первый семестр", "B", "Хорошо", 30, 25, 30, None, None, None),
        (2, 11, "Заочная", "Первая зимняя сессия", "A", "Отлично", 40, 40, 40, None, None, None),
    ])
    df = clean(raw)
    assert set(df["student"]) == {1}
    assert df["fail"].tolist() == [True, False]
    assert pd.isna(df.loc[0, "att1"])  # «Неявка» в баллах превратилась в пропуск


def test_collapse_takes_worst_result():
    raw = _raw([
        (1, 10, "Очная", "Первый семестр", "A", "Хорошо", 30, 30, 30, None, None, None),
        (1, 10, "Очная", "Первый семестр", "A", "Неявка", 10, 10, 10, None, None, None),
    ])
    out = collapse_duplicates(clean(raw))
    assert len(out) == 1 and bool(out.loc[0, "fail"]) is True
    assert out.loc[0, "att1"] == 20


def test_pairs_use_next_semester_as_target():
    raw = _raw([
        (1, 10, "Очная", "Первый семестр", "A", "Хорошо", 30, 30, 30, None, None, None),
        (1, 10, "Очная", "Второй семестр", "A", "Неявка", None, None, None, None, None, None),
        (2, 10, "Очная", "Первый семестр", "A", "Хорошо", 30, 30, 30, None, None, None),
    ])
    pairs = make_pairs(build_panel(collapse_duplicates(clean(raw))))
    assert len(pairs) == 1                       # у студента 2 нет семестра 2
    assert pairs.loc[0, "y"] == 1                # во втором семестре студент 1 должник
    assert pairs.loc[0, "debtor_now"] == 0       # а в первом должником не был

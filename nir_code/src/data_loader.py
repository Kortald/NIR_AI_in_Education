# Загрузка исходной таблицы аттестаций из xlsx из приложения к ГОСТ Р 72391-2025.
import pandas as pd

import config


def load_raw(use_cache: bool = True) -> pd.DataFrame:
    # Читает xlsx (или готовый кэш) и возвращает таблицу как есть, без обработки.
    #
    # Одна строка = аттестация одного студента по одной дисциплине.
    if use_cache and config.CACHE_PKL.exists():
        return pd.read_pickle(config.CACHE_PKL)
    df = pd.read_excel(config.DATA_XLSX)
    config.CACHE_PKL.parent.mkdir(parents=True, exist_ok=True)
    df.to_pickle(config.CACHE_PKL)
    return df

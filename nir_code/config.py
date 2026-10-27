# Все настройки проекта в одном месте: пути, определения, параметры экспериментов.
from pathlib import Path

ROOT = Path(__file__).resolve().parent
DATA_XLSX = ROOT.parent / "Сборник файлов" / "ГОСТ Р 72391-2025 приложение.xlsx"
CACHE_PKL = ROOT / "data" / "raw_cache.pkl"   # xlsx читается долго, кэшируем
RESULTS_DIR = ROOT / "results"

SEED = 42
N_FOLDS = 5
INNER_FOLDS = 3
N_REPEATS = 3

# Основная выборка: формы обучения, у которых периоды идут по семестрам 1-4.
# Заочная форма живёт по сессиям с неоднозначным порядком, в выборку не входит.
SEMESTER_FORMS = ("Очная", "Очно-заочная")
SEMESTER_NUMBER = {
    "Первый семестр": 1,
    "Второй семестр": 2,
    "Третий семестр": 3,
    "Четвертый семестр": 4,
    "Пятый семестр": 5,
}
MAX_SEMESTER = 4     # пятый семестр встречается у 2 студентов, отбрасываем

FAIL_MARKS = ("Неудовлетворительно", "Не зачтено", "Неявка")

# Для прогноза числа должников берём только группы (план x семестр) не меньше этого размера.
MIN_GROUP_SIZE = 10

# Признаки считаются по семестру t и более ранним.
FEATURES = [
    "semester", "n_disc", "n_fail", "fail_share", "n_noshow", "n_retake",
    "n_commission", "mean_att1", "mean_att2", "mean_att3", "mean_exam",
    "cum_debtor_terms", "is_part_time",
]

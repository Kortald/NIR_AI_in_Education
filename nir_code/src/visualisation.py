# Графики для отчёта. Палитра Окабе-Ито, различима при дальтонизме и в ч/б печати.
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import confusion_matrix, roc_curve

from src.evaluation import qc_thresholds

COLORS = ["#0072B2", "#E69F00", "#009E73", "#D55E00", "#CC79A7", "#56B4E9"]
plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False,
                     "figure.dpi": 150})


def plot_qc(summary, title, path, k=2):
    # Столбики Qc по моделям с границами «соответствия» из таблицы 1.
    low, high = qc_thresholds(k)
    order = summary.sort_values("Qc")
    fig, ax = plt.subplots(figsize=(7.5, 3.6))
    ax.barh(order["модель"], order["Qc"], xerr=order["Qc_std"], color=COLORS[0], height=0.6)
    ax.axvline(low, color=COLORS[3], ls="--", lw=1.2, label=f"соответствие (от {low:.2f})")
    ax.axvline(high, color=COLORS[2], ls="--", lw=1.2, label=f"высокое соответствие (от {high:.2f})")
    ax.legend(fontsize=7, loc="lower right")
    ax.set_xlim(min(0.9, order["Qc"].min() - 0.05), 2.0)
    ax.set_xlabel("Qc (максимум 2)")
    ax.set_title(title, fontsize=10)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def plot_roc(y, probas, title, path):
    fig, ax = plt.subplots(figsize=(4.8, 4.2))
    for color, (name, p) in zip(COLORS, probas.items()):
        if name.startswith("Константа"):
            continue
        fpr, tpr, _ = roc_curve(y, p)
        ax.plot(fpr, tpr, color=color, lw=1.6, label=name)
    ax.plot([0, 1], [0, 1], color="grey", lw=0.8, ls=":")
    ax.set_xlabel("Доля ложных тревог")
    ax.set_ylabel("Доля найденных должников")
    ax.set_title(title, fontsize=10)
    ax.legend(fontsize=7, loc="lower right")
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def plot_confusion(y, labels, name, path):
    cm = confusion_matrix(y, labels)
    fig, ax = plt.subplots(figsize=(3.6, 3.2))
    ax.imshow(cm, cmap="Blues")
    for i in range(2):
        for j in range(2):
            ax.text(j, i, cm[i, j], ha="center", va="center",
                    color="white" if cm[i, j] > cm.max() / 2 else "black")
    ax.set_xticks([0, 1], ["не должник", "должник"])
    ax.set_yticks([0, 1], ["не должник", "должник"])
    ax.set_xlabel("Прогноз")
    ax.set_ylabel("Факт")
    ax.set_title(name, fontsize=9)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def plot_loss(history, path):
    fig, ax = plt.subplots(figsize=(4.8, 3.2))
    ax.plot(history, color=COLORS[0], lw=1.6)
    ax.set_xlabel("Итерация градиентного спуска")
    ax.set_ylabel("Функция потерь")
    ax.set_title("Сходимость логистической регрессии с нуля", fontsize=10)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def plot_counts(groups, model, path):
    # Прогноз и факт числа должников по группам (план x семестр).
    fig, ax = plt.subplots(figsize=(4.4, 4.2))
    top = max(groups["actual"].max(), groups[model].max()) * 1.05
    ax.scatter(groups["actual"], groups[model], s=18, color=COLORS[0], alpha=0.75)
    ax.plot([0, top], [0, top], color="grey", lw=0.8, ls=":")
    ax.set_xlabel("Фактическое число должников")
    ax.set_ylabel("Прогноз")
    ax.set_title(f"Число должников по группам: {model}", fontsize=9)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def plot_calibration(tables, path):
    # Диаграмма надёжности: средний прогноз и фактическая доля долгов по группам прогноза.
    fig, ax = plt.subplots(figsize=(4.6, 4.2))
    top = max(max(t["прогноз"].max(), t["доля_долгов"].max()) for t in tables.values()) * 1.1
    ax.plot([0, top], [0, top], color="grey", lw=0.8, ls=":", label="идеальная калибровка")
    for color, (name, table) in zip(COLORS, tables.items()):
        ax.plot(table["прогноз"], table["доля_долгов"], marker="o", ms=4, lw=1.2,
                color=color, label=name)
    ax.set_xlabel("Средний прогноз вероятности долга")
    ax.set_ylabel("Фактическая доля долгов")
    ax.legend(fontsize=7, loc="upper left")
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)

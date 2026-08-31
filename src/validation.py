"""
Неделя 3-4: метрики качества и валидация.

Отток — редкое событие, поэтому accuracy не подходит. Используем метрики
ранжирования: ROC-AUC, PR-AUC, Precision@k, Lift@k.
"""
import pandas as pd
import numpy as np
from sklearn.metrics import roc_auc_score, average_precision_score


def time_split(dataset, train_snaps, val_snaps, test_snaps):
    """Разбиение по времени (out-of-time), три среза.

    Parameters
    ----------
    dataset : датасет со столбцом snapshot_date
    train_snaps : список снапшотов для train
    val_snaps : список снапшотов для val (early stopping)
    test_snaps : список снапшотов для test (финальная оценка)

    Returns
    -------
    train, val, test — три датафрейма
    """
    train = dataset[dataset["snapshot_date"].isin(train_snaps)]
    val = dataset[dataset["snapshot_date"].isin(val_snaps)]
    test = dataset[dataset["snapshot_date"].isin(test_snaps)]
    return train, val, test

def precision_at_k(y_true, y_score, k: float = 0.1) -> float:
    """Доля реально ушедших среди топ-k% клиентов по прогнозу риска."""
    n = max(1, int(len(y_score) * k))
    idx = np.argsort(-np.asarray(y_score))[:n]
    return float(np.asarray(y_true)[idx].mean())


def lift_at_k(y_true, y_score, k: float = 0.1) -> float:
    """Во сколько раз топ-k% лучше случайного выбора."""
    return precision_at_k(y_true, y_score, k) / (np.mean(y_true) + 1e-12)


def evaluate(y_true, y_score, label: str = "") -> dict:
    """Печатает и возвращает основные метрики качества."""
    metrics = {
        "label": label,
        "roc_auc": roc_auc_score(y_true, y_score),
        "pr_auc": average_precision_score(y_true, y_score),
        "precision_at_10": precision_at_k(y_true, y_score, 0.10),
        "lift_at_10": lift_at_k(y_true, y_score, 0.10),
        "churn_rate": float(np.mean(y_true)),
    }
    print(f"--- {label} ---")
    print(f"ROC-AUC: {metrics['roc_auc']:.4f}")
    print(f"PR-AUC: {metrics['pr_auc']:.4f}")
    print(f"Precision@10%: {metrics['precision_at_10']:.4f}")
    print(f"Lift@10%: {metrics['lift_at_10']:.2f}x")
    print(f"Доля оттока: {metrics['churn_rate']:.4f}")
    return metrics


def bootstrap_auc_diff(y_true, s1, s2, n_iter: int = 1000, seed: int = 42) -> dict:
    """Bootstrap-сравнение двух подходов: доля выборок, где подход 1 лучше.

    Используйте на неделе 4 для честного сравнения подходов A/B/C — разница
    в 0.5-1 п.п. AUC может быть просто шумом.
    """
    rng = np.random.default_rng(seed)
    y_true, s1, s2 = map(np.asarray, (y_true, s1, s2))
    wins, diffs = 0, []
    for _ in range(n_iter):
        idx = rng.integers(0, len(y_true), len(y_true))
        if y_true[idx].min() == y_true[idx].max():
            continue  # выборка без обоих классов — пропускаем
        d = roc_auc_score(y_true[idx], s1[idx]) - roc_auc_score(y_true[idx], s2[idx])
        diffs.append(d)
        wins += d > 0
    result = {"win_rate": wins / len(diffs), "mean_diff": float(np.mean(diffs))}
    print(f"Подход 1 лучше в {result['win_rate']:.0%} выборок; "
          f"средняя разница AUC = {result['mean_diff']:+.4f}")
    return result

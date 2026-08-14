"""
Неделя 2-3: сборка обучающей выборки и построение признаков.

Правило №1 проекта: признаки строятся ТОЛЬКО из данных до точки отсчёта
(snapshot), таргет — ТОЛЬКО из данных после. Нарушение этого правила — data
leakage.

Функции ниже — скелеты. Заполните тело под свой датасет, не меняя сигнатуры
(так будет проще ревьюить и сравнивать между студентами).
"""
import numpy as np
import pandas as pd

OBSERVATION_MONTHS = 6  # окно наблюдения: сколько месяцев до snapshot
FORECAST_MONTHS = 3      # окно прогноза: сколько месяцев после snapshot

def shift_month(month: str, delta: int) -> str:
    """Сдвинуть месяц 'YYYY-MM' на delta месяцев.
    shift_month('2025-01', -1) -> '2024-12'"""
    return str(pd.Period(month, freq="M") + delta)

def build_snapshot(usage: pd.DataFrame, clients: pd.DataFrame, snapshot: str) -> pd.DataFrame:
    """Собирает обучающую выборку для одной точки отсчёта.

    Parameters
    ----------
    usage : помесячная таблица (client_id, month, revenue, ...)
    clients : справочник клиентов (client_id, segment, product, ...)
    snapshot : строка вида '2025-01' — точка отсчёта

    Returns
    -------
    DataFrame с признаками, таргетом и метаданными (segment/product/snapshot_date)

    TODO (неделя 2):
    1. Отфильтровать клиентов, активных на конец окна наблюдения ("мёртвые души")
    2. Посчитать признаки по окну наблюдения через make_features()
    3. Определить таргет по окну прогноза
    4. Склеить признаки + таргет + метаданные клиента
    """
    alive = clients[
        (clients["connection_date"] <= snapshot)
        & (clients["termination_date"].isna() | (clients["termination_date"] > snapshot))
        ].copy()

    active_ids = usage.loc[
        (usage["month"] == snapshot) & (usage["revenue"] > 0),
        "client_id",
    ].unique()
    alive = alive[alive["client_id"].isin(active_ids)]

    obs_start = shift_month(snapshot, -(OBSERVATION_MONTHS - 1))  # например, для snapshot='2025-06' -> '2025-01'

    obs = usage[
        (usage["month"] >= obs_start)
        & (usage["month"] <= snapshot)
        & (usage["client_id"].isin(alive["client_id"]))
        ].sort_values("month")

    feats = obs.groupby("client_id").agg(
        revenue_mean_6m=("revenue", "mean"),
        revenue_last=("revenue", "last"),
        traffic_mean_6m=("traffic_gb", "mean"),
        tickets_sum_6m=("n_tickets", "sum"),
        debt_max_6m=("debt", "max"),
        n_sim_last=("n_sim", "last"),
    ).reset_index()

    pred_start = shift_month(snapshot, FORECAST_MONTHS - 1)  # следующий месяц после snapshot
    pred_end = shift_month(snapshot, FORECAST_MONTHS)  # через 3 месяца

    pred = usage[
        (usage["month"] >= pred_start)
        & (usage["month"] <= pred_end)
        & (usage["client_id"].isin(alive["client_id"]))
        ]
    pred_rev = pred.groupby("client_id")["revenue"].sum()

    alive["pred_revenue"] = alive["client_id"].map(pred_rev).fillna(0)

    alive["target"] = (
            (alive["termination_date"].notna() & (alive["termination_date"] <= pred_end))
            | (alive["pred_revenue"] == 0)
    ).astype(int)

    df = alive[["client_id", "segment", "product", "region", "target"]].merge(
        feats, on="client_id", how="left"
    )

    df["snapshot_date"] = snapshot
    return df


def make_features(obs: pd.DataFrame, obs_end) -> pd.DataFrame:
    """Строит признаки по данным окна наблюдения.

    Parameters
    ----------
    obs : помесячные данные окна наблюдения (client_id, month, revenue, ...)
    obs_end : последний месяц окна наблюдения

    Returns
    -------
    DataFrame признаков с индексом client_id

    TODO (неделя 3): реализовать группы признаков:
    - платежи (среднее/мин/макс, отношение последний/среднее)
    - динамика (тренд, волатильность, месяцев подряд снижения)
    - объём отношений (число услуг, срок жизни)
    - сигналы боли (обращения в поддержку, задолженность)
    """
    raise NotImplementedError("Реализуйте на неделе 3")


def trend_slope(s: pd.Series) -> float:
    """Наклон линейного тренда: >0 растёт, <0 падает."""
    y = s.to_numpy(dtype=float)
    if len(y) < 2:
        return 0.0
    x = np.arange(len(y))
    return float(np.polyfit(x, y, 1)[0])


def months_declining(s: pd.Series) -> int:
    """Сколько последних месяцев подряд значение снижалось."""
    diffs = s.diff().to_numpy()[1:]
    cnt = 0
    for d in diffs[::-1]:
        if d < 0:
            cnt += 1
        else:
            break
    return cnt

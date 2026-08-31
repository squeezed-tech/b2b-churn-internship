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

OBSERVATION_MONTHS = 6
FORECAST_MONTHS = 3
GAP_MONTHS = 2
MIN_OBS_MONTHS = 4

def shift_month(month: str, delta: int) -> str:
    """Сдвинуть месяц 'YYYY-MM' на delta месяцев.
    shift_month('2025-01', -1) -> '2024-12'"""
    return str(pd.Period(month, freq="M") + delta)

def build_snapshot(usage: pd.DataFrame, clients: pd.DataFrame, snapshot: str) -> pd.DataFrame:
    """Собирает обучающую выборку для одной точки отсчёта.

    Наблюдение: OBSERVATION_MONTHS месяцев по месяц снапшота включительно -> признаки.
    Зазор: GAP_MONTHS месяцев после снапшота -> не используются вообще.
    Прогноз: FORECAST_MONTHS месяцев после зазора -> только таргет.
    """
    # --- живые на момент снапшота ---
    alive = clients[
        (clients["connection_date"] <= snapshot)
        & (clients["termination_date"].isna() | (clients["termination_date"] > snapshot))
    ].copy()

    active_ids = usage.loc[
        (usage["month"] == snapshot) & (usage["revenue"] > 0),
        "client_id",
    ].unique()
    alive = alive[alive["client_id"].isin(active_ids)]

    # --- окно наблюдения (только признаки) ---
    obs_start = shift_month(snapshot, -(OBSERVATION_MONTHS - 1))
    obs = usage[
        (usage["month"] >= obs_start)
        & (usage["month"] <= snapshot)
        & (usage["client_id"].isin(alive["client_id"]))
    ].sort_values("month")

    # Фильтр короткой истории: минимум MIN_OBS_MONTHS месяцев в окне,
    # иначе "среднее за 6 месяцев" считается по 1-2 месяцам
    obs_months = obs.groupby("client_id")["month"].nunique()
    good_ids = obs_months[obs_months >= MIN_OBS_MONTHS].index
    obs = obs[obs["client_id"].isin(good_ids)]
    alive = alive[alive["client_id"].isin(good_ids)]

    feats = make_features(obs, snapshot)

    # --- окно прогноза (только таргет), через зазор ---
    pred_start = shift_month(snapshot, GAP_MONTHS + 1)
    pred_end = shift_month(snapshot, GAP_MONTHS + FORECAST_MONTHS)

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

    alive["tenure_months"] = alive["connection_date"].apply(
        lambda d: (pd.Period(snapshot, freq="M") - pd.Period(d, freq="M")).n
    )

    df = alive[["client_id", "segment", "product", "region", "tenure_months", "target"]].merge(
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

    obs = obs.sort_values(["client_id", "month"]).copy()

    features = obs.groupby("client_id").agg(

        # Платежи
        revenue_mean_6m=("revenue", "mean"),
        revenue_min_6m=("revenue", "min"),
        revenue_max_6m=("revenue", "max"),
        revenue_last=("revenue", "last"),

        # Использование
        traffic_mean_6m=("traffic_gb", "mean"),
        traffic_last=("traffic_gb", "last"),

        # Обращения в поддержку
        tickets_sum_6m=("n_tickets", "sum"),
        tickets_mean_6m=("n_tickets", "mean"),

        # Задолженность
        debt_max_6m=("debt", "max"),
        debt_mean_6m=("debt", "mean"),
        debt_last=("debt", "last"),

        # Объём услуг
        n_sim_mean_6m=("n_sim", "mean"),
        n_sim_last=("n_sim", "last"),

    )

    # Последняя выручка относительно среднего уровня клиента

    features["revenue_last_to_mean"] = (features["revenue_last"] / features["revenue_mean_6m"].replace(0, np.nan))
    features["revenue_last_to_mean"] = (features["revenue_last_to_mean"].fillna(0))

    # Динамические признаки

    revenue_trend = (obs.groupby("client_id")["revenue"].apply(trend_slope).rename("revenue_trend"))
    revenue_volatility = (obs.groupby("client_id")["revenue"].std().fillna(0).rename("revenue_volatility"))
    declining_months = (obs.groupby("client_id")["revenue"].apply(months_declining).rename("revenue_declining_months"))

    features = features.join(revenue_trend)
    features = features.join(revenue_volatility)
    features = features.join(declining_months)
    features = features.reset_index()
    return features


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

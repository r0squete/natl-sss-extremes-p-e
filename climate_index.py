#!/usr/bin/env python
# Created by arosquete on 2025-05-23

import warnings
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy.stats import pearsonr, spearmanr, ttest_1samp

warnings.filterwarnings(
    "ignore", category=FutureWarning, module="statsmodels.tsa.stattools"
)

PROJECT_ROOT = Path("/home/rosquete/Documents/TFM/Data/data_public/")
WORK_ROOT = PROJECT_ROOT / "sss"
NAO_FILE = PROJECT_ROOT / "climate_index" / "data_index.csv"
NINNO_FILE = PROJECT_ROOT / "climate_index" / "ninno_index.csv"
INDICES_FIG_ROOT = PROJECT_ROOT / "output"
INDEX_COLUMNS = ["nao", "ninno"]


def get_season(month):
    if month in [12, 1, 2]:
        return "DJF"
    elif month in [3, 4, 5]:
        return "MAM"
    elif month in [6, 7, 8]:
        return "JJA"
    elif month in [9, 10, 11]:
        return "SON"


def lagged_corr(x, y, lags, method="pearson"):
    r_vals = []
    p_vals = []
    for lag in lags:
        if lag < 0:
            x_lag = x.iloc[:lag]
            y_lag = y.iloc[-lag:]
        elif lag > 0:
            x_lag = x.iloc[lag:]
            y_lag = y.iloc[:-lag]
        else:
            x_lag = x.copy()
            y_lag = y.copy()
        if method == "pearson":
            r, p = pearsonr(x_lag, y_lag)
        elif method == "spearman":
            r, p = spearmanr(x_lag, y_lag)
        else:
            raise ValueError("Invalid method")
        r_vals.append(r)
        p_vals.append(p)
    return np.array(r_vals), np.array(p_vals)


def plot_lag_corr(lags, r_vals, p_vals, title):
    plt.figure(figsize=(10, 5))
    plt.stem(lags, r_vals, basefmt=" ")
    plt.axhline(0, color="gray", linestyle="--")
    plt.axhline(0.2, color="red", linestyle=":", linewidth=1)
    plt.axhline(-0.2, color="red", linestyle=":", linewidth=1)
    for i, p in enumerate(p_vals):
        if p < 0.05:
            plt.plot(lags[i], r_vals[i], "ro")
    plt.xlabel("Lag (months)")
    plt.ylabel("Correlation coefficient")
    plt.title(title)
    plt.grid(True)
    plt.tight_layout()
    plt.show()


fechas_bajos = pd.read_csv(
    WORK_ROOT / "fechas_mensuales_eventos_sss_bajos.csv", parse_dates=["event_date"]
)
fechas_altos = pd.read_csv(
    WORK_ROOT / "fechas_mensuales_eventos_sss_altos.csv", parse_dates=["event_date"]
)
fechas_bajos["event_date"] = (
    pd.to_datetime(fechas_bajos["event_date"]).dt.to_period("M").dt.to_timestamp()
)
fechas_altos["event_date"] = (
    pd.to_datetime(fechas_altos["event_date"]).dt.to_period("M").dt.to_timestamp()
)

sss = pd.read_csv(WORK_ROOT / "sss_monthly_anomalies.csv", parse_dates=["time"])
sss["time"] = sss["time"].dt.to_period("M").dt.to_timestamp()
sss = sss.set_index("time").sort_index()


data_index = pd.read_csv(NAO_FILE, delimiter=r"\s+", engine="python")
data_index = data_index.rename(columns={data_index.columns[0]: "Year"})

data_index_long = data_index.melt(id_vars="Year", var_name="Month", value_name="nao")
data_index_long["Month_num"] = pd.to_datetime(
    data_index_long["Month"], format="%b"
).dt.month
data_index_long["Date"] = pd.to_datetime(
    dict(year=data_index_long["Year"], month=data_index_long["Month_num"], day=1)
)
nao_df = data_index_long.set_index("Date").sort_index()[["nao"]]

ninno_raw = pd.read_csv(NINNO_FILE, sep="\t", engine="python")
ninno_raw.columns = [col.strip() for col in ninno_raw.columns]

ninno_long = ninno_raw.melt(
    id_vars=ninno_raw.columns[0], var_name="Season", value_name="ninno"
)
ninno_long = ninno_long.rename(columns={ninno_raw.columns[0]: "Year"})
ninno_long = ninno_long.dropna()

season_month = {
    "DJF": 1,
    "JFM": 2,
    "FMA": 3,
    "MAM": 4,
    "AMJ": 5,
    "MJJ": 6,
    "JJA": 7,
    "JAS": 8,
    "ASO": 9,
    "SON": 10,
    "OND": 11,
    "NDJ": 12,
}
ninno_long["Month_num"] = ninno_long["Season"].map(season_month)

ninno_long["Date"] = pd.to_datetime(
    dict(year=ninno_long["Year"].astype(int), month=ninno_long["Month_num"], day=1)
)
ninno_df = ninno_long.set_index("Date").sort_index()[["ninno"]]


df = pd.merge(sss, nao_df, left_index=True, right_index=True, how="inner")
df = pd.merge(df, ninno_df, left_index=True, right_index=True, how="inner")

df = df.rename(columns={"salinity_detrended": "salinity_anomaly"})

df["time"] = df.index
df = df[["time", "salinity_anomaly", "nao", "ninno"]].reset_index(drop=True)

max_lag = 12
lags = np.arange(-max_lag, max_lag + 1)


df["time"] = pd.to_datetime(df["time"])
df = df.set_index("time")
df["season"] = df.index.month.map(get_season)
df["year"] = df.index.year
df.loc[df.index.month == 12, "year"] += 1


df_seasonal = df.groupby(["year", "season"]).mean().reset_index()


def seasonal_lag_corr(df_seasonal, index_name, var_name, max_lag_seasons=6):
    """
    index_name: 'nao' o 'ninno'
    var_name: 'salinity_anomaly'
    """
    estaciones = ["DJF", "MAM", "JJA", "SON"]
    results = []

    for est_idx in estaciones:
        for est_sss in estaciones:
            for lag in range(max_lag_seasons + 1):
                df_ref = df_seasonal[df_seasonal["season"] == est_idx].copy()
                df_sss = df_seasonal[df_seasonal["season"] == est_sss].copy()

                df_ref["target_year"] = (
                    df_ref["year"]
                    + (lag + estaciones.index(est_sss) - estaciones.index(est_idx)) // 4
                )
                df_ref["target_season"] = estaciones[
                    (estaciones.index(est_idx) + lag) % 4
                ]

                merged = pd.merge(
                    df_ref,
                    df_sss,
                    how="inner",
                    left_on=["target_year", "target_season"],
                    right_on=["year", "season"],
                    suffixes=("_idx", "_sss"),
                )

                if len(merged) >= 10:
                    r, p = pearsonr(
                        merged[f"{index_name}_idx"], merged[f"{var_name}_sss"]
                    )
                    results.append(
                        {
                            "Index": index_name.upper(),
                            "Index Season": est_idx,
                            "SSS Season": est_sss,
                            "Season Lag": lag,
                            "r": r,
                            "p": p,
                        }
                    )

    return pd.DataFrame(results)


result_nao = seasonal_lag_corr(
    df_seasonal, "nao", "salinity_anomaly", max_lag_seasons=6
)
result_ninno = seasonal_lag_corr(
    df_seasonal, "ninno", "salinity_anomaly", max_lag_seasons=6
)

result_all = pd.concat([result_nao, result_ninno], ignore_index=True)
significativas = result_all[result_all["p"] < 0.05].sort_values(by="r", ascending=False)


def plot_seasonal_heatmap_adjusted(
    df,
    indice,
    lag_season,
    ax=None,
    subplot_label=None,
    annotate_bbox=True,
    show_colorbar=True,
    title=None,
    vlim=None,
):
    df_plot = df[(df["Index"] == indice) & (df["Season Lag"] == lag_season)].copy()

    tabla = df_plot.pivot(index="Index Season", columns="SSS Season", values="r")
    pvalores = df_plot.pivot(index="Index Season", columns="SSS Season", values="p")

    signif_mask = pvalores < 0.05

    anotaciones = tabla.round(2).astype(str)
    anotaciones = np.where(signif_mask, anotaciones + "*", anotaciones)

    sns.set(style="white")
    if ax is None:
        fig, ax = plt.subplots(figsize=(4.8, 3.5))
    else:
        fig = ax.figure

    vmin, vmax = (vlim if vlim is not None else (None, None))

    heatmap_ax = sns.heatmap(
        tabla,
        annot=anotaciones,
        fmt="",
        cmap="coolwarm",
        center=0,
        linewidths=0.5,
        cbar=show_colorbar,
        cbar_kws={"label": "r coefficient"} if show_colorbar else None,
        annot_kws={"size": 10},
        ax=ax,
        vmin=vmin,
        vmax=vmax,
    )

    ax = heatmap_ax

    if title:
        ax.set_title(title, fontsize=11)
    ax.set_xlabel("SSS season", fontsize=10)
    ax.set_ylabel("Index season", fontsize=10)
    ax.set_xticklabels(ax.get_xticklabels(), fontsize=9)
    ax.set_yticklabels(ax.get_yticklabels(), fontsize=9, rotation=0)

    if subplot_label:
        ax.text(
            0.93,
            1.02,
            subplot_label,
            transform=ax.transAxes,
            fontsize=13,
            fontweight="bold",
            ha="right",
            va="top",
            clip_on=False,
            bbox=(
                dict(
                    facecolor="white",
                    edgecolor="none",
                    alpha=0.6,
                    boxstyle="round,pad=0.2",
                )
                if annotate_bbox
                else None
            ),
        )
    if ax is None:
        fig.tight_layout()
        INDICES_FIG_ROOT.mkdir(parents=True, exist_ok=True)
        fig.savefig(INDICES_FIG_ROOT / f"{indice}_heatmap_compacto.png", dpi=600)
        plt.show()
    return ax


fig, axes = plt.subplots(1, 2, figsize=(9.6, 3.6))

max_abs_r = np.nanmax(np.abs(result_all["r"]))
shared_vlim = (-max_abs_r, max_abs_r)

plot_seasonal_heatmap_adjusted(
    result_all,
    "NINNO",
    lag_season=4,
    ax=axes[0],
    subplot_label="a)",
    show_colorbar=False,
    vlim=shared_vlim,
)
plot_seasonal_heatmap_adjusted(
    result_all,
    "NAO",
    lag_season=5,
    ax=axes[1],
    subplot_label="b)",
    show_colorbar=True,
    vlim=shared_vlim,
)

for ax in axes:
    for label in ax.get_xticklabels():
        label.set_rotation(0)

fig.tight_layout(w_pad=2.0)
INDICES_FIG_ROOT.mkdir(parents=True, exist_ok=True)
fig.savefig(INDICES_FIG_ROOT / "seasonal_heatmaps_adjusted.png", dpi=600)
plt.show()


def calcular_composite_con_lag(df_indices, fechas_eventos, nombre_evento, lag_mes=0):
    fechas_lag = fechas_eventos["event_date"] - pd.DateOffset(months=lag_mes)
    fechas_lag = fechas_lag.to_frame(name="event_date")
    fechas_lag["year"] = fechas_lag["event_date"].dt.year
    fechas_lag["month"] = fechas_lag["event_date"].dt.month

    df_indices = df_indices.copy()
    df_indices["year"] = df_indices.index.year
    df_indices["month"] = df_indices.index.month

    df_merge = pd.merge(
        df_indices.reset_index(), fechas_lag, on=["year", "month"], how="inner"
    )

    composite = df_merge[INDEX_COLUMNS].mean()
    std_eventos = df_merge[INDEX_COLUMNS].std()
    climatologia = df_indices[INDEX_COLUMNS].mean()
    std_clima = df_indices[INDEX_COLUMNS].std()

    df_plot = pd.DataFrame(
        {
            "Composite": composite,
            "Climatology": climatologia,
            "±1σ Events": std_eventos,
            "±1σ Climatology": std_clima,
        }
    )
    df_plot["Event Type"] = nombre_evento
    df_plot = df_plot.reset_index().rename(columns={"index": "Index"})
    return df_plot


def plot_composites_by_event(df_low, df_high):
    df_low = df_low.copy()
    df_high = df_high.copy()

    df_low["Type"] = "Low SSS"
    df_high["Type"] = "High SSS"

    display_names = {"nao": "NAO", "ninno": "ONI"}
    df_low["index_display"] = df_low["Index"].map(display_names)
    df_high["index_display"] = df_high["Index"].map(display_names)

    df_concat = pd.concat([df_low, df_high]).reset_index(drop=True)

    plt.figure(figsize=(10, 4))
    sns.set_context("paper", font_scale=1.2)
    sns.set_style("whitegrid")

    sns.barplot(
        data=df_concat,
        x="index_display",
        y="Composite",
        hue="Type",
        palette="Set2",
        errorbar=None,
    )

    for i, row in df_concat.iterrows():
        yerr = row["±1σ Events"]
        x = i % len(df_low) + (0.2 if row["Type"] == "High SSS" else -0.2)
        plt.errorbar(
            x,
            row["Composite"],
            yerr=yerr,
            fmt="none",
            color="black",
            capsize=4,
            linewidth=1,
        )

    plt.axhline(0, color="gray", linestyle="--", linewidth=0.8)
    plt.ylabel("Mean index value during events", fontsize=14)
    plt.xlabel("Climate index", fontsize=14)
    plt.legend(title="SSS event type", frameon=False, fontsize=13)
    plt.tight_layout()
    INDICES_FIG_ROOT.mkdir(parents=True, exist_ok=True)
    plt.savefig(INDICES_FIG_ROOT / "composites.png", dpi=600)
    plt.show()


lag_enso = 4

comp_bajo = calcular_composite_con_lag(df, fechas_bajos, "SSS baja", lag_mes=lag_enso)
comp_alto = calcular_composite_con_lag(df, fechas_altos, "SSS alta", lag_mes=lag_enso)

plot_composites_by_event(comp_bajo, comp_alto)


def test_composite_con_lag(df, fechas, columna, nombre_evento, lag_mes=0):
    fechas_lag = fechas["event_date"] - pd.DateOffset(months=lag_mes)
    serie = df.loc[df.index.isin(fechas_lag), columna].dropna()
    media_clima = df[columna].mean()
    t_stat, p_val = ttest_1samp(serie, popmean=media_clima)
    print(f"\n📌 {columna.upper()} during {nombre_evento} (lag {lag_mes} seasons):")
    print(f"- Mean during events: {serie.mean():.3f}")
    print(f"- Climatological mean:  {media_clima:.3f}")
    print(f"- t-statistic: {t_stat:.3f}")
    print(f"- p-value: {p_val:.4f}")


test_composite_con_lag(df, fechas_altos, "ninno", "SSS alta", lag_mes=lag_enso)
test_composite_con_lag(df, fechas_bajos, "ninno", "SSS baja", lag_mes=lag_enso)
test_composite_con_lag(df, fechas_altos, "nao", "SSS alta", lag_mes=lag_enso)
test_composite_con_lag(df, fechas_bajos, "nao", "SSS baja", lag_mes=lag_enso)

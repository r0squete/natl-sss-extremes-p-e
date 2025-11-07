#!/usr/bin/env python
# Created by arosquete on 2025-05-20

from pathlib import Path
import matplotlib.pyplot as plt
import pandas as pd

DATA_ROOT = Path("/home/rosquete/Documents/TFM/Data/data_public/sss")
FIGURES_ROOT = Path("/home/rosquete/Documents/TFM/Data/data_public/output")

def load_event_dates() -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load CSV files listing the monthly occurrence of low/high SSS events."""
    dates_low = pd.read_csv(DATA_ROOT / "fechas_mensuales_eventos_sss_bajos.csv", parse_dates=["event_date"])
    dates_high = pd.read_csv(DATA_ROOT / "fechas_mensuales_eventos_sss_altos.csv", parse_dates=["event_date"])
    return dates_low, dates_high

def plot_monthly_frequencies(conteo_bajos: pd.Series, conteo_altos: pd.Series) -> None:
    """Plot the monthly distribution of low/high SSS events."""
    fig, ax = plt.subplots(figsize=(9, 6))
    ax.bar(
        conteo_bajos.index - 0.2,
        conteo_bajos.values,
        width=0.4,
        label="Low SSS Events",
        color="cornflowerblue",
        edgecolor="black",
        alpha=0.7,
    )
    ax.bar(
        conteo_altos.index + 0.2,
        conteo_altos.values,
        width=0.4,
        label="High SSS Events",
        color="indianred",
        edgecolor="black",
        alpha=0.7,
    )
    ax.set_xticks(range(1, 13))
    ax.set_xlabel("Month", fontsize=13)
    ax.set_ylabel("Number of Events", fontsize=13)
    ax.tick_params(axis="both", labelsize=11)
    ax.legend(loc="upper left", fontsize=11)
    ax.grid(True, linestyle="--", alpha=0.5)
    plt.tight_layout()
    FIGURES_ROOT.mkdir(parents=True, exist_ok=True)
    plt.savefig(FIGURES_ROOT / "frecuencia.png", dpi=600)
    plt.show()

def summarize_persistent_events() -> None:
    """Print events that persisted longer than 6 months."""
    events_low = pd.read_csv(DATA_ROOT / "eventos_sss_bajos.csv", parse_dates=["Inicio", "Fin"])
    events_high = pd.read_csv(DATA_ROOT / "eventos_sss_altos.csv", parse_dates=["Inicio", "Fin"])

    events_low["duration_months"] = (
        events_low["Fin"].dt.to_period("M") - events_low["Inicio"].dt.to_period("M")
    ).apply(lambda x: x.n + 1)
    events_high["duration_months"] = (
        events_high["Fin"].dt.to_period("M") - events_high["Inicio"].dt.to_period("M")
    ).apply(lambda x: x.n + 1)

    prolonged_low = events_low[events_low["duration_months"] > 6]
    prolonged_high = events_high[events_high["duration_months"] > 6]

    print('Low SSS events lasting more than six months:')
    print(prolonged_low[["Inicio", "Fin", "duration_months"]])

    print("\nHigh SSS events lasting more than six months:")
    print(prolonged_high[["Inicio", "Fin", "duration_months"]])

def main() -> None:
    dates_low, dates_high = load_event_dates()
    dates_low["month"] = dates_low["event_date"].dt.month
    dates_high["month"] = dates_high["event_date"].dt.month
    counts_low = dates_low["month"].value_counts().sort_index()
    counts_high = dates_high["month"].value_counts().sort_index()

    plot_monthly_frequencies(counts_low, counts_high)
    summarize_persistent_events()

if __name__ == "__main__":
    main()

#!/usr/bin/env python
# Created by arosquete on 2025-05-19

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd
import xarray as xr

file_path_detrended = (
    "/home/rosquete/Documents/TFM/Data/data_public/sss/sss_detrended_1985_2014.nc"
)
sss_var_name_detrended = "salinity_detrended"
time_var_name = "time"
lat_var_name = "lat"
lon_var_name = "lon"

start_year_str = "1985"
end_year_str = "2014"

valid_data_min_threshold = -5
valid_data_max_threshold = +5

try:
    ds_detrended_full = xr.open_dataset(file_path_detrended)
    sss_detrended_all_times = ds_detrended_full[sss_var_name_detrended]

    if (
        "depth" in sss_detrended_all_times.dims
        and sss_detrended_all_times.sizes["depth"] == 1
    ):
        sss_detrended_all_times = sss_detrended_all_times.squeeze("depth", drop=True)

    sss_detrended_periodo_raw = sss_detrended_all_times.sel(
        {time_var_name: slice(start_year_str, end_year_str)}
    )

    if sss_detrended_periodo_raw[time_var_name].size == 0:
        raise ValueError("No detrended data were selected for the study period.")
    print(
        f"Selected raw detrended SSS data for {start_year_str}-{end_year_str}. time steps: {sss_detrended_periodo_raw[time_var_name].size}"
    )

except Exception as e:
    print(f"Error loading or selecting detrended data: {e}")
    exit()

sss_detrended_valid_region = sss_detrended_periodo_raw.where(
    (sss_detrended_periodo_raw > valid_data_min_threshold)
    & (sss_detrended_periodo_raw < valid_data_max_threshold)
)

sss_media_espacial_detrended = sss_detrended_valid_region.mean(
    dim=[lat_var_name, lon_var_name], skipna=True
)
sss_media_espacial_detrended = sss_media_espacial_detrended.squeeze(drop=True)

if sss_media_espacial_detrended.isnull().all():
    print(
        "Error: sss_media_espacial_detrended contains only NaNs. Check the dataset, thresholds, or mask."
    )
    exit()

print("\n🔹 Spatially averaged detrended SSS time series computed.")
mean_check = sss_media_espacial_detrended.mean().item()
print(
    f"   Mean of the spatially averaged detrended time series: {mean_check:.4f} (should be ~0)"
)

clim_detrended = sss_media_espacial_detrended.groupby(f"{time_var_name}.month").mean(
    dim=time_var_name
)
print("\n🔹 Monthly climatology of the detrended series:")
for mes_num in clim_detrended.month.values:
    valor = clim_detrended.sel(month=mes_num).item()
    print(f"   - Month {mes_num:02d}: {valor:.4f}")

sss_anomalies_clean = (
    sss_media_espacial_detrended.groupby(f"{time_var_name}.month") - clim_detrended
)
print("\n🔹 Clean monthly anomalies computed.")
mean_anom_check = sss_anomalies_clean.mean().item()
print(f"   Mean of the clean anomalies: {mean_anom_check:.4f} (should be ~0)")

df = sss_anomalies_clean.to_dataframe().reset_index()
df.to_csv(
    "/home/rosquete/Documents/TFM/Data/data_public/sss/sss_monthly_anomalies.csv",
    index=False,
)

std_dev_anomalies_clean = sss_anomalies_clean.std(dim=time_var_name).item()
print(f"\n🔹 Standard deviation of the clean anomalies: {std_dev_anomalies_clean:.3f}")
print("   (This value defines thresholds for extreme events)")


tiempo_anom = sss_anomalies_clean["time"]

fig, ax = plt.subplots(figsize=(12, 5.5))
ax.plot(
    tiempo_anom.data,
    sss_anomalies_clean.data,
    color="darkslateblue",
    linewidth=1.0,
    alpha=0.9,
    label="SSS Anomalies",
)
ax.axhline(0, color="black", linestyle="-", linewidth=0.8, alpha=0.7, zorder=0)
if not np.isnan(std_dev_anomalies_clean):
    ax.axhline(
        std_dev_anomalies_clean,
        color="red",
        linestyle="--",
        linewidth=1,
        alpha=0.7,
        label=f"+1σ ({std_dev_anomalies_clean:.3f})",
    )
    ax.axhline(
        -std_dev_anomalies_clean,
        color="red",
        linestyle="--",
        linewidth=1,
        alpha=0.7,
        label=f"-1σ (-{std_dev_anomalies_clean:.3f})",
    )
ax.set_xlabel("Year", fontsize=12)
ax.set_ylabel("SSS Anomaly", fontsize=12)
ax.legend(fontsize=12, loc="best")
ax.tick_params(axis="x", labelsize=11)
ax.tick_params(axis="y", labelsize=11)
ax.yaxis.set_major_formatter(mticker.FormatStrFormatter("%.2f"))
ax.grid(True, linestyle=":", alpha=0.6, zorder=0)
if sss_anomalies_clean.data[np.isfinite(sss_anomalies_clean.data)].size > 0:
    max_abs_anomaly_plot = np.nanmax(np.abs(sss_anomalies_clean.data))
    y_limit_anom_plot = max_abs_anomaly_plot * 1.1 if max_abs_anomaly_plot > 0 else 0.5
    ax.set_ylim(-y_limit_anom_plot, y_limit_anom_plot)
plt.tight_layout()
plt.savefig(
    "/home/rosquete/Documents/TFM/Data/data_public/output/figure_B1_sss_anomalies_clean_EN.png",
    dpi=600,
)
plt.show()


def identify_events_with_persistence(
    anomalies_series, magnitude_threshold, persistence_months, event_type="alto"
):
    """Identify events exceeding a magnitude threshold that persist for a minimum number of months."""
    time_coord_name = "time"
    if not isinstance(anomalies_series, (xr.DataArray, pd.Series)):
        raise TypeError("anomalies_series must be an xarray.DataArray or pandas.Series")
    if (
        isinstance(anomalies_series, xr.DataArray)
        and time_coord_name not in anomalies_series.coords
    ):
        raise ValueError(
            f"anomalies_series (xarray.DataArray) does not have a '{time_coord_name}' coordinate"
        )

    if event_type == "alto":
        meets_threshold = anomalies_series > magnitude_threshold
    elif event_type == "bajo":
        meets_threshold = anomalies_series < magnitude_threshold
    else:
        raise ValueError("event_type must be 'alto' or 'bajo'")

    events = []
    in_event = False
    event_start_idx = -1
    month_counter = 0
    event_values = []

    for i in range(len(anomalies_series)):
        current_value = anomalies_series.data[i]

        if meets_threshold.data[i]:
            if not in_event:
                in_event = True
                event_start_idx = i
                month_counter = 0
                event_values = []

            month_counter += 1
            event_values.append(current_value)
        else:
            if in_event and month_counter >= persistence_months:
                events.append(
                    {
                        "Inicio": anomalies_series[time_coord_name].data[
                            event_start_idx
                        ],
                        "Fin": anomalies_series[time_coord_name].data[i - 1],
                        "Duration_months": month_counter,
                        "Mean_Magnitude": np.mean(event_values)
                        if event_values
                        else np.nan,
                        "Peak_Magnitude": (
                            np.max(event_values)
                            if event_type == "alto" and event_values
                            else (
                                np.min(event_values)
                                if event_type == "bajo" and event_values
                                else np.nan
                            )
                        ),
                    }
                )
            in_event = False
            month_counter = 0
            event_values = []

    if in_event and month_counter >= persistence_months:
        events.append(
            {
                "Inicio": anomalies_series[time_coord_name].data[event_start_idx],
                "Fin": anomalies_series[time_coord_name].data[-1],
                "Duration_months": month_counter,
                "Mean_Magnitude": np.mean(event_values) if event_values else np.nan,
                "Peak_Magnitude": (
                    np.max(event_values)
                    if event_type == "alto" and event_values
                    else (
                        np.min(event_values)
                        if event_type == "bajo" and event_values
                        else np.nan
                    )
                ),
            }
        )

    return pd.DataFrame(events)


def get_monthly_event_dates(df_events):
    """Return unique monthly timestamps covered by the input events."""
    if df_events.empty:
        return pd.DatetimeIndex([])

    all_event_dates = []
    for _, event in df_events.iterrows():
        start_date = pd.to_datetime(event["Inicio"])
        end_date = pd.to_datetime(event["Fin"])
        event_date_range = pd.date_range(
            start=start_date.to_period("M").to_timestamp(),
            end=end_date.to_period("M").to_timestamp(),
            freq="MS",
        )
        all_event_dates.extend(event_date_range)

    return pd.DatetimeIndex(sorted(set(all_event_dates)))


print(
    f"Standard deviation of clean anomalies (std_dev_anomalies_clean): {std_dev_anomalies_clean:.3f}\n"
)

umbral_de_1_0 = 1.0 * std_dev_anomalies_clean
persistence_2_months = 2
events_altos_de1_0_p2 = identify_events_with_persistence(
    sss_anomalies_clean, umbral_de_1_0, persistence_2_months, event_type="alto"
)
events_bajos_de1_0_p2 = identify_events_with_persistence(
    sss_anomalies_clean, -umbral_de_1_0, persistence_2_months, event_type="bajo"
)
print(
    f"--- Threshold: +/- 1.0 σ ({umbral_de_1_0:.3f}), Persistence: {persistence_2_months} months ---"
)
print(f"Number of HIGH events: {len(events_altos_de1_0_p2)}")
if not events_altos_de1_0_p2.empty:
    print(events_altos_de1_0_p2.head())  # noqa: E701
print(f"Number of LOW events: {len(events_bajos_de1_0_p2)}")
if not events_bajos_de1_0_p2.empty:
    print(events_bajos_de1_0_p2.head())  # noqa: E701
print("-" * 50)

umbral_de_1_5 = 1.5 * std_dev_anomalies_clean
persistence_3_months = 3
events_altos_de1_5_p3 = identify_events_with_persistence(
    sss_anomalies_clean, umbral_de_1_5, persistence_3_months, event_type="alto"
)
events_bajos_de1_5_p3 = identify_events_with_persistence(
    sss_anomalies_clean, -umbral_de_1_5, persistence_3_months, event_type="bajo"
)
print(
    f"--- Threshold: +/- 1.5 σ ({umbral_de_1_5:.3f}), Persistence: {persistence_3_months} months ---"
)
print(f"Number of HIGH events: {len(events_altos_de1_5_p3)}")
if not events_altos_de1_5_p3.empty:
    print(events_altos_de1_5_p3.head())  # noqa: E701
print(f"Number of LOW events: {len(events_bajos_de1_5_p3)}")
if not events_bajos_de1_5_p3.empty:
    print(events_bajos_de1_5_p3.head())  # noqa: E701
print("-" * 50)

umbral_p05 = sss_anomalies_clean.quantile(0.05, skipna=True).item()
umbral_p95 = sss_anomalies_clean.quantile(0.95, skipna=True).item()
persistence_p_2_months_05_95 = 2
events_altos_p95_p2 = identify_events_with_persistence(
    sss_anomalies_clean, umbral_p95, persistence_p_2_months_05_95, event_type="alto"
)
events_bajos_p05_p2 = identify_events_with_persistence(
    sss_anomalies_clean, umbral_p05, persistence_p_2_months_05_95, event_type="bajo"
)
print(
    f"--- Threshold: Percentile 05 ({umbral_p05:.3f}) and Percentile 95 ({umbral_p95:.3f}), Persistence: {persistence_p_2_months_05_95} months ---"
)
print(f"Number of HIGH events (P95): {len(events_altos_p95_p2)}")
if not events_altos_p95_p2.empty:
    print(events_altos_p95_p2.head())  # noqa: E701
print(f"Number of LOW events (P05): {len(events_bajos_p05_p2)}")
if not events_bajos_p05_p2.empty:
    print(events_bajos_p05_p2.head())  # noqa: E701
print("-" * 50)

umbral_p10 = sss_anomalies_clean.quantile(0.10, skipna=True).item()
umbral_p90 = sss_anomalies_clean.quantile(0.90, skipna=True).item()
persistence_p_2_months_10_90 = 2

events_altos_p90_p2 = identify_events_with_persistence(
    sss_anomalies_clean, umbral_p90, persistence_p_2_months_10_90, event_type="alto"
)
events_bajos_p10_p2 = identify_events_with_persistence(
    sss_anomalies_clean, umbral_p10, persistence_p_2_months_10_90, event_type="bajo"
)

print(
    f"--- Threshold: Percentile 10 ({umbral_p10:.3f}) and Percentile 90 ({umbral_p90:.3f}), Persistence: {persistence_p_2_months_10_90} months ---"
)
print(f"Number of HIGH events (P90): {len(events_altos_p90_p2)}")
if not events_altos_p90_p2.empty:
    print(events_altos_p90_p2.head())  # noqa: E701
print(f"Number of LOW events (P10): {len(events_bajos_p10_p2)}")
if not events_bajos_p10_p2.empty:
    print(events_bajos_p10_p2.head())  # noqa: E701
print("-" * 50)

events_altos_seleccionados = events_altos_de1_0_p2
events_bajos_seleccionados = events_bajos_de1_0_p2

if not events_altos_seleccionados.empty:
    events_altos_seleccionados.to_csv(
        "/home/rosquete/Documents/TFM/Data/data_public/sss/eventos_sss_altos.csv",
        index=False,
    )
if not events_bajos_seleccionados.empty:
    events_bajos_seleccionados.to_csv(
        "/home/rosquete/Documents/TFM/Data/data_public/sss/eventos_sss_bajos.csv",
        index=False,
    )


print("\n--- Saving individual monthly dates for all events ---")

high_event_months = get_monthly_event_dates(events_altos_seleccionados)
low_event_months = get_monthly_event_dates(events_bajos_seleccionados)

if len(high_event_months) > 0:
    pd.DataFrame({"event_date": high_event_months}).to_csv(
        "/home/rosquete/Documents/TFM/Data/data_public/sss/fechas_mensuales_eventos_sss_altos.csv",
        index=False,
        header=["event_date"],
    )
    print(
        f"Monthly dates for HIGH events ({len(high_event_months)} months) saved to 'fechas_mensuales_eventos_sss_altos.csv'"
    )

if len(low_event_months) > 0:
    pd.DataFrame({"event_date": low_event_months}).to_csv(
        "/home/rosquete/Documents/TFM/Data/data_public/sss/fechas_mensuales_eventos_sss_bajos.csv",
        index=False,
        header=["event_date"],
    )
    print(
        f"Monthly dates for LOW events ({len(low_event_months)} months) saved to 'fechas_mensuales_eventos_sss_bajos.csv'"
    )

fig, ax = plt.subplots(figsize=(12, 5.5))

ax.plot(
    tiempo_anom.data,
    sss_anomalies_clean.data,
    color="darkslateblue",
    linewidth=1.0,
    alpha=0.9,
    zorder=2,
    label="SSS Anomalies",
)

ax.axhline(0, color="black", linestyle="-", linewidth=0.8, alpha=0.7, zorder=1)

if not np.isnan(std_dev_anomalies_clean):
    umbral_usado = 1.0 * std_dev_anomalies_clean
    ax.axhline(
        umbral_usado,
        color="dimgray",
        linestyle="--",
        linewidth=1,
        alpha=0.9,
        label=f"Threshold (±1σ = ±{umbral_usado:.3f})",
        zorder=1,
    )
    ax.axhline(
        -umbral_usado, color="dimgray", linestyle="--", linewidth=1, alpha=0.9, zorder=1
    )

label_event_alto_added = False
for idx, event in events_altos_seleccionados.iterrows():
    ax.axvspan(
        pd.to_datetime(event["Inicio"]),
        pd.to_datetime(event["Fin"]) + pd.Timedelta(days=15),
        color="lightcoral",
        alpha=0.5,
        zorder=0,
        label="High SSS Events" if not label_event_alto_added else "_nolegend_",
    )
    label_event_alto_added = True

label_event_bajo_added = False
for idx, event in events_bajos_seleccionados.iterrows():
    ax.axvspan(
        pd.to_datetime(event["Inicio"]),
        pd.to_datetime(event["Fin"]) + pd.Timedelta(days=15),
        color="lightsteelblue",
        alpha=0.5,
        zorder=0,
        label="Low SSS Events" if not label_event_bajo_added else "_nolegend_",
    )
    label_event_bajo_added = True

ax.set_xlabel("Year", fontsize=12)
ax.set_ylabel("SSS Anomaly", fontsize=12)

if sss_anomalies_clean.data[np.isfinite(sss_anomalies_clean.data)].size > 0:
    max_abs_anomaly_plot = np.nanmax(np.abs(sss_anomalies_clean.data))
    y_limit_anom_plot = (
        max(max_abs_anomaly_plot, umbral_usado if not np.isnan(umbral_usado) else 0)
        * 1.15
    )
    if y_limit_anom_plot == 0:
        y_limit_anom_plot = 0.5  # noqa: E701
    ax.set_ylim(-y_limit_anom_plot, y_limit_anom_plot)

ax.tick_params(axis="x", labelsize=11)
ax.tick_params(axis="y", labelsize=11)
ax.yaxis.set_major_formatter(mticker.FormatStrFormatter("%.2f"))

ax.legend(fontsize=12, loc="best")

plt.tight_layout()
plt.savefig(
    "/home/rosquete/Documents/TFM/Data/data_public/output/figure_sss_anomalies_extreme_events_EN.png",
    dpi=1000,
)
plt.show()

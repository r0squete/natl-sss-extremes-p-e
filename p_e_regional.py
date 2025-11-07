#!/usr/bin/env python
# Created by arosquete on 2025-05-27

import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import xarray as xr
import xesmf as xe

output_dir = "/home/rosquete/Documents/TFM/Data/data_public/output/"

file_path_ploa_anom_limpias = (
    "/home/rosquete/Documents/TFM/Data/data_public/p_e/pe_1985-2014_anom_detrended.nc"
)
ploa_anom_var_name = "E_P"
unidades_pe = "(mm/month)"
time_var_name_std = "time"
lat_var_name_std = "lat"
lon_var_name_std = "lon"

csv_eventos_altos = "/home/rosquete/Documents/TFM/Data/data_public/sss/fechas_mensuales_eventos_sss_altos.csv"
csv_eventos_bajos = "/home/rosquete/Documents/TFM/Data/data_public/sss/fechas_mensuales_eventos_sss_bajos.csv"

start_year_str = "1985"
end_year_str = "2014"

lags_a_probar = [-3, -2, -1, 0, 1, 2, 3]

regiones_sumidero_info = {
    "AZO": {
        "mask_path": "/home/rosquete/Documents/TFM/Data/data_public/masks/AZO.nc",
        "mask_var_name": "mask",
        "mask_orig_lat_name": "lat",
        "mask_orig_lon_name": "lon",
    },
    "CAN": {
        "mask_path": "/home/rosquete/Documents/TFM/Data/data_public/masks/CAN.nc",
        "mask_var_name": "mask",
        "mask_orig_lat_name": "lat",
        "mask_orig_lon_name": "lon",
    },
    "CVE": {
        "mask_path": "/home/rosquete/Documents/TFM/Data/data_public/masks/CAV.nc",
        "mask_var_name": "mask",
        "mask_orig_lat_name": "lat",
        "mask_orig_lon_name": "lon",
    },
    "MAD": {
        "mask_path": "/home/rosquete/Documents/TFM/Data/data_public/masks/MAD.nc",
        "mask_var_name": "mask",
        "mask_orig_lat_name": "lat",
        "mask_orig_lon_name": "lon",
    },
    "IP": {
        "mask_path": "/home/rosquete/Documents/TFM/Data/data_public/masks/IP.nc",
        "mask_var_name": "mask",
        "mask_orig_lat_name": "lat",
        "mask_orig_lon_name": "lon",
    },
    "CAM": {
        "mask_path": "/home/rosquete/Documents/TFM/Data/data_public/masks/CAM.nc",
        "mask_var_name": "mask",
        "mask_orig_lat_name": "lat",
        "mask_orig_lon_name": "lon",
    },
    "CAR": {
        "mask_path": "/home/rosquete/Documents/TFM/Data/data_public/masks/CAR.nc",
        "mask_var_name": "mask",
        "mask_orig_lat_name": "lat",
        "mask_orig_lon_name": "lon",
    },
    "ENA": {
        "mask_path": "/home/rosquete/Documents/TFM/Data/data_public/masks/ENA.nc",
        "mask_var_name": "mask",
        "mask_orig_lat_name": "lat",
        "mask_orig_lon_name": "lon",
    },
}

print("--- Loading cleaned P-E anomalies (full domain) ---")
try:
    ds_ploa_anom_full = xr.open_dataset(file_path_ploa_anom_limpias, decode_times=False)
    ploa_anom_raw_time = ds_ploa_anom_full[ploa_anom_var_name]

    time_values_numeric_ploa = ploa_anom_raw_time[time_var_name_std].data

    time_units_ploa = ploa_anom_raw_time[time_var_name_std].attrs.get("units")
    if not time_units_ploa:
        print(
            f"  WARNING: Time units for PLOA were not found. Assuming 'months since {start_year_str}-01-15 00:00:00'. please verify."
        )
        time_units_ploa = f"months since {start_year_str}-01-15 00:00:00"

    reference_date_str_ploa = time_units_ploa.split(" since ")[1]
    reference_date_ploa = pd.to_datetime(reference_date_str_ploa)
    datetime_values_ploa = [
        reference_date_ploa + pd.DateOffset(months=int(m))
        for m in time_values_numeric_ploa
    ]

    dims_originales_ploa = list(ploa_anom_raw_time.dims)
    coord_esp_original_nombres = [
        d for d in dims_originales_ploa if d != time_var_name_std
    ]

    if len(coord_esp_original_nombres) != 2:
        raise ValueError(
            f"Expected 2 spatial dimensions in PLOA, found {len(coord_esp_original_nombres)}: {coord_esp_original_nombres}"
        )

    original_lon_name_ploa = coord_esp_original_nombres[1]
    original_lat_name_ploa = coord_esp_original_nombres[0]

    coords_ploa_dt_time = {
        time_var_name_std: pd.to_datetime(datetime_values_ploa),
        lat_var_name_std: ploa_anom_raw_time.coords[original_lat_name_ploa].data,
        lon_var_name_std: ploa_anom_raw_time.coords[original_lon_name_ploa].data,
    }

    ploa_anom_amplio = xr.DataArray(
        ploa_anom_raw_time.data,
        coords=coords_ploa_dt_time,
        dims=[time_var_name_std, lat_var_name_std, lon_var_name_std],
        name=ploa_anom_raw_time.name,
        attrs=ploa_anom_raw_time.attrs,
    )
    print(
        f"P-E time converted. First dates: {ploa_anom_amplio[time_var_name_std].data[:3]}"
    )

    print(f"P-E anomalies (full domain) ready. Shape: {ploa_anom_amplio.shape}")

except Exception as e:
    print(f"Error loading or processing PLOA anomaly data: {e}")
    exit()

series_temporales_ploa_sumideros = {}
print("\n--- Processing P-E for sink regions ---")

for nombre_sumidero, info_mascara in regiones_sumidero_info.items():
    print(f"\nProcessing sink: {nombre_sumidero}")

    try:
        ds_mask_sumidero_orig = xr.open_dataset(info_mascara["mask_path"])
        mask_sumidero_orig = ds_mask_sumidero_orig[
            info_mascara["mask_var_name"]
        ].squeeze(drop=True)

        if (mask_sumidero_orig["lon"] > 180).any():
            mask_sumidero_orig["lon"] = ((mask_sumidero_orig["lon"] + 180) % 360) - 180
            mask_sumidero_orig = mask_sumidero_orig.sortby("lon")

        mask_sumidero_orig = mask_sumidero_orig.rename(
            {
                info_mascara["mask_orig_lat_name"]: "lat",
                info_mascara["mask_orig_lon_name"]: "lon",
            }
        )
        print(
            f"  Mask for {nombre_sumidero} loaded. Original shape: {mask_sumidero_orig.shape}"
        )

        ploa_lat_min = ploa_anom_amplio["lat"].min().item()
        ploa_lat_max = ploa_anom_amplio["lat"].max().item()
        ploa_lon_min = ploa_anom_amplio["lon"].min().item()
        ploa_lon_max = ploa_anom_amplio["lon"].max().item()
        margin = 2.0

        es_lat_desc_mask = (
            mask_sumidero_orig["lat"][0].item() > mask_sumidero_orig["lat"][-1].item()
        )
        lim_n_slice_mask = min(
            ploa_lat_max + margin, mask_sumidero_orig["lat"].max().item()
        )
        lim_s_slice_mask = max(
            ploa_lat_min - margin, mask_sumidero_orig["lat"].min().item()
        )
        lat_slice_m = (
            slice(lim_n_slice_mask, lim_s_slice_mask)
            if es_lat_desc_mask
            else slice(lim_s_slice_mask, lim_n_slice_mask)
        )

        lim_o_slice_mask = max(
            ploa_lon_min - margin, mask_sumidero_orig["lon"].min().item()
        )
        lim_e_slice_mask = min(
            ploa_lon_max + margin, mask_sumidero_orig["lon"].max().item()
        )
        lon_slice_m = slice(lim_o_slice_mask, lim_e_slice_mask)

        mask_sumidero_subset = mask_sumidero_orig.sel(lat=lat_slice_m, lon=lon_slice_m)

        if (
            mask_sumidero_subset.size == 0
            or mask_sumidero_subset["lat"].size == 0
            or mask_sumidero_subset["lon"].size == 0
        ):
            print(f"  WARNING: Mask subset for {nombre_sumidero} empty. Skipping.")
            series_temporales_ploa_sumideros[nombre_sumidero] = None
            continue
        print(
            f"  Mask for {nombre_sumidero} subset. New shape: {mask_sumidero_subset.shape}"
        )

        print(f"  Regridding mask for {nombre_sumidero} onto the P-E grid...")
        regridder_sumidero = xe.Regridder(
            mask_sumidero_subset,
            ploa_anom_amplio,
            method="nearest_s2d",
            unmapped_to_nan=False,
        )
        mask_sumidero_regridded_float = regridder_sumidero(
            mask_sumidero_subset, keep_attrs=True
        )
        mask_sumidero_final_bool = np.round(mask_sumidero_regridded_float) == 1

        if mask_sumidero_final_bool.sum().item() == 0:
            print(
                f"  WARNING: Mask regridded for {nombre_sumidero} has no True points. Skipping."
            )
            series_temporales_ploa_sumideros[nombre_sumidero] = None
            continue
        print(
            f"  Mask for {nombre_sumidero} regridded. True points: {mask_sumidero_final_bool.sum().item()}"
        )

        ploa_anom_sumidero_enmascarada = ploa_anom_amplio.where(
            mask_sumidero_final_bool
        )

        serie_temporal_actual = ploa_anom_sumidero_enmascarada.mean(
            dim=[lat_var_name_std, lon_var_name_std], skipna=True
        )
        series_temporales_ploa_sumideros[nombre_sumidero] = (
            serie_temporal_actual.squeeze(drop=True)
        )
        print(
            f"  P-E anomaly time series for {nombre_sumidero} computed. Mean: {serie_temporal_actual.mean(skipna=True).item():.4f}"
        )

    except Exception as e:
        print(f"  Error processing sink {nombre_sumidero}: {e}")
        series_temporales_ploa_sumideros[nombre_sumidero] = None

print("\n--- Processing of P-E for all sink regions completed ---")


df = pd.DataFrame(
    {
        region: [float(data.values)] if data.size == 1 else data.values.flatten()
        for region, data in series_temporales_ploa_sumideros.items()
    }
)

df.to_csv("/home/rosquete/Documents/TFM/Data/data_public/sss/anomalias_regionales.csv")

actual_date_column_altos = "event_date"
actual_date_column_bajos = "event_date"

try:
    df_eventos_altos_csv = pd.read_csv(csv_eventos_altos)
    df_eventos_bajos_csv = pd.read_csv(csv_eventos_bajos)

    s_fechas_sss_altas = (
        pd.to_datetime(df_eventos_altos_csv[actual_date_column_altos])
        .dt.to_period("M")
        .dt.to_timestamp()
    )
    s_fechas_sss_bajas = (
        pd.to_datetime(df_eventos_bajos_csv[actual_date_column_bajos])
        .dt.to_period("M")
        .dt.to_timestamp()
    )

    set_fechas_sss_altas = set(s_fechas_sss_altas.dropna())
    set_fechas_sss_bajas = set(s_fechas_sss_bajas.dropna())

except FileNotFoundError:
    print("Error: SSS event CSV file(s) not found. Please check paths.")
    print(f"High-event path: {csv_eventos_altos}")
    print(f"Low-event path: {csv_eventos_bajos}")
    set_fechas_sss_altas = set()
    set_fechas_sss_bajas = set()
except KeyError as e:
    print(
        f"Error: Column {e} not found in SSS CSV. Ensure '{actual_date_column_altos}' or '{actual_date_column_bajos}' are correct."
    )
    set_fechas_sss_altas = set()
    set_fechas_sss_bajas = set()
except Exception as e:
    print(f"An unexpected error occurred while loading SSS event CSVs: {e}")
    set_fechas_sss_altas = set()
    set_fechas_sss_bajas = set()

print(
    f"Loaded {len(set_fechas_sss_altas)} unique high SSS event dates (normalized to 1st of month)."
)
print(
    f"Loaded {len(set_fechas_sss_bajas)} unique low SSS event dates (normalized to 1st of month)."
)

all_lags_data = []
unique_region_namonth_from_data = sorted(list(series_temporales_ploa_sumideros.keys()))

for nombre_sumidero in unique_region_namonth_from_data:
    pe_anom_series = series_temporales_ploa_sumideros.get(nombre_sumidero)
    if pe_anom_series is None:
        print(
            f"Skipping {nombre_sumidero} as it has no P-E data in series_temporales_ploa_sumideros."
        )
        continue

    df_pe_anom = pe_anom_series.to_pandas()
    if not isinstance(df_pe_anom.index, pd.DatetimeIndex):
        df_pe_anom.index = pd.to_datetime(df_pe_anom.index)

    df_pe_anom.index = df_pe_anom.index.to_period("M").to_timestamp()

    for fecha_pe, valor_pe in df_pe_anom.items():
        for lag_months in lags_a_probar:
            target_sss_event_date = fecha_pe - pd.DateOffset(months=lag_months)

            condition = "SSS Neutra"
            if target_sss_event_date in set_fechas_sss_altas:
                condition = "SSS Alta"
            elif target_sss_event_date in set_fechas_sss_bajas:
                condition = "SSS Baja"

            if pd.notna(valor_pe):
                all_lags_data.append(
                    {
                        "P-E Anomaly": valor_pe,
                        "SSS Condition": condition,
                        "Lag (Months)": lag_months,
                        "Region": nombre_sumidero,
                    }
                )

df_all_lags_plot = pd.DataFrame(all_lags_data)

event_types_map = {
    "SSS Baja": "Low_SSS",
    "SSS Neutra": "Normal",
    "SSS Alta": "High_SSS",
}
colors_for_sss = {"Low_SSS": "#1f77b4", "Normal": "#cccccc", "High_SSS": "#d62728"}
sss_conditions_ordered = ["Low_SSS", "Normal", "High_SSS"]
lags_filtrados = [0, 1, 2, 3]

df_all_lags_plot["SSS Display"] = df_all_lags_plot["SSS Condition"].map(event_types_map)

os.makedirs(output_dir, exist_ok=True)

regiones_ordenadas = list(regiones_sumidero_info.keys())


def build_x_layout(lags, conditions):
    x_order = []
    xticks_labels = []
    group_centers = {}
    idx = 0
    for lag_idx, lag in enumerate(lags):
        start_idx = idx
        for cond in conditions:
            key = f"Lag{lag}_{cond}"
            x_order.append(key)
            xticks_labels.append(cond.replace("_", " "))
            idx += 1
        end_idx = idx - 1
        group_centers[lag] = (start_idx + end_idx) / 2
        if lag_idx != len(lags) - 1:
            spacer = f"Spacer_Lag{lag}"
            x_order.append(spacer)
            xticks_labels.append("")
            idx += 1
    return x_order, xticks_labels, group_centers


x_order_global, xticks_labels_global, group_centers_global = build_x_layout(
    lags_filtrados, sss_conditions_ordered
)


def plot_region_panel(
    ax,
    region_name,
    subplot_label,
    show_xticklabels=False,
    show_group_labels=False,
    title=None,
):
    df_region = df_all_lags_plot[
        (df_all_lags_plot["Region"] == region_name)
        & (df_all_lags_plot["Lag (Months)"].isin(lags_filtrados))
    ].copy()

    if df_region.empty:
        ax.set_visible(False)
        return

    df_region["Lag"] = df_region["Lag (Months)"]
    df_region["x_custom"] = df_region.apply(
        lambda row: f"Lag{row['Lag']}_{row['SSS Display']}", axis=1
    )
    df_region["Hue_dummy"] = df_region["SSS Display"]

    # Ensure placeholder categories exist so the spacing between lags is preserved.
    for sep in [k for k in x_order_global if k.startswith("Spacer")]:
        fila_vacia = pd.DataFrame(
            [{"x_custom": sep, "P-E Anomaly": np.nan, "Hue_dummy": np.nan}]
        )
        df_region = pd.concat([df_region, fila_vacia], ignore_index=True)

    sns.boxplot(
        data=df_region,
        x="x_custom",
        y="P-E Anomaly",
        hue="Hue_dummy",
        dodge=False,
        palette=colors_for_sss,
        showmeans=True,
        meanline=True,
        showfliers=False,
        order=x_order_global,
        width=0.7,
        legend=False,
        ax=ax,
    )

    ax.set_xticks(range(len(x_order_global)))
    if show_xticklabels:
        ax.set_xticklabels(xticks_labels_global, rotation=45, ha="center", fontsize=10)
        ax.tick_params(
            axis="x", which="both", bottom=False, top=False, labelbottom=True
        )
    else:
        ax.set_xticklabels([])
        ax.tick_params(
            axis="x", which="both", bottom=False, top=False, labelbottom=False
        )

    ax.tick_params(axis="y", labelsize=9)
    ax.set_ylabel("P–E anomaly (mm/month)", fontsize=10)
    ax.set_xlabel("")
    ax.grid(axis="y", linestyle="--", alpha=0.6)

    ax.text(
        1.05,
        0.5,
        region_name,
        transform=ax.transAxes,
        fontsize=10,
        rotation=270,
        va="center",
        ha="left",
    )

    if show_group_labels:
        max_index = max(1, len(x_order_global) - 1)
        for lag, center in group_centers_global.items():
            center_norm = center / max_index
            ax.text(
                center_norm,
                1.08,
                f"Lag {lag}",
                transform=ax.transAxes,
                ha="center",
                va="bottom",
                fontsize=10,
            )

    if subplot_label:
        ax.text(
            1.05,
            1.0,
            subplot_label,
            transform=ax.transAxes,
            fontsize=10,
            fontweight="bold",
            ha="right",
            va="top",
            bbox=dict(
                facecolor="white", edgecolor="none", alpha=0.6, boxstyle="round,pad=0.2"
            ),
            clip_on=False,
        )


subplot_labels = ["a)", "b)", "c)", "d)"]

group_configs = [
    ("PE_anomalies_panels_group1.png", regiones_ordenadas[:4], subplot_labels),
    ("PE_anomalies_panels_group2.png", regiones_ordenadas[4:], subplot_labels),
]

for filename, region_subset, labels_subset in group_configs:
    if not region_subset:
        continue

    fig, axes = plt.subplots(len(region_subset), 1, figsize=(9, 12.3), sharex=True)
    if len(region_subset) == 1:
        axes = [axes]

    for idx, (ax, region_name, subplot_label) in enumerate(
        zip(axes, region_subset, labels_subset)
    ):
        show_xticklabels = idx == len(region_subset) - 1
        show_group_labels = idx == 0
        plot_region_panel(
            ax,
            region_name,
            subplot_label,
            show_xticklabels=show_xticklabels,
            show_group_labels=show_group_labels,
        )

    fig.subplots_adjust(left=0.10, right=0.93, top=0.95, bottom=0.1, hspace=0.2)
    output_path = os.path.join(output_dir, filename)
    plt.savefig(output_path, dpi=600)
    plt.show()
    plt.close(fig)
    print(f"✅ Saved: {output_path}")

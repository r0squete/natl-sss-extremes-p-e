#!/usr/bin/env python
# Created by arosquete on 2025-05-22

import os
import string

import cartopy.crs as ccrs
import cartopy.feature as cfeature
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd
import xarray as xr
from cartopy.mpl.ticker import LatitudeFormatter, LongitudeFormatter
from scipy.stats import ttest_1samp

csv_eventos_altos = "/home/rosquete/Documents/TFM/Data/data_public/sss/fechas_mensuales_eventos_sss_altos.csv"
csv_eventos_bajos = "/home/rosquete/Documents/TFM/Data/data_public/sss/fechas_mensuales_eventos_sss_bajos.csv"
ds_mask = xr.open_dataset(
    "/home/rosquete/Documents/TFM/Data/data_public/masks/NATL_anual.nc"
)
ocean_mask = ds_mask["mask"]

time_var_name = "time"
lat_var_name = "lat"
lon_var_name = "lon"

start_year_str = "1985"
end_year_str = "2014"

resolucion_costas_mapa = "10m"
dpi_mapa = 600
map_amplio_extent = [-97, 10, 9.5, 55]
OUTPUT_DIR = "/home/rosquete/Documents/TFM/Data/data_public/output/"

anomalias_limpias_hg = xr.open_dataset(
    "/home/rosquete/Documents/TFM/Data/data_public/atm_var/HG_500_anom_detrended.nc"
)
hg_var_name = "HG"
anomalias_limpias_hg = anomalias_limpias_hg[hg_var_name]

anomalias_limpias_uivt = xr.open_dataset(
    "/home/rosquete/Documents/TFM/Data/data_public/atm_var/uq_1985_2014_anom_detrended.nc"
)
uivt_var_name = "uq"
anomalias_limpias_uivt = anomalias_limpias_uivt[uivt_var_name]

anomalias_limpias_vivt = xr.open_dataset(
    "/home/rosquete/Documents/TFM/Data/data_public/atm_var/vq_1985_2014_anom_detrended.nc"
)
vivt_var_name = "vq"
anomalias_limpias_vivt = anomalias_limpias_vivt[vivt_var_name]

anomalias_limpias_mslp = xr.open_dataset(
    "/home/rosquete/Documents/TFM/Data/data_public/atm_var/MSLP_1985-2014_anom_detrended.nc"
)
mslp_var_name = "mslp"
anomalias_limpias_mslp = anomalias_limpias_mslp[mslp_var_name]

anomalias_limpias_omega500 = xr.open_dataset(
    "/home/rosquete/Documents/TFM/Data/data_public/atm_var/OMEGA_WRF_ERA5_monthly_anom_detrended.nc"
)
omega500_var_name = "OMEGA"
anomalias_limpias_omega500 = anomalias_limpias_omega500[omega500_var_name]

anomalias_limpias_evap = xr.open_dataset(
    "/home/rosquete/Documents/TFM/Data/data_public/atm_var/E_WRF_1985-2014_anom_detrended.nc"
)
evap_var_name = "e"
anomalias_limpias_evap = anomalias_limpias_evap[evap_var_name]

df_eventos_altos = pd.read_csv(csv_eventos_altos, parse_dates=["event_date"])
fechas_altas_np = df_eventos_altos["event_date"].to_numpy()
df_eventos_bajos = pd.read_csv(csv_eventos_bajos, parse_dates=["event_date"])
fechas_bajas_np = df_eventos_bajos["event_date"].to_numpy()
print(
    f"Loaded {len(fechas_altas_np)} high-event dates and {len(fechas_bajas_np)} low-event dates."
)

lags_a_probar = [-3, -2, -1, 0, 1, 2, 3]

fechas_eventos_desfasadas_altos = {}
fechas_eventos_desfasadas_bajos = {}

print("\n--- Generating lagged dates for SSS events ---")
for lag_actual in lags_a_probar:
    lista_temporal_fechas_altas_lag = []
    for fecha_original_np in fechas_altas_np:
        fecha_original_timestamp = pd.Timestamp(fecha_original_np)
        lista_temporal_fechas_altas_lag.append(
            fecha_original_timestamp + pd.DateOffset(months=lag_actual)
        )

    fechas_eventos_desfasadas_altos[lag_actual] = pd.to_datetime(
        lista_temporal_fechas_altas_lag
    ).to_numpy()

    lista_temporal_fechas_bajas_lag = []
    for fecha_original_np in fechas_bajas_np:
        fecha_original_timestamp = pd.Timestamp(fecha_original_np)
        lista_temporal_fechas_bajas_lag.append(
            fecha_original_timestamp + pd.DateOffset(months=lag_actual)
        )

    fechas_eventos_desfasadas_bajos[lag_actual] = pd.to_datetime(
        lista_temporal_fechas_bajas_lag
    ).to_numpy()

    print(
        f"Lag {lag_actual} months: {len(fechas_eventos_desfasadas_altos[lag_actual])} lagged high-event dates, {len(fechas_eventos_desfasadas_bajos[lag_actual])} lagged low-event dates"
    )


def calcular_compuesto_mensual(
    data_array_anomalias: xr.DataArray,
    fechas_evento_dt64: np.ndarray,
    time_coord_name: str = "time",
    calcular_significancia: bool = False,
    popmean: float = 0.0,
    alpha_sig: float = 0.05,
) -> tuple[xr.DataArray, xr.DataArray | None]:
    if not isinstance(data_array_anomalias, xr.DataArray):
        raise TypeError(
            "El argumento 'data_array_anomalias' debe ser un xarray.DataArray."
        )
    if time_coord_name not in data_array_anomalias.coords:
        raise ValueError(
            f"El DataArray no tiene la coordenada de tiempo especificada: '{time_coord_name}'."
        )
    if not np.issubdtype(data_array_anomalias[time_coord_name].dtype, np.datetime64):
        raise TypeError(
            f"La coordenada '{time_coord_name}' en 'data_array_anomalias' no es de tipo datetime64[ns]."
        )
    if not isinstance(fechas_evento_dt64, np.ndarray) or not np.issubdtype(
        fechas_evento_dt64.dtype, np.datetime64
    ):
        raise TypeError(
            "'fechas_evento_dt64' must be a NumPy array with dtype datetime64[ns]."
        )
    if len(fechas_evento_dt64) == 0:
        raise ValueError("'fechas_evento_dt64' cannot be empty.")

    print(
        f"\n-- Computing composite for '{data_array_anomalias.name if data_array_anomalias.name else 'unknown variable'}' ---"
    )

    try:
        event_periods_anomes = pd.Series(
            pd.to_datetime(fechas_evento_dt64).to_period("M")
        ).unique()
    except Exception as e:
        raise ValueError(
            f"Error converting 'fechas_evento_dt64' to year-month periods: {e}"
        )

    atm_times_original_dt64 = data_array_anomalias[time_coord_name].data
    atm_periods_anomes = pd.to_datetime(atm_times_original_dt64).to_period("M")
    map_atm_periodo_a_timestamp = {}
    unique_atm_periods, indices_primera_ocurrencia = np.unique(
        atm_periods_anomes, return_index=True
    )
    for i, periodo in enumerate(unique_atm_periods):
        map_atm_periodo_a_timestamp[periodo] = atm_times_original_dt64[
            indices_primera_ocurrencia[i]
        ]
    timestamps_atm_seleccionados = []
    eventos_excluidos = 0
    for periodo_evento in event_periods_anomes:
        if periodo_evento in map_atm_periodo_a_timestamp:
            timestamps_atm_seleccionados.append(
                map_atm_periodo_a_timestamp[periodo_evento]
            )
        else:
            eventos_excluidos += 1
            print(
                f" INFO: Event with year-month {periodo_evento} not found in atmospheric data."
            )

    if eventos_excluidos > 0:
        print(
            f" INFO: {eventos_excluidos} of {len(event_periods_anomes)} unique year-month events were excluded (no matching data)."
        )

    if not timestamps_atm_seleccionados:
        raise ValueError(
            f"CRITICAL: None of the {len(event_periods_anomes)} unique year-month events matched the atmospheric dataset. Check the date ranges."
        )

    target_times_para_sel = xr.DataArray(
        np.unique(np.array(timestamps_atm_seleccionados, dtype="datetime64[ns]")),
        dims=[time_coord_name],
        name=time_coord_name,
    ).sortby(time_coord_name)

    if target_times_para_sel.size == 0 and len(event_periods_anomes) > 0:
        print(
            f" WARNING: target_times_para_sel is empty even though there were {len(event_periods_anomes)} event periods."
        )
        empty_coords = {
            d: data_array_anomalias[d]
            for d in data_array_anomalias.dims
            if d != time_coord_name
        }
        empty_composite = xr.DataArray(
            np.nan,
            coords=empty_coords,
            dims=[d for d in data_array_anomalias.dims if d != time_coord_name],
        )
        return empty_composite, None

    print(
        f" Using {target_times_para_sel.size} unique atmospheric timestamps for the composite."
    )

    try:
        datos_para_compuesto = data_array_anomalias.sel(
            {time_coord_name: target_times_para_sel}
        )
    except KeyError as e:
        raise KeyError(
            f"CRITICAL error in .sel(): {e}. Timestamps attempted (first 5): {target_times_para_sel.data[:5]}"
        )

    if datos_para_compuesto[time_coord_name].size == 0:
        raise ValueError(
            f"CRITICAL error: the selection returned zero records even though {target_times_para_sel.size} timestamps were requested. "
            f"Timestamps (first 5): {target_times_para_sel.data[:5]}"
        )

    compuesto_resultado = datos_para_compuesto.mean(dim=time_coord_name, skipna=True)

    p_values_map = None
    if calcular_significancia:
        if datos_para_compuesto[time_coord_name].size >= 2:
            time_axis_num = datos_para_compuesto.get_axis_num(time_coord_name)

            with np.errstate(invalid="ignore"):
                t_stat, p_values_data = ttest_1samp(
                    datos_para_compuesto.data,
                    popmean=popmean,
                    axis=time_axis_num,
                    nan_policy="omit",
                )

            coords_p_values = {
                k: v
                for k, v in datos_para_compuesto.coords.items()
                if k != time_coord_name
            }
            dims_p_values = [
                d for d in datos_para_compuesto.dims if d != time_coord_name
            ]

            p_values_map = xr.DataArray(
                p_values_data,
                coords=coords_p_values,
                dims=dims_p_values,
                name=f"p_value_{data_array_anomalias.name or 'variable'}",
            )
            print(
                f" Significance test computed. Mean p-value: {np.nanmean(p_values_data):.4f}"
            )
        else:
            print(
                f" INFO: Unable to compute significance; at least two data points are required for the t-test. Available samples: {datos_para_compuesto[time_coord_name].size}."
            )

            coords_p_values = {
                k: v
                for k, v in datos_para_compuesto.coords.items()
                if k != time_coord_name
            }
            dims_p_values = [
                d for d in datos_para_compuesto.dims if d != time_coord_name
            ]
            nan_data = np.full([len(coords_p_values[d]) for d in dims_p_values], np.nan)
            p_values_map = xr.DataArray(
                nan_data,
                coords=coords_p_values,
                dims=dims_p_values,
                name=f"p_value_{data_array_anomalias.name or 'variable'}_insufficient_data",
            )

    try:
        media_espacial_compuesto = compuesto_resultado.mean().item()
        print(
            f" Compuesto calculado. Media espacial del mapa: {media_espacial_compuesto:.4f}"
        )
    except Exception:
        if compuesto_resultado.size == 1 and np.all(np.isnan(compuesto_resultado.data)):
            print(" Composite computed. The result is a single NaN.")
        else:
            print(
                " Composite computed. Unable to compute the spatial mean (e.g., multidimensional NaN or no valid data)."
            )

    return compuesto_resultado, p_values_map


todos_los_compuestos_atmos_altos = {}
todos_los_compuestos_atmos_bajos = {}

todos_los_pvalores_atmos_altos = {}
todos_los_pvalores_atmos_bajos = {}

anomalias_atmos_dict = {
    "HG_500": anomalias_limpias_hg,
    "UIVT": anomalias_limpias_uivt,
    "VIVT": anomalias_limpias_vivt,
    "MSLP": anomalias_limpias_mslp,
    "OMEGA500": anomalias_limpias_omega500,
    "EVAP": anomalias_limpias_evap,
}

print("\n--- Calculating composites for all atmospheric variables and lags ---")

activar_calculo_significancia = False

for var_nombre, data_anom_var in anomalias_atmos_dict.items():
    print(f"\nProcessing variable: {var_nombre}")
    compuestos_altos_actual_var = {}
    compuestos_bajos_actual_var = {}

    p_valores_altos_actual_var = {}
    p_valores_bajos_actual_var = {}

    for lag_actual in lags_a_probar:
        print(f"  Lag: {lag_actual} months")
        compuesto_mean_altos, p_value_map_altos = calcular_compuesto_mensual(
            data_array_anomalias=data_anom_var,
            fechas_evento_dt64=fechas_eventos_desfasadas_altos[lag_actual],
            calcular_significancia=activar_calculo_significancia,
        )
        compuestos_altos_actual_var[lag_actual] = compuesto_mean_altos
        if activar_calculo_significancia and p_value_map_altos is not None:
            p_valores_altos_actual_var[lag_actual] = p_value_map_altos

        compuesto_mean_bajos, p_value_map_bajos = calcular_compuesto_mensual(
            data_array_anomalias=data_anom_var,
            fechas_evento_dt64=fechas_eventos_desfasadas_bajos[lag_actual],
            calcular_significancia=activar_calculo_significancia,
        )
        compuestos_bajos_actual_var[lag_actual] = compuesto_mean_bajos
        if activar_calculo_significancia and p_value_map_bajos is not None:
            p_valores_bajos_actual_var[lag_actual] = p_value_map_bajos

    todos_los_compuestos_atmos_altos[var_nombre] = compuestos_altos_actual_var
    todos_los_compuestos_atmos_bajos[var_nombre] = compuestos_bajos_actual_var

    if activar_calculo_significancia:
        todos_los_pvalores_atmos_altos[var_nombre] = p_valores_altos_actual_var
        todos_los_pvalores_atmos_bajos[var_nombre] = p_valores_bajos_actual_var

print("\n--- Atmospheric composites (optionally with p-values) computed ---")


def plot_escalar_composites_subplot(
    comp_bajos_var,
    comp_altos_var,
    lags=[0, 1, 2, 3],
    vmin_var=None,
    vmax_var=None,
    center_norm_var=None,
    cmap_var="RdBu_r",
    extent=[-97, 10, 9.5, 55],
    cb_format_decimals=2,
    titulo_cb_var="Variable Anomaly Unit",
    output_filename="composites_variable.png",
):
    nrows = len(lags)
    ncols = 2

    fig_width = 12
    fig_height = max(3.0 * nrows, 12)
    fig, axes = plt.subplots(
        nrows=nrows,
        ncols=ncols,
        figsize=(fig_width, fig_height),
        subplot_kw={"projection": ccrs.PlateCarree()},
    )

    if nrows == 1:
        axes = np.expand_dims(axes, axis=0)

    cf = None

    letras = list(string.ascii_lowercase)
    letra_idx = 0

    for i, lag in enumerate(lags):
        for j, evento in enumerate(["bajo", "alto"]):
            var_data_full = (
                comp_bajos_var[lag] if evento == "bajo" else comp_altos_var[lag]
            )

            var_data = var_data_full.sel(
                lat=slice(extent[2], extent[3]), lon=slice(extent[0], extent[1])
            )

            ax = axes[i, j]
            ax.set_extent(extent, crs=ccrs.PlateCarree())
            ax.add_feature(
                cfeature.COASTLINE.with_scale("10m"), linewidth=0.7, zorder=3
            )
            ax.add_feature(cfeature.BORDERS.with_scale("10m"), linewidth=0.5, zorder=3)

            gl = ax.gridlines(
                draw_labels=True, linestyle="--", alpha=0.5, color="gray", zorder=2
            )
            gl.top_labels = False
            gl.right_labels = False
            gl.xlabel_style = {"size": 11}
            gl.ylabel_style = {"size": 11}
            gl.xformatter = LongitudeFormatter(zero_direction_label=True)
            gl.yformatter = LatitudeFormatter()

            lon2d, lat2d = np.meshgrid(var_data.lon, var_data.lat)

            norm = None
            if center_norm_var is not None:
                norm = mcolors.CenteredNorm(halfrange=max(abs(vmin_var), abs(vmax_var)))

            cf = ax.contourf(
                lon2d,
                lat2d,
                var_data,
                levels=np.linspace(vmin_var, vmax_var, 31),
                cmap=cmap_var,
                extend="both",
                norm=norm,
                transform=ccrs.PlateCarree(),
            )

            ax.text(
                0.93,
                0.95,
                f"{letras[letra_idx]})",
                transform=ax.transAxes,
                fontsize=13,
                fontweight="bold",
                ha="right",
                va="top",
                bbox=dict(
                    facecolor="white",
                    edgecolor="none",
                    alpha=0.7,
                    boxstyle="round,pad=0.2",
                ),
            )
            letra_idx += 1

    cbar_ax = fig.add_axes([0.2, 0.04, 0.6, 0.02])
    cb = fig.colorbar(cf, cax=cbar_ax, orientation="horizontal", pad=0.08, aspect=30)
    cb.set_label(titulo_cb_var, fontsize="14")
    cb.ax.tick_params(labelsize=13)
    cb.ax.xaxis.set_major_formatter(
        mticker.FormatStrFormatter(f"%.{cb_format_decimals}f")
    )

    fig.subplots_adjust(
        left=0.05, right=0.95, top=0.96, bottom=0.1, hspace=0.18, wspace=0.12
    )

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    plt.savefig(os.path.join(OUTPUT_DIR, output_filename), dpi=600, bbox_inches="tight")
    plt.show()


mslp_bajos_comp_pa = todos_los_compuestos_atmos_bajos["MSLP"]
mslp_altos_comp_pa = todos_los_compuestos_atmos_altos["MSLP"]

mslp_bajos_comp_hpa = {lag: da / 100 for lag, da in mslp_bajos_comp_pa.items()}
mslp_altos_comp_hpa = {lag: da / 100 for lag, da in mslp_altos_comp_pa.items()}

plot_escalar_composites_subplot(
    comp_bajos_var=mslp_bajos_comp_hpa,
    comp_altos_var=mslp_altos_comp_hpa,
    lags=[0, 1, 2, 3],
    vmin_var=-2,
    vmax_var=2,
    cb_format_decimals=2,
    center_norm_var=0.0,
    cmap_var="RdBu_r",
    extent=[-97, 10, 9.5, 55],
    titulo_cb_var="MSLP Anomaly (hPa)",
    output_filename="composites_mslp.png",
)

hg_bajos_comp = todos_los_compuestos_atmos_bajos["HG_500"]
hg_altos_comp = todos_los_compuestos_atmos_altos["HG_500"]

plot_escalar_composites_subplot(
    comp_bajos_var=hg_bajos_comp,
    comp_altos_var=hg_altos_comp,
    lags=[0, 1, 2, 3],
    vmin_var=-29.0,
    vmax_var=29,
    cb_format_decimals=0,
    center_norm_var=0.0,
    cmap_var="RdBu_r",
    extent=[-97, 10, 9.5, 55],
    titulo_cb_var="HG_500 Anomaly (m)",
    output_filename="composites_HG_500.png",
)


factor_escala_omega = 100
omega_pressure_level = 5e04

omega_bajos_comp_raw_dict = todos_los_compuestos_atmos_bajos["OMEGA500"]
omega_altos_comp_raw_dict = todos_los_compuestos_atmos_altos["OMEGA500"]

omega_bajos_comp_final = {}
for lag, da_3d in omega_bajos_comp_raw_dict.items():
    da_2d_at_level = da_3d.sel(levels=omega_pressure_level)
    omega_bajos_comp_final[lag] = da_2d_at_level * factor_escala_omega

omega_altos_comp_final = {}
for lag, da_3d in omega_altos_comp_raw_dict.items():
    da_2d_at_level = da_3d.sel(levels=omega_pressure_level)
    omega_altos_comp_final[lag] = da_2d_at_level * factor_escala_omega

plot_escalar_composites_subplot(
    comp_bajos_var=omega_bajos_comp_final,
    comp_altos_var=omega_altos_comp_final,
    lags=[0, 1, 2, 3],
    vmin_var=-1.5,
    vmax_var=1.5,
    cmap_var="PRGn_r",
    extent=[-97, 10, 9.5, 55],
    titulo_cb_var="Omega_500 Anomaly (10⁻² Pa/s)",
    output_filename="composites_omega.png",
    center_norm_var=True,
    cb_format_decimals=2,
)


def plot_ivt_composites_subplot_quiver(
    comp_bajos_uivt,
    comp_altos_uivt,
    comp_bajos_vivt,
    comp_altos_vivt,
    lags=[0, 1, 2, 3],
    vmin_ivt=0,
    vmax_ivt=60,
    cmap_ivt="YlGn",
    extent=[-97, 10, 9.5, 55],
    titulo_cb="Vertical Integrated Moisture Flux Anomaly (kg m-1 s-1)",
    quiver_scale=50,
    quiver_width=0.002,
    quiver_spacing=30,
):
    nrows = len(lags)
    ncols = 2

    fig_width = 12
    fig_height = max(3.0 * nrows, 12)
    fig, axes = plt.subplots(
        nrows=nrows,
        ncols=ncols,
        figsize=(fig_width, fig_height),
        subplot_kw={"projection": ccrs.PlateCarree()},
    )

    if nrows == 1:
        axes = np.expand_dims(axes, axis=0)

    cf = None

    letras = list(string.ascii_lowercase)
    letra_idx = 0

    for i, lag in enumerate(lags):
        for j, evento in enumerate(["bajo", "alto"]):
            u_data_full = (
                comp_bajos_uivt[lag] if evento == "bajo" else comp_altos_uivt[lag]
            )
            v_data_full = (
                comp_bajos_vivt[lag] if evento == "bajo" else comp_altos_vivt[lag]
            )

            u_data = u_data_full.sel(lat=slice(extent[2], extent[3]))
            v_data = v_data_full.sel(lat=slice(extent[2], extent[3]))

            u_data = u_data.sel(lon=slice(extent[0], extent[1]))
            v_data = v_data.sel(lon=slice(extent[0], extent[1]))

            ivt_magnitude = np.sqrt(u_data**2 + v_data**2)

            ax = axes[i, j]
            ax.set_extent(extent, crs=ccrs.PlateCarree())
            ax.add_feature(
                cfeature.COASTLINE.with_scale("10m"), linewidth=0.7, zorder=3
            )
            ax.add_feature(cfeature.BORDERS.with_scale("10m"), linewidth=0.5, zorder=3)

            gl = ax.gridlines(
                draw_labels=True, linestyle="--", alpha=0.5, color="gray", zorder=2
            )
            gl.top_labels = False
            gl.right_labels = False
            gl.xlabel_style = {"size": 11}
            gl.ylabel_style = {"size": 11}
            gl.xformatter = LongitudeFormatter(zero_direction_label=True)
            gl.yformatter = LatitudeFormatter()

            lon2d, lat2d = np.meshgrid(u_data.lon, u_data.lat)

            cf = ax.contourf(
                lon2d,
                lat2d,
                ivt_magnitude,
                levels=np.linspace(vmin_ivt, vmax_ivt, 11),
                cmap=cmap_ivt,
                extend="max",
                transform=ccrs.PlateCarree(),
            )

            ax.quiver(
                lon2d[::quiver_spacing, ::quiver_spacing],
                lat2d[::quiver_spacing, ::quiver_spacing],
                u_data.data[::quiver_spacing, ::quiver_spacing],
                v_data.data[::quiver_spacing, ::quiver_spacing],
                scale=quiver_scale,
                scale_units="inches",
                width=quiver_width,
                transform=ccrs.PlateCarree(),
                zorder=4,
            )

            ax.text(
                0.93,
                0.95,
                f"{letras[letra_idx]})",
                transform=ax.transAxes,
                fontsize=13,
                fontweight="bold",
                ha="right",
                va="top",
                bbox=dict(
                    facecolor="white",
                    edgecolor="none",
                    alpha=0.7,
                    boxstyle="round,pad=0.2",
                ),
            )
            letra_idx += 1

    cbar_ax = fig.add_axes([0.2, 0.04, 0.6, 0.02])
    cb = fig.colorbar(cf, cax=cbar_ax, orientation="horizontal", aspect=30)
    cb.set_label(titulo_cb, fontsize="14")
    cb.ax.tick_params(labelsize=13)
    cb.ax.xaxis.set_major_formatter(mticker.FormatStrFormatter(f"%.{0}f"))

    fig.subplots_adjust(
        left=0.05, right=0.95, top=0.96, bottom=0.1, hspace=0.18, wspace=0.12
    )

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    plt.savefig(
        os.path.join(OUTPUT_DIR, "composites_ivt.png"), dpi=600, bbox_inches="tight"
    )
    plt.show()


comp_bajos_uivt = todos_los_compuestos_atmos_bajos["UIVT"]
comp_altos_uivt = todos_los_compuestos_atmos_altos["UIVT"]

comp_bajos_vivt = todos_los_compuestos_atmos_bajos["VIVT"]
comp_altos_vivt = todos_los_compuestos_atmos_altos["VIVT"]

plot_ivt_composites_subplot_quiver(
    comp_bajos_uivt=comp_bajos_uivt,
    comp_altos_uivt=comp_altos_uivt,
    comp_bajos_vivt=comp_bajos_vivt,
    comp_altos_vivt=comp_altos_vivt,
    lags=[0, 1, 2, 3],
    vmin_ivt=0,
    vmax_ivt=60,
    cmap_ivt="YlGn",
    extent=[-97, 10, 9.5, 55],
    titulo_cb="Vertical Integrated Moisture Flux Anomaly (kg m-1 s-1)",
    quiver_scale=50,
    quiver_width=0.002,
    quiver_spacing=30,
)


def plot_evap_composites_subplot(
    comp_bajos,
    comp_altos,
    mask_data,
    lags=[-1, 0, 1],
    vmin=-0.2,
    vmax=0.2,
    cmap="Spectral_r",
    extent=[-70, 0, 5, 45],
    titulo_cb="Evaporation anomalies (mm/day)",
):
    nrows = len(lags)
    fig, axes = plt.subplots(
        nrows=nrows,
        ncols=2,
        figsize=(12, 14),
        subplot_kw={"projection": ccrs.PlateCarree()},
    )

    if nrows == 1:
        axes = np.expand_dims(axes, axis=0)

    mask_lon2d, mask_lat2d = np.meshgrid(mask_data.lon.values, mask_data.lat.values)

    letras = list(string.ascii_lowercase)
    letra_idx = 0

    for i, lag in enumerate(lags):
        for j, evento in enumerate(["bajo", "alto"]):
            data = comp_bajos[lag] if evento == "bajo" else comp_altos[lag]
            ax = axes[i, j]
            ax.set_extent(extent)
            ax.add_feature(
                cfeature.COASTLINE.with_scale("10m"), linewidth=0.7, zorder=3
            )
            ax.add_feature(cfeature.BORDERS.with_scale("10m"), linewidth=0.5, zorder=3)

            gl = ax.gridlines(
                draw_labels=True, linestyle="--", alpha=0.5, color="gray", zorder=2
            )
            gl.top_labels = False
            gl.right_labels = False
            gl.xlabel_style = {"size": 11}
            gl.ylabel_style = {"size": 11}
            gl.xformatter = LongitudeFormatter(zero_direction_label=True)
            gl.yformatter = LatitudeFormatter()

            lon2d, lat2d = np.meshgrid(data.lon, data.lat)
            cf = ax.contourf(
                lon2d,
                lat2d,
                data,
                levels=np.linspace(vmin, vmax, 51),
                cmap=cmap,
                extend="both",
                transform=ccrs.PlateCarree(),
            )

            ax.contour(
                mask_lon2d,
                mask_lat2d,
                mask_data.data,
                levels=[0.5],
                colors="black",
                linewidths=1.2,
                transform=ccrs.PlateCarree(),
                zorder=5,
            )

            ax.text(
                0.93,
                0.95,
                f"{letras[letra_idx]})",
                transform=ax.transAxes,
                fontsize=13,
                fontweight="bold",
                ha="right",
                va="top",
                bbox=dict(
                    facecolor="white",
                    edgecolor="none",
                    alpha=0.7,
                    boxstyle="round,pad=0.2",
                ),
            )
            letra_idx += 1

    cbar_ax = fig.add_axes([0.25, 0.08, 0.5, 0.02])
    cb = fig.colorbar(cf, cax=cbar_ax, orientation="horizontal")
    cb.set_label(titulo_cb, fontsize="14")
    cb.ax.tick_params(labelsize=13)
    cb.ax.xaxis.set_major_formatter(mticker.FormatStrFormatter(f"%.{2}f"))

    plt.tight_layout(rect=[0, 0.1, 1, 1])

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    plt.savefig(os.path.join(OUTPUT_DIR, "evap_composites.png"), dpi=600)
    plt.show()


plot_evap_composites_subplot(
    todos_los_compuestos_atmos_bajos["EVAP"],
    todos_los_compuestos_atmos_altos["EVAP"],
    ocean_mask,
)

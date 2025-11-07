#!/usr/bin/env python
# Created by arosquete on 2025-05-19

from pathlib import Path

import cartopy.crs as ccrs
import cartopy.feature as cfeature
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import xarray as xr
from cartopy.mpl.ticker import LatitudeFormatter, LongitudeFormatter
from scipy.stats import linregress

PROJECT_ROOT = Path("/home/rosquete/Documents/TFM/Data/data_public")
SSS_FIG_ROOT = PROJECT_ROOT / "output"
SSS_DATA_FILE = PROJECT_ROOT / "sss" / "EN_sss_1900_2014_procesed.nc"
REF_DATA_FILE = PROJECT_ROOT / "masks" / "2004050300.nc"
SSS_FIG_ROOT.mkdir(parents=True, exist_ok=True)

ds = xr.open_dataset(SSS_DATA_FILE)
sss = ds["salinity"].sel(time=slice("1985-01-01", "2014-12-31"))


if "depth" in sss.dims and sss.sizes["depth"] == 1:
    sss = sss.squeeze("depth", drop=True)


sss_filtrado = sss.where(sss > 0)

sss_media_espacial = sss_filtrado.mean(dim=["lat", "lon"], skipna=True)
sss_mean = sss_filtrado.mean(dim=["time"])

media_general = sss_media_espacial.mean(dim="time").item()
print(f"🔹 SSS Media General (1985–2014): {media_general:.3f}")

clim_from_timeseries = sss_media_espacial.groupby("time.month").mean(dim="time")
std_clim_from_timeseries = sss_media_espacial.groupby("time.month").std(dim="time")

clim = clim_from_timeseries
std_clim = std_clim_from_timeseries

amplitud = (clim.max() - clim.min()).item()
mes_max = clim.month[clim.argmax(dim="month")].item()
mes_min = clim.month[clim.argmin(dim="month")].item()

print(f"🔹 Seasonal amplitude: {amplitud:.3f}")
print(f"🔹 Month of maximum SSS: {mes_max}  |  Month of minimum SSS: {mes_min}")
print("🔹 Monthly climatology:")
for mes_num_actual in clim.month.values:
    valor = clim.sel(month=mes_num_actual).item()
    std_dev = std_clim.sel(month=mes_num_actual).item()
    print(f"   - Month {mes_num_actual:02d}: {valor:.3f} ± {std_dev:.3f}")

tiempo = sss_media_espacial["time"]
anhos = tiempo.dt.year + (tiempo.dt.month - 1) / 12

sss_valores_numpy = sss_media_espacial.data
anhos_numpy = anhos.data

mascara_valida = np.isfinite(sss_valores_numpy) & np.isfinite(anhos_numpy)

x_para_regresion = anhos_numpy[mascara_valida]
y_para_regresion = sss_valores_numpy[mascara_valida]

if len(x_para_regresion) > 1:
    pendiente, intercepto, r_valor, p_valor, stderr = linregress(
        x_para_regresion, y_para_regresion
    )
    print(f"🔹 Linear trend: {pendiente:.4f} year⁻¹")
    print(f"   - Intercept: {intercepto:.3f}")
    print(f"   - R²: {r_valor**2:.4f}")
    print(f"   - p-value: {p_valor:.4f}")
    print(f"   - Standard error of the slope: {stderr:.4f}")
else:
    print("🔹 Linear trend: Not enough valid data to compute the trend.")


mask_ds = xr.open_dataset(REF_DATA_FILE)

lat = mask_ds["lat"]
lon = mask_ds["lon"]
lat_min = float(lat.min())
lat_max = float(lat.max())
lon_min = float(lon.min())
lon_max = float(lon.max())

lat_margin = (lat_max - lat_min) * 0.1
lon_margin = (lon_max - lon_min) * 0.1
lat_min_ext = lat_min - lat_margin
lat_max_ext = lat_max + lat_margin
lon_min_ext = lon_min - lon_margin
lon_max_ext = lon_max + lon_margin


def plot_mean_sss_map(
    sss_mean_field: xr.DataArray,
    extent: tuple[float, float, float, float],
    output_name: str,
    colorbar_label: str,
) -> None:
    """Helper to plot the mean SSS map for a given geographic extent."""
    fig, ax = plt.subplots(
        figsize=(10, 6), subplot_kw={"projection": ccrs.PlateCarree()}
    )
    ax.set_facecolor("white")
    ax.set_extent(extent, crs=ccrs.PlateCarree())

    mesh = sss_mean_field.squeeze().plot.pcolormesh(
        ax=ax,
        transform=ccrs.PlateCarree(),
        cmap="viridis",
        add_colorbar=True,
        zorder=1,
        shading="auto",
        cbar_kwargs={"label": colorbar_label, "shrink": 0.5, "pad": 0.02},
    )
    mesh.colorbar.ax.tick_params(labelsize=10)
    mesh.colorbar.set_label(colorbar_label, size=12)
    ax.set_title("")

    ax.add_feature(
        cfeature.LAND, facecolor="gainsboro", edgecolor="black", linewidth=0.2, zorder=2
    )
    coastline = cfeature.NaturalEarthFeature(
        "physical", "coastline", "10m", edgecolor="black", facecolor="none"
    )
    borders = cfeature.NaturalEarthFeature(
        "cultural",
        "admin_0_boundary_lines_land",
        "10m",
        edgecolor="black",
        facecolor="none",
        linestyle="-",
    )
    ax.add_feature(coastline, linewidth=0.8, zorder=3)
    ax.add_feature(borders, linewidth=0.5, zorder=3)

    gridlines = ax.gridlines(
        draw_labels=True, linewidth=0.3, color="gray", alpha=0.4, linestyle="--"
    )
    gridlines.top_labels = gridlines.right_labels = False
    gridlines.xformatter = LongitudeFormatter()
    gridlines.yformatter = LatitudeFormatter()

    plt.tight_layout()
    plt.savefig(SSS_FIG_ROOT / output_name, dpi=600, bbox_inches="tight")
    plt.show()


plot_mean_sss_map(
    sss_mean,
    (lon_min_ext, lon_max_ext, lat_min_ext, lat_max_ext),
    "figure1_mean_sss_map_adimensional.png",
    "SSS",
)

plot_mean_sss_map(
    sss_mean,
    (-70, 5, 10, 40),
    "figure1_mean_sss_map_centrado.png",
    "SSS",
)


clim_vals = clim.data
std_vals = std_clim.data
months_num_axis = clim.month.values

fig, ax = plt.subplots(figsize=(10, 5.5))

ax.fill_between(
    months_num_axis,
    clim_vals - std_vals,
    clim_vals + std_vals,
    color="skyblue",
    alpha=0.4,
    label="± 1 Standard Deviation",
)

ax.plot(
    months_num_axis,
    clim_vals,
    color="darkslateblue",
    marker="o",
    linewidth=2.0,
    markersize=6,
    label="Climatological Monthly Mean SSS",
)

ax.set_xticks(months_num_axis)
ax.set_xticklabels(
    ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
)
ax.set_xlabel("Month", fontsize=12)
ax.set_ylabel("SSS", fontsize=12)

ax.legend(fontsize=12, loc="best")
ax.tick_params(labelsize=12)
ax.yaxis.set_major_formatter(mticker.FormatStrFormatter("%.2f"))

min_val_plot_real_scale = np.min(clim_vals - std_vals) - 0.02
max_val_plot_real_scale = np.max(clim_vals + std_vals) + 0.02
if not (np.isnan(min_val_plot_real_scale) or np.isnan(max_val_plot_real_scale)):
    ax.set_ylim(min_val_plot_real_scale, max_val_plot_real_scale)


plt.tight_layout()
plt.savefig(SSS_FIG_ROOT / "figure2_seasonal_cycle_EN.png", dpi=300)
plt.show()


if not (np.isnan(pendiente) or np.isnan(intercepto)):
    trend_line_values = intercepto + pendiente * anhos.data
else:
    trend_line_values = np.full_like(anhos.data, np.nan, dtype=float)

fig, ax = plt.subplots(figsize=(12, 5.5))

ax.plot(
    tiempo.data,
    sss_media_espacial.data,
    color="darkslateblue",
    linewidth=1.2,
    marker=".",
    markersize=4,
    alpha=0.8,
    label="Monthly Mean SSS",
)

if not np.all(np.isnan(trend_line_values)):
    trend_label = f"Linear Trend ({pendiente:.4f} year⁻¹"
    if not np.isnan(p_valor):
        if p_valor < 0.001:
            trend_label += ", p < 0.001"  # noqa: E701
        elif p_valor < 0.01:
            trend_label += ", p < 0.01"  # noqa: E701
        elif p_valor < 0.05:
            trend_label += ", p < 0.05"  # noqa: E701
        else:
            trend_label += f", p = {p_valor:.3f}"
    trend_label += ")"
    ax.plot(
        tiempo.data,
        trend_line_values,
        color="crimson",
        linestyle="--",
        linewidth=2.2,
        label=trend_label,
    )

ax.set_xlabel("Year", fontsize=13)
ax.set_ylabel("SSS", fontsize=13)

ax.legend(fontsize=13, loc="best")
ax.tick_params(axis="x", labelsize=12)
ax.tick_params(axis="y", labelsize=12)

all_y_values_for_plot = np.concatenate((sss_media_espacial.data, trend_line_values))
valid_y_values = all_y_values_for_plot[np.isfinite(all_y_values_for_plot)]
if len(valid_y_values) > 0:
    plot_min_y = np.min(valid_y_values)
    plot_max_y = np.max(valid_y_values)
    y_range = plot_max_y - plot_min_y
    margin = y_range * 0.05 if y_range > 0 else 0.1
    ax.set_ylim(plot_min_y - margin, plot_max_y + margin)

plt.tight_layout()
plt.savefig(SSS_FIG_ROOT / "figure3_timeseries_trend_EN.png", dpi=600)
plt.show()

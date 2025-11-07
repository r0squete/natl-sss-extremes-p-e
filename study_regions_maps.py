#!/usr/bin/env python
# Created by arosquete on 2025-05-17

from __future__ import annotations

from pathlib import Path
from typing import Iterable

import cartopy.crs as ccrs
import cartopy.feature as cfeature
import matplotlib.pyplot as plt
import numpy as np
import xarray as xr
from cartopy.mpl.ticker import LatitudeFormatter, LongitudeFormatter
from matplotlib.colors import ListedColormap
from matplotlib.patches import Patch, Rectangle

DATA_ROOT = Path("/home/rosquete/Documents/TFM/Data/data_public/")
FIGURES_ROOT = Path("/home/rosquete/Documents/TFM/Data/data_public/output")
REFERENCE_FILE = DATA_ROOT / "masks" / "2004050300.nc"

REGION_MASKS = {
    "AZO": DATA_ROOT / "masks" / "AZO.nc",
    "MAD": DATA_ROOT / "masks" / "MAD.nc",
    "CVE": DATA_ROOT / "masks" / "CAV.nc",
    "CAN": DATA_ROOT / "masks" / "CAN.nc",
    "IP": DATA_ROOT / "masks" / "IP.nc",
    "CAM": DATA_ROOT / "masks" / "CAM.nc",
    "CAR": DATA_ROOT / "masks" / "CAR.nc",
    "ENA": DATA_ROOT / "masks" / "ENA.nc",
}

REGION_COLORS = {
    "AZO": "#377eb8",
    "MAD": "#4daf4a",
    "CVE": "#e41a1c",
    "CAN": "#984ea3",
    "IP": "#ff7f00",
    "CAM": "#ffff33",
    "CAR": "#f781bf",
    "ENA": "#a65628",
}


def open_dataset_or_fail(path: Path, description: str) -> xr.Dataset:
    """Open ``path`` and raise a friendly error if the file is missing."""
    try:
        return xr.open_dataset(path)
    except FileNotFoundError as exc:
        raise FileNotFoundError(f"File not found {description}: {path}") from exc


def compute_extent_from_coords(
    lat_values: np.ndarray, lon_values: np.ndarray, padding: float = 0.1
) -> tuple[float, float, float, float]:
    """Compute a geographic extent given latitude and longitude coordinates."""
    lat_min = float(np.min(lat_values))
    lat_max = float(np.max(lat_values))
    lon_min = float(np.min(lon_values))
    lon_max = float(np.max(lon_values))

    lat_pad = (lat_max - lat_min) * padding
    lon_pad = (lon_max - lon_min) * padding
    return lon_min - lon_pad, lon_max + lon_pad, lat_min - lat_pad, lat_max + lat_pad


def configure_map_axes(ax: plt.Axes, extent: Iterable[float]) -> None:
    """Configure coastlines, borders, and gridlines for a PlateCarree map."""
    ax.set_facecolor("white")
    ax.set_extent(extent, crs=ccrs.PlateCarree())
    ax.add_feature(
        cfeature.LAND, facecolor="gainsboro", edgecolor="black", linewidth=0.3, zorder=1
    )
    ax.add_feature(
        cfeature.NaturalEarthFeature(
            "physical", "coastline", "10m", edgecolor="black", facecolor="none"
        ),
        linewidth=0.8,
        zorder=3,
    )
    ax.add_feature(
        cfeature.NaturalEarthFeature(
            "cultural",
            "admin_0_boundary_lines_land",
            "10m",
            edgecolor="black",
            facecolor="none",
        ),
        linewidth=0.5,
        zorder=3,
    )
    gridlines = ax.gridlines(
        draw_labels=True, linewidth=0.4, color="gray", alpha=0.4, linestyle="--"
    )
    gridlines.top_labels = gridlines.right_labels = False
    gridlines.xformatter = LongitudeFormatter()
    gridlines.yformatter = LatitudeFormatter()


def plot_sink_regions() -> None:
    """Plot regional masks highlighting sink areas."""
    ref_ds = open_dataset_or_fail(REFERENCE_FILE, "de referencia")
    lat_coord = "lat" if "lat" in ref_ds.coords else "latitude"
    lon_coord = "lon" if "lon" in ref_ds.coords else "longitude"
    lat = ref_ds[lat_coord].values
    lon = ref_ds[lon_coord].values
    base_extent = compute_extent_from_coords(lat, lon, padding=0.0)
    plot_extent = compute_extent_from_coords(lat, lon, padding=0.1)

    fig, ax = plt.subplots(
        figsize=(10, 6), subplot_kw={"projection": ccrs.PlateCarree()}
    )
    configure_map_axes(ax, plot_extent)

    legend_handles: list[Patch] = []
    for region, mask_path in REGION_MASKS.items():
        mask_ds = open_dataset_or_fail(mask_path, f"for the mask for {region}")
        mask = mask_ds[list(mask_ds.data_vars)[0]]
        if "latitude" in mask.dims:
            mask = mask.rename({"latitude": "lat"})
        if "longitude" in mask.dims:
            mask = mask.rename({"longitude": "lon"})
        mask = mask.sortby(["lat", "lon"])
        mask = mask.astype(int)
        region_mask = mask.where(mask == 1)
        if region_mask.count().item() == 0:
            continue
        cmap = ListedColormap([REGION_COLORS[region]])
        region_mask.plot.pcolormesh(
            ax=ax,
            transform=ccrs.PlateCarree(),
            cmap=cmap,
            alpha=0.6,
            add_colorbar=False,
            zorder=2,
            shading="auto",
        )
        legend_handles.append(
            Patch(facecolor=REGION_COLORS[region], alpha=0.6, label=region)
        )

    if legend_handles:
        ax.legend(
            handles=legend_handles,
            loc="lower left",
            bbox_to_anchor=(1.02, 0),
            frameon=True,
        )

    rectangle = Rectangle(
        (base_extent[0], base_extent[2]),
        base_extent[1] - base_extent[0],
        base_extent[3] - base_extent[2],
        edgecolor="red",
        linewidth=1.5,
        facecolor="none",
        transform=ccrs.PlateCarree(),
        zorder=4,
    )
    ax.add_patch(rectangle)

    plt.tight_layout()
    output_path = FIGURES_ROOT / "mapa_sumideros.png"
    plt.savefig(output_path, dpi=600, bbox_inches="tight")
    print(f"Figura guardada: {output_path}")
    plt.show()


def main() -> None:
    FIGURES_ROOT.mkdir(parents=True, exist_ok=True)
    plot_sink_regions()


if __name__ == "__main__":
    main()

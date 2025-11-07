# Linking-Salinity-and-Precipitation-through-Moisture-Transport
Code for the study “Linking Salinity and Precipitation through Moisture Transport: A Study for the North Atlantic”: reproducible scripts for SSS anomalies, extreme-event detection, atmospheric composites, and regional P–E analysis (1985–2014).

## Data:
external NetCDF/CSV files are not included. Place them under data_public/ (or set DATA_ROOT).

## Quick start
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

## Usage (typical order)
python study_regions_maps.py
python sss_analysis.py
python sss_anomalies.py
python extreme_events_resume.py
python climate_index.py
python atm_var.py
python p_e_regional.py

## Data paths
Scripts expect ./data_public/. You can override once:

macOS/Linux: export DATA_ROOT=./data_public/

Windows (PowerShell): $Env:DATA_ROOT = ".\data_public\"

## Outputs
Written to ./data_public/output/ (adjust in scripts if needed).

## Citing
Please cite the repository and the related article (when available).

## License
MIT (see LICENSE).

https://doi.org/10.5281/zenodo.17550918

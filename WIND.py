import cdsapi
import numpy as np
from netCDF4 import Dataset
import pandas as pd
from pathlib import Path
from datetime import datetime


def create_cdsapirc(api_key: str):
    home = Path.home()
    file_path = home / ".cdsapirc"
    content = (
        "url: https://cds.climate.copernicus.eu/api\n"
        f"key: {api_key}\n")
    with open(file_path, "w") as f:
        f.write(content)
    print(f"Fichier créé : {file_path}")

def get_wind(lat, lon, dt: datetime):
    pressure_levels = {
        "0ft": "1000",
        "1000ft": "975",
        "2000ft": "950",
        "3000ft": "925",
        "4000ft": "900",
        "5000ft": "850",
        "6000ft": "825",
        "7000ft": "800",}
    output_file = "data/raw/temp/wind_era5.nc"
    create_cdsapirc("3fbb3d97-320a-43d3-bb61-43a3ab32216b")
    c = cdsapi.Client()
    c.retrieve(
        "reanalysis-era5-pressure-levels",
        {
            "product_type": "reanalysis",
            "variable": [
                "u_component_of_wind",
                "v_component_of_wind",
            ],
            "pressure_level": list(pressure_levels.values()),
            "year": dt.strftime("%Y"),
            "month": dt.strftime("%m"),
            "day": dt.strftime("%d"),
            "time": dt.strftime("%H:%M"),
            "format": "netcdf",
            "area": [
                lat + 0.1,
                lon - 0.1,
                lat - 0.1,
                lon + 0.1,
            ],
        },
        output_file,
    )

    ds = Dataset(output_file)
    u_all = ds.variables["u"][0, :, :, :]
    v_all = ds.variables["v"][0, :, :, :]
    levels = ds.variables["pressure_level"][:]
    lats = ds.variables["latitude"][:]
    lons = ds.variables["longitude"][:]
    lat_idx = np.abs(lats - lat).argmin()
    lon_idx = np.abs(lons - lon).argmin()
    results = []
    for alt_label, plevel in pressure_levels.items():
        level_idx = np.abs(levels - int(plevel)).argmin()
        u_val = float(u_all[level_idx, lat_idx, lon_idx])
        v_val = float(v_all[level_idx, lat_idx, lon_idx])
        speed_ms = np.sqrt(u_val**2 + v_val**2)
        speed_kmh = speed_ms * 3.6
        direction_rad = np.arctan2(-u_val, -v_val)
        direction_deg = (np.degrees(direction_rad) + 360) % 360
        results.append({
            "wind_altitude": alt_label,
            "wind_speed": speed_kmh,
            "wind_direction": round(direction_deg),
        })
    ds.close()
    df = pd.DataFrame(results)
    return df


# =========================
# EXAMPLE USAGE
# =========================
box_lat = 43.475
box_lon = 3.8328
runway_lat = 43.572
runway_lon = 3.957
dt = datetime(2026, 3, 20, 9, 0)
df_wind_box = get_wind(box_lat, box_lon, dt)
df_wind_runway = get_wind(runway_lat, runway_lon, dt)
print(df_wind_box)
print(df_wind_runway)
import cdsapi
import numpy as np
from netCDF4 import Dataset

#.cdsapirc
#url: https://cds.climate.copernicus.eu/api
#key: 3fbb3d97-320a-43d3-bb61-43a3ab32216b

# =========================
# PARAMETRES
# =========================
lat = 43.475
lon = 3.8328

date = "2026-03-20"
time = "09:00"

# niveau pression correspondant ~1500 m
pressure_level = "850"  # hPa

output_file = "wind_era5.nc"

# =========================
# TELECHARGEMENT ERA5
# =========================
c = cdsapi.Client()

c.retrieve(
    "reanalysis-era5-pressure-levels",
    {
        "product_type": "reanalysis",
        "variable": [
            "u_component_of_wind",
            "v_component_of_wind",
        ],
        "pressure_level": pressure_level,
        "year": date[:4],
        "month": date[5:7],
        "day": date[8:10],
        "time": time,
        "format": "netcdf",
        "area": [
            lat + 0.1,  # N
            lon - 0.1,  # W
            lat - 0.1,  # S
            lon + 0.1,  # E
        ],
    },
    output_file,
)

# =========================
# LECTURE DES DONNEES
# =========================
ds = Dataset(output_file)

u = ds.variables["u"][0, 0, :, :]  # vent zonal
v = ds.variables["v"][0, 0, :, :]  # vent méridien

lats = ds.variables["latitude"][:]
lons = ds.variables["longitude"][:]

# trouver le point le plus proche
lat_idx = np.abs(lats - lat).argmin()
lon_idx = np.abs(lons - lon).argmin()

u_val = float(u[lat_idx, lon_idx])
v_val = float(v[lat_idx, lon_idx])

# =========================
# CALCUL VENT
# =========================

# vitesse (m/s → km/h)
speed_ms = np.sqrt(u_val**2 + v_val**2)
speed_kmh = speed_ms * 3.6

# direction (météo : d'où vient le vent)
direction_rad = np.arctan2(-u_val, -v_val)
direction_deg = (np.degrees(direction_rad) + 360) % 360

# =========================
# RESULTAT
# =========================
print(date, time)
print("Vent à 1500 m (850 hPa)")
print(f"Vitesse : {speed_kmh:.1f} km/h")
print(f"Direction : {direction_deg:.0f}° (d'où vient le vent)")
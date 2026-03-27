import cdsapi
import numpy as np
from netCDF4 import Dataset

#.cdsapirc
#url: https://cds.climate.copernicus.eu/api
#key: 3fbb3d97-320a-43d3-bb61-43a3ab32216b

from pathlib import Path

def create_cdsapirc(api_key: str):
    home = Path.home()
    file_path = home / ".cdsapirc"

    content = (
        "url: https://cds.climate.copernicus.eu/api\n"
        f"key: {api_key}\n"
    )

    with open(file_path, "w") as f:
        f.write(content)

    print(f"Fichier créé : {file_path}")
    # Home/.cdsapirc
    # url: https://cds.climate.copernicus.eu/api
    # key: 3fbb3d97-320a-43d3-bb61-43a3ab32216b


create_cdsapirc("3fbb3d97-320a-43d3-bb61-43a3ab32216b")

# =========================
# PARAMETRES
# =========================
lat = 43.475
lon = 3.8328

date = "2026-03-20"
time = "09:00"

 # niveaux correspondant aux altitudes demandées
pressure_levels = {
    "0ft": "1000",
    "1000ft": "975",
    "2000ft": "950",
    "3000ft": "925",
    "4000ft": "900",
    "5000ft": "850",
    "6000ft": "825",
    "7000ft": "800",
}

output_file = "data/raw/temp/wind_era5.nc"

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
        "pressure_level": list(pressure_levels.values()),
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

# ERA5 utilise 'pressure_level' comme nom de variable
u_all = ds.variables["u"][0, :, :, :]  # tous niveaux
v_all = ds.variables["v"][0, :, :, :]
# ERA5 utilise 'pressure_level' comme nom de variable
levels = ds.variables["pressure_level"][:]

print(levels)

lats = ds.variables["latitude"][:]
lons = ds.variables["longitude"][:]

# trouver le point le plus proche
lat_idx = np.abs(lats - lat).argmin()
lon_idx = np.abs(lons - lon).argmin()

print(date, time)

for alt_label, plevel in pressure_levels.items():
    # trouver l'index du niveau
    level_idx = np.where(levels == int(plevel))[0][0]

    u_val = float(u_all[level_idx, lat_idx, lon_idx])
    v_val = float(v_all[level_idx, lat_idx, lon_idx])

    # vitesse (m/s → km/h)
    speed_ms = np.sqrt(u_val**2 + v_val**2)
    speed_kmh = speed_ms * 3.6

    # direction (météo : d'où vient le vent)
    direction_rad = np.arctan2(-u_val, -v_val)
    direction_deg = (np.degrees(direction_rad) + 360) % 360

    print(f"\n{alt_label} (~{plevel} hPa)")
    print(f"  Vitesse : {speed_kmh:.1f} km/h")
    print(f"  Direction : {direction_deg:.0f}°")

# fermeture fichier
ds.close()
import requests

lat = 43.475
lon = 3.8328

date = "2026-03-21"
time_target = 14  # heure UTC

url = "https://archive-api.open-meteo.com/v1/archive"

params = {
    "latitude": lat,
    "longitude": lon,
    "start_date": date,
    "end_date": date,
    # Use supported levels (10m and 100m). Pressure levels may be unavailable in archive API.
    "hourly": ",".join([
        "windspeed_10m",
        "winddirection_10m",
        "windspeed_100m",
        "winddirection_100m",
    ]),
    "timezone": "UTC"
}

data = requests.get(url, params=params).json()

# debug: check API response
if "hourly" not in data:
    print("Erreur API:", data)
    raise Exception("Open-Meteo n'a pas retourné de données horaires")

times = data["hourly"]["time"]

# trouver l'heure
idx = next(i for i, t in enumerate(times) if f"T{time_target:02d}:00" in t)

# extraction
levels = {
    "10m": ("windspeed_10m", "winddirection_10m"),
    "100m": ("windspeed_100m", "winddirection_100m"),
}

print("Vent en altitude :\n")

for label, (s_key, d_key) in levels.items():
    speed = data["hourly"].get(s_key, [None])[idx]
    direction = data["hourly"].get(d_key, [None])[idx]

    print(f"{label}")

    if speed is None or direction is None:
        print("  Donnée indisponible")
    else:
        print(f"  Vitesse : {speed:.1f} km/h")
        print(f"  Direction : {direction:.0f}°")

    print()

print("Note: Open-Meteo archive API ne fournit pas toujours les niveaux pression (850/925/800 hPa).")
print("Utilisation ici des niveaux 10m et 100m. Pour altitude réelle, ERA5 reste nécessaire.")
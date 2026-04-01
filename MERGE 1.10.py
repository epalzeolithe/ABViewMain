# -------- CONFIG --------
SKIP_X4_EXPORT = False
SKIP_GNS3000_IMPORT = False
SKIP_IPHONE_IMPORT = False
SKIP_METAR = False
SKIP_WIND = False
CONSOLE_WINDOW = False

import os,re
from datetime import time as dt_time
from datetime import datetime, timedelta, timezone
from zoneinfo import ZoneInfo
import subprocess
from scipy.signal import butter, filtfilt
from pymediainfo import MediaInfo
import requests
import cdsapi
import numpy as np
from netCDF4 import Dataset
import pandas as pd
from pathlib import Path
from datetime import datetime
import sys
import gpxpy
from PyQt5.QtWidgets import QApplication, QTextEdit
from PyQt5.QtCore import QObject, pyqtSignal, QThread

# -------- SUBPROCESS STREAM (REAL-TIME) --------
def run_cmd_stream(cmd):
    print("Running:", " ".join(cmd))
    proc = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
        universal_newlines=True
    )

    for line in proc.stdout:
        print(line, end="")

    proc.wait()

    if proc.returncode != 0:
        raise subprocess.CalledProcessError(proc.returncode, cmd)

# -------- QT CONSOLE --------

class ConsoleStream(QObject):
    new_text = pyqtSignal(str)

    def write(self, text):
        self.new_text.emit(str(text))

    def flush(self):
        pass

class ConsoleWindow(QTextEdit):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("MERGE "+__version__+" - Console")
        self.setReadOnly(True)
        self.resize(900, 500)

    def append_text(self, text):
        self.moveCursor(self.textCursor().End)
        self.insertPlainText(text)
        self.ensureCursorVisible()

    def keyPressEvent(self, event):
        # Close on any key press
        QApplication.quit()

class Worker(QThread):
    finished_signal = pyqtSignal(bool)

    def run(self):
        try:
            main()
            print("\nAppuyez sur une touche dans la fenêtre pour fermer...")
            self.finished_signal.emit(True)
        except Exception as e:
            print("\nErreur :", e)
            print("Appuyez sur une touche dans la fenêtre pour fermer...")
            self.finished_signal.emit(False)

def get_last_two_insv_files(directory):
    pattern = re.compile(r"^VID_.*?(\d{3})\.insv$", re.IGNORECASE)

    files_with_index = []

    for f in os.listdir(directory):
        match = pattern.match(f)
        if match:
            index = int(match.group(1))
            files_with_index.append((index, f))

    # Trier par index croissant
    files_with_index.sort(key=lambda x: x[0])

    if not files_with_index:
        return None, "none.insv"

    if len(files_with_index) == 1:
        last_file = files_with_index[0][1]
        return last_file, "none.insv"

    # Avant-dernier et dernier
    second_last = files_with_index[-2][1]
    last = files_with_index[-1][1]

    return second_last, last

def get_last_GPS_log_file(directory):
    pattern = re.compile(r"^LOG.*?(\d{5})\.txt$", re.IGNORECASE)

    files_with_index = []

    for f in os.listdir(directory):
        match = pattern.match(f)
        if match:
            index = int(match.group(1))
            files_with_index.append((index, f))

    if not files_with_index:
        return "none.txt"

    # Tri par index croissant
    files_with_index.sort(key=lambda x: x[0])

    # Retourne le dernier fichier
    return files_with_index[-1][1]

from ver import __version__

# -------- INPUT FILES

SUBDIR="data/raw/"
TMP=SUBDIR+"temp/"
X4_INSV_1, X4_INSV_2 = get_last_two_insv_files(SUBDIR)
GPS_GNS3000=get_last_GPS_log_file(SUBDIR)
#X4_INSV_1 = "VID_20260320_131559_00_053.insv"
#X4_INSV_2 = "VID_20260320_131559_00_054.insv"
#GPS_GNS3000 = "LOG00005.TXT"
IPHONE_SENSORLOG = "sensorlog.csv"

WINDOW = 4 # taille moyenne glissante pour lissage GNS3000
WINDOW_ACCX4 = 50 # taille moyenne glissante pour lissage accéléros X4
GNS3000_PERIOD = 0.25 # 4 Hz
X4_DEC = 10 # 1000 Hz > 100 Hz
IPHONE_DEC = 5 # division données par 100 Hz > 20 Hz
GYROFLOW_BIN = "/Applications/Gyroflow.app/Contents/MacOS/gyroflow"
GYRO2BB = "ressources/gyro2bb-mac-arm64"
EXIFTOOL = "ressources/exiftool"
EXIFFMT = "ressources/gpx.fmt"
MAINDIR="/Users/drax/Down/ABViewMain/"
ACC_SCALE = 9.81 / 20234

def get_bundle_name_from_insv(path):
    name = os.path.basename(path)
    # attendu : VID_20260221_091717_00_050.insv
    parts = name.split("_")
    date = parts[1]  # 20260221
    time = parts[2]  # 091717
    return f"Vol_{date[:4]}_{date[4:6]}_{date[6:8]}.abv"

OUTPUT = "data/"+get_bundle_name_from_insv(X4_INSV_1)+"/merged_data.csv"


# ======================================================
# MP4 creation date (UTC)
# ======================================================
def get_mp4_creation_datetime(path):
    media_info = MediaInfo.parse(path)
    for track in media_info.tracks:
        if track.track_type == "General" and track.encoded_date:
            s = track.encoded_date.strip().replace("UTC", "").strip()
            dt = datetime.strptime(s, "%Y-%m-%d %H:%M:%S")
            return dt.replace(tzinfo=timezone.utc)
    raise RuntimeError(f"Date de création MP4 introuvable : {path}")

# -------- CONVERSION --------
def nmea_to_decimal(coord, direction):
    if not coord:
        return None

    deg = int(float(coord) / 100)
    minutes = float(coord) - deg * 100
    decimal = deg + minutes / 60

    if direction in ["S", "W"]:
        decimal *= -1
    return decimal

def nmea_time_to_time(nmea_time):
    nmea_time = float(nmea_time)
    hh = int(nmea_time // 10000)
    mm = int((nmea_time % 10000) // 100)
    ss = nmea_time % 100
    sec = int(ss)
    microsec = int((ss - sec) * 1_000_000)
    return dt_time(hour=hh, minute=mm, second=sec, microsecond=microsec)

def export_GYROFLOW_CLI(insv_path):
    CURRENT_PATH=os.getcwd()+"/"

    cmd = [GYROFLOW_BIN,
        str(CURRENT_PATH+SUBDIR+insv_path),"--export-metadata","3:"+str(CURRENT_PATH+TMP+insv_path+".cli.csv")]
    print(cmd)
    run_cmd_stream(cmd)

def export_GYRO2BB_gyro_acc_from_insv(insv_path):
    PATH=os.getcwd()+"/"
    cmd = [PATH+GYRO2BB,str(PATH+SUBDIR+insv_path)]
    print(cmd)
    print("Running GYRO2BB :", " ".join(cmd))
    run_cmd_stream(cmd)

    fn=insv_path+".csv"
    nf=PATH+SUBDIR+fn
    import shutil
    shutil.move(nf,TMP+fn)



def read_GYROFLOW_CLI_export_CSV(gyro):
    gf = pd.read_csv(gyro, low_memory=False)
    gf=gf[['timestamp_ms','org_quat_w','org_quat_x','org_quat_y','org_quat_z']]
    return gf

def read_GYRO2BB_CSV(INPUT_FILE):
    df = pd.read_csv(INPUT_FILE, skiprows=66)
    df[["accSmooth[0]", "accSmooth[1]", "accSmooth[2]"]] *= ACC_SCALE # passage en ms-2> g
    df["time"] *=1e-3 # passage en ms
    df = df.rename(columns={'time': 'timestamp_ms'})
    df[["gyroADC[0]", "gyroADC[1]", "gyroADC[2]"]] *= np.pi / 180 # passage en deg/s

    #mean_norm = np.linalg.norm(df[["accSmooth[0]", "accSmooth[1]", "accSmooth[2]"]].iloc[:500], axis=1).mean()
    #print("Gravity mean:", mean_norm) # sur 500 premières valeurs, fps 30s > 15 secondes
    return df

def get_datas_from_insv(insv):
    # insv > Gyroflow CLI > export CSV > cli_df
    if not SKIP_X4_EXPORT:
        export_GYROFLOW_CLI(insv)
        print("Export CLI done")
    else:
        print(".....Skipping CLI export")
    cli_df = read_GYROFLOW_CLI_export_CSV(TMP+insv+".cli.csv") # 30 fps

    # insv > BB tool > export csv > bb_df
    if not SKIP_X4_EXPORT:
        export_GYRO2BB_gyro_acc_from_insv(insv)
        print("Export BB done")
    else:
        print(".....Skipping BB export")
    bb_df = read_GYRO2BB_CSV(TMP+insv+".csv")

    # interpolation CLI datas car 30 FPS
    import numpy as np
    t_cli_df = cli_df["timestamp_ms"].to_numpy()
    t_bb_df = bb_df["timestamp_ms"].to_numpy()
    cli_df_interp = {}
    for col in cli_df.columns:
        if col != "timestamp_ms":
            cli_df_interp[col] = np.interp(t_bb_df, t_cli_df, cli_df[col])
    for k, v in cli_df_interp.items():
        bb_df[k] = v

    return bb_df

def get_datas_from_gyroflow_GUI_export_CSV(csvfile):

    # WARNING OLD FUNCTION

    af = pd.read_csv(csvfile, low_memory=False)
    print("Taille Données X4 avant réduction :", af.shape)
    xdf = af.iloc[::X4_DEC].reset_index(drop=True)
    xdf = xdf[xdf["timestamp_ms"] >= 0]  # filtrage
    xdf = xdf.reset_index(drop=True)
    print("Taille Données X4 après réduction :", xdf.shape)
    st = pd.to_datetime(X4_GYROFLOW_START)
    # xdf['timestamp'] = st + pd.to_timedelta(xdf['timestamp_ms']-xdf['timestamp_ms'][0], unit='ms')
    xdf['timestamp'] = st + pd.to_timedelta(xdf['timestamp_ms'], unit='ms')

    xdf["timestamp"] = xdf["timestamp"].dt.tz_convert("Etc/GMT-1")
    xdf['timestamp'] = xdf['timestamp'].dt.tz_localize(None)
    xdf["timestamp"] = xdf["timestamp"].astype("datetime64[us]")
    xdf_last_timestamp = xdf['timestamp_ms'].iloc[-1]
    xdf = xdf[['timestamp', 'org_acc_x', 'org_acc_y', 'org_acc_z',
               'org_quat_w', 'org_quat_x', 'org_quat_y', 'org_quat_z', 'timestamp_ms']]
    xdf = xdf.rename(columns={'org_acc_x': 'x4_acc_x', 'org_acc_y': 'x4_acc_y', 'org_acc_z': 'x4_acc_z',
                              'org_quat_w': 'x4_quat_w', 'org_quat_x': 'x4_quat_x', 'org_quat_y': 'x4_quat_y',
                              'org_quat_z': 'x4_quat_z'})
    return xdf

def get_datas_from_gns3000(log):
    if not SKIP_GNS3000_IMPORT:
        gdf = pd.DataFrame(columns=["timestamp", "lat", "lon", "alt", "speed", "heading"])
        with open(log) as f:
            for line in f:
                line = line.strip()
                parts = line.split(",")

                if line.startswith("$GNGGA"):
                    timestamp = nmea_time_to_time(parts[1])
                    lat = nmea_to_decimal(parts[2], parts[3])
                    lon = nmea_to_decimal(parts[4], parts[5])
                    alt = float(parts[9]) * 3.28084 if parts[9] else 0

                if line.startswith("$GNRMC"):
                    speed = round(float(parts[7]) * 1.852, 1)
                    heading = round(float(parts[8]), 1)
                    row = [timestamp, lat, lon, alt, speed, heading]
                    gdf.loc[len(gdf)] = row

        gdf = gdf.rename(columns={'lat': 'gps_lat', 'lon': 'gps_lon', 'alt': 'gps_alt', 'speed': 'gps_speed','heading': 'gps_heading'})
        # calcul vitesse Z


        # ajout date à gdf et décaler en GMT
        st1 = get_mp4_creation_datetime(SUBDIR+X4_INSV_1)
        #print(st1)
        offset = st1.replace(tzinfo=ZoneInfo("Europe/Paris")).utcoffset().total_seconds() / 3600
        # offset = idf['timestamp'][0].replace(tzinfo=ZoneInfo("Europe/Paris")).utcoffset().total_seconds() / 3600
        gdf["timestamp"] = pd.to_datetime(st1.strftime("%Y-%m-%d") + " " + gdf["timestamp"].astype(str),
                                          format="mixed") + pd.Timedelta(hours=offset)


        gdf['gps_fpm'] = np.gradient(gdf['gps_alt'], GNS3000_PERIOD)
        gdf['gps_fpm'] = gdf['gps_fpm'] * 60

        gdf.to_csv(TMP+"gps.csv", index=False, encoding="utf-8")
        return gdf
    else:
        gdf = pd.read_csv(TMP+"gps.csv", low_memory=False)
        return gdf

def get_datas_from_iphone_sensorlog(log):
    if not SKIP_IPHONE_IMPORT:
        af = pd.read_csv(log, low_memory=False)
        af = af.rename(columns={'loggingSample(N)': 'Milliseconds', 'loggingTime(txt)': 'UTC',
                                'locationLatitude(WGS84)': 'Latitude', 'locationLongitude(WGS84)': 'Longitude',
                                'locationAltitude(m)': 'Altitude', 'locationSpeed(m/s)': 'Speed',
                                'motionYaw(rad)': 'YawRad', 'motionPitch(rad)': 'PitchRad',
                                'motionRoll(rad)': 'RollRad', 'locationTrueHeading(°)': 'Heading'})
        af = af[['UTC', 'Milliseconds', 'Latitude', 'Longitude', 'Altitude', 'Speed', 'Heading',
                 'accelerometerAccelerationX(G)', 'accelerometerAccelerationY(G)', 'accelerometerAccelerationZ(G)',
                 'YawRad', 'RollRad', 'PitchRad',
                 'motionQuaternionX(R)', 'motionQuaternionY(R)', 'motionQuaternionZ(R)', 'motionQuaternionW(R)']]
        idf = af.iloc[::IPHONE_DEC].reset_index(drop=True)
        idf["timestamp"] = pd.to_datetime(idf["UTC"])
        idf['timestamp'] = idf['timestamp'].dt.tz_localize(None)
        idf = idf[['timestamp', 'Latitude', 'Longitude', 'Altitude', 'Speed', 'Heading',
                   'accelerometerAccelerationX(G)', 'accelerometerAccelerationY(G)', 'accelerometerAccelerationZ(G)',
                   'YawRad', 'RollRad', 'PitchRad',
                   'motionQuaternionX(R)', 'motionQuaternionY(R)', 'motionQuaternionZ(R)', 'motionQuaternionW(R)']]
        idf = idf.rename(columns={'Latitude': 'iphone_lat', 'Longitude': 'iphone_lon', 'Altitude': 'iphone_alt',
                                  'Speed': 'iphone_speed',
                                  'Heading': 'iphone_heading', 'accelerometerAccelerationX(G)': 'iphone_acc_x',
                                  'accelerometerAccelerationY(G)': 'iphone_acc_y',
                                  'accelerometerAccelerationZ(G)': 'iphone_acc_z', 'YawRad': 'iphone_yawrad',
                                  'RollRad': 'iphone_rollrad', 'PitchRad': 'iphone_pitchrad',
                                  'motionQuaternionX(R)': 'iphone_quat_x', 'motionQuaternionY(R)': 'iphone_quat_y',
                                  'motionQuaternionZ(R)': 'iphone_quat_z', 'motionQuaternionW(R)': 'iphone_quat_w'})
        idf['iphone_alt'] = idf['iphone_alt'] * 3.28084  # conversion en feet
        idf['iphone_speed'] = idf['iphone_speed'] * 3.6  # conversion en km/h
        idf.to_csv(TMP+"sensorlog.formatted.csv", index=True, encoding="utf-8")
        return idf
    else:
        print(".....Skipping Iphone Sensorlog direct parsing")
        idf = pd.read_csv(TMP+"sensorlog.formatted.csv", low_memory=False)
        return idf

def download_metar_history(icao, start, end):
    """
    Télécharge les METAR historiques depuis Ogimet.

    start / end : datetime UTC
    """

    begin = start.strftime("%Y%m%d%H%M")
    end_s = end.strftime("%Y%m%d%H%M")

    url = (
        "https://www.ogimet.com/cgi-bin/getmetar"
        f"?icao={icao}"
        f"&begin={begin}"
        f"&end={end_s}"
        "&lang=eng"
        "&header=yes"
    )

    print("Downloading:", url)

    headers = {
        "User-Agent": "Mozilla/5.0"
    }

    r = requests.get(url, headers=headers, timeout=30)
    print("HTTP status:", r.status_code)
    text = r.text

    from io import StringIO

    csv_buffer = StringIO(text)

    try:
        df = pd.read_csv(csv_buffer)
    except Exception:
        print("Returned data:")
        print(text[:500])
        raise

    # Build datetime column (Ogimet returns Spanish column names)
    # ESTACION,ANO,MES,DIA,HORA,MINUTO,PARTE
    df["time"] = pd.to_datetime(
        dict(
            year=df["ANO"],
            month=df["MES"],
            day=df["DIA"],
            hour=df["HORA"],
            minute=df["MINUTO"],
        ),
        utc=True,
    )
    # convert from UTC to Paris timezone
    df["time"] = df["time"].dt.tz_convert("Europe/Paris").dt.tz_localize(None)
    #df["time"] = df["time"].dt.tz_convert("UTC").dt.tz_localize(None)

    df = df.rename(columns={"PARTE": "metar"})

    df = df[["time", "metar"]].sort_values("time").reset_index(drop=True)

    return df


def find_metar_for_time(df, t):

    idx = df["time"].searchsorted(t)

    if idx == 0:
        return df.iloc[0]

    if idx >= len(df):
        return df.iloc[-1]

    before = df.iloc[idx - 1]
    after = df.iloc[idx]

    if abs(t - before.time) < abs(after.time - t):
        return before
    else:
        return after

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
# ADD WIND FUNCTION
# =========================
def add_wind(merged):

    box_lat = 43.475
    box_lon = 3.8328
    runway_lat = 43.572
    runway_lon = 3.957

    ts_local = pd.to_datetime(merged["timestamp"].iloc[0])
    ts_utc = ts_local.tz_localize("Europe/Paris").tz_convert("UTC")
    ts = ts_utc.floor("h").tz_localize(None)
    dt = ts.to_pydatetime()
    print(dt)

    df_wind_box = get_wind(box_lat, box_lon, dt)
    df_wind_runway = get_wind(runway_lat, runway_lon, dt)

    print(df_wind_box)
    print(df_wind_runway)

    def haversine_vec(lat1, lon1, lat2, lon2):
        lat1 = np.asarray(lat1, dtype=np.float64)
        lon1 = np.asarray(lon1, dtype=np.float64)
        lat2 = np.asarray(lat2, dtype=np.float64)
        lon2 = np.asarray(lon2, dtype=np.float64)
        R = 6371e3
        phi1 = np.radians(lat1)
        phi2 = np.radians(lat2)
        dphi = np.radians(lat2 - lat1)
        dlambda = np.radians(lon2 - lon1)
        a = np.sin(dphi / 2) ** 2 + np.cos(phi1) * np.cos(phi2) * np.sin(dlambda / 2) ** 2
        return 2 * R * np.arctan2(np.sqrt(a), np.sqrt(1 - a))

    def prepare_wind_df(df):
        df = df.copy()
        df["alt_ft"] = df["wind_altitude"].str.replace("ft", "").astype(int)
        return df.sort_values("alt_ft")

    df_wind_box_p = prepare_wind_df(df_wind_box)
    df_wind_runway_p = prepare_wind_df(df_wind_runway)

    lat_arr = pd.to_numeric(merged["gps_lat"], errors="coerce").to_numpy(dtype=float)
    lon_arr = pd.to_numeric(merged["gps_lon"], errors="coerce").to_numpy(dtype=float)
    alt_arr = pd.to_numeric(merged["gps_alt"], errors="coerce").to_numpy(dtype=float)

    d_box = haversine_vec(lat_arr, lon_arr, box_lat, box_lon)
    d_runway = haversine_vec(lat_arr, lon_arr, runway_lat, runway_lon)

    use_box = d_box < d_runway

    def extract_arrays(df):
        import numpy as np

        alt = np.asarray(df["alt_ft"], dtype=np.float64)
        spd = np.asarray(df["wind_speed"], dtype=np.float64)

        # conversion ultra-safe
        direction = pd.to_numeric(df["wind_direction"], errors="coerce").to_numpy()
        direction = np.asarray(direction, dtype=np.float64)

        return (
            alt,
            spd,
            np.deg2rad(direction)
        )

    alt_b, spd_b, dir_b = extract_arrays(df_wind_box_p)
    alt_r, spd_r, dir_r = extract_arrays(df_wind_runway_p)

    alt_clamped = np.clip(alt_arr, min(alt_b.min(), alt_r.min()), max(alt_b.max(), alt_r.max()))

    speed_box = np.interp(alt_clamped, alt_b, spd_b)
    speed_runway = np.interp(alt_clamped, alt_r, spd_r)

    def interp_dir(altitudes, dirs_rad, alt_vals):
        sin_comp = np.sin(dirs_rad)
        cos_comp = np.cos(dirs_rad)
        sin_i = np.interp(alt_vals, altitudes, sin_comp)
        cos_i = np.interp(alt_vals, altitudes, cos_comp)
        return (np.degrees(np.arctan2(sin_i, cos_i)) + 360) % 360

    dir_box = interp_dir(alt_b, dir_b, alt_clamped)
    dir_runway = interp_dir(alt_r, dir_r, alt_clamped)

    era5_speed = np.where(use_box, speed_box, speed_runway)
    era5_dir = np.where(use_box, dir_box, dir_runway)

    merged["era5_wind_speed"] = era5_speed
    merged["era5_wind_direction"] = era5_dir

    return merged

# =========================
# ADD IAS FUNCTION
# =========================
def add_ias(merged):

    # arrays
    speed = merged["gps_speed"].to_numpy(dtype=float)
    heading = merged["gps_heading"].to_numpy(dtype=float)
    wind_speed = merged["era5_wind_speed"].to_numpy(dtype=float)
    wind_dir = merged["era5_wind_direction"].to_numpy(dtype=float)

    # handle NaN safely
    valid = ~np.isnan(speed) & ~np.isnan(heading) & ~np.isnan(wind_speed) & ~np.isnan(wind_dir)

    # init result
    ias = np.full_like(speed, np.nan)

    # compute only valid values
    rel_angle = np.radians(wind_dir[valid] - heading[valid])

    # wind direction is FROM, so headwind positive when cos(angle) > 0
    headwind = wind_speed[valid] * np.cos(rel_angle)

    # condition: only apply wind correction if speed >= 50
    valid_fast = valid.copy()
    valid_fast[valid] = speed[valid] >= 50

    # default: IAS = ground speed
    ias[valid] = speed[valid]

    # apply correction only for fast points
    ias[valid_fast] = speed[valid_fast] - headwind[ speed[valid] >= 50 ]

    merged["gps_ias"] = ias

    return merged

def export_EXIFTOOL_GPX_from_insv(insv_path):
    PATH = os.getcwd() + "/"
    input_file = PATH  + SUBDIR + insv_path
    output_file = PATH + TMP + insv_path + ".gpx"
    cmd = [
        PATH + EXIFTOOL,
        "-ee",
        "-p", EXIFFMT,
        input_file]
    print("Running EXIFTOOL:", " ".join(cmd))
    with open(output_file, "w") as f:
        subprocess.run(cmd, stdout=f, check=True)
    print("GPX créé:", output_file)

def read_EXIFTOOL_GPX(gpx_file):
    with open(gpx_file, "r") as f:
        gpx = gpxpy.parse(f)

    data = []

    for track in gpx.tracks:
        for segment in track.segments:
            for p in segment.points:
                data.append({
                    "timestamp": p.time.strftime("%Y-%m-%d %H:%M:%S") if p.time else None,
                    "gps_lat": p.latitude,
                    "gps_lon": p.longitude,
                    "gps_alt": round(p.elevation * 3.28084,0)
                })

    df = pd.DataFrame(data)

    # Convert UTC string to datetime, then to local time (Europe/Paris)
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
    df["timestamp"] = df["timestamp"].dt.tz_convert("Europe/Paris").dt.tz_localize(None)

    # ---- Detect gaps (>= 2 seconds) ----
    df = df.sort_values("timestamp").reset_index(drop=True)

    # Remove duplicate timestamps (can happen with EXIFTOOL GPX)
    before = len(df)
    df = df.drop_duplicates(subset="timestamp", keep="first")
    after = len(df)
    if before != after:
        print(f"Points dupliqués supprimés: {before - after}")
    dt_sec = df["timestamp"].diff().dt.total_seconds()
    gaps = (dt_sec >= 2).sum()
    print(f"Nombre de trous GPS (>=2s): {gaps}")

    # ---- Reindex to 1 Hz timeline and interpolate ----
    df = df.set_index("timestamp")

    full_index = pd.date_range(
        start=df.index.min(),
        end=df.index.max(),
        freq="1s"
    )

    df = df.reindex(full_index)

    # Interpolate core GPS values
    df["gps_lat"] = df["gps_lat"].interpolate(method="linear")
    df["gps_lon"] = df["gps_lon"].interpolate(method="linear")
    df["gps_alt"] = df["gps_alt"].interpolate(method="linear")

    # Restore timestamp column
    df = df.reset_index().rename(columns={"index": "timestamp"})

    import numpy as np

    # Convert timestamp to datetime
    df["dt_obj"] = pd.to_datetime(df["timestamp"], errors="coerce")

    # Compute time difference in seconds
    df["dt"] = df["dt_obj"].diff().dt.total_seconds()

    # Haversine distance (meters)
    def haversine(lat1, lon1, lat2, lon2):
        R = 6371000.0
        lat1 = np.radians(lat1)
        lon1 = np.radians(lon1)
        lat2 = np.radians(lat2)
        lon2 = np.radians(lon2)
        dlat = lat2 - lat1
        dlon = lon2 - lon1
        a = np.sin(dlat/2)**2 + np.cos(lat1)*np.cos(lat2)*np.sin(dlon/2)**2
        return 2 * R * np.arctan2(np.sqrt(a), np.sqrt(1-a))

    # Compute distance between consecutive points
    df["dist"] = haversine(
        df["gps_lat"].shift(1), df["gps_lon"].shift(1),
        df["gps_lat"], df["gps_lon"]
    )

    # Speed in km/h
    df["gps_speed"] = (df["dist"] / df["dt"]) * 3.6

    # Clean
    df["gps_speed"] = df["gps_speed"].fillna(0).round(0)

    # Compute heading (degrees from North)
    def compute_heading(lat1, lon1, lat2, lon2):
        lat1 = np.radians(lat1)
        lat2 = np.radians(lat2)
        dlon = np.radians(lon2 - lon1)

        x = np.sin(dlon) * np.cos(lat2)
        y = np.cos(lat1) * np.sin(lat2) - np.sin(lat1) * np.cos(lat2) * np.cos(dlon)

        heading = np.degrees(np.arctan2(x, y))
        return (heading + 360) % 360

    df["gps_heading"] = compute_heading(
        df["gps_lat"].shift(1),
        df["gps_lon"].shift(1),
        df["gps_lat"],
        df["gps_lon"]
    )

    df["gps_heading"] = df["gps_heading"].bfill().round(1)

    # Compute vertical speed (feet per minute) safely
    dt_safe = df["dt"].replace(0, np.nan)

    dz = df["gps_alt"].diff()
    df["gps_fpm"] = (dz / dt_safe) * 60

    # forward/backward fill small gaps
    df["gps_fpm"] = df["gps_fpm"].bfill().ffill()

    # Clean NaN / inf values
    df["gps_fpm"] = df["gps_fpm"].replace([np.inf, -np.inf], np.nan)
    df["gps_fpm"] = df["gps_fpm"].fillna(0).round(0)

    # Cap unrealistic GPS speeds (km/h)
    df.loc[df["gps_speed"] > 340, "gps_speed"] = 340
    df = df.drop(columns=["dt_obj", "dt", "dist"], errors="ignore")

    # nettoyage optionnel
    df = df.dropna(subset=["timestamp", "gps_lat", "gps_lon", "gps_alt", "gps_speed"])

    return df

def get_gps_from_insv(insv):
    # insv > BB tool > export gpx
    export_EXIFTOOL_GPX_from_insv(insv)
    print("Export GPX done")
    gbb_df = read_EXIFTOOL_GPX(TMP+insv+".gpx")
    gbb_df.to_csv(TMP+insv+".gpx.csv", index=True, encoding="utf-8")
    return gbb_df

def main():

    #print(TMP)
    Path(TMP).mkdir(parents=True, exist_ok=True)

    print("**********************************************************************")
    print("Start Merging for ABView : X4 datas + GPS datas + iPhone sensorlog ...")
    print("X4_INSV_1:", X4_INSV_1)
    print("X4_INSV_2:", X4_INSV_2)
    print("GPS_GNS3000:", GPS_GNS3000)
    print("IPHONE_SENSORLOG:", IPHONE_SENSORLOG)

    print("SKIP_X4_EXPORT",SKIP_X4_EXPORT)
    print("SKIP_GNS3000_IMPORT",SKIP_GNS3000_IMPORT)
    print("SKIP_IPHONE_IMPORT",SKIP_IPHONE_IMPORT)
    print("SKIP_METAR",SKIP_METAR)

    xloaded=x2loaded=gloaded=iloaded=False

    if os.path.exists(SUBDIR+GPS_GNS3000):
        print("Loading Datas from gns3000")
        gdf = get_datas_from_gns3000(SUBDIR+GPS_GNS3000)
        print("Datas from gns3000 loaded")
        gloaded=True
    else:
        print("GNS3000 log file not found")
        print(("Trying with GPX export"))
        if os.path.exists(SUBDIR + X4_INSV_1):
            gdf=get_gps_from_insv(X4_INSV_1)
            gloaded = True
        if os.path.exists(SUBDIR + X4_INSV_2):
            gdf2=get_gps_from_insv(X4_INSV_2)
            print("GPS TO STICH " + str(gdf["timestamp"].iloc[-1]) + " vs " + str(gdf2["timestamp"].iloc[0]))
            gdf = pd.concat([gdf, gdf2], axis=0, ignore_index=True)
        print("INSV>GPX>CSV done")


    if os.path.exists(SUBDIR+IPHONE_SENSORLOG):
        print("Loading Datas from iPhone sensorlog")
        idf = get_datas_from_iphone_sensorlog(SUBDIR+IPHONE_SENSORLOG)
        print("Datas from iPhone sensorlog loaded")
        iloaded=True
    else:
        print("iPhone sensorslog file not found")

    if os.path.exists(SUBDIR+X4_INSV_1):
        print("Loading Datas from .insv")
        xdf = get_datas_from_insv(X4_INSV_1)
        print("Datas from .insv loaded")
        xloaded=True
    else:
        print(".insv file not found")

    if os.path.exists(SUBDIR+X4_INSV_2):
        print("Loading Datas from 2nd insv")
        xdf2 = get_datas_from_insv(X4_INSV_2)
        print("Datas from 2nd .insv loaded")
        xloaded2=True
    else:
        print("2nd .insv file not found")

    if xloaded:
        xdf = xdf[xdf["timestamp_ms"] >= 0]
        xdf = xdf.reset_index(drop=True)
        st1 = get_mp4_creation_datetime(SUBDIR+X4_INSV_1)
        xdf['timestamp'] = st1 + pd.to_timedelta(xdf['timestamp_ms'], unit='ms')
        xdf["timestamp"] = xdf["timestamp"].dt.tz_convert("Etc/GMT-1")
        xdf['timestamp'] = xdf['timestamp'].dt.tz_localize(None)
        xdf["timestamp"] = xdf["timestamp"].astype("datetime64[us]")

        print("*************** CUT END of INSV1 ***************")
        xdf_begin = xdf['timestamp'].iloc[0]
        xdf_end = xdf['timestamp'].iloc[-1]
        cut_time = xdf_end.replace(microsecond=0)
        print("Begin 1        ", xdf_begin)
        print("End 1          ", xdf_end)
        print("Cut time       ", cut_time)

        xdf = xdf[xdf['timestamp'] <= cut_time].reset_index(drop=True)
        xdf_end = xdf['timestamp'].iloc[-1]
        print("End 1 after cut", xdf_end)

        mean_norm = np.linalg.norm(xdf[["accSmooth[0]", "accSmooth[1]", "accSmooth[2]"]].iloc[:500], axis=1).mean()
        print("Gravity mean at start :", round(mean_norm,2))

        xdf['org_acc_x']=xdf['accSmooth[0]']
        xdf['org_acc_y']=xdf['accSmooth[1]']
        xdf['org_acc_z']=xdf['accSmooth[2]']

        xdf = xdf[['timestamp', 'org_acc_x', 'org_acc_y', 'org_acc_z','org_quat_w', 'org_quat_x', 'org_quat_y', 'org_quat_z', 'timestamp_ms']]
        xdf = xdf.rename(columns={'org_acc_x': 'x4_acc_x', 'org_acc_y': 'x4_acc_y', 'org_acc_z': 'x4_acc_z','org_quat_w': 'x4_quat_w', 'org_quat_x': 'x4_quat_x', 'org_quat_y': 'x4_quat_y','org_quat_z': 'x4_quat_z'})

    if xloaded2:
        xdf2 = xdf2[xdf2["timestamp_ms"] >= 0]
        xdf2 = xdf2.reset_index(drop=True)

        xdf_end=xdf['timestamp'].iloc[-1]
        t0 = xdf2['timestamp_ms'].iloc[0]
        xdf2['timestamp'] = xdf_end + pd.to_timedelta(xdf2['timestamp_ms'] - t0, unit='ms')

        xdf2['timestamp'] = xdf2['timestamp'].dt.tz_localize(None)
        xdf2["timestamp"] = xdf2["timestamp"].astype("datetime64[us]")

        xdf2['org_acc_x'] = xdf2['accSmooth[0]']
        xdf2['org_acc_y'] = xdf2['accSmooth[1]']
        xdf2['org_acc_z'] = xdf2['accSmooth[2]']

        xdf2 = xdf2[['timestamp', 'org_acc_x', 'org_acc_y', 'org_acc_z', 'org_quat_w', 'org_quat_x', 'org_quat_y', 'org_quat_z','timestamp_ms']]
        xdf2 = xdf2.rename(columns={'org_acc_x': 'x4_acc_x', 'org_acc_y': 'x4_acc_y', 'org_acc_z': 'x4_acc_z','org_quat_w': 'x4_quat_w', 'org_quat_x': 'x4_quat_x', 'org_quat_y': 'x4_quat_y','org_quat_z': 'x4_quat_z'})

        print("*************** STITCHING INSV CHECK ***************")

        xdf_end = xdf['timestamp'].iloc[-1]
        print("End previous 1 ", xdf_end)
        xdf2_begin = xdf2['timestamp'].iloc[0]
        print("Begin 2        ", xdf2_begin)

        xdf = pd.concat([xdf, xdf2], axis=0, ignore_index=True)

    for col in ['x4_acc_x', 'x4_acc_y', 'x4_acc_z']:
        xdf[col] = xdf[col].rolling(window=WINDOW_ACCX4, min_periods=1,center=True).mean()

    fs = 100
    fc = 5
    b, a = butter(4, fc / (fs / 2), btype='low')
    for col in ['x4_acc_x', 'x4_acc_y', 'x4_acc_z']:
        xdf[col] = filtfilt(b, a, xdf[col])

    xdf = xdf.iloc[::X4_DEC].reset_index(drop=True)

    for df in [gdf, xdf]:
        df.sort_values("timestamp", inplace=True)

    if iloaded:
        idf.sort_values("timestamp", inplace=True)
        idf["timestamp"] = pd.to_datetime(idf["timestamp"]).astype("datetime64[us]")

    gdf["timestamp"] = pd.to_datetime(gdf["timestamp"]).astype("datetime64[us]")
    print("GPS start time", gdf['timestamp'][0])
    if iloaded:
        print("IPHONE start time", idf['timestamp'][0])
    print("X4 start time", xdf['timestamp'][0])

    if iloaded:
        merged = pd.merge_asof(xdf, idf, on="timestamp", direction="nearest")
        merged = pd.merge_asof(merged, gdf, on="timestamp", direction="nearest")
    else:
        merged = pd.merge_asof(xdf, gdf, on="timestamp", direction="nearest")


    merged = add_wind(merged)
    merged = add_ias(merged)


    merged.to_csv(OUTPUT, index=True, encoding="utf-8")
    print("\nMerged for ABView :"+OUTPUT)

    with open("data/" + get_bundle_name_from_insv(X4_INSV_1) + "/version.txt", "w") as f:
        f.write(__version__)

    # extract METAR

    if not SKIP_METAR:
        # start / end time of the video from merged dataframe
        start = pd.to_datetime(merged["timestamp"].iloc[0], utc=True)
        # round start down to previous 3‑hour boundary (00,03,06,09,12,15,18,21 UTC)
        h = (start.hour // 3) * 3 - 3
        start = start.replace(hour=6, minute=0, second=0, microsecond=0)
        end = start + timedelta(hours=12)
        print("Start time", start)
        print("End time", end)

        metar_df = download_metar_history("LFMT", start, end)
        # print(metar_df)
        OUTPUT_METAR = "data/" + get_bundle_name_from_insv(X4_INSV_1) + "/metar.csv"
        metar_df.to_csv(OUTPUT_METAR, index=False, encoding="utf-8")
    else:
        print(".....Skipping METAR export")
        INPUT_METAR = "data/" + get_bundle_name_from_insv(X4_INSV_1) + "/metar.csv"
        metar_df = pd.read_csv(INPUT_METAR, encoding="utf-8")
        # print(metar_df)

    print("Done.")


if __name__ == "__main__":

    if CONSOLE_WINDOW:
        app = QApplication(sys.argv)

        console = ConsoleWindow()
        console.show()

        stream = ConsoleStream()
        stream.new_text.connect(console.append_text)

        sys.stdout = stream
        sys.stderr = stream

        worker = Worker()
        worker.start()

        sys.exit(app.exec_())

    else:
        # mode console simple (pas de Qt)
        try:
            main()
        except Exception as e:
            print("Erreur :", e)
        sys.exit(0)

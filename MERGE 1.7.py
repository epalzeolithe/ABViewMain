
import os,re
from datetime import time as dt_time
from datetime import datetime, timedelta, timezone
from zoneinfo import ZoneInfo
import subprocess
import numpy as np
import pandas as pd
from scipy.signal import butter, filtfilt
from pymediainfo import MediaInfo
from pathlib import Path
import requests

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

SUBDIR="data/raw/"
X4_INSV_1, X4_INSV_2 = get_last_two_insv_files(SUBDIR)
GPS_GNS3000=get_last_GPS_log_file(SUBDIR)

# -------- CONFIG --------
SKIP_X4_EXPORT = True
SKIP_GNS3000_IMPORT = True
SKIP_IPHONE_IMPORT = False
SKIP_METAR = True
#X4_INSV_1 = "VID_20260320_131559_00_053.insv"
#X4_INSV_2 = "VID_20260320_131559_00_054.insv"
#GPS_GNS3000 = "LOG00005.TXT"
IPHONE_SENSORLOG = "sensorlog.csv"

WINDOW = 4 # taille moyenne glissante pour lissage GNS3000
WINDOW_ACCX4 = 50 # taille moyenne glissante pour lissage accéléros X4
GNS3000_PERIOD = 0.25 # 4 Hz
X4_DEC = 10 # 1000 Hz > 100 Hz
IPHONE_DEC = 5 # division données par 100 Hz > 20 Hz
#SUBDIR="data/raw/"
TMP=SUBDIR+"temp/"
GYROFLOW_BIN = "/Applications/Gyroflow.app/Contents/MacOS/gyroflow"
GYRO2BB = "data/ressources/gyro2bb-mac-arm64"
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
    print("Running:", " ".join(cmd))
    subprocess.run(cmd, check=True)

def export_GYRO2BB_gyro_acc_from_insv(insv_path):
    PATH=os.getcwd()+"/"
    cmd = [PATH+GYRO2BB,str(PATH+SUBDIR+insv_path)]
    print(cmd)
    print("Running GYRO2BB :", " ".join(cmd))
    subprocess.run(cmd, check=True)

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
        print(st1)
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


if __name__ == "__main__":

    print(TMP)
    Path(TMP).mkdir(parents=True, exist_ok=True)

    print("#####################################################################")
    print("Start Merging for ABView : X4 datas + GPS datas + iPhone sensorlog...")

    xloaded=x2loaded=gloaded=iloaded=False

    if os.path.exists(SUBDIR+GPS_GNS3000):
        print("Loading Datas from gns3000")
        gdf = get_datas_from_gns3000(SUBDIR+GPS_GNS3000)
        print("Datas from gns3000 loaded")
        gloaded=True
    else:
        print("GNS3000 log file not found")

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
        # xdf process
        xdf = xdf[xdf["timestamp_ms"] >= 0]  # filtrage données à temps négatif
        xdf = xdf.reset_index(drop=True)
        st1 = get_mp4_creation_datetime(SUBDIR+X4_INSV_1)
        xdf['timestamp'] = st1 + pd.to_timedelta(xdf['timestamp_ms'], unit='ms')
        xdf["timestamp"] = xdf["timestamp"].dt.tz_convert("Etc/GMT-1")
        xdf['timestamp'] = xdf['timestamp'].dt.tz_localize(None)
        xdf["timestamp"] = xdf["timestamp"].astype("datetime64[us]")

        print("*************** CUT END of INSV1 ***************")
        #print("Creation time 1", st1)
        xdf_begin = xdf['timestamp'].iloc[0]
        xdf_end = xdf['timestamp'].iloc[-1]
        cut_time = xdf_end.replace(microsecond=0)
        print("Begin 1        ", xdf_begin)
        print("End 1          ", xdf_end)
        print("Cut time       ", cut_time)

        # remove all rows after cut_time
        xdf = xdf[xdf['timestamp'] <= cut_time].reset_index(drop=True)
        xdf_end = xdf['timestamp'].iloc[-1]
        print("End 1 after cut", xdf_end)

        mean_norm = np.linalg.norm(xdf[["accSmooth[0]", "accSmooth[1]", "accSmooth[2]"]].iloc[:500], axis=1).mean()
        print("Gravity mean at start :", round(mean_norm,2))  # sur 500 premières valeurs, fps 30s > 15 secondes

        #accelerations transfer from BB
        xdf['org_acc_x']=xdf['accSmooth[0]']
        xdf['org_acc_y']=xdf['accSmooth[1]'] # warning axe vertical X4
        xdf['org_acc_z']=xdf['accSmooth[2]']
    
        #rearrange
        xdf = xdf[['timestamp', 'org_acc_x', 'org_acc_y', 'org_acc_z','org_quat_w', 'org_quat_x', 'org_quat_y', 'org_quat_z', 'timestamp_ms']]
        xdf = xdf.rename(columns={'org_acc_x': 'x4_acc_x', 'org_acc_y': 'x4_acc_y', 'org_acc_z': 'x4_acc_z','org_quat_w': 'x4_quat_w', 'org_quat_x': 'x4_quat_x', 'org_quat_y': 'x4_quat_y','org_quat_z': 'x4_quat_z'})

    if xloaded2:
        # xdf2 process
        xdf2 = xdf2[xdf2["timestamp_ms"] >= 0]  # filtrage données à temps négatif
        xdf2 = xdf2.reset_index(drop=True)

        xdf_end=xdf['timestamp'].iloc[-1]   # for stitching
        xdf2['timestamp'] = xdf_end + pd.to_timedelta(xdf2['timestamp_ms'], unit='ms')
        #xdf2["timestamp"] = xdf2["timestamp"].dt.tz_convert("Etc/GMT-1")
        xdf2['timestamp'] = xdf2['timestamp'].dt.tz_localize(None)
        xdf2["timestamp"] = xdf2["timestamp"].astype("datetime64[us]")

        # accelerations transfer from BB
        xdf2['org_acc_x'] = xdf2['accSmooth[0]']
        xdf2['org_acc_y'] = xdf2['accSmooth[1]']  # warning axe vertical X4
        xdf2['org_acc_z'] = xdf2['accSmooth[2]']

        # rearrange
        xdf2 = xdf2[['timestamp', 'org_acc_x', 'org_acc_y', 'org_acc_z', 'org_quat_w', 'org_quat_x', 'org_quat_y', 'org_quat_z','timestamp_ms']]
        xdf2 = xdf2.rename(columns={'org_acc_x': 'x4_acc_x', 'org_acc_y': 'x4_acc_y', 'org_acc_z': 'x4_acc_z','org_quat_w': 'x4_quat_w', 'org_quat_x': 'x4_quat_x', 'org_quat_y': 'x4_quat_y','org_quat_z': 'x4_quat_z'})

        print("*************** STITCHING INSV CHECK ***************")

        xdf_end = xdf['timestamp'].iloc[-1]
        print("End previous 1 ", xdf_end)
        xdf2_begin = xdf2['timestamp'].iloc[0]
        print("Begin 2        ", xdf2_begin)

        xdf = pd.concat([xdf, xdf2], axis=0, ignore_index=True)

    # smooth accelerations
    for col in ['x4_acc_x', 'x4_acc_y', 'x4_acc_z']:
        xdf[col] = xdf[col].rolling(window=WINDOW_ACCX4, min_periods=1,center=True).mean()  # min_periods pour régler pb sur les bords
    fs = 100  # fréquence d'échantillonnage (Hz)
    fc = 5  # fréquence de coupure (Hz)
    b, a = butter(4, fc / (fs / 2), btype='low') # filtre Butterworth
    for col in ['x4_acc_x', 'x4_acc_y', 'x4_acc_z']:
        xdf[col] = filtfilt(b, a, xdf[col])


    # decimate to reduce size
    xdf = xdf.iloc[::X4_DEC].reset_index(drop=True)

    # sort for merge
    for df in [gdf, xdf]:
        df.sort_values("timestamp", inplace=True)

    if iloaded:
        idf.sort_values("timestamp", inplace=True)
        # force iphone timestamp dtype si lecture csv
        idf["timestamp"] = pd.to_datetime(idf["timestamp"]).astype("datetime64[us]")

    # force gps timestamp dtype si lecture csv
    gdf["timestamp"] = pd.to_datetime(gdf["timestamp"]).astype("datetime64[us]")

    # check start time
    print("GPS start time", gdf['timestamp'][0])
    if iloaded:
        print("IPHONE start time", idf['timestamp'][0])
    print("X4 start time", xdf['timestamp'][0])
    #print(gdf["timestamp"].dtype, idf["timestamp"].dtype, xdf["timestamp"].dtype)

    if iloaded:
        merged = pd.merge_asof(xdf, idf, on="timestamp", direction="nearest")
        merged = pd.merge_asof(merged, gdf, on="timestamp", direction="nearest")
        merged = merged[
            ['timestamp', 'x4_acc_x', 'x4_acc_y', 'x4_acc_z', 'x4_quat_w', 'x4_quat_x', 'x4_quat_y', 'x4_quat_z',
             'iphone_lat', 'iphone_lon', 'iphone_alt', 'iphone_speed', 'iphone_heading',
             'gps_lat', 'gps_lon', 'gps_alt', 'gps_speed', 'gps_heading', 'gps_fpm']]
    else:
        merged = pd.merge_asof(xdf, gdf, on="timestamp", direction="nearest")
        merged = merged[
            ['timestamp', 'x4_acc_x', 'x4_acc_y', 'x4_acc_z', 'x4_quat_w', 'x4_quat_x', 'x4_quat_y', 'x4_quat_z',
             'gps_lat', 'gps_lon', 'gps_alt', 'gps_speed', 'gps_heading', 'gps_fpm']]



    #gdf.to_csv("data/gdf.csv", index=True,encoding="utf-8")
    #idf.to_csv("data/idf.csv", index=True,encoding="utf-8")
    #xdf.to_csv("data/xdf.csv", index=True,encoding="utf-8")
    merged.to_csv(OUTPUT, index=True, encoding="utf-8")
    print("\nMerged for ABView :"+OUTPUT)

    # extract METAR

    if not SKIP_METAR:
        # start / end time of the video from merged dataframe
        start = pd.to_datetime(merged["timestamp"].iloc[0], utc=True)
        # round start down to previous 3‑hour boundary (00,03,06,09,12,15,18,21 UTC)
        h = (start.hour // 3) * 3 -3
        start = start.replace(hour=6, minute=0, second=0, microsecond=0)
        end = start + timedelta(hours=12)
        print("Start time", start)
        print("End time", end)

        metar_df = download_metar_history("LFMT", start, end)
        #print(metar_df)
        OUTPUT_METAR = "data/" + get_bundle_name_from_insv(X4_INSV_1) + "/metar.csv"
        metar_df.to_csv(OUTPUT_METAR, index=False, encoding="utf-8")
    else:
        print(".....Skipping METAR export")
        INPUT_METAR = "data/" + get_bundle_name_from_insv(X4_INSV_1) + "/metar.csv"
        metar_df = pd.read_csv(INPUT_METAR, encoding="utf-8")
        #print(metar_df)

    print("Done.")




import requests
import pandas as pd
from datetime import datetime, timezone
import re


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

    # Build datetime column
    df["time"] = pd.to_datetime(
        dict(
            year=df["YEAR"],
            month=df["MONTH"],
            day=df["DAY"],
            hour=df["HOUR"],
            minute=df["MIN"],
        ),
        utc=True,
    )

    df = df.rename(columns={"REPORT": "metar"})

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


# ======================================================
# EXEMPLE UTILISATION
# ======================================================

start = datetime(2026, 2, 21, 8, 0, tzinfo=timezone.utc)
end = datetime(2026, 2, 21, 12, 0, tzinfo=timezone.utc)

metar_df = download_metar_history("LFMT", start, end)

print(metar_df)

# exemple : heure vidéo
video_time = datetime(2026, 2, 21, 9, 17, tzinfo=timezone.utc)

metar_row = find_metar_for_time(metar_df, video_time)

print("\nMETAR correspondant :")
print(metar_row.metar)
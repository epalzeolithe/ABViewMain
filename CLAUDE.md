# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

**ABView** is a Python desktop application for aerobatic flight data visualization. It synchronizes dual video feeds (Insta360 X4) with IMU/GPS/accelerometer data, displaying real-time 3D attitude, GPS trajectory, HUD overlays, and an OpenStreetMap tracker.

Two main scripts:
- **`ABView 1.10.py`** — PyQt5 viewer application (~5,100 lines)
- **`MERGE 1.10.py`** — Sensor data fusion pipeline (~1,040 lines)

## Running the Scripts

```bash
# Run the viewer (prompts for .abv bundle selection)
python "ABView 1.10.py"

# Run the data merge pipeline (processes raw sensor files)
python "MERGE 1.10.py"

# Convert raw video files to MP4
python "CONVERT 1.10.py"

# Build macOS .app bundle
python setup.py py2app
```

No test or lint commands exist in this project.

## Configuration (Top of Each Script)

**`ABView 1.10.py`** (lines ~70–100):
- `SKIP_BDL_SELECTION` — if `True`, skips dialog and loads `BDL` path directly
- `BDL` — hardcoded flight bundle path (`data/Vol_YYYY_MM_DD.abv/`)
- `MAINDIR` — absolute project root path
- `STL_FILE` — aircraft 3D model path (`ressources/CAP10.STL`)
- `PITCH_MONTAGE_PAR_DEFAUT` — camera mounting pitch angle in degrees
- `DF_FREQ` — data frequency in Hz (100)
- `TRACE` — GPS trajectory trail length in samples (6000)

**`MERGE 1.10.py`** (lines ~1–160):
- `SKIP_X4_EXPORT`, `SKIP_GNS3000_IMPORT`, `SKIP_IPHONE_IMPORT` — skip stages on reruns
- `SKIP_METAR`, `SKIP_WIND` — skip external API calls (Ogimet, Copernicus CDS)
- `SUBDIR` — raw input directory (`data/raw/`)
- `GYROFLOW_BIN` — path to Gyroflow.app binary
- `GYRO2BB` — path to `ressources/gyro2bb-mac-arm64` binary

## Architecture

### Data Flow

```
Raw inputs (data/raw/):
  VID_*.insv  ──→ Gyroflow CLI → quaternion CSV (30 fps)
               └→ gyro2bb     → gyro+accel CSV (1000 Hz → resample 100 Hz)
  LOG*.TXT    ──→ NMEA parse   → GPS DataFrame (4 Hz)
  sensorlog.csv → parse+decimate → iPhone IMU (100 Hz → 20 Hz)
  ERA5 API    ──→ NetCDF       → wind at altitude
  METAR API   ──→ CSV          → weather observations
                       ↓ MERGE 1.10.py
              data/Vol_YYYY_MM_DD.abv/merged_data.csv
                       ↓ ABView 1.10.py
              PyQt5 GUI: dual video + 3D + HUD + GPS map
```

### Flight Bundle (`.abv` directory)

```
data/Vol_YYYY_MM_DD.abv/
├── merged_data.csv    # Time-aligned sensor data at 100 Hz (~87 MB)
├── front.mp4          # Cockpit camera
├── back.mp4           # External camera
├── metar.csv          # Weather observations
├── bookmark.csv       # User annotations (frame + name + timestamp)
└── version.txt        # Software version used to create bundle
```

### `merged_data.csv` Key Columns

| Column | Description |
|--------|-------------|
| `timestamp` | ISO datetime |
| `x4_quat_w/x/y/z` | Quaternion for 3D attitude |
| `x4_acc_x/y/z` | Acceleration (m/s²) |
| `gps_lat/lon/alt` | Position & altitude |
| `gps_speed`, `gps_heading` | Ground speed & compass |
| `gps_fpm` | Vertical speed (ft/min) |
| `gps_ias` | Indicated airspeed |
| `era5_wind_speed/direction` | Wind at altitude |
| `iphone_*` | Alternative sensor data |

### `ABView 1.10.py` Key Classes

- **`MainWindow`** — Core Qt application; ~4,500 lines with these primary init methods:
  - `_init_video()` — Opens two MP4 containers via PyAV, computes time offsets
  - `_init_audio()` — PyAV audio resampler → Qt PCM output for clock sync
  - `init_UI()` — Menu, slider, buttons, overlays
  - `init_map_OSM_widget()` — Leaflet webview with plane icon tracking
  - `init_gps_pyqtgraph()` — 3D trajectory plot
  - `init_gfx()` — pygfx 3D scene with CAP10 STL aircraft model
  - `main_loop()` — 30 fps timer: decodes frames, looks up sensor row, updates all displays
  - `update_all()` — Per-frame: video fetch → sensor lookup → attitude → overlays → map
- **`VideoYUVOpenGLWidget`** — OpenGL YUV renderer (shader-based H.264 decode)
- **`ArtificialHorizon`** — Transparent pitch/bank overlay widget
- **`AnalogBadin`**, **`AnalogAltimeter`**, **`AnalogVario`** — Circular analog gauge widgets
- **`SCStreamHandler`** — macOS ScreenCaptureKit screen recording

### Real-time Loop Design

The main loop (`main_loop`, 30 fps) is audio-clock driven:
1. Audio output position → target frame index
2. Seek both video streams to frame `i` (PyAV)
3. Decode YUV planes → send to OpenGL shaders
4. `idf = nearest timestamp index` in DataFrame numpy arrays
5. `quat_to_rot()` → rotation matrix → pygfx 3D model transform
6. Update all HUD widgets, map, GPS plot, bookmarks

### External Tool Dependencies

| Tool | Location | Used By |
|------|----------|---------|
| Gyroflow | `/Applications/Gyroflow.app/Contents/MacOS/gyroflow` | MERGE: quaternion export |
| gyro2bb | `ressources/gyro2bb-mac-arm64` | MERGE: raw gyro/accel from .insv |
| exiftool | `ressources/exiftool` | MERGE: GPS metadata extraction |
| Copernicus CDS API | credentials in `.cdsapirc` | MERGE: ERA5 wind data |
| Ogimet.com | HTTP | MERGE: METAR historical weather |

## Key Development Notes

- Files with spaces in names (`ABView 1.10.py`) are intentional — always quote them in shell commands.
- The `.abv` extension is a directory, not a file package — macOS Finder displays it as a folder.
- Video time offset between front/back cameras is computed from MP4 `creation_time` metadata and stored as `VIDEO_OFFSET`.
- Attitude quaternion from Gyroflow is in a different reference frame from GPS heading — a `R_recalage_repere` flag selects calibration mode.
- `ACC_SCALE = 9.81 / 20234` converts raw integer accelerometer counts to m/s²; this constant is hardware-specific to the Insta360 X4.
- `data/raw/` and `data/*.abv/` directories are git-ignored (large binary/CSV files).
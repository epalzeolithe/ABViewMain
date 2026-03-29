# ABView - Aerobatic Flight Viewer

DEMO https://youtu.be/CWQwiJPiQm4

**ABView** is a desktop application for visualizing and analyzing aerobatic flight data. It synchronizes multiple video feeds with inertial (IMU), GPS, and accelerometer data recorded during aerobatic flights, providing a real-time multi-panel view of the aircraft's attitude, trajectory, and flight parameters.



> **Current version:** 1.9 GPX import
---

## Features

- **Dual Video Playback** -- Synchronized side-by-side display of two onboard camera feeds (e.g. cockpit + external) with timestamp overlay.
- **3D GPS Trajectory** -- Interactive Matplotlib 3D plot showing the flight path color-coded by altitude, with ground projection, north indicator, and aerobatic box axis overlay.
- **3D Attitude Visualization** -- Real-time pygfx 3D scene rendering a CAP10 aircraft STL model oriented by quaternion data from the Insta360 X4 IMU.
- **Artificial Horizon HUD** -- Two artificial-horizon widgets overlaid on the 3D view: a standard front-facing horizon and a wingtip (side-view) horizon for aerobatic reference.
- **Flight Parameters Overlay** -- Live readouts of pitch, bank angle, GPS heading, groundspeed (color-coded by airspeed range), altitude, vertical speed (ft/min), and G-load (with min/max tracking).
- **OpenStreetMap Tracking** -- Embedded Leaflet map showing the aircraft position in real time, with the aerobatic box axis drawn on the map.
- **Data Merge Pipeline** -- Companion script (`MERGE 0.9.py`) that fuses data from an Insta360 X4 camera (gyroscope + accelerometer via Gyroflow / gyro2bb), a GNS3000 GPS receiver (NMEA), and an iPhone running SensorLog into a single time-aligned CSV.

---

## Architecture Overview

The project consists of two main scripts:

| File | Role |
|---|---|
| `MERGE 0.9.py` | **Data preparation** -- Extracts, converts, filters, and merges sensor data from multiple sources into a unified `data/merged.csv` file. |
| `ABView 0.9.py` | **Viewer application** -- PyQt5 desktop GUI that loads the merged CSV alongside two MP4 video files and presents synchronized multi-panel visualization. |

### Data Flow

```
Insta360 X4 (.insv)
  |
  +---> Gyroflow CLI ---> quaternion CSV (30 fps)
  |                             |
  +---> gyro2bb tool ---> gyro + accel CSV (1000 Hz)
  |                             |
  +---- interpolate & merge ----+
                |
                v
         X4 DataFrame (100 Hz, decimated to 10 Hz)

GNS3000 GPS (.TXT, NMEA)
  |
  +---> parse GNGGA/GNRMC sentences ---> GPS DataFrame (4 Hz)

iPhone SensorLog (.csv)
  |
  +---> parse & decimate (100 Hz -> 20 Hz) ---> iPhone DataFrame

         |           |            |
         +--- merge_asof (timestamp) ---+
                      |
                      v
              data/merged.csv
                      |
                      v
               ABView 0.9.py  +  video1.mp4  +  video2.mp4
```

---

## Requirements

### System

- Python 3.10+
- macOS (primary target) or Linux with display server
- [Gyroflow](https://gyroflow.xyz/) binary installed (for data export from `.insv` files)

### Python Dependencies

```
PyQt5
PyQtWebEngine
opencv-python
numpy
pandas
scipy
matplotlib
pygfx
pymediainfo
```

Install with:

```bash
pip install PyQt5 PyQtWebEngine opencv-python numpy pandas scipy matplotlib pygfx pymediainfo
```

### External Tools

| Tool | Purpose | Location |
|---|---|---|
| **Gyroflow CLI** | Export quaternion metadata from Insta360 `.insv` files | Configured in `MERGE 0.9.py` as `GYROFLOW_BIN` (default: `/Applications/Gyroflow.app/Contents/MacOS/gyroflow`) |
| **gyro2bb** | Extract raw gyroscope and accelerometer data from `.insv` files | Bundled in `tool/gyro2bb-mac-arm64` |

---

## Data Preparation

### Input Files

Place the following files in the `data/` directory:

| File | Source | Description |
|---|---|---|
| `VID_*.insv` (x2) | Insta360 X4 | Raw video files containing embedded IMU data |
| `LOG*.TXT` | GNS3000 GPS | NMEA log file (GNGGA + GNRMC sentences) |
| `sensorlog.csv` | iPhone SensorLog app | Accelerometer, gyroscope, quaternion, and GPS data |
| `CAP10.stl` | 3D model | STL file of the aircraft for 3D visualization (already included) |

### Running the Merge

```bash
python "MERGE 0.9.py"
```

This will produce `data/merged.csv` containing time-aligned columns from all sensors.

#### Configuration (top of `MERGE 0.9.py`)

```python
X4_INSV_1 = "data/VID_20260221_091717_00_050.insv"   # First .insv file
X4_INSV_2 = "data/VID_20260221_091717_00_051.insv"   # Second .insv file
GPS_GNS3000 = "data/LOG00003.TXT"                     # GNS3000 NMEA log
IPHONE_SENSORLOG = "data/sensorlog.csv"                # iPhone SensorLog export
OUTPUT = "data/merged.csv"                             # Output merged file

SKIP_X4_EXPORT = True       # Set to False on first run to export from .insv
SKIP_GNS3000_IMPORT = True  # Set to False on first run to parse NMEA
SKIP_IPHONE_IMPORT = True   # Set to False on first run to parse SensorLog
```

> **First run:** Set all `SKIP_*` flags to `False` so the raw data is exported and parsed. On subsequent runs, set them to `True` to reuse cached intermediate files and speed up the process.

---

## Running ABView

### Input Files

Ensure the following files are in the `data/` directory:

- `merged.csv` -- Output from the merge step
- `video1.mp4` -- First camera video (e.g. cockpit)
- `video2.mp4` -- Second camera video (e.g. external / wing)
- `CAP10.STL` -- Aircraft 3D model (included)

### Launch

```bash
python "ABView 0.9.py"
```

### Configuration (top of `ABView 0.9.py`)

```python
DATA = "data/merged.csv"        # Merged sensor data
VIDEO1 = "data/video1.mp4"      # First video feed
VIDEO2 = "data/video2.mp4"      # Second video feed
STL_FILE = "data/CAP10.STL"     # 3D aircraft model

BOX = 0.007 * 1.5              # Map box size in degrees latitude
DF_FREQ = 100                   # DataFrame frequency (Hz)
TRACE = 9000                    # Trajectory trail length (samples)
VITESSE_MISE_EN_LIGNE = 80     # Line-up speed threshold (km/h)
PITCH_MONTAGE_PAR_DEFAUT = 15  # Default camera mounting pitch angle (degrees)
```

---

## User Interface

### Layout

```
+---------------------+---------------------+
|                     |                     |
|     Video 1         |     Video 2         |
|    (cockpit)        |    (external)       |
|                     |                     |
+----------+----------+----------+----------+
|                     |          |          |
|      3D View        | GPS 3D   |  OSM    |
|      (pygfx)        | Plot     |  Map    |
|                     |          |          |
+----------+----------+----------+----------+
|  [===================slider===========]   |
|  [Pause][Recalibrer][Start][Mise en       |
|   ligne][BOX][Zoom-][Zoom+][Reset]...     |
+-------------------------------------------+
```

### 3D View Overlays

The pygfx 3D panel displays the CAP10 model with the following HUD overlays:

| Position | Information | Color |
|---|---|---|
| Top-right | GPS Heading | Black |
| Right (upper) | Pitch angle | Green |
| Right (middle) | Bank angle | Blue |
| Right (center) | G-load (with min/max) | Red (color varies with G) |
| Bottom-right | Groundspeed | Color-coded by speed range |
| Bottom-right | Altitude (ft) | Blue |
| Bottom-right | Vertical speed (ft/min) | Blue |
| Top-left | Front artificial horizon | -- |
| Left | Wingtip artificial horizon | -- |

### Keyboard Shortcuts

| Key | Action |
|---|---|
| `Space` | Play / Pause |
| `Home` | Go to start |
| `Ctrl+Q` | Quit |
| `Ctrl+M` | Go to line-up point (speed > 80 km/h) |
| `Ctrl+B` | Go to BOX entry (altitude > 3000 ft) |
| `Ctrl++` | Zoom in (3D GPS plot) |
| `Ctrl+-` | Zoom out (3D GPS plot) |
| `Ctrl+0` | Reset zoom |
| `Ctrl+[` | Decrease trajectory trail length |
| `Ctrl+]` | Increase trajectory trail length |
| `Ctrl+T` | Reset trajectory trail length |

### Mouse

- **Double-click** on the 3D GPS plot to open it in full-screen (playback pauses automatically; resumes on close).
- **Click and drag** on the 3D GPS plot to rotate the view (elevation is locked by default).

---

## Project Structure

```
ABView/
  ABView 0.9.py       # Main viewer application (PyQt5)
  MERGE 0.9.py         # Data merge pipeline
  data/
    CAP10.stl          # 3D aircraft model (CAP10)
    merged.csv         # (generated) Merged sensor data
    video1.mp4         # (user-provided) First video
    video2.mp4         # (user-provided) Second video
    *.insv             # (user-provided) Insta360 X4 raw files
    LOG*.TXT           # (user-provided) GNS3000 GPS log
    sensorlog.csv      # (user-provided) iPhone SensorLog export
  tool/
    gyro2bb-mac-arm64  # gyro2bb binary for macOS ARM64
  docs/
    architecture.md    # Detailed architecture documentation
    data-pipeline.md   # Data pipeline documentation
    user-guide.md      # User guide
```

---

## Version History

| Version | Notes |
|---|---|
| 0.8 | Working version, requires export via Gyroflow GUI |
| 0.9 | Direct export from X4 `.insv` files -- requires Gyroflow binary installed |

---

## License

This project is currently provided without a license. All rights reserved by the author.

---

## Contributing

Contributions are welcome. Please open an issue or pull request on GitHub.

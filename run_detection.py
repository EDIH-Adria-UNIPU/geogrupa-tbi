import bisect
import csv
import json
import math
from pathlib import Path

import cv2
import numpy as np
from ultralytics import YOLO

# ------------------------------------------------------------------------------
# PARAMETERS (tune these one time for your rig and recording)
# ------------------------------------------------------------------------------
horizontal_fov_deg = 90.0  # same FOV you passed to py360convert.e2p
CAM_YAW_OFFSET = 25.0  # measured yaw offset of camera mount (deg)
time_offset = 5.0  # sec: video-to-GPS clock latency (measure ±0.25s)
HDG_DT = 1.0  # sec: centred heading baseline (must be ≥1Hz)

for f in ["detections_geo.csv", "thumbnails"]:
    p = Path(f)
    if p.exists():
        if p.is_file():
            p.unlink()
        else:
            for child in p.iterdir():
                child.unlink()
            p.rmdir()

root = Path("dataset") / "250"
index = json.loads((root / "index.json").read_text())

# read raw GPS data (one fix per second or faster)
telemetry_file = Path("telemetry") / "telemetry_250.json"
gps_raw = json.loads(telemetry_file.read_text())["GPS"]["Data"]

# keep only valid fixes and sort by time
gps_raw = [r for r in gps_raw if r.get("is_acquired")]
gps_raw.sort(key=lambda r: r["unix_timestamp"])

# drop exact millisecond duplicates (optional)
seen = set()
clean = []
for r in gps_raw:
    ts = r["unix_timestamp"]
    if ts not in seen:
        seen.add(ts)
        clean.append(r)
gps_raw = clean

# build numpy arrays for interpolation
t0 = gps_raw[0]["unix_timestamp"]
times = np.array([r["unix_timestamp"] - t0 for r in gps_raw])
lats = np.array([r["lat"] for r in gps_raw])
lons = np.array([r["lon"] for r in gps_raw])


# ------------------------------------------------------------------------------
# INTERPOLATION & BEARING UTILITIES
# ------------------------------------------------------------------------------
def coord_at(t: float) -> tuple[float, float]:
    """
    Return (lat, lon) by linear interpolation on the sorted arrays.
    t = seconds since first GPS fix; we add time_offset to align clocks.
    Args:
        t: seconds since first GPS fix
    Returns:
        lat, lon: interpolated coordinates (degrees)
    """
    t += time_offset
    i = bisect.bisect_left(times, t)
    if i <= 0:
        return lats[0], lons[0]
    if i >= len(times):
        return lats[-1], lons[-1]
    t_lo, t_hi = times[i - 1], times[i]
    w = (t - t_lo) / (t_hi - t_lo)
    lat = lats[i - 1] * (1 - w) + lats[i] * w
    lon = lons[i - 1] * (1 - w) + lons[i] * w
    return lat, lon


def bearing(lat1, lon1, lat2, lon2) -> float:
    """
    Return forward azimuth from (lat1,lon1) to (lat2,lon2) in degrees [0..360).
    Args:
        lat1, lon1: start point (degrees)
        lat2, lon2: end point (degrees)
    Returns:
        azimuth: forward azimuth (degrees [0..360))
    """
    φ1 = math.radians(lat1)
    φ2 = math.radians(lat2)
    Δλ = math.radians(lon2 - lon1)
    y = math.sin(Δλ) * math.cos(φ2)
    x = math.cos(φ1) * math.sin(φ2) - math.sin(φ1) * math.cos(φ2) * math.cos(Δλ)
    return (math.degrees(math.atan2(y, x)) + 360.0) % 360.0


def car_heading(t: float) -> float:
    """
    Args:
        t: seconds since first GPS fix (same as in coord_at())
    Returns:
        heading: forward azimuth of car at time t (degrees [0..360))
    """
    lat1, lon1 = coord_at(t - HDG_DT)
    lat2, lon2 = coord_at(t + HDG_DT)
    return bearing(lat1, lon1, lat2, lon2)


# ------------------------------------------------------------------------------
# RUN DETECTION LOOP
# ------------------------------------------------------------------------------
model = YOLO(str(Path("models") / "yolov8s-traffic-sign.pt"))
w_csv = csv.writer(open("detections_geo.csv", "w", newline=""))
thumb_dir = Path("thumbnails")
thumb_dir.mkdir(exist_ok=True)

# CSV header
w_csv.writerow(["lat", "lon", "bearing", "class", "conf", "frame", "thumb"])

for rec in index:
    # 1) interpolation & heading
    t = rec["t"]  # seconds since video start
    lat0, lon0 = coord_at(t)
    heading = car_heading(t)

    # 2) load image & run YOLO
    img = cv2.imread(str(root / rec["file"]))
    h, w, _ = img.shape
    results = model.predict(img, classes=[1, 6, 8, 10], conf=0.5, verbose=False)[0]

    # 3) process each box
    for box in results.boxes:
        x0, y0, x1, y1 = [int(v) for v in box.xyxy[0]]
        cx = (x0 + x1) / 2.0  # pixel center

        # --- true gnomonic pixel→angle conversion ---
        norm_x = 2.0 * (cx / w - 0.5)  # -1..+1 across image
        half_fov = math.radians(horizontal_fov_deg / 2.0)
        offset = math.degrees(math.atan(norm_x * math.tan(half_fov)))

        # compute global azimuth
        az = (heading + CAM_YAW_OFFSET + rec["yaw"] + offset) % 360.0

        # extract thumbnail
        pad = 0.15
        dx = int((x1 - x0) * pad)
        dy = int((y1 - y0) * pad)
        x0p = max(x0 - dx, 0)
        y0p = max(y0 - dy, 0)
        x1p = min(x1 + dx, w)
        y1p = min(y1 + dy, h)
        thumb = img[y0p:y1p, x0p:x1p]
        thumb_name = f"{rec['location_id']:05d}_{int(cx)}_{int(az)}.jpg"
        cv2.imwrite(str(thumb_dir / thumb_name), thumb)

        # write CSV row
        w_csv.writerow(
            [
                lat0,
                lon0,
                az,
                model.names[int(box.cls[0])],
                float(box.conf[0]),
                rec["location_id"],
                thumb_name,
            ]
        )

print("Done - detections_geo.csv + thumbnails/ generated.")

import bisect
import csv
import json
import math
from pathlib import Path

import cv2
import numpy as np
from ultralytics import YOLO

# Clear detections_geo.csv and thumbnails directory if they exist
for f in ["detections_geo.csv", "thumbnails"]:
    if Path(f).exists():
        if Path(f).is_file():
            Path(f).unlink()
        else:
            for file in Path(f).iterdir():
                file.unlink()
            Path(f).rmdir()

diagonal_fov_deg = 90  # must match the value in create_image_dataset.py
horizontal_fov_deg = 2 * math.degrees(
    math.atan(math.tan(math.radians(diagonal_fov_deg / 2)) / math.sqrt(2))
)

time_offset = 10.0  # seconds to shift your camera timestamps
lookahead = 0.2  # your existing 0.2 s for heading

root = Path("dataset") / "250"
index = json.loads((root / "index.json").read_text())

gps_raw = json.load(open(Path("telemetry") / "telemetry_250.json"))["GPS"]["Data"]
t0 = gps_raw[0]["unix_timestamp"]
times = np.array([g["unix_timestamp"] - t0 for g in gps_raw])
lats = np.array([g["lat"] for g in gps_raw])
lons = np.array([g["lon"] for g in gps_raw])


def coord_at(t):
    t = t + time_offset
    i = bisect.bisect_left(times, t)
    if i == 0:
        return lats[0], lons[0]
    if i == len(times):
        return lats[-1], lons[-1]
    t0, t1 = times[i - 1], times[i]
    w = (t - t0) / (t1 - t0)
    return (1 - w) * lats[i - 1] + w * lats[i], (1 - w) * lons[i - 1] + w * lons[i]


def bearing(lat1, lon1, lat2, lon2):
    φ1, φ2 = map(math.radians, (lat1, lat2))
    dλ = math.radians(lon2 - lon1)
    y = math.sin(dλ) * math.cos(φ2)
    x = math.cos(φ1) * math.sin(φ2) - math.sin(φ1) * math.cos(φ2) * math.cos(dλ)
    return (math.degrees(math.atan2(y, x)) + 360) % 360


model = YOLO(str(Path("models") / "yolov8s-traffic-sign.pt"))
w_csv = csv.writer(open("detections_geo.csv", "w", newline=""))
w_csv.writerow(["lat", "lon", "bearing", "class", "conf", "frame", "thumb"])

thumb_dir = Path("thumbnails")
thumb_dir.mkdir(exist_ok=True)

for rec in index:
    t = rec["t"]
    yaw_panel = rec["yaw"]
    lat0, lon0 = coord_at(t)
    lat1, lon1 = coord_at(t + lookahead)
    heading = bearing(lat0, lon0, lat1, lon1)

    img = cv2.imread(str(root / rec["file"]))
    h, w, _ = img.shape
    preds = model.predict(img, classes=[1, 6, 8, 10], conf=0.5, verbose=False)[0]

    for box in preds.boxes:
        cx = (box.xyxy[0][0] + box.xyxy[0][2]).item() / 2
        offset = horizontal_fov_deg * (cx / w - 0.5)
        az = (heading + yaw_panel + offset) % 360

        x0, y0, x1, y1 = [int(v) for v in box.xyxy[0]]
        pad = 0.15
        dx, dy = int((x1 - x0) * pad), int((y1 - y0) * pad)
        x0, y0, x1, y1 = (
            max(x0 - dx, 0),
            max(y0 - dy, 0),
            min(x1 + dx, w),
            min(y1 + dy, h),
        )
        thumb = img[y0:y1, x0:x1]
        thumb_name = f"{rec['location_id']:05d}_{int(cx)}_{int(az)}.jpg"
        cv2.imwrite(str(thumb_dir / thumb_name), thumb)

        w_csv.writerow(
            [
                lat0,
                lon0,
                az,
                model.names[int(box.cls)],
                float(box.conf),
                rec["location_id"],
                thumb_name,
            ]
        )

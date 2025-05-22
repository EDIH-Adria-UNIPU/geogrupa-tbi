import bisect
import csv
import json
import math
from pathlib import Path

import cv2
import numpy as np
from ultralytics import YOLO

from classify_traffic_sign import classify_sign

# PARAMETERS
horizontal_fov_deg = 90.0
time_offset = 5.0
HDG_DT = 1.0
THUMB_SIZE = (120, 120)  # standard thumbnail size (width, height)

root = Path("dataset") / "250"
index = json.loads((root / "index.json").read_text())

telemetry = json.loads((Path("telemetry") / "telemetry_250.json").read_text())["GPS"][
    "Data"
]
valid_fixes = [r for r in telemetry if r.get("is_acquired")]
valid_fixes.sort(key=lambda r: r["unix_timestamp"])

t0 = valid_fixes[0]["unix_timestamp"]
times = np.array([r["unix_timestamp"] - t0 for r in valid_fixes])
lats = np.array([r["lat"] for r in valid_fixes])
lons = np.array([r["lon"] for r in valid_fixes])


def coord_at(sec: float) -> tuple[float, float]:
    sec += time_offset
    idx = bisect.bisect_left(times, sec)
    if idx <= 0:
        return lats[0], lons[0]
    if idx >= len(times):
        return lats[-1], lons[-1]
    t_lo, t_hi = times[idx - 1], times[idx]
    w = (sec - t_lo) / (t_hi - t_lo)
    return (
        lats[idx - 1] * (1 - w) + lats[idx] * w,
        lons[idx - 1] * (1 - w) + lons[idx] * w,
    )


def bearing(lat1, lon1, lat2, lon2) -> float:
    phi1, phi2 = map(math.radians, (lat1, lat2))
    dlon = math.radians(lon2 - lon1)
    y = math.sin(dlon) * math.cos(phi2)
    x = math.cos(phi1) * math.sin(phi2) - math.sin(phi1) * math.cos(phi2) * math.cos(
        dlon
    )
    return (math.degrees(math.atan2(y, x)) + 360) % 360


models = {
    "traffic-sign": YOLO(str(Path("models") / "yolov8s-traffic-sign.pt")),
    "street-light": YOLO(str(Path("models") / "yolov8n-oiv7.pt")),
}
class_filters = {"traffic-sign": [1, 6, 8, 10], "street-light": [497]}
conf_thresholds = {"traffic-sign": 0.5, "street-light": 0.05}

detection_dir = Path("detections")
detection_dir.mkdir(exist_ok=True)

csv_writer = csv.writer(open(detection_dir / "detections_geo.csv", "w", newline=""))
csv_writer.writerow(["lat", "lon", "bearing", "class", "conf", "frame", "thumb"])

thumb_dir = detection_dir / "thumbnails"
thumb_dir = Path(thumb_dir)
thumb_dir.mkdir(exist_ok=True)

for record in index:
    sec = record["t"]
    lat0, lon0 = coord_at(sec)
    heading = bearing(*coord_at(sec - HDG_DT), *coord_at(sec + HDG_DT))

    img = cv2.imread(str(root / record["file"]))
    h, w, _ = img.shape

    for obj_type, model in models.items():
        results = model.predict(
            img,
            classes=class_filters[obj_type],
            conf=conf_thresholds[obj_type],
            verbose=False,
        )[0]
        for box in results.boxes:
            x0, y0, x1, y1 = [int(v) for v in box.xyxy[0]]
            cx = (x0 + x1) / 2.0
            norm_x = 2 * (cx / w - 0.5)
            half_fov = math.radians(horizontal_fov_deg / 2)
            offset = math.degrees(math.atan(norm_x * math.tan(half_fov)))
            az = (heading + record["yaw"] + offset) % 360

            # padded crop
            pad = 0.15
            dx = int((x1 - x0) * pad)
            dy = int((y1 - y0) * pad)
            x0p, y0p = max(x0 - dx, 0), max(y0 - dy, 0)
            x1p, y1p = min(x1 + dx, w), min(y1 + dy, h)
            thumb = img[y0p:y1p, x0p:x1p]
            thumb = cv2.resize(thumb, THUMB_SIZE, interpolation=cv2.INTER_AREA)
            thumb_name = (
                f"{record['location_id']:05d}_{int(cx)}_{int(az)}_{obj_type}.jpg"
            )
            cv2.imwrite(str(thumb_dir / thumb_name), thumb)

            if obj_type == "traffic-sign":
                detected_category = classify_sign(Path(thumb_dir) / thumb_name)
                print(f"Detected category: {detected_category}")
            else:
                detected_category = obj_type

            csv_writer.writerow(
                [
                    lat0,
                    lon0,
                    az,
                    detected_category,
                    float(box.conf[0]),
                    record["location_id"],
                    thumb_name,
                ]
            )

print("run_detection completed: detections_geo.csv + thumbnails/")

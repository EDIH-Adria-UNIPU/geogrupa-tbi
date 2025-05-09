import bisect
import csv
import json
import math
from pathlib import Path

import cv2
from geopy.distance import geodesic
from ultralytics import YOLO

root = Path("dataset") / "250"
index = json.loads((root / "index.json").read_text())
gps_table = json.load(open(Path("telemetry") / "telemetry_250.json"))["GPS"]["Data"]
t0 = gps_table[0]["unix_timestamp"]
gps = {int(round(e["unix_timestamp"] - t0)): (e["lat"], e["lon"]) for e in gps_table}

seconds, coords = [], []
for e in gps_table:
    s = int(round(e["unix_timestamp"] - t0))
    if s not in seconds:  # keep first fix in that second
        seconds.append(s)
        coords.append((e["lat"], e["lon"]))


def coord_at(sec):
    """return the most recent fix at or before <sec>"""
    i = bisect.bisect_right(seconds, sec) - 1
    if i < 0:
        raise ValueError(f"no GPS fix before second {sec}")
    return coords[i]


def bearing(lat1, lon1, lat2, lon2):
    φ1, φ2 = map(math.radians, (lat1, lat2))
    dλ = math.radians(lon2 - lon1)
    y = math.sin(dλ) * math.cos(φ2)
    x = math.cos(φ1) * math.sin(φ2) - math.sin(φ1) * math.cos(φ2) * math.cos(dλ)
    return (math.degrees(math.atan2(y, x)) + 360) % 360


model = YOLO(str(Path("models") / "yolov8s-traffic-sign.pt"))
writer = csv.writer(open("detections_geo.csv", "w", newline=""))
writer.writerow(["lat", "lon", "class", "conf", "frame"])

for rec in index:
    loc = rec["location_id"]
    yaw = rec["yaw"]
    lat0, lon0 = coord_at(loc)
    lat1, lon1 = coord_at(loc + 1)  # OK even if +1 has no fix: falls back
    road_dir = bearing(lat0, lon0, lat1, lon1)

    img = cv2.imread(str(root / rec["file"]))
    h, w, _ = img.shape
    for box in model.predict(img, classes=[1, 6, 8, 10], verbose=False, conf=0.6)[0].boxes:
        xc = (box.xyxy[0][0].item() + box.xyxy[0][2].item()) / 2
        offset = 90 * (xc / w - 0.5)  # −45 … +45
        theta = (road_dir + yaw + offset) % 360
        point = geodesic(meters=5).destination((lat0, lon0), theta)
        writer.writerow(
            [
                point.latitude,
                point.longitude,
                model.names[int(box.cls)],
                float(box.conf),
                loc,
            ]
        )

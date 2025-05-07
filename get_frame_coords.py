"""
Creates a CSV file with the frame coordinates from the telemetry data.
"""

import csv
import json

TELEMTRY_FILE = "telemetry_250.json"
FRAME_COORDS_FILE = "frame_coords_250.csv"

t = json.load(open(TELEMTRY_FILE))["GPS"]["Data"]
t0 = t[0]["unix_timestamp"]
rows = {}

for e in t:
    sec = int(round(e["unix_timestamp"] - t0))
    rows.setdefault(sec, (e["lat"], e["lon"], e["altitude"]))

with open(FRAME_COORDS_FILE, "w", newline="") as f:
    w = csv.writer(f)
    w.writerow(["second", "lat", "lon", "alt"])
    for s, (la, lo, al) in sorted(rows.items()):
        w.writerow([s, la, lo, al])

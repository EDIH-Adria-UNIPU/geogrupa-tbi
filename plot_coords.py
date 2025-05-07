"""
Plot the GPS coordinates of the frame captures on a map.
"""

import json
import sys

import folium

TELEMETRY_FILE = "telemetry_250.json"
OUTPUT_FILE = "gps_track_250.html"

# ------------------------------------------------------------------
# load the GPS fixes (one per second) straight from the CAMM dump
rows = {}
t = json.load(open(TELEMETRY_FILE))["GPS"]["Data"]

if not t:
    print(f"Telemetry file {TELEMETRY_FILE} is empty or invalid.")
    sys.exit(1)

t0 = t[0]["unix_timestamp"]
for e in t:
    if not all(key in e for key in ("unix_timestamp", "lat", "lon", "altitude")):
        print(f"Telemetry file {TELEMETRY_FILE} is missing some keys.")
        continue
    sec = int(round(e["unix_timestamp"] - t0))
    rows.setdefault(sec, (e["lat"], e["lon"], e["altitude"]))

if not rows:
    print(f"No valid GPS data points were processed from {TELEMETRY_FILE}.")
    sys.exit(1)

track = [value for key, value in sorted(rows.items())]

# ------------------------------------------------------------------
# build a folium map
lat0, lon0, _ = track[0]
m = folium.Map(location=[lat0, lon0], zoom_start=17, tiles="OpenStreetMap")

# poly‑line of the whole path
folium.PolyLine(
    [(lat, lon) for lat, lon, _ in track], color="blue", weight=3, opacity=0.7
).add_to(m)

# numbered marker for every panorama (simple popup with frame index)
for sec, (lat, lon, alt) in enumerate(track):
    folium.Marker(
        [lat, lon],
        popup=f"frame {sec}",
        icon=folium.DivIcon(html=f"<div style='font-size:9pt'>{sec}</div>"),
    ).add_to(m)

m.save(OUTPUT_FILE)
print(f"Map saved to {OUTPUT_FILE}.")

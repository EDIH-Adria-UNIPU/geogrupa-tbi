import json
import math
from pathlib import Path

import folium
import numpy as np
import pandas as pd
from geopy.distance import geodesic
from sklearn.cluster import DBSCAN


def bearing_deg(lat1, lon1, lat2, lon2):
    φ1, φ2 = map(math.radians, (lat1, lat2))
    dλ = math.radians(lon2 - lon1)
    y = math.sin(dλ) * math.cos(φ2)
    x = math.cos(φ1) * math.sin(φ2) - math.sin(φ1) * math.cos(φ2) * math.cos(dλ)
    return (math.degrees(math.atan2(y, x)) + 360) % 360


def llh_to_enu(lat, lon, lat0, lon0):
    d = geodesic((lat0, lon0), (lat, lon)).meters
    brg = math.radians(bearing_deg(lat0, lon0, lat, lon))
    return d * math.sin(brg), d * math.cos(brg)


def enu_to_llh(e, n, lat0, lon0):
    az = math.degrees(math.atan2(e, n)) % 360
    p = geodesic(meters=math.hypot(e, n)).destination((lat0, lon0), az)
    return p.latitude, p.longitude


def intersection(pts, bearings):
    A = np.column_stack((-np.sin(bearings), np.cos(bearings)))
    c = (A * pts).sum(1)
    return np.linalg.lstsq(A, c, rcond=None)[0]


det = pd.read_csv("detections_geo.csv")
with open(Path("telemetry") / "telemetry_250.json") as f:
    path = np.array([(e["lat"], e["lon"]) for e in json.load(f)["GPS"]["Data"]])

lat0, lon0 = path[0]
enu = det.apply(
    lambda r: llh_to_enu(r.lat, r.lon, lat0, lon0), axis=1, result_type="expand"
)
det["e"], det["n"] = enu[0], enu[1]

clusterer = DBSCAN(eps=5, min_samples=2).fit(det[["e", "n"]].to_numpy())
det["cluster"] = clusterer.labels_

# Initialize map and add polyline before iterating through clusters
m = folium.Map(location=[lat0, lon0], zoom_start=17)
folium.PolyLine(path, color="blue").add_to(m)

objects = []
for k, g in det.groupby("cluster"):
    if k == -1:
        continue
    pos = intersection(g[["e", "n"]].to_numpy(), np.deg2rad(g.bearing.to_numpy()))
    lat, lon = enu_to_llh(pos[0], pos[1], lat0, lon0)
    cls = g.iloc[0]["class"]  # Define cls from the current group
    objects.append((lat, lon, cls))  # Keep populating objects for the CSV

    # Your new marker code integrated here
    # This assumes that your DataFrame 'g' (and thus 'det') has a 'thumb' column
    # containing filenames for the thumbnails.
    thumbs = "<br>".join(
        f"<img src='thumbnails/{t}' width='120'>" for t in g.thumb.head(3)
    )
    popup_html = f"<b>{cls}</b><br>{thumbs}"
    folium.Marker(
        [lat, lon],
        popup=folium.Popup(popup_html, max_width=400),
        icon=folium.Icon(color="red"),
    ).add_to(m)

pd.DataFrame(objects, columns=["lat", "lon", "class"]).to_csv(
    "objects.csv", index=False
)

m.save("objects_map.html")
print("wrote objects.csv and objects_map.html")

import json
import math
from pathlib import Path

import folium
import numpy as np
import pandas as pd
from geopy.distance import geodesic
from sklearn.cluster import DBSCAN

pd.options.mode.chained_assignment = None

detection_dir = Path("detections")
detection_df = pd.read_csv(detection_dir / "detections_geo.csv")
telemetry = json.loads((Path("telemetry") / "telemetry_250.json").read_text())["GPS"][
    "Data"
]
path = np.array([(e["lat"], e["lon"]) for e in telemetry])
lat0, lon0 = path[0]


def bearing_deg(lat1, lon1, lat2, lon2):
    phi1, phi2 = map(math.radians, (lat1, lat2))
    dl = math.radians(lon2 - lon1)
    y = math.sin(dl) * math.cos(phi2)
    x = math.cos(phi1) * math.sin(phi2) - math.sin(phi1) * math.cos(phi2) * math.cos(dl)
    return (math.degrees(math.atan2(y, x)) + 360) % 360


def llh_to_enu(lat, lon, lat0, lon0):
    d = geodesic((lat0, lon0), (lat, lon)).meters
    br = math.radians(bearing_deg(lat0, lon0, lat, lon))
    return d * math.sin(br), d * math.cos(br)


def intersection(pts, bearings):
    A = np.column_stack((-np.sin(bearings), np.cos(bearings)))
    c = (A * pts).sum(1)
    return np.linalg.lstsq(A, c, rcond=None)[0]


def enu_to_llh(e, n, lat0, lon0):
    az = math.degrees(math.atan2(e, n)) % 360
    p = geodesic(meters=math.hypot(e, n)).destination((lat0, lon0), az)
    return p.latitude, p.longitude


map_obj = folium.Map(location=[lat0, lon0], zoom_start=17)
folium.PolyLine(path, color="blue").add_to(map_obj)

objects = []
for obj_type in detection_df["class"].unique():
    subset = detection_df[detection_df["class"] == obj_type]
    enu = subset.apply(
        lambda r: llh_to_enu(r.lat, r.lon, lat0, lon0), axis=1, result_type="expand"
    )
    subset["e"], subset["n"] = enu[0], enu[1]
    subset["cluster"] = DBSCAN(eps=5, min_samples=2).fit_predict(subset[["e", "n"]])

    for cluster_id, group in subset.groupby("cluster"):
        if cluster_id < 0:
            continue
        pts = group[["e", "n"]].to_numpy()
        bears = np.deg2rad(group.bearing.to_numpy())
        e_int, n_int = intersection(pts, bears)
        lat, lon = enu_to_llh(e_int, n_int, lat0, lon0)
        objects.append((lat, lon, obj_type))

        thumbs_html = "<br>".join(
            f"<img src=''thumbnails/{t}' width='120'>" for t in group.thumb.head(3)
        )
        popup = folium.Popup(f"<b>{obj_type}</b><br>{thumbs_html}", max_width=400)
        color = "red" if obj_type == "traffic-sign" else "orange"
        folium.Marker([lat, lon], popup=popup, icon=folium.Icon(color=color)).add_to(
            map_obj
        )

pd.DataFrame(objects, columns=["lat", "lon", "class"]).to_csv(
    detection_dir / "objects.csv", index=False
)
map_obj.save(detection_dir / "objects_map.html")
print("triangulation complete: objects_map.html + objects.csv")

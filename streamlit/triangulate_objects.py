import json
import math
from pathlib import Path

import folium
import numpy as np
import pandas as pd
from geopy.distance import geodesic
from sklearn.cluster import DBSCAN

def triangulate_objects():
    pd.options.mode.chained_assignment = None

    detection_dir = Path("./streamlit/detections/")
    detection_df = pd.read_csv(detection_dir / "detections_geo.csv")
    telemetry = json.loads(Path("telemetry_835.json").read_text())["GPS"][
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


    def intersection(
        pts_enu_vehicle, bearings_rad
    ):  # pts_enu_vehicle is [E_vehicle, N_vehicle]
        """
        Calculates the intersection point of multiple lines.
        Each line is defined by a point (pts_enu_vehicle[i]) and a bearing (bearings_rad[i]).
        Args:
            pts_enu_vehicle (np.ndarray): Nx2 array of ENU coordinates of the vehicle for N detections.
                                        Column 0 is East, Column 1 is North.
            bearings_rad (np.ndarray): Nx1 array of bearings in radians (clockwise from North).
        Returns:
            tuple: (e_intersect, n_intersect) or (np.nan, np.nan) if error.
        """
        # --- Initial checks ---
        if not isinstance(pts_enu_vehicle, np.ndarray) or not isinstance(
            bearings_rad, np.ndarray
        ):
            # print("Warning: Inputs to intersection must be numpy arrays.") # Optional: for non-debug, might remove print
            return np.nan, np.nan
        if pts_enu_vehicle.ndim != 2 or pts_enu_vehicle.shape[1] != 2:
            # print(f"Warning: `pts_enu_vehicle` array has wrong shape {pts_enu_vehicle.shape}. Expected (N, 2).")
            return np.nan, np.nan
        if bearings_rad.ndim != 1:
            # print(f"Warning: `bearings_rad` array has wrong shape {bearings_rad.shape}. Expected (N,).")
            return np.nan, np.nan
        if (
            pts_enu_vehicle.shape[0] < 2
            or bearings_rad.shape[0] < 2
            or pts_enu_vehicle.shape[0] != bearings_rad.shape[0]
        ):
            # print("Warning: Not enough points or mismatched shapes for intersection.")
            return np.nan, np.nan
        # --- End initial checks ---

        cos_b = np.cos(bearings_rad)
        sin_b = np.sin(bearings_rad)

        # Line equation for each detection i:
        # E_intersect * cos(bearing_i) - N_intersect * sin(bearing_i) = E_vehicle_i * cos(bearing_i) - N_vehicle_i * sin(bearing_i)
        # This is of the form A * x_solution = b_solution_vector
        # where x_solution = [E_intersect, N_intersect]'

        # A_matrix columns are coefficients of E_intersect and N_intersect respectively
        A_matrix = np.column_stack((cos_b, -sin_b))

        # b_vector elements are the right-hand side of the equation
        # pts_enu_vehicle[:, 0] is E_vehicle
        # pts_enu_vehicle[:, 1] is N_vehicle
        b_vector = pts_enu_vehicle[:, 0] * cos_b - pts_enu_vehicle[:, 1] * sin_b

        try:
            result, residuals, rank, singular_values = np.linalg.lstsq(
                A_matrix, b_vector, rcond=None
            )
            return result[0], result[1]  # E_intersect, N_intersect
        except np.linalg.LinAlgError as e:
            print(
                f"Linear algebra error during intersection: {e}"
            )  # Keep this for production issues
            return np.nan, np.nan


    def enu_to_llh(e, n, lat0, lon0):
        az = math.degrees(math.atan2(e, n)) % 360
        p = geodesic(meters=math.hypot(e, n)).destination((lat0, lon0), az)
        return p.latitude, p.longitude


    map_obj = folium.Map(location=[lat0, lon0], zoom_start=17)
    folium.PolyLine(path, color="blue").add_to(map_obj)

    objects = []
    for obj_type in detection_df["class"].unique():
        print(obj_type)
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

            # draw the cluster’s rays
            for _, detection in group.iterrows():
                folium.PolyLine(
                    [(detection.lat, detection.lon), (lat, lon)],
                    color="green",
                    weight=1,
                    opacity=0.4,
                ).add_to(map_obj)

            if obj_type != "not_a_sign":
                thumbs = "<br>".join(
                    f"<img src='thumbnails/{t}' width='120'>" for t in group.thumb.head(3)
                )
                popup = folium.Popup(f"<b>{obj_type}</b><br>{thumbs}", max_width=400)
                color = "orange" if obj_type == "street-light" else "red"
                folium.Marker(
                    [lat, lon], popup=popup, icon=folium.Icon(color=color)
                ).add_to(map_obj)

            folium.CircleMarker(
                location=(lat, lon),
                radius=5,
                color="black",
                fill=True,
                fill_color="yellow",
                fill_opacity=0.9,
                popup=f"intersection of {obj_type}",
            ).add_to(map_obj)
        objects.append([lat, lon, obj_type])

    pd.DataFrame(objects, columns=["lat", "lon", "class"]).to_csv(
        detection_dir / "objects.csv", index=False
    )
    map_obj.save(detection_dir / "objects_map.html")
    print("triangulation complete: objects_map.html + objects.csv")

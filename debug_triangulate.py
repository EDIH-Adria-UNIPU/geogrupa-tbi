# ==== triangulate_objects_debug_interactive.py ====
import json
import math
import sys
from pathlib import Path

import folium
import numpy as np
import pandas as pd
from geopy.distance import geodesic
from sklearn.cluster import DBSCAN

pd.options.mode.chained_assignment = None

# --- CONFIGURATION ---
DETECTION_DIR = Path("detections")
TELEMETRY_FILE_PATH = Path("telemetry") / "telemetry_250.json"
OUTPUT_DEBUG_DIR = Path("triangulation_interactive_debug")
DBSCAN_EPS = 10  # Epsilon for DBSCAN in meters. Tune this!
DBSCAN_MIN_SAMPLES = 2
# --- END CONFIGURATION ---

OUTPUT_DEBUG_DIR.mkdir(exist_ok=True)


# --- Utility Functions ---
def load_json_data(file_path, description):
    if not file_path.exists():
        print(f"ERROR: {description} file not found at {file_path}")
        sys.exit(1)
    try:
        return json.loads(file_path.read_text())
    except json.JSONDecodeError as e:
        print(f"ERROR: Could not decode JSON from {file_path}. Error: {e}")
        sys.exit(1)


def bearing_deg(lat1, lon1, lat2, lon2):
    # Check for identical points to avoid NaN from atan2(0,0)
    if lat1 == lat2 and lon1 == lon2:
        return 0.0  # Or handle as an error/special case depending on context

    phi1, phi2 = map(math.radians, (lat1, lat2))
    dl = math.radians(lon2 - lon1)

    y = math.sin(dl) * math.cos(phi2)
    x = math.cos(phi1) * math.sin(phi2) - math.sin(phi1) * math.cos(phi2) * math.cos(dl)

    bearing = math.degrees(math.atan2(y, x))
    return (bearing + 360) % 360


def llh_to_enu(lat, lon, ref_lat, ref_lon):
    if pd.isna(lat) or pd.isna(lon) or pd.isna(ref_lat) or pd.isna(ref_lon):
        # print(f"Warning: NaN value encountered in llh_to_enu ({lat}, {lon}, {ref_lat}, {ref_lon})")
        return np.nan, np.nan
    try:
        # Handle cases where lat,lon might be identical to ref_lat, ref_lon
        if lat == ref_lat and lon == ref_lon:
            return 0.0, 0.0

        dist_m = geodesic((ref_lat, ref_lon), (lat, lon)).meters
        br_rad = math.radians(bearing_deg(ref_lat, ref_lon, lat, lon))

        # Standard ENU: East = d * sin(bearing), North = d * cos(bearing)
        e = dist_m * math.sin(br_rad)
        n = dist_m * math.cos(br_rad)
        return e, n
    except ValueError as e_geo:  # Geodesic can raise ValueError for invalid points
        print(
            f"Error in geodesic/bearing calculation: {e_geo} with inputs ({lat}, {lon}), ref ({ref_lat}, {ref_lon})"
        )
        return np.nan, np.nan


def intersection(pts, bearings_rad):  # pts is [E_vehicle, N_vehicle]
    # --- Initial checks (same as your old function) ---
    if not isinstance(pts, np.ndarray) or not isinstance(bearings_rad, np.ndarray):
        print("Warning: Inputs to intersection must be numpy arrays.")
        return np.nan, np.nan
    if pts.ndim != 2 or pts.shape[1] != 2:
        print(f"Warning: `pts` array has wrong shape {pts.shape}. Expected (N, 2).")
        return np.nan, np.nan
    if bearings_rad.ndim != 1:
        print(
            f"Warning: `bearings_rad` array has wrong shape {bearings_rad.shape}. Expected (N,)."
        )
        return np.nan, np.nan
    if (
        pts.shape[0] < 2
        or bearings_rad.shape[0] < 2
        or pts.shape[0] != bearings_rad.shape[0]
    ):
        # print("Warning: Not enough points or mismatched shapes for intersection.")
        return np.nan, np.nan
    # --- End initial checks ---

    cos_b = np.cos(bearings_rad)
    sin_b = np.sin(bearings_rad)

    # A_matrix rows: [cos(beta_i), -sin(beta_i)]
    A_matrix = np.column_stack((cos_b, -sin_b))

    # b_vector elements: E_vehicle_i * cos(beta_i) - N_vehicle_i * sin(beta_i)
    # pts[:, 0] corresponds to E_vehicle
    # pts[:, 1] corresponds to N_vehicle
    b_vector = pts[:, 0] * cos_b - pts[:, 1] * sin_b

    # --- Debug prints (optional, but useful if issues persist) ---
    # print(f"        DEBUG Intersection Internal: pts (Vehicle ENU) = \n{pts}")
    # print(f"        DEBUG Intersection Internal: bearings_rad = \n{bearings_rad}")
    # print(f"        DEBUG Intersection Internal: A_matrix = \n{A_matrix}")
    # print(f"        DEBUG Intersection Internal: b_vector = \n{b_vector}")
    # --- End debug prints ---

    try:
        # Solve A_matrix * [E_intersect, N_intersect]' = b_vector
        # result will be a 2-element array: [E_intersect, N_intersect]
        result, residuals, rank, singular_values = np.linalg.lstsq(
            A_matrix, b_vector, rcond=None
        )
        # print(f"        DEBUG Intersection Internal: lstsq result = \n{result}")
        return result[0], result[1]  # E_intersect, N_intersect
    except np.linalg.LinAlgError as e:
        print(f"Linear algebra error during intersection: {e}")
        return np.nan, np.nan


def enu_to_llh(e, n, ref_lat, ref_lon):
    if pd.isna(e) or pd.isna(n) or pd.isna(ref_lat) or pd.isna(ref_lon):
        return np.nan, np.nan

    dist_meters = math.hypot(e, n)
    if dist_meters < 1e-6:  # If point is essentially at the reference
        return ref_lat, ref_lon

    # Azimuth from North, clockwise for geopy.destination
    # atan2(East, North) gives angle from North axis, positive for East.
    az_rad = math.atan2(e, n)
    az_deg = (math.degrees(az_rad) + 360) % 360

    try:
        destination_point = geodesic(meters=dist_meters).destination(
            (ref_lat, ref_lon), az_deg
        )
        return destination_point.latitude, destination_point.longitude
    except ValueError as e_geo:
        print(
            f"Error in geodesic destination: {e_geo} with inputs e={e}, n={n}, ref ({ref_lat}, {ref_lon}), dist={dist_meters}, az={az_deg}"
        )
        return np.nan, np.nan


# --- End Utility Functions ---

# --- Load Data ---
detections_csv_path = DETECTION_DIR / "detections_geo.csv"
if not detections_csv_path.exists():
    print(f"ERROR: detections_geo.csv not found at {detections_csv_path}")
    sys.exit(1)
detection_df = pd.read_csv(detections_csv_path)

telemetry_data = load_json_data(TELEMETRY_FILE_PATH, "Telemetry")
if "GPS" not in telemetry_data or "Data" not in telemetry_data["GPS"]:
    print(
        f"ERROR: Telemetry file {TELEMETRY_FILE_PATH} does not have the expected GPS.Data structure."
    )
    sys.exit(1)
telemetry_gps_entries = telemetry_data["GPS"]["Data"]

path_list = []
for e_gps in telemetry_gps_entries:
    if isinstance(e_gps, dict) and "lat" in e_gps and "lon" in e_gps:
        path_list.append((e_gps["lat"], e_gps["lon"]))
if not path_list:
    print("ERROR: No valid lat/lon pairs found in telemetry data.")
    sys.exit(1)
vehicle_path_coords = np.array(path_list)
ref_lat0, ref_lon0 = vehicle_path_coords[0]  # Reference point for all ENU conversions
# --- End Load Data ---


# --- Main Map for All Triangulated Objects ---
map_all_objects = folium.Map(
    location=[ref_lat0, ref_lon0], zoom_start=17, tiles="OpenStreetMap"
)
folium.PolyLine(
    vehicle_path_coords.tolist(),
    color="blue",
    weight=2.5,
    opacity=0.8,
    popup="Vehicle Path",
).add_to(map_all_objects)

triangulated_objects_data = []  # To store data for the final CSV

print(
    f"Starting triangulation. DBSCAN eps={DBSCAN_EPS}m, min_samples={DBSCAN_MIN_SAMPLES}"
)
print(f"Reference ENU origin (lat0, lon0): {ref_lat0:.6f}, {ref_lon0:.6f}")
print("----------------------------------------------------")

for obj_type_being_processed in sorted(
    detection_df["class"].unique()
):  # Sorted for consistent order
    print(f"\nProcessing Class: '{obj_type_being_processed}'")
    # Create a working copy for each class type
    subset_df = detection_df[detection_df["class"] == obj_type_being_processed].copy()

    if subset_df.empty:
        print("  No detections for this class.")
        continue

    # ENU conversion for vehicle positions at detection time
    # This converts the (lat, lon) of the vehicle when the object was detected to local ENU
    enu_coords = subset_df.apply(
        lambda r: pd.Series(
            llh_to_enu(r.lat, r.lon, ref_lat0, ref_lon0),
            index=["e_vehicle", "n_vehicle"],
        ),
        axis=1,
    )
    subset_df["e_vehicle"] = enu_coords["e_vehicle"]
    subset_df["n_vehicle"] = enu_coords["n_vehicle"]

    # Drop rows where ENU conversion might have failed (e.g., due to NaN input GPS for a detection)
    subset_df.dropna(subset=["e_vehicle", "n_vehicle"], inplace=True)

    if subset_df.empty:
        print(
            f"  No valid ENU coordinates for class '{obj_type_being_processed}' after NaN drop."
        )
        continue
    if len(subset_df) < DBSCAN_MIN_SAMPLES:
        print(
            f"  Not enough detections ({len(subset_df)}) for DBSCAN for class '{obj_type_being_processed}'. Need at least {DBSCAN_MIN_SAMPLES}."
        )
        continue

    # DBSCAN Clustering on vehicle ENU positions
    try:
        valid_enu_for_dbscan = subset_df[["e_vehicle", "n_vehicle"]].to_numpy()
        clustering_model = DBSCAN(eps=DBSCAN_EPS, min_samples=DBSCAN_MIN_SAMPLES)
        subset_df["cluster_id"] = clustering_model.fit_predict(valid_enu_for_dbscan)
    except ValueError as e_dbscan:  # Should not happen if we check len(subset_df)
        print(
            f"  Error during DBSCAN for class '{obj_type_being_processed}': {e_dbscan}. Skipping this class."
        )
        continue

    num_clusters_found = subset_df["cluster_id"].nunique() - (
        1 if -1 in subset_df["cluster_id"].unique() else 0
    )
    print(
        f"  DBSCAN found {num_clusters_found} potential object clusters (eps={DBSCAN_EPS}m)."
    )

    for cluster_label, cluster_group_df in subset_df.groupby("cluster_id"):
        if cluster_label == -1:  # Noise points from DBSCAN
            continue

        print(
            f"\n  Cluster ID: {cluster_label} (Class: '{obj_type_being_processed}', Detections in cluster: {len(cluster_group_df)})"
        )
        print(
            "    Thumbnails in this cluster:",
            ", ".join(cluster_group_df["thumb"].head(10).tolist()),
        )  # Show first 10

        if (
            len(cluster_group_df) < DBSCAN_MIN_SAMPLES
        ):  # Should be caught by DBSCAN min_samples, but good check
            print(
                f"    Skipping cluster {cluster_label}, not enough samples ({len(cluster_group_df)})."
            )
            continue

        # --- Interactive Debug Option ---
        user_choice = ""
        while user_choice not in ["y", "n", "q"]:
            user_choice = (
                input(
                    f"    Generate detailed debug map for this cluster? (y/n/q to quit all): "
                )
                .strip()
                .lower()
            )
            if user_choice not in ["y", "n", "q"]:
                print("    Invalid input. Please enter 'y', 'n', or 'q'.")

        if user_choice == "q":
            print("Quitting script as per user request.")
            sys.exit(0)

        # Prepare data for triangulation
        vehicle_enu_pts_cluster = cluster_group_df[
            ["e_vehicle", "n_vehicle"]
        ].to_numpy()
        bearings_deg_cluster = (
            cluster_group_df.bearing.to_numpy()
        )  # Keep degrees for printing
        bearings_rad_cluster = np.deg2rad(bearings_deg_cluster)

        print(
            f"      DEBUG: Inputs to INTERSECTION function for Cluster {cluster_label} (Class: {obj_type_being_processed}):"
        )
        for i in range(len(vehicle_enu_pts_cluster)):
            print(
                f"        Point {i+1} (Veh ENU): E={vehicle_enu_pts_cluster[i,0]:.2f}, N={vehicle_enu_pts_cluster[i,1]:.2f} | Bearing (deg): {bearings_deg_cluster[i]:.2f} | Bearing (rad): {bearings_rad_cluster[i]:.3f}"
            )
            # Also print the original vehicle lat/lon for this point for cross-referencing
            original_lat = cluster_group_df.iloc[i]["lat"]
            original_lon = cluster_group_df.iloc[i]["lon"]
            print(
                f"          Original Veh Lat/Lon: {original_lat:.6f}, {original_lon:.6f} (Thumb: {cluster_group_df.iloc[i]['thumb']})"
            )

        # Perform triangulation
        e_intersect, n_intersect = intersection(
            vehicle_enu_pts_cluster, bearings_rad_cluster
        )  # This is the call we're interested in

        if not (pd.isna(e_intersect) or pd.isna(n_intersect)):
            print(
                f"      DEBUG: Output from INTERSECTION (ENU): E_intersect={e_intersect:.2f}, N_intersect={n_intersect:.2f}"
            )
            obj_lat_triangulated, obj_lon_triangulated = enu_to_llh(
                e_intersect, n_intersect, ref_lat0, ref_lon0
            )
            # ... rest of the logic ...
        else:
            print(
                f"      DEBUG: INTERSECTION function returned NaN for Cluster {cluster_label}."
            )

        if user_choice == "y":
            map_cluster_detail = folium.Map(
                location=[ref_lat0, ref_lon0], zoom_start=18
            )
            folium.PolyLine(
                vehicle_path_coords.tolist(), color="blue", weight=1.5, opacity=0.6
            ).add_to(map_cluster_detail)

            print("      Visualizing individual rays for this cluster on detail map:")
            for _, det_row in cluster_group_df.iterrows():
                veh_lat_det, veh_lon_det = (
                    det_row["lat"],
                    det_row["lon"],
                )  # Original vehicle LL for this detection
                bearing_to_obj_det = det_row["bearing"]

                folium.Marker(
                    [veh_lat_det, veh_lon_det],
                    tooltip=f"Veh: {det_row['thumb']}\nBearing: {bearing_to_obj_det:.1f}",
                    icon=folium.Icon(color="gray", icon="camera", prefix="fa"),
                ).add_to(map_cluster_detail)

                ray_end_point = geodesic(meters=150).destination(
                    (veh_lat_det, veh_lon_det), bearing_to_obj_det
                )
                folium.PolyLine(
                    [
                        (veh_lat_det, veh_lon_det),
                        (ray_end_point.latitude, ray_end_point.longitude),
                    ],
                    color="purple",
                    weight=2,
                    opacity=0.9,
                    popup=f"Ray from {det_row['thumb']}\nBearing: {bearing_to_obj_det:.1f}°",
                ).add_to(map_cluster_detail)

            if not (pd.isna(e_intersect) or pd.isna(n_intersect)):
                obj_lat_triangulated_detail, obj_lon_triangulated_detail = enu_to_llh(
                    e_intersect, n_intersect, ref_lat0, ref_lon0
                )
                if not (
                    pd.isna(obj_lat_triangulated_detail)
                    or pd.isna(obj_lon_triangulated_detail)
                ):
                    print(
                        f"      Triangulated ENU: ({e_intersect:.2f}, {n_intersect:.2f})"
                    )
                    print(
                        f"      Triangulated LLH: ({obj_lat_triangulated_detail:.6f}, {obj_lon_triangulated_detail:.6f})"
                    )
                    folium.Marker(
                        [obj_lat_triangulated_detail, obj_lon_triangulated_detail],
                        popup=f"Triangulated: {obj_type_being_processed}\nCluster: {cluster_label}",
                        icon=folium.Icon(color="red", icon="star"),
                    ).add_to(map_cluster_detail)
                else:
                    print(
                        "      ENU to LLH conversion failed for triangulated point on detail map."
                    )
            else:
                print(
                    "      Triangulation failed for this cluster (intersection returned NaN) for detail map."
                )

            cluster_map_filename = f"debug_{obj_type_being_processed.replace(' ', '_')}_cluster{cluster_label}.html"
            cluster_map_path = OUTPUT_DEBUG_DIR / cluster_map_filename
            map_cluster_detail.save(cluster_map_path)
            print(f"      Saved detailed cluster map to: {cluster_map_path}")
        # --- End Interactive Debug Option ---

        # Add to main map and CSV if triangulation was successful
        if not (pd.isna(e_intersect) or pd.isna(n_intersect)):
            obj_lat_triangulated, obj_lon_triangulated = enu_to_llh(
                e_intersect, n_intersect, ref_lat0, ref_lon0
            )

            if not (pd.isna(obj_lat_triangulated) or pd.isna(obj_lon_triangulated)):
                triangulated_objects_data.append(
                    {
                        "lat": obj_lat_triangulated,
                        "lon": obj_lon_triangulated,
                        "class": obj_type_being_processed,
                        "cluster_id": cluster_label,
                        "num_detections": len(cluster_group_df),
                    }
                )

                # Draw rays from vehicle positions to the successfully triangulated point on the main map
                for _, det_row_main in cluster_group_df.iterrows():
                    folium.PolyLine(
                        [
                            (det_row_main.lat, det_row_main.lon),
                            (obj_lat_triangulated, obj_lon_triangulated),
                        ],
                        color="darkgreen",
                        weight=0.7,
                        opacity=0.4,
                        dash_array="5,5",
                    ).add_to(map_all_objects)

                if (
                    obj_type_being_processed != "not_a_sign"
                ):  # Filter out 'not_a_sign' from getting prominent markers
                    # Prepare thumbnail HTML for popup. Assuming thumbnails are in 'detections/thumbnails/'
                    # The HTML paths should be relative to where 'map_all_triangulated_objects.html' will be saved.
                    html_thumbs_list = [
                        f"<img src='../{DETECTION_DIR}/thumbnails/{thumb_name}' width='80' style='margin:1px; border:1px solid #ddd;' alt='{thumb_name}'>"
                        for thumb_name in cluster_group_df.thumb.head(
                            min(len(cluster_group_df), 4)
                        ).tolist()  # Show up to 4 thumbs
                    ]
                    popup_html_content = (
                        f"<b>{obj_type_being_processed}</b> (Cluster {cluster_label})<br>"
                        f"Detections: {len(cluster_group_df)}<br>"
                        + "".join(html_thumbs_list)
                    )
                    popup = folium.Popup(popup_html_content, max_width=350)

                    marker_color = (
                        "orange"
                        if obj_type_being_processed == "street-light"
                        else "purple"
                    )
                    folium.Marker(
                        [obj_lat_triangulated, obj_lon_triangulated],
                        popup=popup,
                        icon=folium.Icon(
                            color=marker_color,
                            icon=(
                                "map-signs"
                                if "sign" in obj_type_being_processed
                                else "lightbulb-o"
                            ),
                            prefix="fa",
                        ),
                    ).add_to(map_all_objects)

                # Add a small circle marker for all triangulated points for consistency
                folium.CircleMarker(
                    location=(obj_lat_triangulated, obj_lon_triangulated),
                    radius=3,
                    color="black",
                    weight=0.5,
                    fill=True,
                    fill_color="yellow",
                    fill_opacity=0.7,
                    tooltip=f"{obj_type_being_processed} - C{cluster_label}",
                ).add_to(map_all_objects)
            else:
                if (
                    user_choice != "y"
                ):  # Avoid double printing if already printed for detail map
                    print(
                        f"    ENU to LLH conversion failed for cluster {cluster_label} for main map."
                    )
        else:
            if user_choice != "y":
                print(
                    f"    Intersection failed for cluster {cluster_label} for main map."
                )

    print("----------------------------------------------------")

# --- Save Outputs ---
main_map_filename = "map_all_triangulated_objects.html"
map_all_objects.save(OUTPUT_DEBUG_DIR / main_map_filename)
print(
    f"\nMain map with all triangulated objects saved to: {OUTPUT_DEBUG_DIR / main_map_filename}"
)

if triangulated_objects_data:
    objects_output_df = pd.DataFrame(triangulated_objects_data)
    csv_output_path = OUTPUT_DEBUG_DIR / "objects_triangulated_interactive.csv"
    objects_output_df.to_csv(csv_output_path, index=False)
    print(f"Triangulated objects CSV saved to: {csv_output_path}")
else:
    print("No objects were successfully triangulated to save to CSV.")

print("\nInteractive triangulation script finished.")

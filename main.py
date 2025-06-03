#!/usr/bin/env python3
"""
Unified pipeline for processing 360-degree video files to detect and triangulate objects.

Usage: python main.py <video_file_path>

This script combines the following steps:
1. Extract dataset (image frames) from the video
2. Extract telemetry data from the video
3. Run object detection on the frames
4. Triangulate objects and create an interactive map
"""

import argparse
import bisect
import csv
import json
import math
import re
import sys
from pathlib import Path

import cv2
import folium
import numpy as np
import pandas as pd
import py360convert as c2
import telemetry_parser
from geopy.distance import geodesic
from sklearn.cluster import DBSCAN
from ultralytics import YOLO

from classify_traffic_sign import classify_sign

# PARAMETERS
horizontal_fov_deg = 90.0
time_offset = 3.0  # time offset in seconds to adjust telemetry timestamps
HDG_DT = 1.0
THUMB_SIZE = (120, 120)


def extract_video_id(video_path: Path) -> str:
    """Extract last 3 digits from filename as video ID (e.g., '220919_111512835.mp4' -> '835')"""
    name = video_path.stem
    match = re.search(r"(\d{3})\D*$", name)
    if match:
        return match.group(1)
    else:
        # Fallback: use the whole filename without extension
        return name


def create_dataset(video_path: Path, dataset_id: str) -> Path:
    """Create image dataset from 360-degree video."""
    print(f"Creating dataset from {video_path}...")

    out_dir = Path("dataset") / dataset_id
    out_dir.mkdir(parents=True, exist_ok=True)

    cap = cv2.VideoCapture(str(video_path))
    fps = cap.get(cv2.CAP_PROP_FPS)
    print(f"FPS: {fps}")

    stride = max(1, int(round(fps * 0.5)))
    print(f"Frames to skip between snapshots: {stride}")

    fov, yaws, pitch = 90, [0, 90, 180, 270], 0
    frames_seen, pano_id = 0, 0
    meta = []

    while True:
        ok, frame = cap.read()
        if not ok:
            break
        if frames_seen % stride == 0:
            stamp = frames_seen / fps  # seconds since start
            for yaw in yaws:
                perspective = c2.e2p(
                    frame, fov_deg=fov, u_deg=yaw, v_deg=pitch, out_hw=(1024, 1024)
                )
                name = f"loc{pano_id:05d}_yaw{yaw:03d}.jpg"
                cv2.imwrite(
                    str(out_dir / name), perspective, [cv2.IMWRITE_JPEG_QUALITY, 92]
                )
                meta.append(
                    {"t": stamp, "yaw": yaw, "location_id": pano_id, "file": name}
                )
            pano_id += 1
        frames_seen += 1

    cap.release()
    (out_dir / "index.json").write_text(json.dumps(meta, indent=2))
    print(f"Dataset created: {out_dir}")
    return out_dir


def extract_telemetry(video_path: Path, dataset_id: str) -> Path:
    """Extract telemetry data from video."""
    print(f"Extracting telemetry from {video_path}...")

    telemetry_dir = Path("telemetry")
    telemetry_dir.mkdir(exist_ok=True)

    telemetry_file = telemetry_dir / f"telemetry_{dataset_id}.json"

    rec = telemetry_parser.Parser(str(video_path)).telemetry()[0]

    with open(telemetry_file, "w") as f:
        json.dump(rec, f, indent=2)

    print(f"Telemetry extracted: {telemetry_file}")
    return telemetry_file


def coord_at(
    sec: float, times: np.ndarray, lats: np.ndarray, lons: np.ndarray
) -> tuple[float, float]:
    """Get coordinates at a specific time."""
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
    """Calculate bearing between two points."""
    phi1, phi2 = map(math.radians, (lat1, lat2))
    dlon = math.radians(lon2 - lon1)
    y = math.sin(dlon) * math.cos(phi2)
    x = math.cos(phi1) * math.sin(phi2) - math.sin(phi1) * math.cos(phi2) * math.cos(
        dlon
    )
    return (math.degrees(math.atan2(y, x)) + 360) % 360


def run_detection(dataset_dir: Path, telemetry_file: Path, dataset_id: str) -> Path:
    """Run object detection on the dataset."""
    print("Running object detection...")

    # Load dataset index
    index = json.loads((dataset_dir / "index.json").read_text())

    # Load telemetry
    telemetry = json.loads(telemetry_file.read_text())["GPS"]["Data"]
    valid_fixes = [r for r in telemetry if r.get("is_acquired")]
    valid_fixes.sort(key=lambda r: r["unix_timestamp"])

    t0 = valid_fixes[0]["unix_timestamp"]
    times = np.array([r["unix_timestamp"] - t0 for r in valid_fixes])
    lats = np.array([r["lat"] for r in valid_fixes])
    lons = np.array([r["lon"] for r in valid_fixes])

    # Load models
    models = {
        "traffic-sign": YOLO(str(Path("models") / "yolov8s-traffic-sign.pt")),
        "street-light": YOLO(str(Path("models") / "yolov8n-oiv7.pt")),
    }
    class_filters = {"traffic-sign": [1, 6, 8, 10], "street-light": [497]}
    conf_thresholds = {"traffic-sign": 0.5, "street-light": 0.05}

    # Create detection directory
    detection_dir = Path("detections_" + dataset_id)
    detection_dir.mkdir(exist_ok=True)

    csv_writer = csv.writer(open(detection_dir / "detections_geo.csv", "w", newline=""))
    csv_writer.writerow(["lat", "lon", "bearing", "class", "conf", "frame", "thumb"])

    thumb_dir = detection_dir / "thumbnails"
    thumb_dir.mkdir(exist_ok=True)

    for record in index:
        sec = record["t"]
        lat0, lon0 = coord_at(sec, times, lats, lons)
        heading = bearing(
            *coord_at(sec - HDG_DT, times, lats, lons),
            *coord_at(sec + HDG_DT, times, lats, lons),
        )

        img = cv2.imread(str(dataset_dir / record["file"]))
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

    print(f"Detection completed: {detection_dir}")
    return detection_dir


def bearing_deg(lat1, lon1, lat2, lon2):
    """Calculate bearing in degrees."""
    phi1, phi2 = map(math.radians, (lat1, lat2))
    dl = math.radians(lon2 - lon1)
    y = math.sin(dl) * math.cos(phi2)
    x = math.cos(phi1) * math.sin(phi2) - math.sin(phi1) * math.cos(phi2) * math.cos(dl)
    return (math.degrees(math.atan2(y, x)) + 360) % 360


def llh_to_enu(lat, lon, lat0, lon0):
    """Convert lat/lon to East-North-Up coordinates."""
    d = geodesic((lat0, lon0), (lat, lon)).meters
    br = math.radians(bearing_deg(lat0, lon0, lat, lon))
    return d * math.sin(br), d * math.cos(br)


def intersection(pts_enu_vehicle, bearings_rad):
    """Calculate intersection point of multiple lines."""
    if not isinstance(pts_enu_vehicle, np.ndarray) or not isinstance(
        bearings_rad, np.ndarray
    ):
        return np.nan, np.nan
    if pts_enu_vehicle.ndim != 2 or pts_enu_vehicle.shape[1] != 2:
        return np.nan, np.nan
    if bearings_rad.ndim != 1:
        return np.nan, np.nan
    if (
        pts_enu_vehicle.shape[0] < 2
        or bearings_rad.shape[0] < 2
        or pts_enu_vehicle.shape[0] != bearings_rad.shape[0]
    ):
        return np.nan, np.nan

    cos_b = np.cos(bearings_rad)
    sin_b = np.sin(bearings_rad)

    A_matrix = np.column_stack((cos_b, -sin_b))
    b_vector = pts_enu_vehicle[:, 0] * cos_b - pts_enu_vehicle[:, 1] * sin_b

    try:
        result, residuals, rank, singular_values = np.linalg.lstsq(
            A_matrix, b_vector, rcond=None
        )
        return result[0], result[1]  # E_intersect, N_intersect
    except np.linalg.LinAlgError as e:
        print(f"Linear algebra error during intersection: {e}")
        return np.nan, np.nan


def enu_to_llh(e, n, lat0, lon0):
    """Convert East-North-Up to lat/lon."""
    az = math.degrees(math.atan2(e, n)) % 360
    p = geodesic(meters=math.hypot(e, n)).destination((lat0, lon0), az)
    return p.latitude, p.longitude


def triangulate_objects(detection_dir: Path, telemetry_file: Path, dataset_id: str):
    """Triangulate objects and create interactive map."""
    print("Triangulating objects...")

    # Suppress pandas warnings
    pd.options.mode.chained_assignment = None

    detection_df = pd.read_csv(detection_dir / "detections_geo.csv")
    telemetry = json.loads(telemetry_file.read_text())["GPS"]["Data"]
    path = np.array([(e["lat"], e["lon"]) for e in telemetry])
    lat0, lon0 = path[0]

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

            # draw the cluster's rays
            for _, detection in group.iterrows():
                folium.PolyLine(
                    [(detection.lat, detection.lon), (lat, lon)],
                    color="green",
                    weight=1,
                    opacity=0.4,
                ).add_to(map_obj)

            if obj_type != "not_a_sign":
                thumbs = "<br>".join(
                    f"<img src='thumbnails/{t}' width='120'>"
                    for t in group.thumb.head(3)
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

            objects.append(
                {
                    "lat": lat,
                    "lon": lon,
                    "class": obj_type,
                    "num_images": len(group),
                }
            )

    pd.DataFrame(objects, columns=["lat", "lon", "class", "num_images"]).to_csv(
        detection_dir / "objects.csv", index=False
    )
    map_obj.save(detection_dir / "objects_map.html")
    print(f"Triangulation complete: {detection_dir / 'objects_map.html'}")


def main():
    """Main pipeline function."""
    parser = argparse.ArgumentParser(
        description="Process 360-degree video for object detection and triangulation"
    )
    parser.add_argument("video_path", help="Path to the input video file")

    args = parser.parse_args()

    video_path = Path(args.video_path)
    if not video_path.exists():
        print(f"Error: Video file {video_path} does not exist.")
        sys.exit(1)

    # Extract dataset ID from video filename
    dataset_id = extract_video_id(video_path)
    print(f"Processing video: {video_path} (ID: {dataset_id})")

    try:
        # Step 1: Create dataset
        dataset_dir = create_dataset(video_path, dataset_id)

        # Step 2: Extract telemetry
        telemetry_file = extract_telemetry(video_path, dataset_id)

        # Step 3: Run detection
        detection_dir = run_detection(dataset_dir, telemetry_file, dataset_id)

        # Step 4: Triangulate objects
        triangulate_objects(detection_dir, telemetry_file, dataset_id)

        print(f"\nPipeline completed successfully!")
        print(f"Results saved in: {detection_dir}")
        print(f"Open {detection_dir / 'objects_map.html'} to view the interactive map")

    except Exception as e:
        print(f"Error during processing: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()

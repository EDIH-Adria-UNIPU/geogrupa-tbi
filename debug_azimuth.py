# ==== run_detection_debug_all_yaws.py ====
import bisect
import json
import math
from pathlib import Path

import cv2
import folium  # For map visualization
import numpy as np
from geopy.distance import geodesic  # For drawing lines of specific length on map
from ultralytics import YOLO

# --- DEBUG PARAMETERS ---
DEBUG_TARGET_LOCATION_ID = 81  # Panorama/location ID to debug
# The script will process all yaws for this location_id

# Set to True to enable detailed debugging.
ENABLE_DETAILED_DEBUG = True
# --- END DEBUG PARAMETERS ---


# --- CONFIGURATION PARAMETERS (Adjust as needed) ---
CONFIG_CAM_YAW_OFFSET = (
    0.0  # Start with 0.0 for debugging. Adjust if you know the actual offset.
)
CONFIG_TIME_OFFSET = 5.0  # GPS-to-video time offset in seconds
CONFIG_HDG_DT = 1.0  # Time delta for heading calculation
CONFIG_HORIZONTAL_FOV_DEG = 90.0
# --- END CONFIGURATION PARAMETERS ---

# --- PATHS (Adjust if your structure is different) ---
DATASET_ROOT_DIR = Path("dataset") / "250"
TELEMETRY_FILE_PATH = Path("telemetry") / "telemetry_250.json"
MODELS_DIR = Path("models")
OUTPUT_DEBUG_DIR = Path("detections_debug_all_yaws")
# --- END PATHS ---

OUTPUT_DEBUG_DIR.mkdir(exist_ok=True)


def load_json_data(file_path, description):
    if not file_path.exists():
        print(f"ERROR: {description} file not found at {file_path}")
        exit(1)
    try:
        return json.loads(file_path.read_text())
    except json.JSONDecodeError as e:
        print(f"ERROR: Could not decode JSON from {file_path}. Error: {e}")
        exit(1)


index_data = load_json_data(DATASET_ROOT_DIR / "index.json", "Index")
telemetry_full_data = load_json_data(TELEMETRY_FILE_PATH, "Telemetry")

if "GPS" not in telemetry_full_data or "Data" not in telemetry_full_data["GPS"]:
    print(
        f"ERROR: Telemetry file {TELEMETRY_FILE_PATH} does not have the expected GPS.Data structure."
    )
    exit(1)
telemetry_gps_entries = telemetry_full_data["GPS"]["Data"]

valid_fixes = [
    r
    for r in telemetry_gps_entries
    if r.get("is_acquired") and all(k in r for k in ["unix_timestamp", "lat", "lon"])
]
if not valid_fixes:
    print(f"ERROR: No valid GPS fixes found in {TELEMETRY_FILE_PATH}.")
    exit(1)

valid_fixes.sort(key=lambda r: r["unix_timestamp"])
t0_gps = valid_fixes[0]["unix_timestamp"]
gps_times_rel = np.array([r["unix_timestamp"] - t0_gps for r in valid_fixes])
gps_lats = np.array([r["lat"] for r in valid_fixes])
gps_lons = np.array([r["lon"] for r in valid_fixes])


def get_interpolated_coord(video_time_sec: float) -> tuple[float, float]:
    # Apply the fixed time offset to align video time with GPS time system
    # The video_time_sec is relative to video start, gps_times_rel is relative to t0_gps.
    # We assume video_time_sec = 0 corresponds to (t0_gps - CONFIG_TIME_OFFSET) in absolute unix time.
    # So, the target absolute unix time for GPS lookup is (t0_gps - CONFIG_TIME_OFFSET) + video_time_sec.
    # The time to look up in gps_times_rel is:
    # ((t0_gps - CONFIG_TIME_OFFSET) + video_time_sec) - t0_gps = video_time_sec - CONFIG_TIME_OFFSET

    # Correction: The original `time_offset` logic in `run_detection.py` added the offset.
    # `sec += time_offset` means if video frame is at `sec`, we look for GPS at `sec + time_offset`.
    # This implies `t0` in `run_detection.py` was `valid_fixes[0]["unix_timestamp"] - time_offset`.
    # Let's stick to that logic for consistency.
    # `t_video_start_abs = valid_fixes[0]["unix_timestamp"] - time_offset`
    # `t_gps_lookup_abs = t_video_start_abs + video_time_sec`
    # `t_gps_lookup_rel_to_t0_gps = t_gps_lookup_abs - t0_gps`
    #                              `= (valid_fixes[0]["unix_timestamp"] - time_offset) + video_time_sec - valid_fixes[0]["unix_timestamp"]`
    #                              `= video_time_sec - time_offset`
    # This seems to be what `times = np.array([r["unix_timestamp"] - t0 for r in valid_fixes])` and `sec += time_offset` achieved.
    # However, the `coord_at` function in `run_detection.py` had `sec += time_offset`
    # and `times` was `r["unix_timestamp"] - t0_gps`. So, `idx = bisect.bisect_left(times, sec_adjusted)`.
    # This means we are looking up `video_timestamp_from_index_json + CONFIG_TIME_OFFSET` in the `gps_times_rel` array.

    gps_lookup_time = video_time_sec + CONFIG_TIME_OFFSET  # This matches original logic

    idx = bisect.bisect_left(gps_times_rel, gps_lookup_time)

    if idx <= 0:
        return gps_lats[0], gps_lons[0]
    if idx >= len(gps_times_rel):
        return gps_lats[-1], gps_lons[-1]

    t_lo, t_hi = gps_times_rel[idx - 1], gps_times_rel[idx]
    if t_hi == t_lo:  # Avoid division by zero if timestamps are identical
        return gps_lats[idx - 1], gps_lons[idx - 1]

    w = (gps_lookup_time - t_lo) / (t_hi - t_lo)
    lat_interp = gps_lats[idx - 1] * (1 - w) + gps_lats[idx] * w
    lon_interp = gps_lons[idx - 1] * (1 - w) + gps_lons[idx] * w
    return lat_interp, lon_interp


def calculate_bearing(lat1, lon1, lat2, lon2) -> float:
    phi1, phi2 = map(math.radians, (lat1, lat2))
    dlon = math.radians(lon2 - lon1)
    y = math.sin(dlon) * math.cos(phi2)
    x = math.cos(phi1) * math.sin(phi2) - math.sin(phi1) * math.cos(phi2) * math.cos(
        dlon
    )
    return (math.degrees(math.atan2(y, x)) + 360) % 360


# Load models
try:
    models = {
        "traffic-sign": YOLO(str(MODELS_DIR / "yolov8s-traffic-sign.pt")),
        "street-light": YOLO(str(MODELS_DIR / "yolov8n-oiv7.pt")),
    }
except Exception as e:
    print(f"Error loading YOLO models from {MODELS_DIR}: {e}")
    exit(1)

# These should match your model's classes
# For yolov8s-traffic-sign.pt, typical classes might be {0: 'priority road', 1: 'stop', ...}
# For yolov8n-oiv7.pt, street light is often 497. Check your model's .yaml or model.names
class_filters = {
    "traffic-sign": None,
    "street-light": [497],
}  # None means detect all classes for traffic signs
conf_thresholds = {
    "traffic-sign": 0.3,
    "street-light": 0.05,
}  # Adjust confidence as needed

print(f"--- Debugging Location ID: {DEBUG_TARGET_LOCATION_ID} ---")
print(
    f"Using CAM_YAW_OFFSET: {CONFIG_CAM_YAW_OFFSET}, TIME_OFFSET: {CONFIG_TIME_OFFSET}"
)

found_target_location = False
for record in index_data:
    if record["location_id"] != DEBUG_TARGET_LOCATION_ID:
        continue

    found_target_location = True
    pano_loc_id = record["location_id"]
    frame_yaw = record["yaw"]
    img_filename = record["file"]
    video_time_at_frame = record["t"]  # Timestamp from index.json

    print(
        f"\nProcessing: LocationID={pano_loc_id}, Yaw={frame_yaw}, File='{img_filename}', VideoTime={video_time_at_frame:.2f}s"
    )

    # 1. Vehicle Position at frame time
    cam_lat, cam_lon = get_interpolated_coord(video_time_at_frame)
    print(f"  Vehicle Position (Lat, Lon): {cam_lat:.6f}, {cam_lon:.6f}")

    # 2. Vehicle Heading
    # For heading, GPS points are queried relative to video_time_at_frame
    pos_before_heading = get_interpolated_coord(video_time_at_frame - CONFIG_HDG_DT)
    pos_after_heading = get_interpolated_coord(video_time_at_frame + CONFIG_HDG_DT)
    vehicle_heading = calculate_bearing(*pos_before_heading, *pos_after_heading)
    print(f"  Vehicle Heading: {vehicle_heading:.2f}°")

    img_path = DATASET_ROOT_DIR / img_filename
    if not img_path.exists():
        print(f"  ERROR: Image file not found: {img_path}")
        continue
    img = cv2.imread(str(img_path))
    img_h, img_w, _ = img.shape

    # --- Object Detection Loop ---
    object_debugged_for_this_yaw = False
    for obj_model_type, yolo_model in models.items():
        if (
            object_debugged_for_this_yaw and ENABLE_DETAILED_DEBUG
        ):  # Only debug one object per yaw angle
            break

        print(f"  Running model: {obj_model_type}")
        results = yolo_model.predict(
            img,
            classes=class_filters.get(obj_model_type),
            conf=conf_thresholds[obj_model_type],
            verbose=False,
        )[0]

        if not results.boxes:
            print(f"    No objects detected by {obj_model_type} model.")
            continue

        print(f"    {obj_model_type} model detected {len(results.boxes)} objects.")

        # Debug the FIRST object detected by this model in this image
        first_box = results.boxes[0]
        box_data = first_box.xyxy[0]
        x0, y0, x1, y1 = [int(v) for v in box_data]
        obj_cx = (x0 + x1) / 2.0
        detected_class_id = int(first_box.cls[0])
        detected_class_name = yolo_model.names[detected_class_id]
        obj_confidence = float(first_box.conf[0])

        print(
            f"    Debugging FIRST detected object: Class='{detected_class_name}', Conf={obj_confidence:.3f}, CX={obj_cx:.1f}"
        )

        # 3. Angular offset within FoV
        norm_x = 2 * (obj_cx / img_w - 0.5)
        half_fov_rad = math.radians(CONFIG_HORIZONTAL_FOV_DEG / 2)
        angular_offset_in_fov = math.degrees(math.atan(norm_x * math.tan(half_fov_rad)))
        print(
            f"      Normalized X: {norm_x:.3f}, Angular Offset in FoV: {angular_offset_in_fov:.2f}°"
        )

        # 4. Final Azimuth Calculation
        # Base direction of camera's 0-deg panoramic view = vehicle_heading + CONFIG_CAM_YAW_OFFSET
        # Center of current perspective view = base_camera_direction + frame_yaw
        # Direction to object = perspective_center_direction + angular_offset_in_fov
        object_azimuth = (
            vehicle_heading + CONFIG_CAM_YAW_OFFSET + frame_yaw + angular_offset_in_fov
        ) % 360
        print(
            f"      Base Cam Dir (VehHdg+CamYawOff): {(vehicle_heading + CONFIG_CAM_YAW_OFFSET) % 360:.2f}°"
        )
        print(
            f"      Perspective Center Dir (BaseCamDir+FrameYaw): {(vehicle_heading + CONFIG_CAM_YAW_OFFSET + frame_yaw) % 360:.2f}°"
        )
        print(f"      FINAL OBJECT AZIMUTH: {object_azimuth:.2f}°")

        # 5. Create Visualization Map
        debug_map = folium.Map(
            location=[cam_lat, cam_lon], zoom_start=19, tiles="OpenStreetMap"
        )
        folium.Marker(
            [cam_lat, cam_lon],
            popup=f"Vehicle @ LocID {pano_loc_id}, Yaw {frame_yaw}<br>Time: {video_time_at_frame:.2f}s<br>"
            f"LL: {cam_lat:.6f}, {cam_lon:.6f}",
            icon=folium.Icon(color="blue", icon="car", prefix="fa"),
        ).add_to(debug_map)

        # Vehicle Heading (Green)
        ep_heading = geodesic(meters=50).destination(
            (cam_lat, cam_lon), vehicle_heading
        )
        folium.PolyLine(
            [(cam_lat, cam_lon), (ep_heading.latitude, ep_heading.longitude)],
            color="green",
            weight=3,
            opacity=0.8,
            popup=f"Veh.Heading: {vehicle_heading:.1f}°",
        ).add_to(debug_map)

        # Camera System "Front" (Light Blue Dashed)
        cam_sys_front_bearing = (vehicle_heading + CONFIG_CAM_YAW_OFFSET) % 360
        ep_csf = geodesic(meters=45).destination(
            (cam_lat, cam_lon), cam_sys_front_bearing
        )
        folium.PolyLine(
            [(cam_lat, cam_lon), (ep_csf.latitude, ep_csf.longitude)],
            color="lightblue",
            weight=3,
            opacity=0.8,
            dash_array="5, 5",
            popup=f"CamSysFront (H+CYO): {cam_sys_front_bearing:.1f}°",
        ).add_to(debug_map)

        # Perspective View Center (Orange)
        persp_center_bearing = (
            vehicle_heading + CONFIG_CAM_YAW_OFFSET + frame_yaw
        ) % 360
        ep_pc = geodesic(meters=40).destination(
            (cam_lat, cam_lon), persp_center_bearing
        )
        folium.PolyLine(
            [(cam_lat, cam_lon), (ep_pc.latitude, ep_pc.longitude)],
            color="orange",
            weight=3,
            opacity=0.8,
            popup=f"Persp.Center (H+CYO+FY): {persp_center_bearing:.1f}°",
        ).add_to(debug_map)

        # Object Azimuth (Red)
        ep_obj = geodesic(meters=50).destination((cam_lat, cam_lon), object_azimuth)
        folium.PolyLine(
            [(cam_lat, cam_lon), (ep_obj.latitude, ep_obj.longitude)],
            color="red",
            weight=4,
            opacity=0.9,
            popup=f"Obj.Azimuth: {object_azimuth:.1f}°<br>Class: {detected_class_name}<br>"
            f"FoV Offset: {angular_offset_in_fov:.1f}°",
        ).add_to(debug_map)

        map_filename = f"debug_loc{pano_loc_id}_yaw{frame_yaw}_obj{detected_class_name.replace(' ', '_')}_cx{int(obj_cx)}_az{int(object_azimuth)}.html"
        map_save_path = OUTPUT_DEBUG_DIR / map_filename
        debug_map.save(map_save_path)
        print(f"      Saved debug map: {map_save_path}")

        # 6. Show image with detection
        img_display = img.copy()
        cv2.rectangle(img_display, (x0, y0), (x1, y1), (0, 0, 255), 2)  # BBox in Red
        cv2.line(
            img_display, (img_w // 2, 0), (img_w // 2, img_h), (0, 255, 0), 1
        )  # Image center (Green)
        cv2.line(
            img_display, (int(obj_cx), 0), (int(obj_cx), img_h), (255, 100, 0), 1
        )  # Object cx (Blue)
        cv2.putText(
            img_display,
            f"{detected_class_name} ({obj_confidence:.2f}) @cx{obj_cx:.0f}",
            (x0, y0 - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (255, 0, 0),
            2,
        )

        # Resize for display if too large
        max_disp_h, max_disp_w = 800, 1000
        scale = min(max_disp_h / img_h, max_disp_w / img_w, 1.0)
        if scale < 1.0:
            img_display = cv2.resize(img_display, None, fx=scale, fy=scale)

        cv2.imshow(
            f"Debug: Loc{pano_loc_id} Yaw{frame_yaw} Obj:{detected_class_name}",
            img_display,
        )
        print(
            "      Showing image with detection. Press any key in the image window to continue to next yaw/model..."
        )
        cv2.waitKey(0)
        cv2.destroyWindow(
            f"Debug: Loc{pano_loc_id} Yaw{frame_yaw} Obj:{detected_class_name}"
        )

        object_debugged_for_this_yaw = (
            True  # Mark that we've debugged one object for this yaw angle
        )
        if (
            ENABLE_DETAILED_DEBUG
        ):  # if we want to break after the first model finds something for this yaw
            break


if not found_target_location:
    print(
        f"ERROR: No records found for Location ID {DEBUG_TARGET_LOCATION_ID} in index file."
    )
else:
    print(f"\n--- Debugging for Location ID {DEBUG_TARGET_LOCATION_ID} completed. ---")
    print(
        f"Debug maps and console output generated. Check the '{OUTPUT_DEBUG_DIR}' folder."
    )

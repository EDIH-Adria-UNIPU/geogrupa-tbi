import json
import pathlib

import cv2
import py360convert as c2

video_path = pathlib.Path("220919_111512835.mp4")
output_root = pathlib.Path("dataset")
output_root.mkdir(exist_ok=True)

capture = cv2.VideoCapture(str(video_path))
fps = capture.get(cv2.CAP_PROP_FPS)
stride = int(round(fps * 1))  # one panorama every second
fov = 90  # degrees
yaws = [0, 90, 180, 270]  # four headings
pitch = 0  # looking straight ahead

index, panorama_id = 0, 0
metadata = []

while True:
    grabbed, frame = capture.read()
    if not grabbed:
        break
    if index % stride == 0:
        for yaw in yaws:
            perspective = c2.e2p(
                frame, fov_deg=fov, u_deg=yaw, v_deg=pitch, out_hw=(1024, 1024)
            )
            name = f"loc{panorama_id:05d}_yaw{yaw:03d}.jpg"
            cv2.imwrite(
                str(output_root / name), perspective, [cv2.IMWRITE_JPEG_QUALITY, 92]
            )
            metadata.append({"location_id": panorama_id, "yaw": yaw, "file": name})
        panorama_id += 1
    index += 1

capture.release()
(output_root / "index.json").write_text(json.dumps(metadata, indent=2))

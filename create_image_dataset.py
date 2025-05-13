import json
import pathlib

import cv2
import py360convert as c2

video_path = pathlib.Path("220919_111808250.mp4")
out_dir = pathlib.Path("250")
out_dir.mkdir(exist_ok=True)

cap = cv2.VideoCapture(str(video_path))
fps = cap.get(cv2.CAP_PROP_FPS)
stride = max(1, int(round(fps * 0.5)))  # 0.5 s
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
            meta.append({"t": stamp, "yaw": yaw, "location_id": pano_id, "file": name})
        pano_id += 1
    frames_seen += 1

cap.release()
(out_dir / "index.json").write_text(json.dumps(meta, indent=2))

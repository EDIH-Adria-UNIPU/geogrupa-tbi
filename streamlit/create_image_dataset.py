import json
import pathlib
import imageio
import py360convert as c2

def create_dataset(video_path):
    out_dir = pathlib.Path("250")
    out_dir.mkdir(exist_ok=True)

    reader = imageio.get_reader(video_path)
    fps = reader.get_meta_data().get("fps", 30)  # Fallback ako nema FPS-a

    print(f"FPS: {fps}")

    stride = max(1, int(round(fps * 30)))
    print(f"How many frames to skip between snapshots: {stride}")

    fov, yaws, pitch = 90, [0, 90, 180, 270], 0

    pano_id = 0
    meta = []

    for frame_idx, frame in enumerate(reader):
        if frame_idx % stride != 0:
            continue

        stamp = frame_idx / fps  # seconds since start

        for yaw in yaws:
            perspective = c2.e2p(
                frame, fov_deg=fov, u_deg=yaw, v_deg=pitch, out_hw=(1024, 1024)
            )
            name = f"loc{pano_id:05d}_yaw{yaw:03d}.jpg"
            imageio.imwrite(out_dir / name, perspective)
            meta.append({
                "t": stamp,
                "yaw": yaw,
                "location_id": pano_id,
                "file": name
            })
        pano_id += 1

    reader.close()

    (out_dir / "index.json").write_text(json.dumps(meta, indent=2))

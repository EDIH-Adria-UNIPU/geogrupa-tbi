import json
import pathlib
import imageio
import py360convert as c2

def process_batch(reader, start_idx, end_idx, fps, stride, out_dir, pano_id_start):
    fov, yaws, pitch = 90, [0, 90, 180, 270], 0
    meta = []
    pano_id = pano_id_start
    
    for frame_idx in range(start_idx, min(end_idx, reader.count_frames())):
        if frame_idx % stride != 0:
            continue
        
        frame = reader.get_data(frame_idx)
        stamp = frame_idx / fps

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
    
    return meta, pano_id

def create_dataset(video_path, batch_size=10):
    out_dir = pathlib.Path("250")
    out_dir.mkdir(exist_ok=True)

    reader = imageio.get_reader(video_path)
    fps = reader.get_meta_data().get("fps", 30)
    
    try:
        num_frames = reader.count_frames()
    except Exception:
        # fallback ako count_frames ne radi
        print("Warning: count_frames nije podržan, procjena broja frameova na 1000")
        num_frames = 1000
    
    stride = max(1, int(round(fps * 0.5)))
    
    total_batches = (num_frames + batch_size - 1) // batch_size
    
    pano_id = 0
    all_meta = []

    for batch_num in range(total_batches):
        start_idx = batch_num * batch_size
        end_idx = start_idx + batch_size
        print(f"Processing batch {batch_num + 1}/{total_batches} frames {start_idx} to {end_idx}")
        batch_meta, pano_id = process_batch(reader, start_idx, end_idx, fps, stride, out_dir, pano_id)
        all_meta.extend(batch_meta)

    reader.close()

    (out_dir / "index.json").write_text(json.dumps(all_meta, indent=2))
    print("Gotovo!")


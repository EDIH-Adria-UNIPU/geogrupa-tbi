def create_dataset(video_path, batch_size=10, out_dir="250", max_batches=None):
    import pathlib
    import json
    import imageio
    import py360convert as c2

    out_dir = pathlib.Path(out_dir)
    out_dir.mkdir(exist_ok=True)

    reader = imageio.get_reader(video_path)
    fps = reader.get_meta_data().get("fps", 30)

    try:
        num_frames = reader.count_frames()
    except Exception:
        num_frames = 1000

    stride = max(1, int(round(fps * 2)))  # uzmi frame svakih 2 sekunde

    total_batches = (num_frames + batch_size - 1) // batch_size
    if max_batches:
        total_batches = min(total_batches, max_batches)

    all_meta = []
    pano_id = 0

    for batch_index in range(total_batches):
        print(f"Obrađujem batch {batch_index + 1}/{total_batches}")

        start_idx = batch_index * batch_size
        end_idx = start_idx + batch_size

        for frame_idx in range(start_idx, min(end_idx, num_frames)):
            if frame_idx % stride != 0:
                continue

            try:
                frame = reader.get_data(frame_idx)
            except Exception as e:
                print(f"Greška u frameu {frame_idx}: {e}")
                continue

            stamp = frame_idx / fps
            fov, yaws, pitch = 90, [0, 90, 180, 270], 0

            for yaw in yaws:
                try:
                    perspective = c2.e2p(
                        frame, fov_deg=fov, u_deg=yaw, v_deg=pitch, out_hw=(512, 512)
                    )
                    name = f"loc{pano_id:05d}_yaw{yaw:03d}.jpg"
                    imageio.imwrite(out_dir / name, perspective)
                    all_meta.append({
                        "t": stamp,
                        "yaw": yaw,
                        "location_id": pano_id,
                        "file": name
                    })
                except Exception as e:
                    print(f"Greška kod yaw {yaw} frame {frame_idx}: {e}")
            pano_id += 1

    reader.close()

    # Spremi samo glavni index
    index_file = out_dir / "index.json"
    index_file.write_text(json.dumps(all_meta, indent=2))
    print("✅ Gotovo.")

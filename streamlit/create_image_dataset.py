def create_dataset(video_path, batch_size=10, out_dir="250"):
    import pathlib
    import json
    import imageio
    import py360convert as c2

    out_dir = pathlib.Path(out_dir)
    out_dir.mkdir(exist_ok=True)

    # Pokušaj dohvatiti broj frameova
    reader = imageio.get_reader(video_path)
    fps = reader.get_meta_data().get("fps", 30)

    try:
        num_frames = reader.count_frames()
    except Exception:
        num_frames = 1000  # fallback ako count_frames ne radi

    reader.close()

    stride = max(1, int(round(fps * 0.5)))
    total_batches = (num_frames + batch_size - 1) // batch_size

    all_meta = []

    for batch_index in range(total_batches):
        print(f"Obrađujem batch {batch_index + 1}/{total_batches}")
        # Pokreni batch unutar ove funkcije
        reader = imageio.get_reader(video_path)
        start_idx = batch_index * batch_size
        end_idx = start_idx + batch_size
        pano_id_start = batch_index * batch_size
        meta = []
        fov, yaws, pitch = 90, [0, 90, 180, 270], 0
        pano_id = pano_id_start

        for frame_idx in range(start_idx, min(end_idx, num_frames)):
            if frame_idx % stride != 0:
                continue
            try:
                frame = reader.get_data(frame_idx)
            except Exception as e:
                print(f"Greška pri čitanju framea {frame_idx}: {e}")
                continue

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

        reader.close()
        # Spremi privremeni index
        batch_file = out_dir / f"index_batch{batch_index:03d}.json"
        batch_file.write_text(json.dumps(meta, indent=2))
        all_meta.extend(meta)

    # Spoji sve u jedan index.json
    final_index_file = out_dir / "index.json"
    final_index_file.write_text(json.dumps(all_meta, indent=2))
    print("✅ Svi batch-evi obrađeni. index.json generiran.")

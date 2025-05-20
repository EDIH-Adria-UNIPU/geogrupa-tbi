from pathlib import Path

from PIL import Image, ImageDraw
from ultralytics import YOLO

root = Path("dataset") / "250"
outdir = Path("detections")
outdir.mkdir(parents=True, exist_ok=True)

model = YOLO(str(Path("models") / "yolov8s-traffic-sign.pt"))
# model = YOLO(str(Path("models") / "yolov8n-oiv7.pt"))

results = model.predict(
    source=str(root / "*.jpg"),
    save=False,
    classes=[1, 6, 8, 10],  # traffic signs (yolov8s-traffic-sign)
    # classes=[497],  # street lights (yolov8n-oiv7)
    imgsz=640,
    conf=0.5,  # traffic signs
    # conf=0.1,  # street lights
    verbose=False,
)

for result in results:
    if len(result.boxes) > 0:
        img_path = result.path
        img = Image.open(img_path).convert("RGB")
        draw = ImageDraw.Draw(img)
        for box in result.boxes:
            xyxy = box.xyxy[0].tolist()
            class_id = int(box.cls[0])
            class_name = model.names[class_id]
            draw.rectangle(xyxy, outline="red", width=2)
            draw.text((xyxy[0], xyxy[1] - 10), class_name, fill="red")
        save_path = outdir / Path(img_path).name
        img.save(save_path)

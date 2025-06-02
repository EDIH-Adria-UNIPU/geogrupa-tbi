from pathlib import Path

from PIL import Image, ImageDraw
from ultralytics import YOLO

root = Path("dataset") / "835"
outdir = Path("visualizations")
outdir.mkdir(parents=True, exist_ok=True)

CONFIG = {
    "traffic-sign": {
        "model": str(Path("models") / "yolov8s-traffic-sign.pt"),
        "classes": [1, 6, 8, 10],
        "conf": 0.5,
    },
    "street-light": {
        "model": str(Path("models") / "yolov8n-oiv7.pt"),
        "classes": [497],
        "conf": 0.05,
    },
}

CLASS_TO_DETECT = "street-light" 

model = YOLO(CONFIG[CLASS_TO_DETECT]["model"])
classes = CONFIG[CLASS_TO_DETECT]["classes"]
conf = CONFIG[CLASS_TO_DETECT]["conf"]

results = model.predict(
    source=str(root / "*.jpg"),
    save=False,
    classes=classes,
    imgsz=640,
    conf=conf,
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

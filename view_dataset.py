import itertools
import json
import operator
import pathlib

import cv2

root = pathlib.Path("dataset/250")
meta = json.loads((root / "index.json").read_text())
meta.sort(key=operator.itemgetter("location_id", "yaw"))

groups = itertools.groupby(meta, key=lambda r: r["location_id"])
locations = [list(g[1]) for g in groups]

orientation = {0: "Front", 90: "Right", 180: "Back", 270: "Left"}


def build_mosaic(recs, scale=0.4):
    tiles = []
    for entry in recs:
        img = cv2.imread(str(root / entry["file"]))
        label = orientation.get(entry["yaw"], f"Yaw {entry['yaw']}")
        cv2.putText(
            img,
            label,
            (20, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.2,
            (255, 255, 255),
            3,
            cv2.LINE_AA,
        )
        tiles.append(img)
    top = cv2.hconcat(tiles[:2])
    bottom = cv2.hconcat(tiles[2:])
    mosaic = cv2.vconcat([top, bottom])
    return cv2.resize(mosaic, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)


idx = 0
while True:
    cv2.imshow("4-view", build_mosaic(locations[idx]))
    key = cv2.waitKeyEx(0)
    if key in (27, ord("q")):
        break
    if key in (0x250000, ord("a")):
        idx = (idx - 1) % len(locations)
    if key in (0x270000, ord("d")):
        idx = (idx + 1) % len(locations)


cv2.destroyAllWindows()

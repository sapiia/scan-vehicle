import argparse
import os
import cv2
import math
from collections import defaultdict
from ultralytics import YOLO

def parse_args():
    ap = argparse.ArgumentParser(description="YOLOv8 tracking + line counting.")
    ap.add_argument("--source", type=str, default="0", help="Camera index or video path (use 0 for webcam)")
    ap.add_argument("--weights", type=str, default="", help="Path to weights. If empty, uses config or yolov8n.pt")
    ap.add_argument("--conf", type=float, default=None, help="Confidence threshold override")
    ap.add_argument("--iou", type=float, default=None, help="NMS IoU threshold override")
    ap.add_argument("--imgsz", type=int, default=None, help="Inference image size")
    ap.add_argument("--save", action="store_true", help="Save annotated video to outputs/")
    return ap.parse_args()

def load_config():
    import json, pathlib
    cfg_path = pathlib.Path(__file__).resolve().parents[1] / "config.json"
    if cfg_path.exists():
        with open(cfg_path, "r") as f:
            return json.load(f)
    return {}

def open_source(src_str):
    if src_str.isdigit():
        return int(src_str)
    return src_str

def point_line_side(px, py, x1, y1, x2, y2):
    # Returns the sign of the cross product -> which side of the line a point is on
    return (x2 - x1) * (py - y1) - (y2 - y1) * (px - x1)

def main():
    args = parse_args()
    cfg = load_config()

    model_path = args.weights or cfg.get("model", "")
    if not model_path or not os.path.isfile(model_path):
        model_path = "yolov8n.pt"

    conf = args.conf if args.conf is not None else cfg.get("conf", 0.25)
    iou = args.iou if args.iou is not None else cfg.get("iou", 0.45)
    imgsz = args.imgsz if args.imgsz is not None else cfg.get("imgsz", 640)

    wanted_classes = set([c.lower() for c in cfg.get("classes", ["car", "motorcycle", "bicycle", "tuk-tuk"])])

    model = YOLO(model_path)

    # Prepare video writer lazily
    writer = None
    save_path = None

    # Tracking state
    last_centroid_side = {}
    counted_ids = set()
    counts = defaultdict(int)

    # Counting line
    x1, y1, x2, y2 = cfg.get("line", [100, 360, 1180, 360])

    # Map Ultralytics model class indices to names
    class_names = model.model.names if hasattr(model, "model") else {}

    source = open_source(args.source)

    for result in model.track(source=source, conf=conf, iou=iou, imgsz=imgsz, stream=True, persist=True):
        frame = result.orig_img.copy()

        # Draw counting line
        cv2.line(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(frame, "Counting Line", (x1, y1 - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2, cv2.LINE_AA)

        # Initialize writer once we know frame size
        if writer is None and args.save:
            os.makedirs("outputs", exist_ok=True)
            h, w = frame.shape[:2]
            save_path = os.path.join("outputs", "track_count_annotated.mp4")
            writer = cv2.VideoWriter(save_path, cv2.VideoWriter_fourcc(*"mp4v"), 30, (w, h))

        if result.boxes is not None:
            boxes = result.boxes
            ids = boxes.id.cpu().numpy().astype(int).tolist() if boxes.id is not None else [None] * len(boxes)
            xyxy = boxes.xyxy.cpu().numpy().tolist()
            clss = boxes.cls.cpu().numpy().astype(int).tolist()

            for box, tid, cls_idx in zip(xyxy, ids, clss):
                xA, yA, xB, yB = map(int, box)
                cx, cy = (xA + xB) // 2, (yA + yB) // 2

                # Class name and filter
                cname = str(class_names.get(cls_idx, str(cls_idx))).lower()
                if cname not in wanted_classes:
                    continue

                # Draw bbox + label
                cv2.rectangle(frame, (xA, yA), (xB, yB), (255, 255, 255), 2)
                label = f"{cname}#{tid}" if tid is not None else cname
                cv2.putText(frame, label, (xA, max(20, yA - 6)), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA)
                cv2.circle(frame, (cx, cy), 3, (255, 255, 255), -1)

                if tid is None:
                    # Can't count without a track id
                    continue

                # Count crossing
                side = math.copysign(1, point_line_side(cx, cy, x1, y1, x2, y2))
                prev_side = last_centroid_side.get(tid, None)

                if prev_side is not None and tid not in counted_ids and side != prev_side:
                    counted_ids.add(tid)
                    counts[cname] += 1

                last_centroid_side[tid] = side

        # Render counts
        y = 30
        for cname in sorted(wanted_classes):
            cv2.putText(frame, f"{cname}: {counts.get(cname,0)}", (12, y), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 200, 255), 2, cv2.LINE_AA)
            y += 28

        cv2.imshow("YOLOv8 Track & Count", frame)
        if writer is not None:
            writer.write(frame)

        if (cv2.waitKey(1) & 0xFF) == 27:
            break

    if writer is not None:
        writer.release()
        print(f"Saved: {save_path}")
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
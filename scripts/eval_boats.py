r"""Boat-only detection benchmark on LaRS val, for comparing models under identical rules
(e.g. COCO-pretrained yolo11n.pt vs. our LaRS-trained weights). Appends a row per run
to runs/eval/boat_eval.csv.

Boats only because it's the one class COCO and LaRS share (~74% of LaRS val objects).

Rules:
  - GT is read from LaRS's raw JSON, not our converted labels, so converter bugs can't
    leak into the score. Boat/ship + Row boats -> "boat".
  - Predicted classes named "boat"/"row_boat" -> "boat"; everything else ignored.
  - Unmatched boat predictions with IoU >= 0.5 to a non-boat object or a crowd box are
    ignored rather than counted as FPs; crowd boxes aren't counted as FNs. This favours
    the COCO baseline, so reported gains over it are conservative.
  - Matching and AP use Ultralytics' own logic (match_predictions / ap_per_class), so
    numbers are consistent with training-time val.
"""

import argparse
import csv
import json
import sys
from datetime import datetime
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
LARS_BOAT_IDS = {11, 12}  # Boat/ship, Row boats
LARS_FIRST_OBJECT_ID = 11  # lower ids are stuff classes
BOAT_NAMES = {"boat", "row_boat"}
IOU_THRESHOLDS = np.linspace(0.5, 0.95, 10)
DEFAULT_CONF = 0.25  # Ultralytics predict() default


def box_iou(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Pairwise IoU of xyxy boxes: (M,4) x (N,4) -> (M,N)."""
    if len(a) == 0 or len(b) == 0:
        return np.zeros((len(a), len(b)))
    lt = np.maximum(a[:, None, :2], b[None, :, :2])
    rb = np.minimum(a[:, None, 2:], b[None, :, 2:])
    inter = np.clip(rb - lt, 0, None).prod(2)
    area_a = (a[:, 2:] - a[:, :2]).prod(1)
    area_b = (b[:, 2:] - b[:, :2]).prod(1)
    return inter / (area_a[:, None] + area_b[None, :] - inter + 1e-9)


def match(iou: np.ndarray) -> np.ndarray:
    """TP matrix (n_pred, n_iou_thresholds) from IoU (n_gt, n_pred), preds sorted by conf desc.

    Greedy: each pred claims its best unclaimed GT, per threshold. Mirrors Ultralytics'
    BaseValidator.match_predictions.
    """
    correct = np.zeros((iou.shape[1], len(IOU_THRESHOLDS)), dtype=bool)
    claimed = np.zeros((iou.shape[0], len(IOU_THRESHOLDS)), dtype=bool)
    cols = range(len(IOU_THRESHOLDS))
    for j in np.flatnonzero((iou >= IOU_THRESHOLDS.min()).any(0)):
        available = np.where(claimed, 0, iou[:, j, None])
        k = available.argmax(0)
        correct[j] = available[k, cols] >= IOU_THRESHOLDS
        claimed[k, cols] |= correct[j]
    return correct


def load_ground_truth(json_path: Path) -> dict:
    """{file_name: (boat_boxes, ignore_boxes)}, pixel xyxy."""
    ann = json.loads(json_path.read_text(encoding="utf-8"))
    file_names = {im["id"]: im["file_name"] for im in ann["images"]}
    gt = {}
    for a in ann["annotations"]:
        boats, ignore = [], []
        for seg in a["segments_info"]:
            if seg["category_id"] < LARS_FIRST_OBJECT_ID:
                continue
            x, y, w, h = seg["bbox"]
            box = [x, y, x + w, y + h]
            if seg["category_id"] in LARS_BOAT_IDS and not seg.get("iscrowd", 0):
                boats.append(box)
            else:
                ignore.append(box)
        gt[file_names[a["image_id"]]] = (
            np.array(boats, dtype=float).reshape(-1, 4),
            np.array(ignore, dtype=float).reshape(-1, 4),
        )
    return gt


def score_image(boats: np.ndarray, ignore: np.ndarray, pred: np.ndarray, conf: np.ndarray):
    """(TP matrix, confs) for one image, with the ignore rule applied."""
    order = np.argsort(-conf, kind="stable")
    pred, conf = pred[order], conf[order]
    correct = match(box_iou(boats, pred))
    if len(ignore) and len(pred):
        on_other_object = box_iou(ignore, pred).max(0) >= 0.5
        keep = correct[:, 0] | ~on_other_object
        correct, conf = correct[keep], conf[keep]
    return correct, conf


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model", default="yolo11n.pt")
    p.add_argument("--imgsz", type=int, default=640, help="match the model's training size")
    p.add_argument("--lars", type=Path, default=ROOT / "datasets" / "lars_raw" / "val",
                   help="LaRS split dir (images/ + panoptic_annotations.json)")
    p.add_argument("--device", default=None)
    args = p.parse_args()

    for stream in (sys.stdout, sys.stderr):  # Windows: see train.py
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8")

    from ultralytics import YOLO
    from ultralytics.utils.metrics import ap_per_class

    model = YOLO(args.model)
    boat_ids = [i for i, n in model.names.items() if n in BOAT_NAMES]
    if not boat_ids:
        sys.exit(f"{args.model} has no class named any of {sorted(BOAT_NAMES)}")

    gt = load_ground_truth(args.lars / "panoptic_annotations.json")
    n_boats = sum(len(boats) for boats, _ in gt.values())

    all_correct, all_conf = [], []
    for file_name, (boats, ignore) in gt.items():
        r = model.predict(
            args.lars / "images" / file_name,
            imgsz=args.imgsz,
            conf=0.001,  # conf/iou/max_det match Ultralytics val defaults
            iou=0.7,
            max_det=300,
            classes=boat_ids,
            agnostic_nms=True,  # boat and row_boat would otherwise both fire on one object
            device=args.device,
            verbose=False,
        )[0]
        correct, conf = score_image(boats, ignore, r.boxes.xyxy.cpu().numpy(), r.boxes.conf.cpu().numpy())
        all_correct.append(correct)
        all_conf.append(conf)

    tp, conf = np.concatenate(all_correct), np.concatenate(all_conf)
    ap = ap_per_class(tp, conf, np.zeros(len(conf)), np.zeros(n_boats))[5][0]  # AP per IoU threshold, shape (10,)
    at_default = conf >= DEFAULT_CONF
    found = tp[at_default, 0].sum()
    recall = found / n_boats
    precision = found / max(at_default.sum(), 1)

    print(f"\nModel: {args.model}  (counted as boat: {[model.names[i] for i in boat_ids]})")
    print(f"LaRS val: {len(gt)} images, {n_boats} boats to find")
    print(f"  mAP50     {ap[0]:.3f}   standard detection score (0 to 1, higher is better)")
    print(f"  mAP50-95  {ap.mean():.3f}   stricter version that also rewards tight boxes")
    print(f"  At YOLO's default confidence ({DEFAULT_CONF}): finds {recall:.0%} of the boats, "
          f"and {precision:.0%} of its boat detections are real boats")

    out_csv = ROOT / "runs" / "eval" / "boat_eval.csv"
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    is_new = not out_csv.exists()
    with out_csv.open("a", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        if is_new:
            w.writerow(["date", "model", "imgsz", "boats", "mAP50", "mAP50-95",
                        f"recall@{DEFAULT_CONF}", f"precision@{DEFAULT_CONF}"])
        w.writerow([datetime.now().strftime("%Y-%m-%d %H:%M"), args.model, args.imgsz, n_boats,
                    f"{ap[0]:.4f}", f"{ap.mean():.4f}", f"{recall:.4f}", f"{precision:.4f}"])
    print(f"Saved to {out_csv}")


if __name__ == "__main__":
    main()

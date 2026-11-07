r"""Train a YOLO11 object detector on the converted LaRS dataset.

Run from the project folder. On this laptop, use the project's Python
environment (.venv\Scripts\python instead of python):

    # Quick check that everything works (same as our first test run):
    # 2 epochs on 10% of the training images. ~8 min on a laptop CPU.
    python scripts/train.py --check

    # Quick check, but with more epochs
    python scripts/train.py --check --epochs 5

    # Real training: all training images, 100 epochs (needs a GPU to be practical)
    python scripts/train.py --epochs 100

    # Bigger model with a custom run name
    python scripts/train.py --model yolo11s.pt --epochs 100 --name yolo11s_lars

Results go to runs/detect/<name>/:
    weights/best.pt   the trained model (best epoch on the val split)
    results.png       loss and accuracy graphs per epoch
    *_batch*.jpg      preview images with boxes drawn on them
    args.yaml         every setting the run used
"""

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent  # the project folder


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--check", action="store_true",
                   help="quick pipeline test: 2 epochs on 10%% of the training images "
                        "(--epochs and --fraction still override this)")
    p.add_argument("--epochs", type=int, default=None,
                   help="passes over the training images (default: 100, or 2 with --check)")
    p.add_argument("--fraction", type=float, default=None,
                   help="share of training images to use, 0-1 (default: 1.0, or 0.1 with --check)")
    p.add_argument("--model", default="yolo11n.pt",
                   help="starting weights: yolo11n.pt (nano), yolo11s.pt (small), ... (default: yolo11n.pt)")
    p.add_argument("--data", type=Path, default=ROOT / "datasets" / "lars_yolo" / "lars.yaml",
                   help="dataset yaml (default: datasets/lars_yolo/lars.yaml)")
    p.add_argument("--imgsz", type=int, default=640, help="image size in pixels (default: 640)")
    p.add_argument("--batch", type=int, default=16,
                   help="images per training step (default: 16; -1 = fit to GPU memory)")
    p.add_argument("--device", default=None,
                   help="'cpu', or a GPU number like 0 (default: GPU if there is one, else CPU)")
    p.add_argument("--name", default=None,
                   help="run folder under runs/detect/ (default: 'check', or '<model>_lars')")
    args = p.parse_args()

    if args.epochs is None:
        args.epochs = 2 if args.check else 100
    if args.fraction is None:
        args.fraction = 0.1 if args.check else 1.0
    if args.name is None:
        args.name = "check" if args.check else f"{Path(args.model).stem}_lars"
    return args


def main() -> None:
    args = parse_args()

    # On Windows, output saved to a file defaults to an encoding that can't
    # print YOLO's progress-bar characters. Force UTF-8 so that can't crash a run.
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8")

    from ultralytics import YOLO  # imported here so --help is instant

    model = YOLO(args.model)  # downloads the pretrained weights if they're missing
    model.train(
        data=str(args.data),
        epochs=args.epochs,
        fraction=args.fraction,
        imgsz=args.imgsz,
        batch=args.batch,
        device=args.device,
        project=str(ROOT / "runs" / "detect"),
        name=args.name,  # if the folder exists, YOLO adds a number: check2, check3, ...
    )
    print(f"\nTrained model: {model.trainer.best}")
    print(f"Graphs, metrics and preview images: {model.trainer.save_dir}")


if __name__ == "__main__":  # required on Windows, where data loading uses subprocesses
    main()

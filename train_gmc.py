"""Portable command-line wrapper around the supplied direct GMC training script.

The original script is kept byte-for-byte. This wrapper overrides its existing
configuration variables so local absolute paths need not be edited.
"""
import argparse
import sys
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description="Train the supplied standalone grid counter.")
    parser.add_argument("--image-root", type=Path, required=True)
    parser.add_argument("--label-root", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, default=Path("runs/gmc"))
    parser.add_argument("--grid", type=int, choices=(3, 5), default=5)
    parser.add_argument("--image-size", type=int, default=300)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--label-format", choices=("yolo", "point", "xywh", "count", "auto"), default="yolo")
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--cpu", action="store_true")
    parser.add_argument("--skip-test-images", action="store_true")
    args = parser.parse_args()

    gmc_dir = Path(__file__).resolve().parents[1] / "gmc"
    sys.path.insert(0, str(gmc_dir))
    import run_grid_matrix_counter_direct as source

    source.IMAGE_ROOT = str(args.image_root.resolve())
    source.LABEL_ROOT = str(args.label_root.resolve())
    source.OUT_DIR = args.out_dir.resolve()
    source.GRID = args.grid
    source.IMAGE_SIZE = args.image_size
    source.EPOCHS = args.epochs
    source.BATCH_SIZE = args.batch_size
    source.LABEL_FORMAT = args.label_format
    source.NUM_WORKERS = args.num_workers
    source.FORCE_CPU = args.cpu
    source.PREDICT_TEST_AFTER_TRAIN = not args.skip_test_images
    source.train_direct()


if __name__ == "__main__":
    main()

from pathlib import Path
from types import SimpleNamespace

import numpy as np
from PIL import Image
import torch
from torch import nn

from grid_matrix_counter import (
    CountImageTransform,
    GridMatrixCounter,
    build_records,
    count_loss,
    draw_matrix_overlay,
    evaluate,
    load_checkpoint,
    make_loader,
    save_matrix_csv,
    set_seed,
)


# =========================
# 只需要改这里
# =========================

# 你的图片根目录，下面应该有 train、val、test 三个文件夹。
IMAGE_ROOT = r"D:\experients\My_dataset\DroneVehicle-IR\images"

# 你的标签根目录，下面也应该有 train、val、test 三个文件夹。
LABEL_ROOT = r"D:\experients\My_dataset\DroneVehicle-IR\labels"

# 标签格式：
# yolo  : 每行 class x_center y_center width height
# point : 每行 x y，或者 class x y
# xywh  : 每行 x y width height，其中 x y 是左上角
# count : 每个 txt 只有一个总数量
# auto  : 自动判断，简单数据可用；正式训练更建议明确写 yolo/point/xywh/count
LABEL_FORMAT = "yolo"

# 输出矩阵大小：3 表示 3x3，5 表示 5x5。
GRID = 5

# 图像会统一缩放到 IMAGE_SIZE x IMAGE_SIZE。
# 默认 300 同时适合 3x3 和 5x5。
IMAGE_SIZE = 300

# 训练参数。
EPOCHS = 50
BATCH_SIZE = 64
LR = 1e-3
WEIGHT_DECAY = 1e-4
DROPOUT = 0.15
ATTENTION_TEMPERATURE = 1.0
GRID_LOSS_WEIGHT = 1.0
TOTAL_LOSS_WEIGHT = 0.2
HFLIP_PROB = 0.5
NUM_WORKERS = 0
SEED = 42

# 没有 GPU 或者想强制 CPU 时，改成 True。
FORCE_CPU = False

# 训练结果和测试输出保存位置。
OUT_DIR = Path(__file__).resolve().parent / "direct_matrix_run"

# 训练结束后是否自动对 test 文件夹里的图片输出矩阵和可视化图。
PREDICT_TEST_AFTER_TRAIN = True

# None 表示预测全部 test 图片；如果只想先看几张，可以改成 10、20 等。
MAX_TEST_IMAGES = None


# =========================
# 通常不需要改下面
# =========================


def make_args():
    return SimpleNamespace(
        grid=GRID,
        image_size=IMAGE_SIZE,
        label_format=LABEL_FORMAT,
        batch_size=BATCH_SIZE,
        num_workers=NUM_WORKERS,
        hflip_prob=HFLIP_PROB,
    )


def require_dir(path, name):
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"{name} does not exist: {path}")
    if not path.is_dir():
        raise NotADirectoryError(f"{name} is not a folder: {path}")
    return path


def get_split_dirs():
    image_root = require_dir(IMAGE_ROOT, "IMAGE_ROOT")
    label_root = require_dir(LABEL_ROOT, "LABEL_ROOT")
    return {
        "train_images": require_dir(image_root / "train", "train image folder"),
        "val_images": require_dir(image_root / "val", "val image folder"),
        "test_images": require_dir(image_root / "test", "test image folder"),
        "train_labels": require_dir(label_root / "train", "train label folder"),
        "val_labels": require_dir(label_root / "val", "val label folder"),
        "test_labels": require_dir(label_root / "test", "test label folder"),
    }


def train_direct():
    if IMAGE_SIZE % GRID != 0:
        raise ValueError("IMAGE_SIZE must be divisible by GRID. Default IMAGE_SIZE=300 works for GRID=3 or GRID=5.")

    set_seed(SEED)
    split_dirs = get_split_dirs()
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    train_records = build_records(split_dirs["train_images"], split_dirs["train_labels"])
    val_records = build_records(split_dirs["val_images"], split_dirs["val_labels"])
    test_records = build_records(split_dirs["test_images"], split_dirs["test_labels"])

    args = make_args()
    train_loader = make_loader(train_records, args, train=True)
    val_loader = make_loader(val_records, args, train=False)
    test_loader = make_loader(test_records, args, train=False)

    device = torch.device("cuda" if torch.cuda.is_available() and not FORCE_CPU else "cpu")
    print(f"device={device}")
    print(f"grid={GRID} label_format={LABEL_FORMAT}")
    print(f"train={len(train_records)} val={len(val_records)} test={len(test_records)}")

    model = GridMatrixCounter(
        grid_size=GRID,
        dropout=DROPOUT,
        attention_temperature=ATTENTION_TEMPERATURE,
    ).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)

    best_path = OUT_DIR / "best_grid_matrix_counter.pt"
    best_metric = float("inf")

    for epoch in range(1, EPOCHS + 1):
        model.train()
        running_loss = 0.0
        samples = 0

        for images, target_matrix, target_total, has_grid in train_loader:
            images = images.to(device, non_blocking=True)
            target_matrix = target_matrix.to(device, non_blocking=True)
            target_total = target_total.to(device, non_blocking=True)
            has_grid = has_grid.to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)
            outputs = model(images)
            loss, _ = count_loss(
                outputs,
                target_matrix,
                target_total,
                has_grid,
                grid_loss_weight=GRID_LOSS_WEIGHT,
                total_loss_weight=TOTAL_LOSS_WEIGHT,
            )
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            optimizer.step()

            batch_size = images.shape[0]
            running_loss += loss.item() * batch_size
            samples += batch_size

        train_loss = running_loss / max(samples, 1)
        val_metrics = evaluate(model, val_loader, device)
        metric_for_save = val_metrics["total_mae"]

        log = (
            f"epoch={epoch:03d} "
            f"train_loss={train_loss:.4f} "
            f"val_total_mae={val_metrics['total_mae']:.4f} "
            f"val_total_rmse={val_metrics['total_rmse']:.4f}"
        )
        if "grid_cell_mae" in val_metrics:
            log += f" val_grid_cell_mae={val_metrics['grid_cell_mae']:.4f}"
        print(log)

        if metric_for_save < best_metric:
            best_metric = metric_for_save
            torch.save(
                {
                    "model_state": model.state_dict(),
                    "config": {
                        "grid": GRID,
                        "image_size": IMAGE_SIZE,
                        "dropout": DROPOUT,
                        "attention_temperature": ATTENTION_TEMPERATURE,
                    },
                    "best_metric": best_metric,
                },
                best_path,
            )

    print(f"best_checkpoint={best_path}")

    best_model, _ = load_checkpoint(best_path, device)
    test_metrics = evaluate(best_model, test_loader, device)
    print(
        f"test_total_mae={test_metrics['total_mae']:.4f} "
        f"test_total_rmse={test_metrics['total_rmse']:.4f}"
    )
    if "grid_cell_mae" in test_metrics:
        print(f"test_grid_cell_mae={test_metrics['grid_cell_mae']:.4f}")

    if PREDICT_TEST_AFTER_TRAIN:
        predict_test_images(best_path, split_dirs["test_images"], test_records, device)


@torch.no_grad()
def predict_test_images(checkpoint_path, test_image_dir, test_records, device):
    model, config = load_checkpoint(checkpoint_path, device)
    image_size = int(config.get("image_size", IMAGE_SIZE))
    transform = CountImageTransform(image_size=image_size)

    matrix_dir = OUT_DIR / "test_matrices"
    overlay_dir = OUT_DIR / "test_overlays"
    matrix_dir.mkdir(parents=True, exist_ok=True)
    overlay_dir.mkdir(parents=True, exist_ok=True)

    records = test_records if MAX_TEST_IMAGES is None else test_records[:MAX_TEST_IMAGES]
    for image_path, _ in records:
        image_path = Path(image_path)
        rel = image_path.relative_to(test_image_dir)
        matrix_path = (matrix_dir / rel).with_suffix(".csv")
        overlay_path = (overlay_dir / rel).with_suffix(".jpg")

        with Image.open(image_path) as img:
            tensor = transform(img).unsqueeze(0).to(device)

        outputs = model(tensor)
        matrix = outputs["matrix"][0].detach().cpu().numpy()
        total = float(outputs["total_count"][0].detach().cpu())

        save_matrix_csv(matrix, matrix_path)
        draw_matrix_overlay(image_path, matrix, overlay_path)

        np.set_printoptions(precision=4, suppress=True)
        print(f"\nimage={image_path}")
        print(f"pred_total_count={total:.4f}")
        print(matrix)

    print(f"\ntest matrix csv folder={matrix_dir}")
    print(f"test overlay image folder={overlay_dir}")


if __name__ == "__main__":
    train_direct()

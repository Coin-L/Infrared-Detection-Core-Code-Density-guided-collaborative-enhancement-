import argparse
import math
import random
import re
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader, Dataset


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def split_tokens(line: str):
    return [x for x in re.split(r"[\s,]+", line.strip()) if x]


def as_floats(tokens, label_path, line_number):
    try:
        return [float(x) for x in tokens]
    except ValueError as exc:
        raise ValueError(f"Cannot parse numbers in {label_path}, line {line_number}: {tokens}") from exc


def normalize_xy(x, y, image_width, image_height):
    if 0.0 <= x <= 1.0 and 0.0 <= y <= 1.0:
        x_norm, y_norm = x, y
    else:
        x_norm, y_norm = x / image_width, y / image_height
    x_norm = min(max(float(x_norm), 0.0), 0.999999)
    y_norm = min(max(float(y_norm), 0.0), 0.999999)
    return x_norm, y_norm


def add_point_to_grid(grid_counts, x, y, image_width, image_height):
    grid = grid_counts.shape[0]
    x_norm, y_norm = normalize_xy(x, y, image_width, image_height)
    col = min(int(x_norm * grid), grid - 1)
    row = min(int(y_norm * grid), grid - 1)
    grid_counts[row, col] += 1.0


def add_normalized_point_to_grid(grid_counts, x_norm, y_norm):
    grid = grid_counts.shape[0]
    x_norm = min(max(float(x_norm), 0.0), 0.999999)
    y_norm = min(max(float(y_norm), 0.0), 0.999999)
    col = min(int(x_norm * grid), grid - 1)
    row = min(int(y_norm * grid), grid - 1)
    grid_counts[row, col] += 1.0


def parse_label_file(label_path, image_size, grid, label_format="auto"):
    """Return (grid_counts, total_count, has_grid_target).

    Supported txt label formats:
    - yolo: class x_center y_center width height, normalized to 0..1
    - point: x y, or class x y; values may be normalized or pixel coordinates
    - xywh: x y width height; values may be normalized or pixel coordinates
    - count: one number in the file, only total-count supervision
    """

    image_width, image_height = image_size
    label_path = Path(label_path)
    lines = []
    if label_path.exists():
        with label_path.open("r", encoding="utf-8-sig") as f:
            lines = [
                (line_number, line.strip())
                for line_number, line in enumerate(f, start=1)
                if line.strip() and not line.lstrip().startswith("#")
            ]

    grid_counts = np.zeros((grid, grid), dtype=np.float32)
    if not lines:
        return grid_counts, 0.0, True

    parsed_lines = [(line_number, as_floats(split_tokens(line), label_path, line_number)) for line_number, line in lines]

    if label_format == "count" or (
        label_format == "auto" and len(parsed_lines) == 1 and len(parsed_lines[0][1]) == 1
    ):
        return grid_counts, float(parsed_lines[0][1][0]), False

    for line_number, values in parsed_lines:
        if label_format == "yolo" or (
            label_format == "auto"
            and len(values) >= 5
            and all(0.0 <= v <= 1.0 for v in values[1:5])
        ):
            x_center, y_center = values[1], values[2]
            add_point_to_grid(grid_counts, x_center, y_center, image_width, image_height)
            continue

        if label_format == "xywh" or (label_format == "auto" and len(values) == 4):
            x, y, w, h = values[:4]
            if 0.0 <= x <= 1.0 and 0.0 <= y <= 1.0 and 0.0 <= w <= 1.0 and 0.0 <= h <= 1.0:
                x_center, y_center = x + w / 2.0, y + h / 2.0
                add_normalized_point_to_grid(grid_counts, x_center, y_center)
            else:
                x_center, y_center = x + w / 2.0, y + h / 2.0
                add_point_to_grid(grid_counts, x_center, y_center, image_width, image_height)
            continue

        if label_format == "point" or label_format == "auto":
            if len(values) == 2:
                x, y = values
            elif len(values) >= 3:
                x, y = values[1], values[2]
            else:
                raise ValueError(f"Unsupported label line in {label_path}, line {line_number}: {values}")
            add_point_to_grid(grid_counts, x, y, image_width, image_height)
            continue

        raise ValueError(
            f"Unsupported label_format={label_format!r}. "
            "Use auto, yolo, point, xywh, or count."
        )

    return grid_counts, float(grid_counts.sum()), True


def list_images(image_dir):
    image_dir = Path(image_dir)
    return sorted(p for p in image_dir.rglob("*") if p.suffix.lower() in IMAGE_EXTENSIONS)


def find_label_path(image_path, image_dir, label_dir):
    image_path = Path(image_path)
    image_dir = Path(image_dir)
    label_dir = Path(label_dir)
    rel = image_path.relative_to(image_dir)
    candidates = [
        label_dir / rel.with_suffix(".txt"),
        label_dir / f"{image_path.stem}.txt",
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return candidates[0]


def build_records(image_dir, label_dir):
    images = list_images(image_dir)
    records = []
    missing = 0
    for image_path in images:
        label_path = find_label_path(image_path, image_dir, label_dir)
        if label_path.exists():
            records.append((str(image_path), str(label_path)))
        else:
            missing += 1
    if not records:
        raise ValueError(f"No image/label pairs found. image_dir={image_dir}, label_dir={label_dir}")
    if missing:
        print(f"warning: skipped {missing} image(s) without matching .txt label.")
    return records


def split_records(records, val_ratio, seed):
    records = list(records)
    rng = random.Random(seed)
    rng.shuffle(records)
    val_count = max(1, int(round(len(records) * val_ratio))) if len(records) > 1 else 0
    return records[val_count:], records[:val_count]


class CountImageTransform:
    def __init__(self, image_size=300):
        self.image_size = int(image_size)
        self.mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
        self.std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)

    def __call__(self, image):
        image = image.convert("RGB")
        image = image.resize((self.image_size, self.image_size), Image.BILINEAR)
        arr = np.asarray(image, dtype=np.float32) / 255.0
        tensor = torch.from_numpy(arr).permute(2, 0, 1)
        return (tensor - self.mean) / self.std


class GridCountDataset(Dataset):
    def __init__(self, records, grid, image_size, label_format="auto", train=False, hflip_prob=0.5):
        self.records = list(records)
        self.grid = int(grid)
        self.label_format = label_format
        self.train = train
        self.hflip_prob = float(hflip_prob)
        self.transform = CountImageTransform(image_size)

    def __len__(self):
        return len(self.records)

    def __getitem__(self, index):
        image_path, label_path = self.records[index]
        with Image.open(image_path) as img:
            image = img.convert("RGB")
            grid_counts, total_count, has_grid = parse_label_file(
                label_path=label_path,
                image_size=image.size,
                grid=self.grid,
                label_format=self.label_format,
            )
            if self.train and random.random() < self.hflip_prob:
                image = image.transpose(Image.FLIP_LEFT_RIGHT)
                if has_grid:
                    grid_counts = np.ascontiguousarray(grid_counts[:, ::-1])
            image_tensor = self.transform(image)

        return (
            image_tensor,
            torch.from_numpy(grid_counts.astype(np.float32)),
            torch.tensor(float(total_count), dtype=torch.float32),
            torch.tensor(bool(has_grid), dtype=torch.bool),
        )


class ConvEncoder(nn.Module):
    def __init__(self, out_dim=256):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, out_dim, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_dim),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d(1),
        )

    def forward(self, x):
        return self.features(x).flatten(1)


class GridMatrixCounter(nn.Module):
    def __init__(self, grid_size=3, feature_dim=256, dropout=0.15, attention_temperature=1.0):
        super().__init__()
        if grid_size not in (3, 5):
            raise ValueError("grid_size must be 3 or 5.")
        self.grid_size = int(grid_size)
        self.attention_temperature = float(attention_temperature)
        self.encoder = ConvEncoder(feature_dim)
        self.patch_count_head = nn.Sequential(
            nn.Linear(feature_dim, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(128, 1),
        )

    def make_patches(self, x):
        batch, channels, height, width = x.shape
        grid = self.grid_size
        if height % grid != 0 or width % grid != 0:
            raise ValueError(f"Input size {(height, width)} must be divisible by grid_size={grid}.")
        patch_h = height // grid
        patch_w = width // grid
        patches = x.unfold(2, patch_h, patch_h).unfold(3, patch_w, patch_w)
        patches = patches.permute(0, 2, 3, 1, 4, 5).contiguous()
        return patches.view(batch * grid * grid, channels, patch_h, patch_w)

    def forward(self, x):
        batch = x.shape[0]
        grid = self.grid_size
        patches = self.make_patches(x)
        features = self.encoder(patches)
        patch_counts = F.softplus(self.patch_count_head(features).view(batch, grid, grid))
        total_count = patch_counts.flatten(1).sum(dim=1)
        attention = F.softmax(
            patch_counts.flatten(1) / max(self.attention_temperature, 1e-6),
            dim=1,
        ).view(batch, grid, grid)
        return {
            "matrix": patch_counts,
            "total_count": total_count,
            "attention": attention,
        }


def count_loss(outputs, target_matrix, target_total, has_grid, grid_loss_weight, total_loss_weight):
    pred_matrix = outputs["matrix"]
    pred_total = outputs["total_count"]
    total_loss = F.smooth_l1_loss(pred_total, target_total)
    loss = total_loss_weight * total_loss

    if has_grid.any():
        grid_loss = F.smooth_l1_loss(pred_matrix[has_grid], target_matrix[has_grid])
        loss = loss + grid_loss_weight * grid_loss
    else:
        grid_loss = torch.zeros((), device=pred_matrix.device)

    return loss, {"grid_loss": float(grid_loss.detach().cpu()), "total_loss": float(total_loss.detach().cpu())}


@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    total_abs_errors = []
    total_sq_errors = []
    grid_abs_errors = []

    for images, target_matrix, target_total, has_grid in loader:
        images = images.to(device, non_blocking=True)
        target_matrix = target_matrix.to(device, non_blocking=True)
        target_total = target_total.to(device, non_blocking=True)
        has_grid = has_grid.to(device, non_blocking=True)
        outputs = model(images)

        total_error = outputs["total_count"] - target_total
        total_abs_errors.append(total_error.abs().cpu())
        total_sq_errors.append(total_error.square().cpu())
        if has_grid.any():
            grid_abs_errors.append((outputs["matrix"][has_grid] - target_matrix[has_grid]).abs().flatten().cpu())

    total_abs_errors = torch.cat(total_abs_errors)
    total_sq_errors = torch.cat(total_sq_errors)
    metrics = {
        "total_mae": total_abs_errors.mean().item(),
        "total_rmse": math.sqrt(total_sq_errors.mean().item()),
    }
    if grid_abs_errors:
        metrics["grid_cell_mae"] = torch.cat(grid_abs_errors).mean().item()
    return metrics


def make_loader(records, args, train):
    dataset = GridCountDataset(
        records=records,
        grid=args.grid,
        image_size=args.image_size,
        label_format=args.label_format,
        train=train,
        hflip_prob=args.hflip_prob if train else 0.0,
    )
    return DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=train,
        num_workers=args.num_workers,
        pin_memory=torch.cuda.is_available(),
    )


def train(args):
    set_seed(args.seed)
    if args.image_size % args.grid != 0:
        raise ValueError("--image_size must be divisible by --grid. Default 300 works for both 3 and 5.")

    records = build_records(args.image_dir, args.label_dir)
    train_records, val_records = split_records(records, args.val_ratio, args.seed)
    print(f"train_images={len(train_records)} val_images={len(val_records)} label_format={args.label_format}")

    train_loader = make_loader(train_records, args, train=True)
    val_loader = make_loader(val_records, args, train=False) if val_records else None

    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")
    model = GridMatrixCounter(
        grid_size=args.grid,
        dropout=args.dropout,
        attention_temperature=args.attention_temperature,
    ).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    best_path = out_dir / "best_grid_matrix_counter.pt"
    best_metric = float("inf")

    for epoch in range(1, args.epochs + 1):
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
                grid_loss_weight=args.grid_loss_weight,
                total_loss_weight=args.total_loss_weight,
            )
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            optimizer.step()

            batch_size = images.shape[0]
            running_loss += loss.item() * batch_size
            samples += batch_size

        train_loss = running_loss / max(samples, 1)
        log = f"epoch={epoch:03d} train_loss={train_loss:.4f}"
        metric_for_save = train_loss
        if val_loader is not None:
            metrics = evaluate(model, val_loader, device)
            metric_for_save = metrics["total_mae"]
            log += f" val_total_mae={metrics['total_mae']:.4f} val_total_rmse={metrics['total_rmse']:.4f}"
            if "grid_cell_mae" in metrics:
                log += f" val_grid_cell_mae={metrics['grid_cell_mae']:.4f}"
        print(log)

        if metric_for_save < best_metric:
            best_metric = metric_for_save
            torch.save(
                {
                    "model_state": model.state_dict(),
                    "config": {
                        "grid": args.grid,
                        "image_size": args.image_size,
                        "dropout": args.dropout,
                        "attention_temperature": args.attention_temperature,
                    },
                    "best_metric": best_metric,
                },
                best_path,
            )

    print(f"saved_best={best_path}")


def load_checkpoint(checkpoint_path, device):
    checkpoint = torch.load(checkpoint_path, map_location=device)
    config = checkpoint.get("config", {})
    model = GridMatrixCounter(
        grid_size=int(config.get("grid", 3)),
        dropout=float(config.get("dropout", 0.15)),
        attention_temperature=float(config.get("attention_temperature", 1.0)),
    ).to(device)
    model.load_state_dict(checkpoint["model_state"])
    model.eval()
    return model, config


def save_matrix_csv(matrix, out_path):
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.savetxt(out_path, matrix, delimiter=",", fmt="%.6f")
    return out_path


def draw_matrix_overlay(image_path, matrix, out_path):
    image = Image.open(image_path).convert("RGB")
    draw = ImageDraw.Draw(image, "RGBA")
    width, height = image.size
    grid = matrix.shape[0]
    patch_w = width / grid
    patch_h = height / grid
    max_value = float(matrix.max()) if float(matrix.max()) > 0 else 1.0

    for row in range(grid):
        for col in range(grid):
            x0 = int(round(col * patch_w))
            y0 = int(round(row * patch_h))
            x1 = int(round((col + 1) * patch_w))
            y1 = int(round((row + 1) * patch_h))
            value = float(matrix[row, col])
            alpha = int(25 + 190 * value / max_value)
            draw.rectangle([x0, y0, x1, y1], fill=(255, 70, 30, alpha), outline=(255, 255, 255, 210), width=2)
            text = f"{value:.2f}"
            draw.rectangle([x0 + 5, y0 + 5, x0 + 62, y0 + 25], fill=(0, 0, 0, 145))
            draw.text((x0 + 9, y0 + 8), text, fill=(255, 255, 255, 255), font=ImageFont.load_default())

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    image.save(out_path)
    return out_path


@torch.no_grad()
def predict(args):
    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")
    model, config = load_checkpoint(args.checkpoint, device)
    image_size = int(config.get("image_size", args.image_size))
    transform = CountImageTransform(image_size=image_size)

    with Image.open(args.image) as img:
        tensor = transform(img).unsqueeze(0).to(device)

    outputs = model(tensor)
    matrix = outputs["matrix"][0].detach().cpu().numpy()
    attention = outputs["attention"][0].detach().cpu().numpy()
    total = float(outputs["total_count"][0].detach().cpu())

    np.set_printoptions(precision=4, suppress=True)
    print(f"pred_total_count={total:.4f}")
    print("pred_grid_matrix:")
    print(matrix)
    print("attention_matrix:")
    print(attention)

    if args.save_matrix_csv:
        out_path = save_matrix_csv(matrix, args.save_matrix_csv)
        print(f"saved_matrix_csv={out_path}")
    if args.save_overlay:
        out_path = draw_matrix_overlay(args.image, matrix, args.save_overlay)
        print(f"saved_overlay={out_path}")


def build_parser():
    parser = argparse.ArgumentParser(description="Train/predict a 3x3 or 5x5 object-count matrix from images.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    train_parser = subparsers.add_parser("train")
    train_parser.add_argument("--image_dir", required=True, help="Folder containing training images.")
    train_parser.add_argument("--label_dir", required=True, help="Folder containing txt labels with matching names.")
    train_parser.add_argument("--label_format", default="auto", choices=("auto", "yolo", "point", "xywh", "count"))
    train_parser.add_argument("--grid", type=int, choices=(3, 5), default=3)
    train_parser.add_argument("--image_size", type=int, default=300)
    train_parser.add_argument("--epochs", type=int, default=50)
    train_parser.add_argument("--batch_size", type=int, default=16)
    train_parser.add_argument("--lr", type=float, default=1e-3)
    train_parser.add_argument("--weight_decay", type=float, default=1e-4)
    train_parser.add_argument("--dropout", type=float, default=0.15)
    train_parser.add_argument("--attention_temperature", type=float, default=1.0)
    train_parser.add_argument("--grid_loss_weight", type=float, default=1.0)
    train_parser.add_argument("--total_loss_weight", type=float, default=0.2)
    train_parser.add_argument("--val_ratio", type=float, default=0.2)
    train_parser.add_argument("--hflip_prob", type=float, default=0.5)
    train_parser.add_argument("--num_workers", type=int, default=0)
    train_parser.add_argument("--out_dir", default="outputs/grid_matrix_run")
    train_parser.add_argument("--seed", type=int, default=42)
    train_parser.add_argument("--cpu", action="store_true")
    train_parser.set_defaults(func=train)

    pred_parser = subparsers.add_parser("predict")
    pred_parser.add_argument("--checkpoint", required=True)
    pred_parser.add_argument("--image", required=True)
    pred_parser.add_argument("--image_size", type=int, default=300)
    pred_parser.add_argument("--save_matrix_csv", default=None)
    pred_parser.add_argument("--save_overlay", default=None)
    pred_parser.add_argument("--cpu", action="store_true")
    pred_parser.set_defaults(func=predict)

    return parser


def main():
    parser = build_parser()
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()

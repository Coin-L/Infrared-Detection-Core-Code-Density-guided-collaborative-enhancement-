import argparse
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont
import torch

from grid_matrix_counter import CountImageTransform, load_checkpoint


# =========================
# Edit these values when running directly.
# =========================

# This is the image-to-matrix checkpoint, not an Ultralytics YOLO checkpoint.
WEIGHTS_PATH = r"D:\experients\YOLO26\Ultralytics_1754622989\handwork\direct_matrix_run\best_grid_matrix_counter.pt"

IMAGE_PATH = r"D:\experients\My_dataset\DroneVehicle-IR\images\val\00325.jpg"

# Only used as a fallback if the checkpoint has no grid config.
GRID = 3

DEVICE = None  # None=auto, "cpu"=CPU, "0"=GPU 0.

OUT_DIR = Path(__file__).resolve().parent / "yolo_matrix_collage_out"


# =========================
# Usually no need to edit below.
# =========================


def load_font(size=24):
    candidates = [
        r"C:\Windows\Fonts\msyh.ttc",
        r"C:\Windows\Fonts\simhei.ttf",
        r"C:\Windows\Fonts\arial.ttf",
    ]
    for path in candidates:
        if Path(path).exists():
            return ImageFont.truetype(path, size=size)
    return ImageFont.load_default()


def select_device(device):
    if device is None or str(device).strip() == "":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = str(device).strip()
    if device.lower() == "cpu":
        return torch.device("cpu")
    return torch.device(f"cuda:{device}" if device.isdigit() else device)


@torch.no_grad()
def run_matrix_model(weights_path, image_path, device):
    weights_path = Path(weights_path)
    image_path = Path(image_path)

    if not weights_path.exists():
        raise FileNotFoundError(f"checkpoint file does not exist: {weights_path}")
    if not image_path.exists():
        raise FileNotFoundError(f"image file does not exist: {image_path}")

    device = select_device(device)
    model, config = load_checkpoint(weights_path, device)
    image_size = int(config.get("image_size", 300))
    transform = CountImageTransform(image_size=image_size)

    with Image.open(image_path) as img:
        tensor = transform(img).unsqueeze(0).to(device)

    outputs = model(tensor)
    matrix = outputs["matrix"][0].detach().cpu().numpy()
    attention = outputs["attention"][0].detach().cpu().numpy()
    total = float(outputs["total_count"][0].detach().cpu())

    return matrix, attention, total, config, device


def draw_matrix_overlay(image, matrix):
    output = image.copy()
    draw = ImageDraw.Draw(output, "RGBA")
    width, height = output.size
    grid = matrix.shape[0]
    patch_w = width / grid
    patch_h = height / grid
    max_value = float(matrix.max()) if float(matrix.max()) > 0 else 1.0
    font = load_font(max(14, min(width, height) // 36))

    for row in range(grid):
        for col in range(grid):
            x0 = int(round(col * patch_w))
            y0 = int(round(row * patch_h))
            x1 = int(round((col + 1) * patch_w))
            y1 = int(round((row + 1) * patch_h))
            value = float(matrix[row, col])
            alpha = int(30 + 180 * value / max_value)
            draw.rectangle(
                [x0, y0, x1, y1],
                fill=(255, 70, 30, alpha),
                outline=(255, 255, 255, 220),
                width=max(2, min(width, height) // 260),
            )

            text = f"{value:.2f}"
            bbox = draw.textbbox((0, 0), text, font=font)
            text_w = bbox[2] - bbox[0]
            text_h = bbox[3] - bbox[1]
            draw.rectangle(
                [x0 + 6, y0 + 6, x0 + text_w + 18, y0 + text_h + 14],
                fill=(0, 0, 0, 150),
            )
            draw.text((x0 + 12, y0 + 9), text, fill=(255, 255, 255, 255), font=font)

    return output


def draw_matrix_panel(matrix, panel_size):
    width, height = panel_size
    grid = matrix.shape[0]
    image = Image.new("RGB", (width, height), (248, 250, 252))
    draw = ImageDraw.Draw(image, "RGBA")
    font_big = load_font(max(22, min(width, height) // 18))
    font_small = load_font(max(15, min(width, height) // 34))

    margin = max(18, min(width, height) // 28)
    title_h = max(44, min(width, height) // 13)
    grid_x0 = margin
    grid_y0 = margin + title_h
    grid_w = width - margin * 2
    grid_h = height - margin * 2 - title_h
    cell_w = grid_w / grid
    cell_h = grid_h / grid
    max_value = float(matrix.max()) if float(matrix.max()) > 0 else 1.0

    title = f"Predicted {grid}x{grid} Count Matrix"
    total = f"Total = {matrix.sum():.2f}"
    draw.text((margin, margin), title, fill=(15, 23, 42, 255), font=font_big)
    total_bbox = draw.textbbox((0, 0), total, font=font_small)
    total_w = total_bbox[2] - total_bbox[0]
    draw.text((width - margin - total_w, margin + 4), total, fill=(51, 65, 85, 255), font=font_small)

    for row in range(grid):
        for col in range(grid):
            value = float(matrix[row, col])
            ratio = value / max_value
            r = 255
            g = int(245 - 155 * ratio)
            b = int(235 - 205 * ratio)
            x0 = int(round(grid_x0 + col * cell_w))
            y0 = int(round(grid_y0 + row * cell_h))
            x1 = int(round(grid_x0 + (col + 1) * cell_w))
            y1 = int(round(grid_y0 + (row + 1) * cell_h))
            draw.rectangle([x0, y0, x1, y1], fill=(r, g, b, 255), outline=(255, 255, 255, 255), width=3)

            text = f"{value:.2f}"
            bbox = draw.textbbox((0, 0), text, font=font_big)
            text_w = bbox[2] - bbox[0]
            text_h = bbox[3] - bbox[1]
            tx = x0 + (x1 - x0 - text_w) / 2
            ty = y0 + (y1 - y0 - text_h) / 2
            draw.text((tx, ty), text, fill=(15, 23, 42, 255), font=font_big)

    draw.rectangle(
        [grid_x0, grid_y0, grid_x0 + grid_w, grid_y0 + grid_h],
        outline=(30, 41, 59, 255),
        width=3,
    )
    return image


def add_title(image, title, title_height=52):
    font = load_font(24)
    out = Image.new("RGB", (image.width, image.height + title_height), (255, 255, 255))
    out.paste(image, (0, title_height))
    draw = ImageDraw.Draw(out)
    draw.rectangle([0, 0, image.width, title_height], fill=(15, 23, 42))
    draw.text((18, 13), title, fill=(255, 255, 255), font=font)
    return out


def resize_to_height(image, target_height):
    if image.height == target_height:
        return image
    new_width = max(1, int(round(image.width * target_height / image.height)))
    return image.resize((new_width, target_height), Image.BILINEAR)


def make_collage(original_image, overlay_image, matrix_panel, output_path, max_panel_height=720):
    target_height = min(max_panel_height, original_image.height)
    original_panel = resize_to_height(original_image, target_height)
    overlay_panel = resize_to_height(overlay_image, target_height)
    matrix_panel = resize_to_height(matrix_panel, target_height)

    panels = [
        add_title(original_panel, "Original Image"),
        add_title(overlay_panel, "Predicted Matrix Overlay"),
        add_title(matrix_panel, "Predicted Matrix"),
    ]

    gap = 14
    out_width = sum(panel.width for panel in panels) + gap * (len(panels) + 1)
    out_height = max(panel.height for panel in panels) + gap * 2
    collage = Image.new("RGB", (out_width, out_height), (226, 232, 240))

    x = gap
    for panel in panels:
        y = gap + (out_height - gap * 2 - panel.height) // 2
        collage.paste(panel, (x, y))
        x += panel.width + gap

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    collage.save(output_path)
    return output_path


def save_matrix_csv(matrix, output_path):
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.savetxt(output_path, matrix, delimiter=",", fmt="%.6f")
    return output_path


def predict_and_make_collage(weights_path, image_path, grid, device, out_dir):
    weights_path = Path(weights_path)
    image_path = Path(image_path)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    matrix, attention, total, config, used_device = run_matrix_model(weights_path, image_path, device)
    checkpoint_grid = int(config.get("grid", matrix.shape[0]))
    if grid is not None and int(grid) != checkpoint_grid:
        print(f"warning: --grid={grid} ignored because checkpoint grid={checkpoint_grid}")

    original_image = Image.open(image_path).convert("RGB")
    overlay_image = draw_matrix_overlay(original_image, matrix)
    matrix_panel = draw_matrix_panel(matrix, original_image.size)

    stem = image_path.stem
    overlay_path = out_dir / f"{stem}_matrix_prediction_overlay.jpg"
    matrix_image_path = out_dir / f"{stem}_{checkpoint_grid}x{checkpoint_grid}_pred_matrix.jpg"
    matrix_csv_path = out_dir / f"{stem}_{checkpoint_grid}x{checkpoint_grid}_pred_matrix.csv"
    attention_csv_path = out_dir / f"{stem}_{checkpoint_grid}x{checkpoint_grid}_attention_matrix.csv"
    collage_path = out_dir / f"{stem}_original_pred_matrix_collage.jpg"

    overlay_image.save(overlay_path)
    matrix_panel.save(matrix_image_path)
    save_matrix_csv(matrix, matrix_csv_path)
    save_matrix_csv(attention, attention_csv_path)
    make_collage(original_image, overlay_image, matrix_panel, collage_path)

    np.set_printoptions(precision=4, suppress=True)
    print(f"checkpoint={weights_path}")
    print(f"image={image_path}")
    print(f"device={used_device}")
    print(f"checkpoint_config={config}")
    print(f"pred_total_count={total:.4f}")
    print("pred_grid_matrix:")
    print(matrix)
    print("attention_matrix:")
    print(attention)
    print(f"saved_collage={collage_path}")
    print(f"saved_overlay={overlay_path}")
    print(f"saved_matrix_image={matrix_image_path}")
    print(f"saved_matrix_csv={matrix_csv_path}")
    print(f"saved_attention_csv={attention_csv_path}")

    return {
        "matrix": matrix,
        "attention": attention,
        "total": total,
        "collage_path": collage_path,
        "overlay_path": overlay_path,
        "matrix_image_path": matrix_image_path,
        "matrix_csv_path": matrix_csv_path,
        "attention_csv_path": attention_csv_path,
    }


def build_parser():
    parser = argparse.ArgumentParser(description="Image-to-grid-matrix prediction collage.")
    parser.add_argument("--weights", "--checkpoint", dest="weights", default=WEIGHTS_PATH, help="Grid matrix checkpoint path.")
    parser.add_argument("--image", default=IMAGE_PATH, help="Input image path.")
    parser.add_argument("--grid", type=int, default=GRID, choices=(3, 5), help="Fallback grid size; checkpoint config wins.")
    parser.add_argument("--device", default=DEVICE, help='Use "0" for GPU 0, "cpu" for CPU, or omit for auto.')
    parser.add_argument("--out_dir", default=str(OUT_DIR), help="Output folder.")

    # Backward-compatible ignored options from the old YOLO inference version.
    parser.add_argument("--conf", type=float, default=None, help=argparse.SUPPRESS)
    parser.add_argument("--iou", type=float, default=None, help=argparse.SUPPRESS)
    parser.add_argument("--imgsz", type=int, default=None, help=argparse.SUPPRESS)
    return parser


def main():
    args = build_parser().parse_args()
    predict_and_make_collage(
        weights_path=args.weights,
        image_path=args.image,
        grid=args.grid,
        device=args.device,
        out_dir=args.out_dir,
    )


if __name__ == "__main__":
    main()

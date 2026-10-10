"""Small engineering check; does not train or evaluate a YOLO detector."""
from __future__ import annotations

import argparse
import json
import sys
from contextlib import nullcontext
from pathlib import Path

import torch
from torch import nn

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from gmc import GridMatrixCounter, count_loss
from modules import C3k2_GF, TD_CFEM


def check_gradients(model: nn.Module, x: torch.Tensor) -> None:
    if x.grad is None or not torch.isfinite(x.grad).all():
        raise AssertionError("Input gradient is missing or non-finite")
    for name, parameter in model.named_parameters():
        if parameter.requires_grad:
            if parameter.grad is None or not torch.isfinite(parameter.grad).all():
                raise AssertionError(f"Missing or non-finite gradient: {name}")


def check_enhancement(device: torch.device, amp: bool) -> list[dict]:
    results = []
    # Includes different channel widths, inner CSP use, and odd rectangular maps.
    cases = [(32, 48, 2, False, 0.5, 21, 27), (24, 40, 1, True, 0.65, 15, 19)]
    for c1, c2, repeats, c3k, expansion, h, w in cases:
        def make_model():
            return nn.Sequential(
                C3k2_GF(c1, c2, n=repeats, c3k=c3k, e=expansion),
                TD_CFEM(c2),
            ).to(device)

        model = make_model().train()
        x = torch.randn(2, c1, h, w, device=device, requires_grad=True)
        context = torch.autocast("cuda", dtype=torch.float16) if amp else nullcontext()
        with context:
            y = model(x)
            loss = y.float().square().mean()
        if tuple(y.shape) != (2, c2, h, w) or not torch.isfinite(y).all():
            raise AssertionError("Unexpected output shape or non-finite output")
        loss.backward()
        check_gradients(model, x)

        # Inference uses deterministic frequency selection; weights round-trip.
        model.eval()
        clone = make_model().eval()
        clone.load_state_dict(model.state_dict())
        with torch.no_grad():
            expected = model(x.detach())
            torch.testing.assert_close(model(x.detach()), expected, rtol=0, atol=0)
            torch.testing.assert_close(clone(x.detach()), expected, rtol=0, atol=0)
            gf = model[0]
            torch.testing.assert_close(gf.forward(x.detach()), gf.forward_split(x.detach()))
        results.append({"c3k": c3k, "input": list(x.shape), "output": list(y.shape)})
    return results


def check_counter(device: torch.device) -> list[dict]:
    results = []
    for grid in (3, 5):
        model = GridMatrixCounter(grid_size=grid).to(device).train()
        x = torch.randn(2, 3, 90, 90, device=device, requires_grad=True)
        output = model(x)
        target = torch.zeros(2, grid, grid, device=device)
        target[0, 0, 0] = 2
        target[1, -1, -1] = 1
        loss, _ = count_loss(
            output, target, target.flatten(1).sum(1),
            torch.tensor([True, False], device=device), 1.0, 1.0,
        )
        loss.backward()
        check_gradients(model, x)
        torch.testing.assert_close(output["total_count"], output["matrix"].flatten(1).sum(1))
        torch.testing.assert_close(output["attention"].flatten(1).sum(1), torch.ones(2, device=device))
        if not (output["matrix"] > 0).all():
            raise AssertionError("Softplus counts should be positive")
        results.append({"grid": grid, "matrix": list(output["matrix"].shape)})
    return results


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", choices=("cpu", "cuda", "auto"), default="cpu")
    parser.add_argument("--amp", action="store_true", help="Check CUDA float16 autocast.")
    args = parser.parse_args()
    device_name = "cuda" if args.device == "auto" and torch.cuda.is_available() else args.device
    if device_name == "auto":
        device_name = "cpu"
    if device_name == "cuda" and not torch.cuda.is_available():
        parser.error("CUDA is not available in this PyTorch environment")
    if args.amp and device_name != "cuda":
        parser.error("--amp requires a CUDA device")
    torch.set_num_threads(min(4, torch.get_num_threads()))
    torch.manual_seed(42)
    device = torch.device(device_name)
    results = {
        "torch": torch.__version__, "device": device_name, "amp": args.amp,
        "enhancement_modules": check_enhancement(device, args.amp),
        "supplied_counter": check_counter(device),
        "status": "passed",
    }
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()

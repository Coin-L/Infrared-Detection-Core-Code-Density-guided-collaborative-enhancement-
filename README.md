# Infrared Detection Core Code

This collection preserves the supplied GMC source code and provides runnable reference implementations of **C3k2-GF** and **TD-CFEM** based on the manuscript. Implementation details not specified in the manuscript are documented below and in the source. The added modules are reconstructions; their correspondence with the original code used to produce the reported experimental results has not been established.

## Files

| File | Contents and origin |
|---|---|
| `modules/c3k2_gf.py` | Reconstructed thermal-noise gating, multiscale depthwise convolution, complementary fusion, and CSP structure |
| `modules/td_cfem.py` | Reconstructed spatial, channel, and parameterized frequency gating with inverted residual enhancement |
| `modules/common.py` | Supporting Conv-BN-SiLU layer, without an Ultralytics dependency |
| `gmc/run_grid_matrix_counter_direct.py` | Supplied standalone grid-counter training script, preserved unchanged |
| `gmc/yolo_matrix_collage.py` | Supplied prediction and matrix visualization script, preserved unchanged |
| `gmc/grid_matrix_counter.py` | Original dependency imported by both supplied scripts, included unchanged |
| `examples/train_gmc.py` | Training entry point with configurable paths, using the original training procedure |
| `examples/check_modules.py` | Synthetic-tensor checks for shapes, gradients, and module composition |
| `PROVENANCE.json` | Source provenance and SHA-256 checksums |

## Reconstructed modules

**C3k2-GF.** `TNSG` uses a local mean computed with 7x7 average pooling at stride 1 to form `sigmoid(1 - AvgPool(X))`. Multiplying this gate by the input produces `X1`. A second branch splits the channels into three groups, applies 3x3, 5x5, and 7x7 depthwise convolutions, and concatenates their outputs to produce `X2`. The two branches are fused as `X1 * X2 + X1 + X2`. The inner CSP subunit splits the output of a single 1x1 projection into two paths; the outer C3k2/C2f structure retains intermediate-output concatenation. The local mean serves as a low-frequency proxy. No FFT is added beyond the operations described in the manuscript.

**TD-CFEM.** A fixed Laplacian convolution followed by Conv-BN-ReLU produces spatial gating, while ECA produces channel gating. The frequency branch splits the channels into four groups and uses cosine projections with learnable phase offsets, Gumbel-Softmax selection, and relative-energy gating. The three gates are multiplied with the input feature. The gated feature then passes through a 1x1-DW3x3-1x1 transform with an expansion ratio of 2, followed by Swish and addition to the original input. Frequency selection is stochastic during training and uses deterministic argmax in `eval()` mode.

The manuscript does not specify every implementation hyperparameter. This reference implementation uses the following conventions:

| Setting | Default convention |
|---|---|
| GF channel groups | As evenly sized as possible, with remainder channels assigned in order; at least one channel per group |
| GF pooling boundaries and depthwise convolutions | Zero padding is excluded from the pooling average; depthwise convolutions have no additional BN, activation, or bias |
| Additional GF shortcut | `shortcut=False`; an extra residual connection can be enabled explicitly |
| PFSG candidate frequencies | All 16 pairs `(u, v)` with `u, v` in `{0, 1, 2, 3}`; configurable through `frequencies` |
| PFSG phase and selection | Parameters shared across four groups; phases and logits initialized to zero; temperature 1; hard straight-through selection during training |
| PFSG projection and MLP | Raw projection sums without additional normalization; MLP reduction ratio 16 |
| EAFS and ECA | Energy mapping initialized with alpha=1, beta=0, and epsilon=1e-6; ECA gamma=2 and b=1, with kernel size rounded down to an odd integer |
| TD inverted residual convolutions | No additional BN by default; Swish is applied only after the final projection |

See the constructors and docstrings for optional settings. These defaults require comparison with the actual training configuration before attributing the manuscript's reported parameter count or FLOPs to this code.

## Usage

Python 3.10 or later is required. The dependencies are PyTorch, NumPy, and Pillow. `requirements.txt` records the versions used for the engineering checks. Install a GPU build of PyTorch that matches the local CUDA environment. Run the following commands from the collection's root directory:

```powershell
python examples/check_modules.py --device cpu
python examples/check_modules.py --device cuda --amp
```

The second command requires CUDA. These checks use synthetic tensors, require no dataset or pretrained weights, and do not evaluate detection metrics.

```python
import torch
from modules import C3k2_GF, TD_CFEM

gf = C3k2_GF(64, 128, n=2, c3k=True)
td = TD_CFEM(128)
x = torch.randn(2, 64, 40, 60)
y = td(gf(x))  # [2, 128, 40, 60]

# Use deterministic frequency selection for inference.
gf.eval()
td.eval()
with torch.no_grad():
    y = td(gf(x))
```

TD-CFEM requires a channel count divisible by four and preserves the input channel count. GF requires at least three internal channels. To integrate these classes into Ultralytics, register them in the module imports and `parse_model`, pass the actual input/output channel counts and repeat count, and verify the insertion locations. This collection does not include a complete detector YAML, a density-injection path, or a detection trainer.

## Supplied GMC code

The supplied source implements a **standalone patch counter**. It divides RGB or three-channel infrared images into a 3x3 or 5x5 grid of patches. A shared CNN and MLP predict nonnegative patch counts, from which total counts and a softmax attention matrix are computed. With YOLO box annotations, grid-count labels are generated by assigning each box center to a cell. Training uses a weighted sum of grid-count and total-count Smooth L1 losses. Grayscale images are converted to three channels during preprocessing.

This version does not implement the manuscript's 7/5/3 multiscale branches, grid-relation Transformer, DGA, or detection-feature injection. Its `attention` matrix is an output and is not used to enhance detection features in the supplied source. `yolo_matrix_collage.py` loads grid-counter weights. These files therefore represent the supplied partial implementation.

The training entry point reads existing `train`, `val`, and `test` subdirectories under the image and label roots. It does not repartition the dataset. The resized image dimensions must be divisible by the selected grid size. Machine-specific paths in the original scripts are preserved; the added entry point overrides them through command-line arguments:

```powershell
python examples/train_gmc.py --image-root "D:/dataset/images" --label-root "D:/dataset/labels" --out-dir "runs/gmc" --grid 5 --epochs 50
python gmc/yolo_matrix_collage.py --weights "runs/gmc/best_grid_matrix_counter.pt" --image "D:/dataset/images/test/example.jpg" --device cpu --out_dir "runs/collage"
```

The training entry point uses YOLO labels by default. Use `--label-format` to select another format supported by the original code. Use `--help` to list all arguments.

## Verification scope

The reconstructed modules passed formula comparisons, non-square input checks, finite-gradient checks, deterministic inference checks, and CUDA mixed-precision checks. The included example also checks module composition and both grid sizes of the supplied GMC counter. The added training entry point and visualization command passed a one-epoch synthetic-data check covering training, checkpoint saving/loading, and visualization.

These checks establish engineering functionality. They do not reproduce the manuscript's mAP, FPS, or ablation results. Trained weights are not included.

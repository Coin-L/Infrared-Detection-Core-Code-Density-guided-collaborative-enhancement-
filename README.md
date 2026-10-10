# 无人机红外检测核心代码合集

本合集保留所提供的 GMC 源码，并依据当前论文的方法描述补写 **C3k2-GF** 和 **TD-CFEM**。补写文件是可运行的参考实现；论文未明确的配置已在下文和源码中注明，不能将其认定为产生论文实验结果的原始代码。

## 文件

| 文件 | 内容与来源 |
|---|---|
| `modules/c3k2_gf.py` | 按论文补写：热噪声门控、三尺度深度卷积、互补融合及 CSP 结构 |
| `modules/td_cfem.py` | 按论文补写：空间、通道、参数化频率门控与倒残差增强 |
| `modules/common.py` | 所需的 Conv–BN–SiLU 基础层，无 Ultralytics 依赖 |
| `gmc/run_grid_matrix_counter_direct.py` | 所提供的独立网格计数器训练脚本，原样保留 |
| `gmc/yolo_matrix_collage.py` | 所提供的预测、矩阵可视化脚本，原样保留 |
| `gmc/grid_matrix_counter.py` | 两份原脚本导入的同目录依赖，原样补齐 |
| `examples/train_gmc.py` | 路径可配置的训练入口，调用原训练流程 |
| `examples/check_modules.py` | 随机张量的形状、反传及串联检查 |
| `PROVENANCE.json` | 源码来源及 SHA-256 校验记录 |

## 两个补写模块

**C3k2-GF。** `TNSG` 以 7×7、步长 1 的局部均值形成 `sigmoid(1 − AvgPool(X))`，得到 `X1`。另一分支按通道分成三组，分别执行 3×3、5×5、7×7 深度卷积并拼接得到 `X2`，按 `X1 * X2 + X1 + X2` 融合。内层 CSP 采用单次 1×1 投影后分支；外层保留 C3k2/C2f 的中间输出拼接。这里的局部均值是低频代理，未添加论文公式之外的 FFT。

**TD-CFEM。** 固定拉普拉斯卷积配合 Conv–BN–ReLU 产生空间门控；ECA 产生通道门控。频率分支将通道分成四组，通过带可学习相位的余弦投影、Gumbel-Softmax 选择和相对能量门控形成频率门控。三路门控相乘后，执行扩张比 2 的 1×1–DW3×3–1×1 变换，最后施加 Swish 并与原输入相加。训练采用随机选择，`eval()` 下采用确定性 argmax。

论文没有明确列出全部实现超参数，本参考实现作如下约定：

| 项目 | 默认约定 |
|---|---|
| GF 三组通道 | 尽量均分，余数依次分配；每组至少 1 个通道 |
| GF 池化边界 / 深度卷积 | 平均时不计零填充；深度卷积无额外 BN、激活和偏置 |
| GF 额外残差 | `shortcut=False`；可显式启用，但不作为默认论文路径 |
| PFSG 候选频率 | `(u,v) ∈ {0,1,2,3}²`，共 16 个；可通过 `frequencies` 指定 |
| PFSG 相位 / 选择 | 四组共享参数；相位和 logits 初始为 0；温度 1，训练采用 hard straight-through |
| PFSG 投影 / MLP | 原始求和、不额外归一化；MLP reduction=16 |
| EAFS / ECA | 能量映射初值 α=1、β=0，ε=10⁻⁶；ECA γ=2、b=1，核长向下取奇数 |
| TD 倒残差卷积 | 默认不附加 BN，Swish 仅置于最终投影之后 |

可选配置的含义见构造器和 docstring。这些默认值需要与实际训练配置核对，不能据此推断论文报告的参数量或 FLOPs。

## 使用

Python ≥3.10。依赖仅为 PyTorch、NumPy、Pillow；`requirements.txt` 记录了本次检查环境的版本。GPU 版 PyTorch 应按本机 CUDA 环境安装。在解压后的合集根目录运行：

```powershell
python examples/check_modules.py --device cpu
python examples/check_modules.py --device cuda --amp
```

第二条用于有 CUDA 的环境。检查使用合成张量，不需要数据集或已有权重，也不进行检测指标评估。

```python
import torch
from modules import C3k2_GF, TD_CFEM

gf = C3k2_GF(64, 128, n=2, c3k=True)
td = TD_CFEM(128)
x = torch.randn(2, 64, 40, 60)
y = td(gf(x))  # [2, 128, 40, 60]

# 推理时切换到确定性的频率选择。
gf.eval()
td.eval()
with torch.no_grad():
    y = td(gf(x))
```

TD-CFEM 要求通道数能被 4 整除，输入输出通道相同；GF 的内部通道数至少为 3。接入 Ultralytics 时，需在模块导入和 `parse_model` 中登记这两个类，传入实际的输入/输出通道及重复次数，并确认各插入位置。本合集未包含完整检测器 YAML、密度注入路径或检测训练器。

## GMC 原码说明

提供的源码实现的是**独立图块计数器**：将 RGB/三通道红外图像分成 3×3 或 5×5 图块，经共享 CNN 与 MLP 输出非负计数，再计算总数和 softmax 注意力矩阵。YOLO 框标注按框中心落入的网格生成计数标签；训练损失为网格计数与总计数的加权 Smooth L1。灰度图在预处理时转换为三通道。

该版本没有实现论文中描述的 7/5/3 三尺度分支、网格关系 Transformer、DGA 或检测特征注入。其 `attention` 是输出矩阵，未在该源码中参与检测特征增强；`yolo_matrix_collage.py` 使用网格计数器权重。应将它作为所提供的部分实现进行说明。

训练入口读取已有的 `images/train、val、test` 与对应 `labels/train、val、test`，不重新划分数据。图片尺寸需能被所选网格数整除。原脚本中的本机路径保留，通过新增入口覆盖：

```powershell
python examples/train_gmc.py --image-root "D:/dataset/images" --label-root "D:/dataset/labels" --out-dir "runs/gmc" --grid 5 --epochs 50
python gmc/yolo_matrix_collage.py --weights "runs/gmc/best_grid_matrix_counter.pt" --image "D:/dataset/images/test/example.jpg" --device cpu --out_dir "runs/collage"
```

训练入口默认 YOLO 标签格式，可用 `--label-format` 指定原码支持的其他格式。`--help` 显示全部参数。

## 检查范围

已对两个补写模块进行公式对照、非方形输入、有限梯度、推理确定性及 CUDA 混合精度检查；合集示例另检查模块串联和原 GMC 的两种网格输出。新增训练入口与可视化命令还通过了合成数据的一轮训练、保存和加载检查。检查确认工程可运行性，不代表复现了论文的 mAP、FPS 或消融结果。合集不包含训练权重。

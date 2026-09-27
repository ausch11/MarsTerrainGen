# Mars LoRA：SD1.5 基线与 B1 地貌特征监督

本项目由 `lora_对比_V10-4_TE_unet_lora_v6.ipynb` 按功能重构，保留 UNet + Text Encoder LoRA 的训练路线，增加可关闭的 **B1：用冻结 DINOv2-base 的目标图像特征监督 UNet 中间表征**。原 notebook 未修改。

**训练输入：图像、对应文本；生成输入：只有文本。** DINOv2 和投影头只参与训练，导出的生成模型不需要参考图。当前实现是全局特征监督，不包括参考图条件分支、局部 patch 对齐、FID 计算或下游分类训练。

## 1. 项目文件与原 cell 的对应关系

Cell 编号按 notebook 的全部 cell 从 1 开始。

| 原 cell | 项目文件 | 功能 |
|---|---|---|
| 1 | `requirements.txt`、`pyproject.toml`、各模块导入 | 依赖与运行入口 |
| 2 | `mars_lora/config.py`、`configs/*.json` | 参数、配置检查、B0/B1 对照 |
| 3 | `mars_lora/data.py`、`cache.py` | 图像/文本配对、采样、预处理、latent 缓存、batch 拼接 |
| 4 | `mars_lora/validation.py`、`checkpoint.py` | 验证采样、保存/加载、断点管理 |
| 5 | 不执行 | 原注释/探索内容 |
| 6 | `mars_lora/models.py`、`optim.py`、`training.py` | 模型冻结、LoRA 注入、优化器、Accelerate 初始化 |
| 7 | 不执行 | 原注释内容 |
| 8 | `mars_lora/training.py`、`losses.py` | 扩散训练、累积梯度、记录、检查点 |
| 9 | `mars_lora/checkpoint.py`、`inference.py` | 最终导出与纯文本生成 |
| 10–11 | 不执行 | 空 cell |
| B1 新增 | `mars_lora/teacher.py`、`alignment.py` | DINOv2 目标、可训练投影头、余弦对齐损失 |

入口为 `inspect_data.py`、`train.py`、`generate.py`，不依赖 notebook 执行顺序或交互变量。

## 2. 环境安装

面向单 GPU。保留你已有的 `torch 2.6.0+cu124`、`peft 0.18.0`、`accelerate 1.12.0` 环境。代码按 Diffusers 0.36 系列 API 实现；本地实际验证版本为 **0.36.0 正式版**，并未验证你安装的 `0.36.0.dev0` 对应的具体 Git 提交。

在项目目录运行：

```bash
python -m pip install -r requirements.txt
python -m pip check
python -c "import torch,diffusers,transformers,peft,accelerate; print(torch.__version__,diffusers.__version__,transformers.__version__,peft.__version__,accelerate.__version__); print('CUDA:',torch.cuda.is_available())"
```

`requirements.txt` 不直接安装或替换 Torch，不要求 torchvision。已有的开发版 Diffusers 若被 pip 替换，先在独立环境复现，不要覆盖唯一的原实验环境。需要固定本次实际测试的软件栈时，可使用 `diffusers==0.36.0`、`transformers==4.57.6`，其余版本见 `TEST_REPORT.md`。用脚本入口无需安装本项目；需要 `mars-lora-train` 等命令时另执行 `python -m pip install -e . --no-deps`。

`xformers` 为可选项，加载失败时保留模型原有 attention processor；默认 AdamW，不要求 bitsandbytes。多进程分布式训练会明确报错。无需启动多个 accelerate 进程。

## 3. 数据与路径配置

首先修改 `configs/baseline.json` 和 `configs/b1.json` 中的：

- `pretrained_model_name_or_path`：本地 **Diffusers 格式** SD1.5 目录，包含 `unet/`、`vae/`、`text_encoder/`、`tokenizer/`、`scheduler/`；或对应的 Hub 模型 ID。
- `train_data_dir`：真实训练样本目录。
- `validation_data_dir`：独立验证集目录；暂时没有时保持 `null`，此时不选取 best 模型。
- `output_dir`：基线与 B1 必须使用不同目录。

推荐的文件布局：

```text
train/
  crater/
    sample_001.png
    sample_001.txt
  ridges/
    sample_002.png
    sample_002.txt
```

文本是 UTF-8 同名 `.txt`，一张图对应一个描述。类别取训练根目录下的第一级目录名；目录名开头的数字 **不再自动解释成 repeats**。如需重复采样，显式配置 `repeats`，例如 `{"crater": 2}`。

默认检查去掉 repeats 前的图像文件数为 17000、类别数为 17、每类 1000 张；检查结果会打印。若从这 17000 张中划分验证集，必须同步修改 `expected_unique_images` 与 `expected_images_per_class`，或设为 `null`。平铺目录也能读取，但类别会统一记作 `__root__`；这种布局需要调整类别数量检查，无法自动推断 17 类。

```bash
python inspect_data.py --config configs/b1.json
```

缺少/空白 caption 默认报错；图像内容的 SHA256 用于训练/验证重叠检查和缓存失效检测。同一区域的相邻火星影像切片仍可能发生地理泄漏，不能仅依靠文件哈希划分训练/验证/测试。

所有相对路径相对于**运行命令的当前目录**。配置中保留了原 notebook 的 Linux 数据路径；在其他机器必须修改。也可用参数覆盖：

```bash
python inspect_data.py --config configs/b1.json --set "train_data_dir=/data/mars/train" --set "pretrained_model_name_or_path=/models/sd15"
```

## 4. DINOv2-base 与缓存

默认 `teacher_model_name_or_path="facebook/dinov2-base"`。首次运行需要下载其模型及预处理配置。离线服务器可先在联网环境下载完整 Hugging Face 模型目录，再将该参数改成本地目录，并设置 `local_files_only=true`；此时 SD1.5 和 DINOv2 都必须本地可用。

可先独立完成缓存：

```bash
python train.py --config configs/b1.json --cache-only
```

缓存目录包含独立的 `latent/` 和 `feature/`，依据模型身份、预处理、分辨率、随机种子、图像内容等创建版本签名。正式训练会检查并补齐损坏/缺失的缓存。Dataset worker 只读取 CPU 数据，不携带 VAE 或 DINOv2 CUDA 模型。

“缓存一次裁剪结果和一次 VAE 后验采样结果”在本项目中的具体含义：

1. 每张图先缩放并确定一个可复现的裁剪窗口。VAE 和 DINOv2 来自同一个图像视野。你的原图若恰好是 256×256、训练也为 256，则没有随机裁剪位置的变化。
2. VAE 编码得到均值/标准差，以该图像独立的固定随机数采样一次 `z = mean + std * noise`，乘一次 `scaling_factor` 后缓存。
3. 后续 epoch 重用这个 latent；**训练的扩散噪声和 timestep 每次仍重新采样**，文本片段也可随 epoch 重新洗牌。
4. DINOv2 将相同图像视野缩放至 224×224，再使用模型自己的归一化参数，缓存一个 768 维全局特征。不会再次隐式中心裁剪。

若设置 `cache_latents=false`，VAE 后验在训练时重新采样，图像视野仍固定以匹配教师缓存。当前不支持随机翻转；启用 `random_flip` 会报错。改变缓存算法或手动替换且保留文件时间戳的本地模型时，应更改 `cache_tag` 强制创建新缓存。本地模型文件身份记录大小和修改时间，未对全部模型权重做内容哈希。

## 5. 先试运行，再训练对照组

配置保留原主要超参数：256 分辨率、batch 80、累积 4、70 epochs、UNet rank/alpha 64、TE rank/alpha 32、LR 分别为 4e-4 / 1e-4、BF16。Dataloader workers 从 96 调整为 8，可按 CPU 和磁盘情况再调。

对于 17000 张、无 repeats、单卡且不丢尾 batch：每 epoch 有 `ceil(17000/80)=213` 个 microbatch，`ceil(213/4)=54` 次更新，共 3780 次。完整累积组对应 320 张，末组不足 320 张。首次可用完整计划运行 10 次更新后安全暂停：

```bash
python train.py --config configs/b1.json --stop-after-steps 10
python train.py --config configs/b1.json --set resume_from_checkpoint=latest
```

`--stop-after-steps` 不改变完整学习率计划；`max_train_steps` 会改变计划。完成试运行后也可以给基线、B1 各自使用新的输出目录：

```bash
python train.py --config configs/baseline.json
python train.py --config configs/b1.json
```

同一输出目录发现已有 `run.json` 时不会静默覆盖，需显式续训或改目录。续训保存 LoRA、B1 投影头、optimizer、scheduler、随机数状态、下一 epoch/batch 位置，并检查数据/配置签名；原 notebook 的检查点不视作本项目可完整续训的检查点。

若 48GB GPU 上仍出现 OOM，先尝试 `--set gradient_checkpointing=true`；必要时同时将 B0/B1 改为 batch 40、累积 8。后一种方式保留完整组的有效 batch，但浮点计算顺序和尾组有所不同，需要重新开始对照实验。B1 的 DINOv2 在缓存结束后释放，训练时额外保留中间激活和投影头。**本地未实测该 vGPU 上的峰值显存。**

## 6. B1 的确切实现

```text
真实图像 x ── 同一视野 ── VAE ── z0 ── 加噪 ── zt ──────────┐
对应文本 c ── CLIP + TE LoRA ── 文本 embedding ─────────────┤
                                                        UNet + LoRA
                                                         ├─ 噪声预测 → 扩散损失
                                                         └─ mid_block → GAP → LN + MLP → p
真实图像 x ── 同一视野 ── 冻结 DINOv2-base ── CLS 特征 d ───────→ 余弦对齐损失
```

总损失为：

```text
L = L_diffusion + lambda(s) * mean[1 - cosine(p, stop_gradient(d))]
lambda(s) = 0.05 * min(1, (s + 1) / 378)
```

`s` 为已完成更新次数。默认取 DINOv2 的 CLS token，也可以配置 `teacher_pooling="patch_mean"`，但它仍是全局特征。标准 SD1.5 mid-block 的 1280 通道先空间平均，投影头为 `LayerNorm → Linear(1280,1024) → GELU → Linear(1024,768)`，实际维度由模型配置获取。

只有 UNet LoRA、可选 TE LoRA、投影头可训练；VAE、DINOv2、原始 UNet/CLIP 权重被冻结。LoRA 和投影头参数保持 FP32，前向计算用配置的混合精度。DINOv2 缓存分支禁用梯度，UNet mid-block 特征不 detach，因此 B1 损失能真正更新 LoRA。梯度检查点使用非 reentrant 方式，且有实际反传测试。

当前对所有随机 timestep 使用同一对齐规则，没有加入基于 timestep 的筛选或局部纹理损失。`alignment_weight=0.05` 是起始实验值，不是已在火星数据上调优的值；可用 0.01/0.05/0.1 做权重消融。`b1_enabled=false` 会完全跳过教师与投影头，是正确的 B0 对照配置。

方法借鉴了 [REPA 的表征对齐思想](https://arxiv.org/abs/2410.06940)，这里将其改为 SD1.5 UNet LoRA 的全局监督；不能据此宣称复现了原论文的 patch 对齐或性能收益。教师的 CLS/patch 表征接口见 [DINOv2 官方文档](https://huggingface.co/docs/transformers/model_doc/dinov2)。LoRA 的 PEFT 接入参考 [Diffusers 文档](https://huggingface.co/docs/diffusers/training/lora)。

## 7. 验证、导出与文本生成

有独立验证集时，使用固定随机种子的验证去噪 MSE，B0/B1 使用同一指标；`best-validation/` 依据该指标保存，**不代表最佳 FID**。默认每 10 epoch 生成一次类别预览：每个实际类别取一个 caption，每个 prompt 4 张，分辨率明确为 256。可用 `validation_prompts_file` 指定 UTF-8 JSON 字符串列表，或形如 `[{"class_name":"ridges","prompt":"..."}]` 的列表。

```text
outputs/b1/
  run.json                  完整配置、环境版本、数据与缓存摘要
  metrics.jsonl             扩散损失、对齐损失、权重、学习率等
  logs/                     TensorBoard
  checkpoint-N/             可恢复训练状态，包含 projector.safetensors
  best-validation/          有验证集时保存的推理适配器
  final/                    完成训练后的推理适配器，不含投影头
  validation/step-N/        各类别固定种子生成图及 prompt
```

```bash
tensorboard --logdir outputs
python generate.py --adapter outputs/b1/final --prompt "Mars surface, ridges, parallel ridges, fine texture" --output outputs/generated_b1 --num-images 16 --steps 30 --seed 1337
```

请将示例 prompt 换成与你的训练描述一致的地貌术语。迁移模型目录时用 `--base-model /new/path/to/sd15`；离线用 `--local-files-only`。同一路径、同一 seed 的输出文件会覆盖，实验之间使用不同的生成目录。

导出保存原始 PEFT LoRA 状态与完整 rank/alpha 配置，以保证训练与加载一致。请用本项目 `generate.py` / `load_pipeline` 加载，**不要把该目录直接当作标准单文件 `pipeline.load_lora_weights()` 包或 Kohya 格式**。推理仍需要原 SD1.5 基座，但不加载 DINOv2、投影头或目标图像。

## 8. 与原 notebook 的实现差异

除拆分文件和增加 B1，还修正了会影响对照可靠性的实现细节：

- 不因缓存子目录存在而漏掉根目录图像，不将数字类别名自动变成 repeats；明确核对数据量。
- latent 与教师目标来自同一确定性视野，缓存签名包含图像内容与模型/预处理信息；损坏缓存先重建，不在 collate 中混合 latent 与 `None`。
- VAE 默认 FP32、可训练参数 FP32；LoRA alpha 和实际 target modules 一并保存。
- 不再按训练 loss 选择“最佳生成模型”；验证不改变后续训练随机数流，不隐式切换为 512×512。
- 修正 Min-SNR 在 `v_prediction` 下的分母、累积更新计数、检查点保留数量与显式续训行为。
- 原注释中的可选探索/未启用优化器没有作为新的默认训练行为引入。

这些公共修正同样作用于 B0、B1。公平比较应使用本项目的 B0 与 B1；原 notebook 的历史结果可另作参考，不能把全部差异归因于 B1。

## 9. 验证范围与后续实验

```bash
python -m pip install pytest
python -m pytest -q
```

自动测试使用真实 Diffusers/Transformers/PEFT 组件构成的微型随机模型与合成数据，无需下载模型。涵盖 B1 梯度、缓存、保存重载、断点恢复、数据 worker、纯文本生成，详见 `TEST_REPORT.md`。

**这些测试验证软件行为，不验证图像质量。** 本机没有你的 17k 数据、SD1.5 实际权重和指定 vGPU，尚未执行完整训练、FID 或下游分类实验。DINOv2 在火星域的有效性也需要实测。

正式比较固定数据划分、训练预算、生成数量、prompt、seed、采样器、步数与 CFG；降低 FID 才表示该指标改善。全局 FID 与分类别 FID 应分别报告，并注明 Inception 特征提取与预处理。避免只用作为监督的 DINOv2 特征证明提升；增加独立的形态检查与下游分类验证。下游比较真实数据、真实+B0 生成、真实+B1 生成三组，保持分类器与训练预算一致，测试集必须是真实且独立的数据。

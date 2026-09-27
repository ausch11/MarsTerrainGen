# 验证报告

验证日期：2026-09-20。验证对象为本目录最终代码，原 notebook 未执行、未修改。

## 结果

最终测试进程返回码 **0**：`16 passed, 1 warning in 21.41s`。机器可读结果见 `test-results.xml`。测试没有下载实际 SD1.5 / DINOv2-base 权重，使用本地构造并保存的微型真实模型。

| 验证项 | 结果与范围 |
|---|---|
| B1 独立梯度 | 只反传 alignment loss，UNet LoRA 和投影头均有非零梯度；冻结的基础参数及教师目标无梯度 |
| 梯度检查点 | 上述反传分别在关闭/开启 UNet 非 reentrant checkpoint 时通过 |
| 保存与重载 | LoRA 与投影头重载后输出精确一致；验证 alpha 不等于 rank 的情况 |
| Min-SNR | epsilon / v_prediction 分支分别与计算公式一致 |
| 图像枚举 | 根目录图像、含数字前缀的类别目录都能读取，未隐式重复样本 |
| 缓存 | 修改图像内容、配置版本或扩展名不会误用旧项；损坏缓存能检测；混合缺失/有效缓存明确报错 |
| 可复现预处理 | 单次裁剪和 epoch 索引可复现；文本处理不依赖 worker 的随机数进度 |
| B1 完整流程 | 微型 VAE/UNet/CLIP/DINOv2 实际缓存、训练 4 次更新、固定种子验证、生成预览、最终导出 |
| 断点恢复 | 第 1 次更新后暂停，从 epoch 中途恢复至第 4 次；最终 UNet/TE LoRA 及投影头与连续训练逐张量比较，`rtol=1e-5, atol=1e-6` 通过 |
| 纯文本推理 | 移走教师模型目录后加载 `final/`，实际完成 2 步 DDIM 采样并输出 32×32 图像；final 不含投影头 |
| 基线开关 | 将教师路径设为无效值，B0 仍完成训练，且没有投影头 |
| 多 worker | Windows 下实际使用 2 个持久 worker，完成跨 epoch 的 3 次更新 |
| 配置与入口 | B0/B1 解析后仅 `output_dir`、`b1_enabled` 不同；三个脚本的 `--help` 成功 |

附带的一条警告为 CPU 环境下某 CUDA autocast 上下文被自动禁用。测试中的 tokenizer 故意缩小到 16 token，预览长描述发生截断；正式配置为 SD1.5 的 77 token。微型随机模型生成图仅检查软件流程，没有图像质量意义。

首次测试的结束阶段遇到本机沙箱对报告写入的权限限制。获得授权后，在沙箱外运行同一组本地测试并成功保存报告；该权限问题不属于训练代码测试失败。最终报告已经包含后续 LoRA 保存逻辑修改。

## 本地实际环境

| 组件 | 版本 |
|---|---|
| 系统 | Windows，CPU 执行 |
| Python | 3.12.14 |
| Torch | 2.6.0+cpu |
| Diffusers | 0.36.0 |
| Transformers | 4.57.6 |
| PEFT | 0.18.0 |
| Accelerate | 1.12.0 |
| NumPy | 2.3.5 |
| Pillow | 12.3.0 |
| Pytest | 9.1.1 |

本地测试解释器位于工作区外层 `.mars-b1-test-venv`，不会打包进项目。未改动用户远端 GPU 环境。

复验命令（项目目录）：

```bash
python -m pytest -q --junitxml test-results.xml
```

本次为隔离宿主环境的第三方 pytest 插件，设置了 `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1`，并将临时模型目录指定到工作区的 `.mars-b1-pytest-run3`；测试结束后保留这些文件供本机审查，不打包交付。

## 尚未验证

- 用户 vGPU-48GB-425W 上 CUDA 12.4 / BF16 的完整训练、速度、显存上限和 xformers 后端。
- 用户 `diffusers 0.36.0.dev0` 的具体 Git 提交。正式版本通过不能等价为所有开发版提交都通过。
- 实际 17000 张图像的数据完整性、caption 品质和区域划分。
- 真实 SD1.5 / DINOv2-base 权重组合的长时间数值稳定性、最优超参数。
- FID 变化、文本控制性提升、下游分类精度提升。不能根据上述单元/集成测试得出这些结论。
- 未启用的可选配置组合（如 8-bit Adam、非默认 VAE dtype）和跨操作系统/GPU 的逐位续训一致性。

## 源文件追踪

原文件：`C:/Users/29626/Downloads/lora_对比_V10-4_TE_unet_lora_v6.ipynb`。

原文件 SHA256：

```text
D0AA80D3ABFD9DAF9F535634E1DBAA6DE0E217CF8197267270AF6A82F9A2DF2E
```

Cell 对应关系与公共修正见 README。ZIP 内的 `PACKAGE_MANIFEST.json` 记录交付文件的 SHA256；不包含模型、用户数据或训练产物。

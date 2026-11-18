# POC：图像清晰化最小骨架（方案一）

对应设计文档：[`docs/image-deblur/03-plan-a-cpu-lightweight.md`](../docs/image-deblur/03-plan-a-cpu-lightweight.md)

**目标不是做出产品，而是用真实数据替换报告里所有标注为"估算"的耗时数字。**

---

## 1. 环境准备

```bash
cd poc
python -m venv .venv && .venv/Scripts/activate     # Windows
pip install -r requirements.txt
```

依赖说明：`onnxruntime`（CPU）、`opencv-contrib-python`（`dnn_superres` 在 contrib 中）、`numpy`、`Pillow`。**不需要 torch**。

## 2. 放置权重（可选）

见 [`models/README.md`](models/README.md)。**不放权重也能跑**——会自动降级到 OpenCV / Lanczos。

## 3. 运行

```bash
# 单图，默认 scale=2、交互档
python deblur_tool.py --input path/to/blurry.png --scale 2 --out out/sharp.png

# 输出完整分析指标与耗时（用于阈值标定）
python deblur_tool.py --input path/to/blurry.png --json

# 显式允许生成式先验（仅纯照片；文字/UI 会被自动拦截并告警）
python deblur_tool.py --input path/to/photo.jpg --allow-generative-prior

# 调整预算
python deblur_tool.py --input path/to/4k.png --max-latency-ms 8000
```

### 输出示例

```text
output      : poc/out/sharp.png
degradation : mixed  content: text_ui
chain       : lanczos
elapsed     : 189 ms
warnings    : fell_back_to_lanczos
```

> 上面是**未放权重**时的真实输出（`lanczos` 兜底链路）。放入 `realesr-general-x4v3.onnx` 后，
> `chain` 会变成 `realesr-general-x4v3`，耗时按第 6 节方法实测。

## 4. 作为库调用（接入 Agent）

```python
from deblur_tool import SharpenRequest, sharpen, build_tool_schema

# 注册给 LLM
tools = [build_tool_schema()]

# 处理用户图片
res = sharpen(SharpenRequest(
    image_uri="/path/to/blurry.png",
    scale=2,
    tier="interactive",
    max_latency_ms=3000,
))
print(res.output_uri, res.degradation_type, res.content_type, res.elapsed_ms, res.warnings)
```

`sharpen()` **不会抛异常**：图像无法解码时返回 `output_uri=None` + `warnings=["decode_failed: ..."]`；
模型不可用时自动降级并在 `model_chain` 中如实反映。

## 5. 模块结构

| 函数 / 类 | 作用 | 对应文档章节 |
|---|---|---|
| `analyze()` | 6 项退化指标 + 内容类型判定 | 01 第 3、4 节 |
| `resolve_backends()` | 按内容类型与档位生成后端候选链 | 03 第 4.2 节 |
| `tiled_infer()` | 分块推理 + 放大后空间羽化融合 | 03 第 4.3 节 |
| `protect_text_regions()` | 锐度衰减 + 过冲钳位 + 色彩对齐 | 03 第 4.4 节 |
| `sharpen()` | 主流程与四级兜底 | 03 第 4.5 节 |
| `build_tool_schema()` | LLM function-calling 定义 | 03 第 3.2 节 |

## 6. 用它产出实测数据（替换报告中的估算）

报告 `03` 第 6.3 节与 `04` 第 5.2 节的耗时均为估算。实测方法：

1. 准备 30 张真实截图（建议覆盖：纯 UI、图文混排、照片、4K 大图各 7~8 张）。
2. 逐张运行 `--json`，收集 `elapsed_ms` 与 `metrics`。
3. 计算 p50 / p95，回填到文档表格，并把"估算"字样改为"实测（机器型号 / 日期）"。
4. 同时核对 `degradation_type` 与人工判断是否一致——**判定准确率不达标时，先调阈值再谈模型**。

建议的记录表：

| 图片 | 人工判定 | 机器判定 | text_ratio | elapsed_ms | chain | warnings |
|---|---|---|---|---|---|---|
| shot_01.png | lowres+text_ui | | | | | |

## 7. 实测记录（2026-11-18，Windows / Python 3.13 / OpenCV 5.1 / numpy 2.5）

**未安装 onnxruntime、未放任何权重的条件下**跑通，同时验证了"依赖缺失 → 降级"路径。

| 用例 | 输入 | 判定结果 | 链路 | 耗时 |
|---|---|---|---|---|
| 清晰文字截图 | 800×600 | `none` / `text_ui` | `passthrough` | < 30 ms |
| 退化截图（缩放 + JPEG q45 + σ1.2 模糊） | 400×300 | `mixed` / `text_ui` | `lanczos` 兜底 | 189 ms |
| 重度模糊 | 400×300 | `lowres` / `text_ui` | — | — |
| 纯噪声图 | 400×300 | `mixed` / `photo` | — | — |
| 文件不存在 | — | — | `output_uri=None` + `warnings` | 无异常 |

### 带真实权重的实测（`realesr-general-x4v3.onnx`，4.64 MB）

| 输入 | 像素量 | 链路 | 耗时 | 说明 |
|---|---|---|---|---|
| 400×300 退化截图 | 0.12 MP | `realesr-general-x4v3` | **1.69 s** | 2× 输出，无告警 |
| 1280×720 退化截图 | 0.92 MP | `realesr-general-x4v3`（预算 8s） | **14.48 s** | 原文档估算 1.5s，**低估约 10 倍** |
| 1280×720 退化截图 | 0.92 MP | `lanczos`（预算 3s） | **1.55 s** | 触发 `heavy_model_skipped_over_budget(est=15045ms)` |
| 800×600 清晰图 | 0.48 MP | `passthrough` | 0.08 s | 判退生效 |

模型单块推理基准（ONNX Runtime 1.32，8 线程）：192×192 输入 → 398 ms，输出 768×768，输入输出均为动态 H/W。

**由此得出 16 ms / 千像素 的经验系数**，`_estimate_cost_ms()` 已按此重写（原先 1.3 ms/千像素，误差 12 倍）。

### 开发过程中发现并修复的 4 个缺陷

1. **`blur_score` 用了 `ksize=3`** → 数值放大一个数量级（清晰图 844 → 9845），阈值 500 完全失效。改为 `ksize=1`。
2. **`edge_width` 量纲错误**：原实现乘以 1.5 并截断到 12，实测清晰图就被截断。改为直接输出梯度/拉普拉斯比值（`edge_spread`，清晰 2.73 / 模糊 6.69 / 重模糊 27.3），阈值改为 4.0。
3. **`halo_ratio` 判据失效**：低阈值下文字图的边缘带恒为高值（清晰图 0.906），无法区分失焦。改为「各向异性 < 0.35 + edge_spread > 4.0」判失焦，`detail_energy`（阈值 32）仅作"已足够锐"佐证。
4. **文本占比代理指标命中率仅 0.07%~0.5%**：Canny+闭运算把文字连成整行大块（面积 1600~6000），被"面积 ≤ 400"过滤器全部丢弃。改为黑帽/顶帽形态学响应，实测文字图稳定 4%~6%、噪声图 100%（截断为 0）。

### 分块融合正确性验证

`tiled_infer(tile=256, overlap=32)` 在 800×600 输入上放大 2×，与整图一次性 Lanczos 的**平均绝对差为 0.0**，无零权重空洞，说明羽化权重与归一化实现正确。

> 以上均为**兜底链路**的实测数据。ONNX 主链路的耗时需放入权重后按第 6 节方法重测，
> 并回填至 `docs/image-deblur/03-plan-a-cpu-lightweight.md` 第 6.3 节。

## 8. 已知限制（POC 范围）

- **只有交互档**：`tier="offline"` 参数被接受但走同一链路；异步队列、文本分治、一致性校验属于方案二，见 `04` 文档。
- **无 NAFNet 去模糊**：判定为 `motion`/`defocus` 时目前仍走 SR 链路，需先完成 NAFNet ONNX 导出再接第 2 个后端。
- **文本占比是代理指标**：无 OCR，细碎纹理会被误判为文字（已用平坦区占比约束，但仍会误判）。
- **阈值未标定**：`BLUR_SCORE_*`、`ANISOTROPY_MOTION` 等全为文献经验值，**必须**用本地样本标定。
- **INT8 量化未启用**：需额外跑 `onnxruntime.quantization.quantize_dynamic` 并做精度回归。

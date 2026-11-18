# 模型权重目录

**权重不入库**（体积 + 许可证归档需要）。把下载好的文件按下列文件名放入本目录即可，缺失时 POC 会自动降级。

---

## 交互档（方案一）必需

| 文件名 | 来源 | 体积 | 许可证 | 用途 |
|---|---|---|---|---|
| `realesr-general-x4v3.onnx` | Real-ESRGAN 官方权重的社区 ONNX 导出（fp32, opset 17, 动态 H/W），HuggingFace `jonathanst29/tinier-upscale-models` | 4.64 MB | BSD-3-Clause | 主力超分（1.2M 参数，SRVGGNetCompact） |

**已就位**（2026-11-18 通过 `hf-mirror.com` 下载）：

```
文件: realesr-general-x4v3.onnx
大小: 4,866,396 bytes
SHA256: 924ebad6532777303582d4ce7811849b88869231a3ae7093e0f21200249df8d5
```

校验命令：

```powershell
(Get-FileHash models/realesr-general-x4v3.onnx -Algorithm SHA256).Hash.ToLower()
```

下载命令（换机器时复用）：

```powershell
curl.exe -L -o models/realesr-general-x4v3.onnx `
  https://hf-mirror.com/jonathanst29/tinier-upscale-models/resolve/main/realesr-general-x4v3.onnx
```

> 国内网络直连 `huggingface.co` 常失败，`hf-mirror.com` 实测约 30s 下完（~500 KB/s）。

若该文件缺失，POC 会尝试 OpenCV dnn_superres，再退到 Lanczos。

## 交互档可选

| 文件名 | 来源 | 体积 | 许可证 | 用途 |
|---|---|---|---|---|
| `EDSR_x4.pb` | `Saafke/EDSR_TensorFlow` | ~38 MB | Apache-2.0 | 非 GAN 保真超分（文字/UI 首选，CPU 单图约 3s） |
| `FSRCNN_x4.pb` | `Saafke/FSRCNN_TensorFlow` | ~40 KB | Apache-2.0 | 极速超分（毫秒级，质量一般，适合兜底） |

> 文件名必须与上表一致，`deblur_tool.resolve_backends()` 按此查找。

## 离线档（方案二）额外需要

| 文件名 | 来源 | 体积 | 许可证 | 用途 |
|---|---|---|---|---|
| `RealESRGAN_x4plus.onnx` | 同上 HF 仓库 | 64 MB | BSD-3-Clause | 重型超分（16.7M RRDB） |
| `NAFNet-GoPro-width64.onnx` | 由 NAFNet 官方权重自行导出，见下 | ~260 MB | MIT | 运动模糊去模糊 |
| `NAFNet-DPDD-width64.onnx` | 由 NAFNet 官方权重自行导出 | ~260 MB | MIT | 失焦（defocus）去模糊 |

### NAFNet ONNX 导出注意

1. 官方 `basicsr` 中的自定义 `LayerNorm2d` 在导出时会报
   `onnx::LayerNormalization ... not constant`，需替换为标准等价实现，或降至 opset 11 并先固定 shape。
2. 建议先导出**静态 shape**（如 1×3×256×256）验证数值一致性，再尝试 `dynamic_axes`；NAFNet 的
   `SimpleGate`/`LayerNorm2d` 组合对动态形状支持需逐个 patch 验证。
3. 验收标准：与 PyTorch 参考实现在同一输入上的 PSNR > 40 dB（fp32）。

---

## 许可与合规

- 下载权重后请一并归档其 LICENSE 文本与来源 URL、SHA256，纳入 `THIRD_PARTY_LICENSES`。
- **禁止**放入以下来源的权重：
  - `4x-UltraSharp` 等社区超分权重（多为 **CC BY-NC-SA 4.0**，不可商用）
  - SUPIR / StableSR / CodeFormer 相关权重（**非商用许可**）
  - GFPGAN 及其 DFDNet 相关权重（含 **CC BY-NC-SA** 与 NVIDIA 非商用条款）
- 详细结论见 [`docs/image-deblur/02-github-survey.md`](../../docs/image-deblur/02-github-survey.md) 第 3 节。

---

## 无权重时的行为

链路会自动按 `非GAN模型 → GAN模型 → Lanczos` 顺序降级，因此**不下载任何权重也能跑通**，
用于验证判定逻辑与耗时基线（此时 `model_chain` 会显示 `lanczos`，`warnings` 含 `fell_back_to_lanczos`）。

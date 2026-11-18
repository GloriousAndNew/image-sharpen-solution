# 02 · GitHub 开源项目调研

> 数据口径：**stars 与最后推送时间均于 2026-11-18 通过 GitHub REST API（`api.github.com/repos/{owner}/{repo}`）核实**，许可证通过 API `license.spdx_id` + `raw.githubusercontent.com` 的 LICENSE 原文双向核对。star 数随时间变化，本文数值不可当作实时数据使用。

---

## 0. 筛选标准

| 维度 | 要求 | 说明 |
|---|---|---|
| 星标 | ≥ 1000 为软门槛 | 低于此的项目多为论文一次性代码，权重与 issue 无人维护 |
| 维护活跃度 | 近 3 年有推送，或生态已稳定（权重可获取、社区有可用导出） | "停止更新"与"已完成"需区分：Real-ESRGAN 2024-08 后无推送，但权重与 ONNX 导出齐全，仍可用 |
| 许可证 | 允许商用（MIT / Apache-2.0 / BSD-3-Clause），或可明确隔离风险 | 排除 CC BY-NC、NTU S-Lab、自定义非商用许可 |
| CPU 可行性 | 能在无 GPU 环境下以可接受耗时出图 | 硬约束（本项目仅 CPU） |
| 与本场景相关性 | 面向真实退化，而非仅双三次下采样 | 只用合成 bicubic 训练的模型在截图上表现远不如真实退化模型 |

> 说明：星标与活跃度是**初筛**手段，不是质量结论。最终选型以"在该场景下的实际出图质量 + 许可证 + CPU 成本"三者为准。

---

## 1. 候选项目总表

### 1.1 通用超分 / 真实退化复原

| 项目 | 许可证 | Stars | 最后推送 | 规模 | CPU 可行性 | 适用退化 | 文字场景风险 |
|---|---|---|---|---|---|---|---|
| [`xinntao/Real-ESRGAN`](https://github.com/xinntao/Real-ESRGAN) | **BSD-3-Clause** | 37.5k | 2024-08-06 | `x4plus` RRDBNet 16.7M<br>`general-x4v3` SRVGGNetCompact 1.2M | ★★★★☆（x4v3）<br>★★☆☆☆（x4plus） | lowres / jpeg / mixed | 中高（x4plus）<br>中（x4v3） |
| [`cszn/BSRGAN`](https://github.com/cszn/BSRGAN) | **Apache-2.0** | 1.4k | 2024-04-12 | RRDBNet 16.7M | ★★☆☆☆ | lowres / jpeg | 高 |
| [`JingyunLiang/SwinIR`](https://github.com/JingyunLiang/SwinIR) | **Apache-2.0** | 5.6k | 2024-05-14 | SwinIR ~11.8M（classical SR） | ★☆☆☆☆（慢） | lowres（含 real-SR 权重） | 中 |
| [`XPixelGroup/HAT`](https://github.com/XPixelGroup/HAT) | **Apache-2.0** | 1.6k | 2024-06-02 | ~20.8M | ☆（很慢） | lowres | 中 |
| OpenCV `dnn_superres`（EDSR / ESPCN / FSRCNN / LapSRN） | **Apache-2.0** | 随 OpenCV | 随 OpenCV | 0.01M~43M | ★★★★★ | lowres | 低（保守，不幻觉） |

### 1.2 运动模糊 / 失焦模糊去模糊

| 项目 | 许可证 | Stars | 最后推送 | 规模 | CPU 可行性 | 适用退化 | 文字场景风险 |
|---|---|---|---|---|---|---|---|
| [`megvii-research/NAFNet`](https://github.com/megvii-research/NAFNet) | **MIT**（LICENSE 内含 BasicSR 的 Apache-2.0 段落） | 3.2k | 2024-07-03 | width32 ≈ 17.1M<br>width64 ≈ 67.9M | ★★★☆☆（width32, 小尺寸） | motion / defocus | 中 |
| [`swz30/Restormer`](https://github.com/swz30/Restormer) | **MIT** | 2.7k | 2025-10-23 | 26.1M | ★★☆☆☆ | motion / defocus / denoise | 中 |
| [`swz30/MPRNet`](https://github.com/swz30/MPRNet) | **NOASSERTION（Other）** | 1.4k | 2024-04-18 | 20.0M | ★★☆☆☆ | motion | 中 |

### 1.3 生成式复原（扩散模型）

| 项目 | 许可证 | Stars | 最后推送 | CPU 可行性 | 结论 |
|---|---|---|---|---|---|
| [`XPixelGroup/DiffBIR`](https://github.com/XPixelGroup/DiffBIR) | 代码 **Apache-2.0**；基座 SD 2.1 为 **OpenRAIL++-M** | 4.1k | 2025-07-29 | ☆（单图分钟级） | 仅作 GPU 演进分支；文字幻觉风险高 |
| [`Fanghua-Yu/SUPIR`](https://github.com/Fanghua-Yu/SUPIR) | **SUPIR Software License（仅非商用）** | 5.7k | 2025-05-12 | ☆ | **排除** |
| [`IceClear/StableSR`](https://github.com/IceClear/StableSR) | **NTU S-Lab License 1.0（非商用）** | — | 2023-05（停更） | ☆ | **排除** |

### 1.4 人脸专项（本场景不使用，列出以说明排除原因）

| 项目 | 许可证 | Stars | 最后推送 | 结论 |
|---|---|---|---|---|
| [`TencentARC/GFPGAN`](https://github.com/TencentARC/GFPGAN) | 代码 Apache-2.0，但 LICENSE 明确 "except for the third-party components"：**DFDNet 为 CC BY-NC-SA 4.0**、**NVIDIA stylegan2 含 3.3 Use Limitation（仅非商业）** | 37.7k | 2024-07-26 | 不可简化为"可商用"，需法务审查 |
| [`sczhou/CodeFormer`](https://github.com/sczhou/CodeFormer) | **NTU S-Lab License 1.0（非商用）** | 18.2k | 2025-11-18 | **排除** |
| `RestoreFormer` | Apache-2.0（据公开说明） | — | — | 未深入评估；本场景不涉及人脸 |

### 1.5 推理与封装工具

| 项目 | 许可证 | 说明 |
|---|---|---|
| `microsoft/onnxruntime` | MIT | CPU EP + 图优化 + INT8 动态量化，本项目主推理运行时 |
| OpenCV `dnn_superres` | Apache-2.0（随 OpenCV） | 零额外模型依赖的兜底路径 |
| Real-ESRGAN 官方 ncnn 实现（仓库内 `realesrgan-ncnn-vulkan`） | BSD-3-Clause | 以 GPU（Vulkan）为主；CPU 后端可跑但非最优 |
| `upscayl/upscayl` | **AGPL-3.0**（52.8k stars，2026-11-17 推送） | GUI 应用；AGPL 对 SaaS 分发有传染性，**不作为组件引入** |
| HuggingFace `jonathanst29/tinier-upscale-models` | BSD-3-Clause | Real-ESRGAN 官方权重的 **ONNX 导出**（fp32, opset 17, 动态 H/W）：x4v3 1.2M / 4.6MB，x4plus 16.7M / 64MB。可直接用，省去一次导出 |

### 1.6 辅助组件（非复原模型）

| 项目 | 许可证 | 用途 |
|---|---|---|
| `PaddleOCR` | Apache-2.0 | 离线档的文本区域检测（给出精确文本框，用于分治），不需要其识别分支 |

---

## 2. 关键项目点评

### 2.1 Real-ESRGAN —— 本场景的主力首选

- **为什么选它**：唯一同时满足"BSD-3 可商用 + 真实退化训练 + 官方/社区 ONNX 权重齐备 + 有 1.2M 轻量分支"的项目。`general-x4v3` 的 4.6MB ONNX 在 CPU 上分块推理是当前最优性价比组合。
- **为什么不能只用它**：`x4plus`（RRDBNet 16.7M）在文字边缘会产生"虫状/水彩"纹理，把 `il` 画成 `ii`。`general-x4v3` 温和得多，但仍会加锐边缘。
- **两个权重必须区别对待**：

| 权重 | 结构 | 参数量 | ONNX 体积 | CPU 定位 |
|---|---|---|---|---|
| `realesr-general-x4v3` | SRVGGNetCompact | 1.2M | 4.6 MB | **交互档主力** |
| `RealESRGAN_x4plus` | RRDBNet | 16.7M | 64 MB | **离线档**，文字区禁用 |
| `RealESRGAN_x4plus_anime_6B` | RRDBNet（6 blocks） | 较小 | — | 二次元专用，本场景不用 |

- **已知坑**：ncnn 实现会先把输入切 tile，可能产生轻微块间不一致；`x4v3` 需要配合 `dni`（denoise strength）参数调节去噪强度，纯放大场景应调低。

### 2.2 NAFNet —— 去模糊的正确选择

- 非线性激活自由（NAFBlock）设计，同等效果下算力显著低于 Restormer/MPRNet，是 CPU 侧唯一有实用可能的去模糊网络。
- **width32（17.1M）** 用于交互档的小尺寸去模糊；**width64（67.9M）** 仅离线档。
- 训练任务分 GoPro（运动模糊）、SIDD（噪声）、DPDD（散焦）三套权重，**必须按判定结果选权重**，用错权重会显著掉点。
- 仓库自带 `cog.yaml` + `predict.py`，是官方为 Replicate 准备的封装，说明其工程友好度较高。
- **已知坑（ONNX 导出）**：自定义 `LayerNorm2d` 在导出时报 `onnx::LayerNormalization ... not constant`，需替换为标准等价实现（或降到 opset 11 并保持静态 shape）。社区有可用的导出实践，但**动态 shape 需自行验证**。

### 2.3 Restormer / SwinIR / HAT —— 质量上限，CPU 不实用

- 三者的共同问题不是质量，而是 CPU 成本：Transformer 类结构在 CPU 上单张 1080p 级图像耗时通常在**数十秒到数分钟**量级（结构本身与自注意力的内存带宽开销决定，非实现问题）。
- 因此定位为：**离线档的"质量天花板"备选**，且只在"图像面积不大 + 用户明确接受长耗时"时启用。
- SwinIR 额外约束：输入需 pad 到 window size（8）的倍数，否则报 shape mismatch。

### 2.4 生成式路线的诚实结论

| 事实 | 影响 |
|---|---|
| SUPIR / StableSR / CodeFormer 明确非商用 | 直接排除，不需进一步评估 |
| DiffBIR 代码 Apache-2.0，但基座 SD 2.1 为 OpenRAIL++-M（带使用限制条款，非纯商用友好） | 需法务确认；且需一并履行基座的使用限制 |
| 扩散模型在文字区域会**幻觉出不存在的笔画**，且无法通过参数完全抑制 | 对文档/UI 场景是**功能性错误**，不是画质问题 |
| CPU 上单图 512² 通常需数分钟 | 交互档完全不可行，离线档的吞吐也不经济 |

**结论：本场景（CPU + 截图/文档/UI）不采用扩散模型作为主链路。** 仅在"未来获得 GPU 且处理对象为纯照片"时，作为可选增强分支接入 DiffBIR。

---

## 3. 许可证合规结论

| 结论 | 项目 |
|---|---|
| ✅ **可商用，直接使用** | Real-ESRGAN（BSD-3）、NAFNet（MIT）、Restormer（MIT）、SwinIR（Apache-2.0）、HAT（Apache-2.0）、BSRGAN（Apache-2.0）、OpenCV（Apache-2.0）、onnxruntime（MIT）、PaddleOCR（Apache-2.0） |
| ⚠️ **有条件使用** | DiffBIR（代码 Apache-2.0，但基座 OpenRAIL++-M 需履行使用限制）；GFPGAN（代码 Apache-2.0，但含 DFDNet CC BY-NC-SA 与 NVIDIA 非商用条款，**不得**将相关组件随产品分发） |
| ❌ **禁止商用** | SUPIR、StableSR、CodeFormer |
| ❌ **许可证存疑** | MPRNet（SPDX = NOASSERTION，LICENSE 非标准协议，商用前必须人工核原文；建议不进入主链路） |
| ❌ **不引入为组件** | Upscayl（AGPL-3.0，若以 SaaS 形式提供服务并分发修改版，需开源整个服务端） |

### 3.1 分发时的义务

- **BSD-3-Clause**：保留版权声明、许可文本、免责声明；不得用作者名义背书。
- **MIT**：保留版权声明与许可文本。
- **Apache-2.0**：保留 LICENSE、NOTICE（如有）、标注修改过的文件；含专利授权条款，对商用友好。
- 若最终以容器/二进制分发，需在产品内附 `THIRD_PARTY_LICENSES` 清单。

> 本表仅为工程侧合规初筛，不构成法律意见。正式商用前建议由法务复核每一条 LICENSE 原文。

---

## 4. 排除清单与理由

| 项目 | 排除理由 |
|---|---|
| SUPIR | 自定义许可仅授权非商业使用；权重为厂商专有 |
| StableSR | NTU S-Lab License 1.0 非商用；2023-05 后停更 |
| CodeFormer | NTU S-Lab License 1.0 非商用 |
| MPRNet | SPDX NOASSERTION，许可不明；且质量/速度均不如 NAFNet |
| GFPGAN | 许可证内含非商业第三方组件，分发风险高；本场景无人脸需求 |
| Upscayl | AGPL-3.0 传染性；且它是 GUI 应用，非可嵌入组件 |
| `4x-UltraSharp` 等社区权重 | 常见为 CC BY-NC-SA 4.0，**不可商用**（社区流传最广、也最容易踩的坑） |

---

## 5. 事实核验清单（含出处）

| # | 核验项 | 结论 | 出处 |
|---|---|---|---|
| 1 | Real-ESRGAN 许可证 | BSD-3-Clause | `api.github.com/repos/xinntao/Real-ESRGAN` → `license.spdx_id` |
| 2 | Real-ESRGAN 活跃度 | 最后推送 2024-08-06，未归档 | 同上 → `pushed_at` / `archived` |
| 3 | realesr-general-x4v3 规模 | SRVGGNetCompact，1.21M 参数，ONNX 4.6MB，fp32/opset17/动态 HW | HuggingFace `jonathanst29/tinier-upscale-models` 模型卡 |
| 4 | RealESRGAN_x4plus 规模 | RRDBNet，16.7M 参数，ONNX 64MB | 同上 |
| 5 | NAFNet 许可证 | LICENSE 首段为 MIT（megvii-model），文件后半含 BasicSR 的 Apache-2.0 | `raw.githubusercontent.com/megvii-research/NAFNet/main/LICENSE` |
| 6 | NAFNet 活跃度 | 最后推送 2024-07-03；仓库含 `cog.yaml`/`predict.py` | `api.github.com/repos/megvii-research/NAFNet` + 根目录列表 |
| 7 | NAFNet ONNX 导出可行性 | 可行，但 `LayerNorm2d` 需改造，社区有实践（opset 11） | 社区导出实践文章（CSDN/技术博客），**动态 shape 待自行验证** |
| 8 | Restormer 许可证与活跃度 | MIT；最后推送 2025-10-23 | `api.github.com/repos/swz30/Restormer` |
| 9 | SwinIR 许可证 | Apache-2.0；最后推送 2024-05-14 | `api.github.com/repos/JingyunLiang/SwinIR` |
| 10 | BSRGAN 许可证 | Apache-2.0；最后推送 2024-04-12 | `api.github.com/repos/cszn/BSRGAN` |
| 11 | HAT 许可证 | Apache-2.0；最后推送 2024-06-02 | `api.github.com/repos/XPixelGroup/HAT` |
| 12 | MPRNet 许可证 | NOASSERTION / Other | `api.github.com/repos/swz30/MPRNet` |
| 13 | DiffBIR 许可证 | Apache-2.0；最后推送 2025-07-29 | `api.github.com/repos/XPixelGroup/DiffBIR` |
| 14 | SUPIR 许可证 | Other（NOASSERTION）；公开资料为仅非商用 | `api.github.com/repos/Fanghua-Yu/SUPIR` + 公开许可说明 |
| 15 | CodeFormer 许可证 | Other（NOASSERTION）；公开资料为 NTU S-Lab 1.0 非商用 | `api.github.com/repos/sczhou/CodeFormer` |
| 16 | GFPGAN 许可细节 | Apache-2.0，但排除第三方组件；DFDNet = CC BY-NC-SA 4.0；NVIDIA stylegan2 含 3.3 Use Limitation | `raw.githubusercontent.com/TencentARC/GFPGAN/master/LICENSE` 原文 |
| 17 | Upscayl 许可证 | AGPL-3.0；52.8k stars | `api.github.com/repos/upscayl/upscayl` |
| 18 | OpenCV dnn_superres CPU 基准 | i7-9700K, 768×512 放大 4×：ESPCN 0.0116s / FSRCNN 0.0130s / LapSRN 0.283s / EDSR 3.268s；PSNR 分别 26.55 / 26.56 / 26.73 / 29.24 | OpenCV 官方 dnn_superres 示例 README 基准 |
| 19 | PyTorch vs NCNN（CPU） | 256×256 输入：PyTorch 1218ms ≈ NCNN 189ms（约 6× 差距） | 社区实测文章，**仅供参考，非本项目实测** |

> 标注为"社区实践/公开资料"的条目未通过一手 API 直接验证，已在对应章节注明。所有**性能数字**均为外部参考值或结构推算，**不是本项目实测结果**；POC 脚本可用于产出实测值替换。

---

## 6. 调研结论（供方案引用）

1. **交互档唯一可行组合**：`realesr-general-x4v3`（ONNX, 1.2M）+ 可选 NAFNet-width32；兜底 OpenCV FSRCNN/ESPCN（毫秒级）。
2. **离线档可行组合**：`RealESRGAN_x4plus`（16.7M）+ NAFNet-width64/64 + 可选 SwinIR，配合文本区域分治。
3. **扩散模型不进入本场景**，仅作为未来 GPU 环境的演进分支。
4. **AGPL 与 CC BY-NC 是最容易踩的两个坑**，分别对应 Upscayl 与社区超分权重。
5. **ONNX 化是 CPU 场景的关键前置工作**：Real-ESRGAN 有现成导出可直接用；NAFNet 需自行导出并验证动态 shape。

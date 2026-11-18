# 图片清晰方案

面向 **Python 智能体（Agent）** 的图像去模糊 / 超分能力选型：在 GitHub 开源项目调研基础上，给出两套可落地的 CPU 集成方案，并提供可运行的验证骨架与本地测试台。

> 关键约束：**纯 CPU、无 GPU、以截图/文档/UI 为主、需交互与离线两档时延**。

---

## 结论速览

| | 方案一（推荐先做） | 方案二（按需追加） |
|---|---|---|
| 一句话 | 小模型 + 强编排，同步秒级 | 大模型 + 文本分治，异步分钟级 |
| 核心模型 | Real-ESRGAN `general-x4v3`（1.2M）+ NAFNet-w32 | Real-ESRGAN `x4plus`（16.7M）+ NAFNet-w64 + SwinIR |
| 实测时延 | 400×300 ≈ 1.7~2.1s / 720p ≈ 14.5s / 1080p ≈ 33s | 25~80s，异步队列 |
| 许可证 | BSD-3 / MIT / Apache-2.0，可商用 | 同左 |

**三条红线**

1. 文字场景禁用生成式先验——GAN 会把 `rn` 重绘成 `m`，属内容错误而非画质问题。
2. 不使用扩散模型（SUPIR/StableSR/CodeFormer 非商用；DiffBIR 在文字上幻觉笔画；CPU 单图分钟级）。
3. 永不抛异常——模型缺失/超时/算子不支持一律降级到 Lanczos。

---

## 目录结构

```
.
├── docs/image-deblur/          # 技术调研与选型报告
│   ├── README.md               # 索引、TL;DR、决策速查表
│   ├── 01-scenarios.md         # 五类退化判据、内容判定、路由决策树
│   ├── 02-github-survey.md     # 18 个开源项目对比 + 许可证核验出处
│   ├── 03-plan-a-cpu-lightweight.md
│   ├── 04-plan-b-offline-heavy.md
│   └── 05-decision-and-risk.md # 对比矩阵、合规清单、风险与演进
└── poc/                        # 最小可运行骨架 + 本地测试台
    ├── deblur_tool.py          # 核心：判定 / 路由 / 分块推理 / 兜底
    ├── server.py               # FastAPI 服务
    ├── static/index.html       # 上传测试页（前后对比滑块）
    ├── models/                 # 权重目录（含 x4v3，4.6MB）
    └── requirements.txt
```

---

## 快速开始

```powershell
cd poc
python -m venv .venv
.venv\Scripts\python -m pip install -r requirements.txt
.venv\Scripts\python server.py
# 打开 http://127.0.0.1:8000 上传图片
```

命令行方式：

```powershell
.venv\Scripts\python deblur_tool.py --input path/to/blurry.png --scale 2 --json
```

### 交互档容量提示

实测 `realesr-general-x4v3` 在 CPU 上约 **16 ms / 千像素**，因此 3 秒预算内只装得下 **≤ 0.15 MP（约 400×300）**。
更大图片会触发预算裁决，自动跳过重模型并返回 `heavy_model_skipped_over_budget(est=...)` 告警。

---

## 许可证

代码部分为本项目原创。随附的 `poc/models/realesr-general-x4v3.onnx` 来自
[Real-ESRGAN](https://github.com/xinntao/Real-ESRGAN)（BSD-3-Clause），ONNX 转换版由
[HuggingFace](https://huggingface.co/jonathanst29/tinier-upscale-models) 提供，分发时请保留其版权声明。
详见 [`poc/models/README.md`](poc/models/README.md)。

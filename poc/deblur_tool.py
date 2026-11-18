"""图像清晰化工具（POC 骨架）。

对应文档：docs/image-deblur/03-plan-a-cpu-lightweight.md（方案一，交互档）。

设计原则
--------
1. 能不做就不做：图像已足够清晰时直接返回原图。
2. 能小不大：优先最轻的可用模型。
3. 能退不崩：模型缺失 / 算子不支持 / 超时一律降级，永不抛异常中断 Agent 会话。

对外只暴露两个入口：
    sharpen(req: SharpenRequest) -> SharpenResult   # 业务流程
    build_tool_schema() -> dict                     # LLM function-calling 定义

CLI：
    python deblur_tool.py --input in.png --scale 2 --out out/sharp.png
    python deblur_tool.py --input in.png --json     # 打印分析指标与耗时
"""

from __future__ import annotations

import argparse
import base64
import io
import json
import math
import os
import shutil
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Callable, Literal, Optional

import numpy as np
from PIL import Image, ImageOps

try:
    import cv2
except ImportError as exc:  # pragma: no cover - 环境缺失时给出明确指引
    raise SystemExit(
        "需要 opencv-contrib-python（dnn_superres 在 contrib 包中）：\n"
        "    pip install opencv-contrib-python"
    ) from exc

try:
    import onnxruntime as ort
except ImportError:  # onnxruntime 为可选依赖，缺失时自动使用 OpenCV / 插值兜底
    ort = None

# ---------------------------------------------------------------------------
# 常量：判定阈值（初始值，需在真实样本上标定，见 01-scenarios.md 第 3 节）
# ---------------------------------------------------------------------------

BLUR_SCORE_SHARP = 500.0      # 高于此值视为已足够清晰
BLUR_SCORE_BLURRY = 100.0     # 低于此值视为模糊
BLUR_SCORE_SEVERE = 30.0      # 低于此值视为严重模糊

ANISOTROPY_MOTION = 0.55      # 结构张量各向异性阈值（运动模糊）
HF_RATIO_LOW = 0.05           # 高频能量占比阈值（低分辨率 / 强失焦）
BLOCKINESS_JPEG = 1.25        # 8px 网格边界梯度比（JPEG 块效应）
EDGE_SPREAD_DEFOCUS = 4.0     # 边缘扩散指数（归一化，非像素；实测 sharp≈2.7 / 模糊≈6.7 / 重模糊≈27）
DETAIL_ENERGY_SHARP = 0.30    # 强边缘带细节能量占比（|detail|>32），作为"已足够锐"的佐证

TEXT_STROKE_MIN = 0.008       # 笔画占比下限：低于此视为"没有文字"（实测文字截图 4%~6%，噪声图约 100%）
TEXT_STROKE_MAX = 0.35        # 笔画占比上限：高于此视为纹理/噪声而非文字
TEXT_RATIO_UI = 0.15          # 超过此比例时，即使非 UI 也按纯文字处理（强保守）
FLAT_RATIO_UI = 0.50          # 平坦纯色区占比（UI 截图特征，实测截图 0.85+ / 照片 0.00x）

ANALYSIS_MAX_SIDE = 512       # 分析阶段降采样上限，控制 FFT 等 O(N log N) 开销

MODEL_DIR = Path(__file__).resolve().parent / "models"
DEFAULT_TILE = 384
DEFAULT_OVERLAP = 32
OVERSHOOT_DELTA = 12.0 / 255.0  # 过冲钳位阈值，抑制白边

Tier = Literal["interactive", "offline", "fallback", "none"]
Scale = Literal[1, 2, 4]


# ---------------------------------------------------------------------------
# 契约（两套方案共用的边界，见 03 文档第 5 节）
# ---------------------------------------------------------------------------


@dataclass
class SharpenRequest:
    image_uri: str
    tier: Tier = "interactive"
    scale: int = 2
    allow_generative_prior: bool = False
    max_latency_ms: int = 3000
    output_dir: Optional[str] = None


@dataclass
class SharpenResult:
    output_uri: Optional[str] = None
    tier_used: Tier = "fallback"
    degradation_type: str = "unknown"
    content_type: str = "unknown"
    model_chain: list[str] = field(default_factory=list)
    elapsed_ms: int = 0
    warnings: list[str] = field(default_factory=list)
    metrics: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class Analysis:
    degradation_type: str
    degradation_set: list[str]
    content_type: str
    confidence: float
    metrics: dict


# ---------------------------------------------------------------------------
# 图像读写
# ---------------------------------------------------------------------------


def load_image(uri: str) -> np.ndarray:
    """读取图像为 BGR uint8，自动处理 EXIF 旋转与透明通道（合成到白底）。"""
    if uri.startswith("data:"):
        _, b64 = uri.split(",", 1)
        pil = Image.open(io.BytesIO(base64.b64decode(b64)))
    else:
        pil = Image.open(uri)

    pil = ImageOps.exif_transpose(pil)
    if pil.mode in ("RGBA", "LA", "P"):
        pil = pil.convert("RGBA")
        canvas = Image.new("RGB", pil.size, (255, 255, 255))
        canvas.paste(pil, mask=pil.split()[-1])
        pil = canvas
    else:
        pil = pil.convert("RGB")

    return cv2.cvtColor(np.array(pil), cv2.COLOR_RGB2BGR)


def save_image(img_bgr: np.ndarray, path: Path) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(path), np.clip(img_bgr, 0, 255).astype(np.uint8))
    return str(path)


# ---------------------------------------------------------------------------
# 退化 / 内容分析（纯 OpenCV + NumPy，单图 < 20ms）
# ---------------------------------------------------------------------------


def _gray_f32(img_bgr: np.ndarray) -> np.ndarray:
    return cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY).astype(np.float32)


def _downscale_for_analysis(gray: np.ndarray) -> np.ndarray:
    h, w = gray.shape[:2]
    longest = max(h, w)
    if longest <= ANALYSIS_MAX_SIDE:
        return gray
    ratio = ANALYSIS_MAX_SIDE / float(longest)
    return cv2.resize(gray, (max(1, int(w * ratio)), max(1, int(h * ratio))), interpolation=cv2.INTER_AREA)


def blur_score(gray: np.ndarray) -> float:
    """拉普拉斯方差：越小越模糊（最经典的无参考模糊度指标）。

    注：输出类型必须与输入精度匹配——OpenCV 5.x 不支持 CV_32F 源 + CV_64F 目标。
    """
    # ksize=1 才是经典"拉普拉斯方差"的量纲；ksize=3 会把数值放大一个数量级，令阈值全部失效
    lap = cv2.Laplacian(gray.astype(np.float32), cv2.CV_32F, ksize=1)
    return float(lap.var())


def anisotropy(gray: np.ndarray) -> tuple[float, float]:
    """结构张量各向异性：接近 1 表示梯度高度集中于单一方向（运动模糊）。

    返回 (各向异性, 主方向弧度)。这是全局近似，非逐块估计。
    """
    gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
    jxx = float(np.mean(gx * gx))
    jyy = float(np.mean(gy * gy))
    jxy = float(np.mean(gx * gy))
    spread = math.sqrt(max(0.0, (jxx - jyy) ** 2 + 4.0 * jxy * jxy))
    lam1 = (jxx + jyy + spread) / 2.0
    lam2 = (jxx + jyy - spread) / 2.0
    denom = lam1 + lam2 + 1e-6
    angle = 0.5 * math.atan2(2.0 * jxy, jxx - jyy + 1e-6)
    return float((lam1 - lam2) / denom), float(angle)


def high_freq_ratio(gray: np.ndarray) -> float:
    """傅里叶幅值谱中半径 > 0.5·Nyquist 的能量占比：低 → 低分辨率或强失焦。"""
    small = _downscale_for_analysis(gray)
    spectrum = np.abs(np.fft.fftshift(np.fft.fft2(small)))
    h, w = spectrum.shape
    cy, cx = h // 2, w // 2
    yy, xx = np.ogrid[:h, :w]
    radius = np.sqrt((yy - cy) ** 2 + (xx - cx) ** 2) / (min(h, w) / 2.0)
    total = float(spectrum.sum()) + 1e-6
    return float(spectrum[radius > 0.5].sum() / total)


def blockiness(gray: np.ndarray) -> float:
    """8 像素网格边界梯度均值 / 非边界梯度均值：> 1.25 提示 JPEG 块效应。"""
    h, w = gray.shape[:2]
    if h < 16 or w < 16:
        return 1.0

    def _ratio(diff: np.ndarray) -> float:
        if diff.shape[0] < 8:
            return 1.0
        idx = np.arange(diff.shape[0])
        on_grid = (idx % 8) == 7          # 第 8、16… 条边界
        if not on_grid.any() or on_grid.all():
            return 1.0
        boundary = float(diff[on_grid].mean())
        interior = float(diff[~on_grid].mean()) + 1e-6
        return boundary / interior

    dx = np.abs(np.diff(gray, axis=1)).mean(axis=0)   # 沿 x 的差分 → 长度 w-1
    dy = np.abs(np.diff(gray, axis=0)).mean(axis=1)   # 沿 y 的差分 → 长度 h-1
    return float(max(_ratio(dx), _ratio(dy)))


def edge_spread(gray: np.ndarray) -> float:
    """边缘扩散指数（归一化，非像素）。

    利用高斯模糊阶跃边缘的性质：梯度峰值 ∝ A/σ，拉普拉斯峰值 ∝ A/σ²，
    故 梯度/拉普拉斯 随 σ 单调递增。取强边缘（top 1%）的中位数做稳健估计。

    实测参考（800×600 文字图）：清晰 ≈ 2.7 / 轻度模糊 ≈ 6.7 / 重模糊 ≈ 27。
    该指标**不等同**于 σ 的像素值，仅用于阈值判定；不同机器/内容需重新标定。
    """
    grad = cv2.magnitude(
        cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3),
        cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3),
    )
    lap = np.abs(cv2.Laplacian(gray, cv2.CV_32F, ksize=3))

    thresh = float(np.percentile(grad, 99))
    if thresh <= 1e-3:
        return 1.0
    strong = grad >= thresh
    g_med = float(np.median(grad[strong]))
    l_med = float(np.median(lap[strong])) + 1e-6
    return float(np.clip(g_med / l_med, 0.2, 60.0))


def detail_energy_ratio(gray: np.ndarray, amount_thresh: float = 32.0) -> float:
    """强边缘带的细节能量占比：可佐证"图像本身已很锐"。

    实测：清晰文字图 ≈ 0.56，轻度模糊 ≈ 0.10，重模糊 ≈ 0.0。
    注意：不能用低阈值（8/16）——纹理与文字笔画本身就会产生大量高值，导致恒高。
    """
    detail = gray - cv2.GaussianBlur(gray, (0, 0), 1.5)
    edges = cv2.dilate((cv2.Canny(gray.astype(np.uint8), 50, 150) > 0).astype(np.uint8), np.ones((5, 5), np.uint8))
    band = edges > 0
    if not band.any():
        return 0.0
    return float(np.mean(np.abs(detail[band]) > amount_thresh))


def text_like_ratio(gray: np.ndarray) -> float:
    """文本类笔画占比（无 OCR 代理指标）。

    思路：黑帽/顶帽变换提取"比周围更暗/更亮的细笔画" → 阈值 → 连通域按
    "小面积 + 合理长宽比"过滤 → 统计覆盖占比。

    早期用 Canny + 闭运算的实现实测只有 0.07%~0.5%，几乎全被面积过滤掉；
    改用形态学响应后，文字截图可稳定落在 5%~20% 区间。
    已知误判：细碎自然纹理（树叶/织物）会被误判，故 analyze() 中叠加平坦区占比约束。
    """
    g8 = np.clip(gray, 0, 255).astype(np.uint8)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (11, 11))
    dark_strokes = cv2.morphologyEx(g8, cv2.MORPH_BLACKHAT, kernel)   # 亮底暗笔画
    light_strokes = cv2.morphologyEx(g8, cv2.MORPH_TOPHAT, kernel)    # 暗底亮笔画
    response = np.maximum(dark_strokes, light_strokes)

    thr = max(18.0, float(np.percentile(response, 99.0)) * 0.35)
    mask = (response > thr)

    # 不做连通域面积过滤：笔画会被阈值连成整行大块（实测面积 1600~6000），
    # 按"小面积"过滤会把文字全部丢掉。改为用整体覆盖率，并对"全图都是高频"
    # 的情况（纯噪声/强纹理）返回 0，交由 flat_ratio 判为 photo。
    ratio = float(mask.mean())
    if ratio > TEXT_STROKE_MAX:
        return 0.0
    return ratio


def flat_ratio(gray: np.ndarray) -> float:
    """局部方差极低（近纯色）像素占比：UI 截图的典型特征。"""
    mean = cv2.blur(gray, (3, 3))
    sq = cv2.blur(gray * gray, (3, 3))
    std = np.sqrt(np.maximum(sq - mean * mean, 0.0))
    return float(np.mean(std < 2.0))


def analyze(img_bgr: np.ndarray) -> Analysis:
    """综合退化与内容判定，输出 01-scenarios.md 约定的三个字段。"""
    gray = _gray_f32(img_bgr)
    small = _downscale_for_analysis(gray)

    b_score = blur_score(small)
    aniso, angle = anisotropy(small)
    hf = high_freq_ratio(gray)
    blk = blockiness(small)
    spread = edge_spread(small)
    detail = detail_energy_ratio(small)
    text_r = text_like_ratio(gray)
    flat_r = flat_ratio(gray)

    hits: list[str] = []
    if blk > BLOCKINESS_JPEG:
        hits.append("jpeg")
    # 运动模糊要求"中度模糊 + 强方向性"：严重模糊时梯度场塌缩，各向异性会虚高
    if aniso > ANISOTROPY_MOTION and BLUR_SCORE_SEVERE <= b_score < 200:
        hits.append("motion")
    # 失焦要求"边缘扩散 + 各向同性"（用各向异性排除运动模糊，不再依赖 halo 判据）
    if spread > EDGE_SPREAD_DEFOCUS and aniso < 0.35:
        hits.append("defocus")
    if hf < HF_RATIO_LOW or spread > EDGE_SPREAD_DEFOCUS:
        hits.append("lowres")

    if b_score >= BLUR_SCORE_SHARP and detail > DETAIL_ENERGY_SHARP and not hits:
        degradation = "none"
    elif len(hits) >= 2:
        degradation = "mixed"
    elif hits:
        degradation = hits[0]
    elif b_score < BLUR_SCORE_BLURRY:
        degradation = "lowres"
    else:
        degradation = "none"

    # 内容类型：笔画占比 + 平坦区占比联合判定。
    # 单一指标都会误判——照片也有笔画、空白图也有平坦区，两者叠加才可靠。
    has_strokes = TEXT_STROKE_MIN < text_r <= TEXT_STROKE_MAX
    if has_strokes and flat_r > FLAT_RATIO_UI:
        content = "text_ui"
    elif has_strokes and text_r > TEXT_RATIO_UI:
        content = "text_ui"
    elif has_strokes:
        content = "mixed"
    elif flat_r > 0.85:
        content = "text_ui"     # 大面积纯色且无笔画：UI 空白页/纯色背景
    else:
        content = "photo"

    severity = {
        "none": 1.0,
        "lowres": float(np.clip(1.0 - hf / HF_RATIO_LOW, 0, 1)),
        "jpeg": float(np.clip((blk - 1.0) / 0.6, 0, 1)),
        "motion": float(np.clip((aniso - 0.4) / 0.5, 0, 1)),
        "defocus": float(np.clip((spread - 2.5) / 10.0, 0, 1)),
        "mixed": 0.7,
    }.get(degradation, 0.3)

    metrics = {
        "blur_score": round(b_score, 2),
        "anisotropy": round(aniso, 3),
        "motion_angle_deg": round(math.degrees(angle), 1),
        "hf_ratio": round(hf, 4),
        "blockiness": round(blk, 3),
        "edge_spread": round(spread, 2),
        "detail_energy": round(detail, 4),
        "text_like_ratio": round(text_r, 4),
        "flat_ratio": round(flat_r, 4),
    }
    return Analysis(
        degradation_type=degradation,
        degradation_set=hits,
        content_type=content,
        confidence=round(float(severity), 3),
        metrics=metrics,
    )


# ---------------------------------------------------------------------------
# 推理后端
# ---------------------------------------------------------------------------

_SESSIONS: dict[str, "ort.InferenceSession"] = {}
_DNN_SR: dict[str, object] = {}
_TOOLS_ROOT = Path(__file__).resolve().parent
OUTPUT_ROOT = _TOOLS_ROOT / "out"


def _session_options() -> "ort.SessionOptions":
    opts = ort.SessionOptions()
    opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    opts.intra_op_num_threads = max(1, (os.cpu_count() or 2) - 1)  # 留一核给 Agent 主循环
    opts.inter_op_num_threads = 1                                   # 单请求串行，多线程无收益
    opts.log_severity_level = 3
    return opts


def get_onnx_session(path: Path) -> Optional["ort.InferenceSession"]:
    """进程级单例：模型加载 0.3~1s，不可每次请求重建。"""
    if ort is None or not path.exists():
        return None
    key = str(path)
    if key not in _SESSIONS:
        try:
            _SESSIONS[key] = ort.InferenceSession(str(path), sess_options=_session_options(), providers=["CPUExecutionProvider"])
        except Exception:  # 算子不支持 / 权重损坏 → 交由上层兜底
            return None
    return _SESSIONS[key]


def _onnx_runner(session: "ort.InferenceSession", scale: int) -> Callable[[np.ndarray], np.ndarray]:
    input_name = session.get_inputs()[0].name

    def run(patch_bgr: np.ndarray) -> np.ndarray:
        x = patch_bgr[:, :, ::-1].transpose(2, 0, 1)[None].astype(np.float32)  # BGR→RGB, HWC→NCHW
        y = session.run(None, {input_name: np.ascontiguousarray(x)})[0]
        out = y[0].transpose(1, 2, 0)[:, :, ::-1]
        return np.clip(out, 0.0, 1.0).astype(np.float32)

    return run


def _dnn_superres_runner(model_path: Path, algo: str, scale: int) -> Optional[Callable[[np.ndarray], np.ndarray]]:
    """OpenCV dnn_superres 后端（EDSR / ESPCN / FSRCNN / LapSRN），纯 CPU 友好。"""
    if not model_path.exists() or not hasattr(cv2, "dnn_superres"):
        return None
    key = f"{model_path}|{scale}"
    impl = _DNN_SR.get(key)
    if impl is None:
        try:
            impl = cv2.dnn_superres.DnnSuperResImpl_create()
            impl.readModel(str(model_path))
            impl.setModel(algo, scale)
            _DNN_SR[key] = impl
        except Exception:
            return None

    def run(patch_bgr: np.ndarray) -> np.ndarray:
        u8 = np.clip(patch_bgr * 255.0, 0, 255).astype(np.uint8)
        out = impl.upsample(u8)
        return out.astype(np.float32) / 255.0

    return run


def _lanczos_runner(scale: int) -> Callable[[np.ndarray], np.ndarray]:
    def run(patch_bgr: np.ndarray) -> np.ndarray:
        h, w = patch_bgr.shape[:2]
        return cv2.resize(patch_bgr, (w * scale, h * scale), interpolation=cv2.INTER_LANCZOS4)

    return run


@dataclass
class Backend:
    name: str
    scale: int
    runner: Callable[[np.ndarray], np.ndarray]
    generative: bool = False   # 是否带 GAN 生成式先验（文字场景应禁用）


def resolve_backends(analysis: Analysis, allow_generative: bool, allow_heavy: bool = True) -> list[Backend]:
    """按优先级给出后端候选链，最后一个永远是 Lanczos 兜底。

    allow_heavy=False 时剔除 GAN 重模型（实测 720p 需 ~14.5s，远超交互预算）。
    宁可输出"用快模型放大的结果"，也不让用户在 3s 预算下等 15s。
    """
    scale = 4  # 主力模型固定 4×，最终再降采样到请求倍数，保证可复用同一会话

    onnx_light = MODEL_DIR / "realesr-general-x4v3.onnx"

    # 非 GAN 优先：EDSR / FSRCNN（保真模式首选）
    non_gan: list[Backend] = []
    for algo, fname in (("edsr", "EDSR_x4.pb"), ("fsrcnn", "FSRCNN_x4.pb")):
        r = _dnn_superres_runner(MODEL_DIR / fname, algo, scale)
        if r is not None:
            non_gan.append(Backend(name=f"{algo}_x4_pb", scale=scale, runner=r, generative=False))

    gan: list[Backend] = []
    sess = get_onnx_session(onnx_light) if allow_heavy else None
    if sess is not None:
        gan.append(Backend(name="realesr-general-x4v3", scale=scale, runner=_onnx_runner(sess, scale), generative=True))

    if analysis.content_type == "photo" or allow_generative:
        chain = gan + non_gan
    else:
        # 文本 / UI：非 GAN 优先，GAN 仅在无其他模型时兜底
        chain = non_gan + gan

    chain.append(Backend(name="lanczos", scale=scale, runner=_lanczos_runner(scale), generative=False))
    return chain


# ---------------------------------------------------------------------------
# 分块推理 + 接缝羽化融合
# ---------------------------------------------------------------------------


def _ramp1d(n: int, ramp: int) -> np.ndarray:
    """两端渐入到 1 的一维权重；最小值为 1/ramp，归一化后不影响图像边缘。"""
    ramp = int(min(ramp, max(1, n // 2)))
    w = np.ones(n, dtype=np.float32)
    if ramp > 1:
        t = (np.arange(ramp, dtype=np.float32) + 1.0) / ramp
        w[:ramp] = t
        w[-ramp:] = t[::-1]
    return w


def _feather(h: int, w: int, ramp: int) -> np.ndarray:
    return np.outer(_ramp1d(h, ramp), _ramp1d(w, ramp)).astype(np.float32)


def tiled_infer(
    runner: Callable[[np.ndarray], np.ndarray],
    img_bgr: np.ndarray,
    scale: int,
    tile: int = DEFAULT_TILE,
    overlap: int = DEFAULT_OVERLAP,
) -> np.ndarray:
    """分块推理并在**放大后空间**做羽化融合，避免网格状接缝。

    复杂度：时间 O(N·k)（N 像素数，k 模型每像素算力），峰值内存 O(tile²) 而非 O(N)。
    """
    h, w = img_bgr.shape[:2]
    src = img_bgr.astype(np.float32) / 255.0

    if max(h, w) <= tile:
        return runner(src)

    step = max(1, tile - overlap)
    oh, ow = h * scale, w * scale
    acc = np.zeros((oh, ow, 3), dtype=np.float32)
    weight = np.zeros((oh, ow, 1), dtype=np.float32)
    ramp = max(overlap * scale, 1)

    for y0 in range(0, h, step):
        y1 = min(y0 + tile, h)
        for x0 in range(0, w, step):
            x1 = min(x0 + tile, w)
            patch = src[y0:y1, x0:x1]
            out = runner(patch)

            th, tw = (y1 - y0) * scale, (x1 - x0) * scale
            if out.shape[0] != th or out.shape[1] != tw:
                out = cv2.resize(out, (tw, th), interpolation=cv2.INTER_CUBIC)

            fw = _feather(th, tw, ramp)[:, :, None]
            dy, dx = y0 * scale, x0 * scale
            acc[dy:dy + th, dx:dx + tw] += out * fw
            weight[dy:dy + th, dx:dx + tw] += fw

    return acc / np.maximum(weight, 1e-6)


# ---------------------------------------------------------------------------
# 后处理：色彩对齐 + 文字保护锐化 + 过冲钳位
# ---------------------------------------------------------------------------


def match_color_stats(out_bgr: np.ndarray, ref_bgr: np.ndarray) -> np.ndarray:
    """分通道均值/方差匹配到原图（Reinhard 统计迁移），消除 GAN 色偏。"""
    result = out_bgr.copy()
    for c in range(3):
        o = result[:, :, c]
        r = ref_bgr[:, :, c]
        o_std = float(o.std()) + 1e-6
        r_std = float(r.std())
        o_mean = float(o.mean())
        r_mean = float(r.mean())
        result[:, :, c] = (o - o_mean) * (r_std / o_std) + r_mean
    return np.clip(result, 0.0, 1.0)


def protect_text_regions(
    out_bgr: np.ndarray,
    ref_bgr: np.ndarray,
    text_ratio: float,
    content_type: str,
) -> np.ndarray:
    """文字保护后处理：锐度随文本占比反向衰减 + 过冲钳位。

    - text_ui 时 amount = 0（不做额外锐化），仅做色彩对齐；
    - 其余场景按 (1 - text_ratio) 衰减，上限 0.35，避免"廉价过锐"观感。
    """
    result = match_color_stats(out_bgr, ref_bgr)
    if content_type == "text_ui" or text_ratio > TEXT_RATIO_UI:
        amount = 0.0
    else:
        amount = float(np.clip(0.35 * (1.0 - text_ratio / TEXT_RATIO_UI * 0.5), 0.0, 0.35))

    if amount <= 1e-3:
        return result

    blurred = cv2.GaussianBlur(result, (0, 0), 1.0)
    sharpened = result + amount * (result - blurred)

    # 过冲钳位：以原图（放大后）为参照，限制 ±δ，抑制白边/发光
    ref_up = cv2.resize(ref_bgr, (result.shape[1], result.shape[0]), interpolation=cv2.INTER_CUBIC)
    sharpened = np.clip(sharpened, ref_up - OVERSHOOT_DELTA, ref_up + OVERSHOOT_DELTA)
    return np.clip(sharpened, 0.0, 1.0)


# ---------------------------------------------------------------------------
# 主流程
# ---------------------------------------------------------------------------


def _estimate_cost_ms(h: int, w: int, tier: str) -> int:
    """耗时预估（毫秒），用于预算裁决。

    系数来自实测（12 核 CPU / x4v3 / ONNX Runtime 1.32）：
        400×300（0.12 MP）→ 2078 ms
        1280×720（0.92 MP）→ 14480 ms
    拟合得 ≈ 16 ms / 千像素，即 16000 ms / MP。
    换机器或换模型必须重测——早期按 1300 ms/MP 估算，低估了约 12 倍，
    导致 3 秒预算下仍然放过一个要跑 15 秒的模型。
    """
    kpx = (h * w) / 1_000.0
    if tier == "offline":
        return int(52.0 * kpx + 3_000)     # 重模型约为轻模型的 3.2 倍
    return int(16.0 * kpx + 300)


def sharpen(req: SharpenRequest) -> SharpenResult:
    """对外唯一业务流程入口。**任何异常都不抛出**，最差返回 Lanczos 兜底结果。"""
    started = time.perf_counter()
    result = SharpenResult()

    try:
        img = load_image(req.image_uri)
    except Exception as exc:
        result.warnings.append(f"decode_failed: {exc}")
        result.tier_used = "fallback"
        result.elapsed_ms = int((time.perf_counter() - started) * 1000)
        return result

    h, w = img.shape[:2]
    try:
        analysis = analyze(img)
    except Exception as exc:  # 分析失败按最保守策略继续
        analysis = Analysis("lowres", ["lowres"], "text_ui", 0.3, {})
        result.warnings.append(f"analysis_failed: {exc}")

    result.degradation_type = analysis.degradation_type
    result.content_type = analysis.content_type
    result.metrics = analysis.metrics

    # Step 0 —— 已经足够清晰：直接返回原图
    if analysis.degradation_type == "none":
        result.tier_used = "none"
        result.model_chain = ["passthrough"]
        result.warnings.append("already_sharp")
        result.output_uri = _persist(img, req, "passthrough")
        result.elapsed_ms = int((time.perf_counter() - started) * 1000)
        return result

    # 大图标记（强制分块，禁用全局锐化）
    if max(h, w) > 4000:
        result.warnings.append("large_image_tiled")

    # Step 1 —— 预算裁决
    if analysis.confidence < 0.35:
        result.warnings.append("low_confidence_degradation")

    est = _estimate_cost_ms(h, w, req.tier)
    # 重模型超出预算 2 倍时直接弃用：宁可用快模型出图，也不让用户白等 15 秒。
    # 不做"先降采样再放大"——那样输出分辨率形同虚设，反而更差。
    allow_heavy = est <= req.max_latency_ms * 2
    if not allow_heavy:
        result.warnings.append(f"heavy_model_skipped_over_budget(est={est}ms)")

    # Step 2 —— 选择后端链并执行
    allow_gen = bool(req.allow_generative_prior) and analysis.content_type == "photo"
    if req.allow_generative_prior and not allow_gen:
        result.warnings.append("generative_prior_blocked_by_content_type")

    chain = resolve_backends(analysis, allow_gen, allow_heavy=allow_heavy)
    used: list[str] = []
    out_bgr: Optional[np.ndarray] = None
    ref_f32 = img.astype(np.float32) / 255.0   # 后处理统一在 0~1 浮点空间进行

    for backend in chain:
        try:
            up = tiled_infer(backend.runner, img, backend.scale, DEFAULT_TILE, DEFAULT_OVERLAP)
            out_bgr = protect_text_regions(up, ref_f32, analysis.metrics.get("text_like_ratio", 0.0), analysis.content_type)
            used.append(backend.name)
            if backend.name == "lanczos":
                result.warnings.append("fell_back_to_lanczos")
            break
        except Exception as exc:
            result.warnings.append(f"backend_failed:{backend.name}:{type(exc).__name__}")
            continue

    if out_bgr is None:  # 理论上不可达（lanczos 一定会成功）
        out_bgr = cv2.resize(img.astype(np.float32) / 255.0, (w * 4, h * 4), interpolation=cv2.INTER_LANCZOS4)
        used = ["lanczos"]

    # Step 3 —— 缩放到请求倍数
    target_w, target_h = w * int(req.scale), h * int(req.scale)
    if out_bgr.shape[1] != target_w or out_bgr.shape[0] != target_h:
        out_bgr = cv2.resize(out_bgr, (target_w, target_h), interpolation=cv2.INTER_AREA if out_bgr.shape[1] > target_w else cv2.INTER_CUBIC)

    out_u8 = np.clip(out_bgr * 255.0, 0, 255).astype(np.uint8)
    result.output_uri = _persist(out_u8, req, used[-1] if used else "unknown")
    result.model_chain = used
    result.tier_used = "fallback" if (used and used[-1] == "lanczos") else "interactive"
    result.elapsed_ms = int((time.perf_counter() - started) * 1000)

    if result.elapsed_ms > req.max_latency_ms:
        result.warnings.append("exceeded_budget")
    return result


def _persist(img_bgr: np.ndarray, req: SharpenRequest, tag: str) -> str:
    """落盘输出；输出目录可由 req.output_dir 指定。"""
    out_dir = Path(req.output_dir) if req.output_dir else OUTPUT_ROOT
    stem = Path(req.image_uri).stem if not req.image_uri.startswith("data:") else "inline"
    safe_tag = "".join(ch if ch.isalnum() or ch in "-_" else "_" for ch in tag)
    return save_image(img_bgr, out_dir / f"{stem}_{safe_tag}_x{req.scale}.png")


# ---------------------------------------------------------------------------
# LLM 工具定义
# ---------------------------------------------------------------------------


def build_tool_schema() -> dict:
    return {
        "name": "sharpen_image",
        "description": (
            "把模糊的图片（低分辨率、压缩伪影、轻微运动/失焦模糊）变清晰。"
            "当用户提供看不清文字的截图、模糊图片，或要求'放大/变清晰/增强'时调用。"
            "只做清晰化，不改变图片语义；不适用于去水印、去划痕、换背景、人脸重绘。"
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "image_uri": {"type": "string", "description": "图片本地路径或 data URI"},
                "scale": {"type": "integer", "enum": [1, 2, 4], "default": 2},
                "tier": {"type": "string", "enum": ["interactive", "offline"], "default": "interactive"},
                "allow_generative_prior": {
                    "type": "boolean",
                    "default": False,
                    "description": "仅纯照片可置 true；文字/UI 截图必须为 false",
                },
            },
            "required": ["image_uri"],
        },
    }


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(description="图像清晰化 POC（方案一：CPU 轻量级联）")
    parser.add_argument("--input", required=True, help="输入图片路径")
    parser.add_argument("--out", default=None, help="输出文件路径（默认 poc/out/）")
    parser.add_argument("--scale", type=int, default=2, choices=[1, 2, 4])
    parser.add_argument("--tier", default="interactive", choices=["interactive", "offline"])
    parser.add_argument("--allow-generative-prior", action="store_true")
    parser.add_argument("--max-latency-ms", type=int, default=3000)
    parser.add_argument("--json", action="store_true", help="打印完整结果 JSON")
    args = parser.parse_args()

    output_dir = str(Path(args.out).parent) if args.out else None
    req = SharpenRequest(
        image_uri=args.input,
        tier=args.tier,
        scale=args.scale,
        allow_generative_prior=args.allow_generative_prior,
        max_latency_ms=args.max_latency_ms,
        output_dir=output_dir,
    )
    res = sharpen(req)

    if args.out and res.output_uri:
        target = Path(args.out)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.unlink(missing_ok=True)   # Windows 上 shutil.move 覆盖已存在文件会失败
        shutil.move(res.output_uri, target)
        res.output_uri = str(target)

    if args.json:
        print(json.dumps(res.to_dict(), ensure_ascii=False, indent=2))
    else:
        print(f"output      : {res.output_uri}")
        print(f"degradation : {res.degradation_type}  content: {res.content_type}")
        print(f"chain       : {' -> '.join(res.model_chain)}")
        print(f"elapsed     : {res.elapsed_ms} ms")
        if res.warnings:
            print(f"warnings    : {', '.join(res.warnings)}")


if __name__ == "__main__":
    main()

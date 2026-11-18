"""图像清晰化 POC 的本地 Web 服务。

启动：
    python server.py
    # 或  python -m uvicorn server:app --host 127.0.0.1 --port 8000

打开 http://127.0.0.1:8000 即可拖拽上传测试。
本文件只做「HTTP 适配层」，全部算法逻辑复用 deblur_tool.sharpen()。
"""

from __future__ import annotations

import uuid
from pathlib import Path

import cv2
import numpy as np
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

import deblur_tool as dt

BASE_DIR = Path(__file__).resolve().parent
STATIC_DIR = BASE_DIR / "static"
UPLOAD_DIR = BASE_DIR / "uploads"
OUTPUT_DIR = BASE_DIR / "out"

for _d in (UPLOAD_DIR, OUTPUT_DIR):
    _d.mkdir(parents=True, exist_ok=True)

MAX_BYTES = 25 * 1024 * 1024     # 单图上限，防止误传超大文件把 CPU 打满
ALLOWED_SCALES = {1, 2, 4}

app = FastAPI(title="Image Deblur POC", docs_url="/api/docs")

app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")
# 输入图与输出图同目录，方便前端做前后对比
app.mount("/out", StaticFiles(directory=str(OUTPUT_DIR)), name="out")


@app.get("/")
def index() -> FileResponse:
    return FileResponse(str(STATIC_DIR / "index.html"))


@app.get("/api/health")
def health() -> dict:
    """前端启动时探测环境，用于显示"当前是否带模型"的横幅。"""
    onnx_model = dt.MODEL_DIR / "realesr-general-x4v3.onnx"
    dnn_models = [p.name for p in sorted(dt.MODEL_DIR.glob("*.pb"))]
    return {
        "opencv": cv2.__version__,
        "onnxruntime": getattr(dt.ort, "__version__", None) if dt.ort is not None else None,
        "dnn_superres": hasattr(cv2, "dnn_superres"),
        "onnx_model": onnx_model.exists(),
        "dnn_models": dnn_models,
        "cpu_count": __import__("os").cpu_count(),
    }


@app.post("/api/sharpen")
async def api_sharpen(
    file: UploadFile = File(...),
    scale: int = Form(2),
    tier: str = Form("interactive"),
    allow_generative_prior: bool = Form(False),
    max_latency_ms: int = Form(3000),
) -> dict:
    if scale not in ALLOWED_SCALES:
        raise HTTPException(status_code=400, detail=f"scale 只能是 {sorted(ALLOWED_SCALES)}")
    if tier not in ("interactive", "offline"):
        raise HTTPException(status_code=400, detail="tier 只能是 interactive / offline")

    raw = await file.read()
    if not raw:
        raise HTTPException(status_code=400, detail="空文件")
    if len(raw) > MAX_BYTES:
        raise HTTPException(status_code=413, detail="图片超过 25MB，请先缩小再上传")

    img = cv2.imdecode(np.frombuffer(raw, np.uint8), cv2.IMREAD_COLOR)
    if img is None:
        raise HTTPException(status_code=400, detail="无法解码该图片（格式不支持或文件损坏）")

    uid = uuid.uuid4().hex[:12]
    # 统一转 PNG 落盘：避免原图是 CMYK / 带透明 / 动图时下游读取失败
    src_path = OUTPUT_DIR / f"{uid}_src.png"
    cv2.imwrite(str(src_path), img)

    result = dt.sharpen(
        dt.SharpenRequest(
            image_uri=str(src_path),
            scale=scale,
            tier=tier,            # type: ignore[arg-type]
            allow_generative_prior=allow_generative_prior,
            max_latency_ms=max_latency_ms,
            output_dir=str(OUTPUT_DIR),
        )
    )

    payload = result.to_dict()
    payload["source_url"] = f"/out/{src_path.name}"
    payload["output_url"] = f"/out/{Path(result.output_uri).name}" if result.output_uri else None
    payload["source_size"] = [img.shape[1], img.shape[0]]
    payload["request"] = {
        "scale": scale,
        "tier": tier,
        "allow_generative_prior": allow_generative_prior,
        "max_latency_ms": max_latency_ms,
    }
    return payload


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="127.0.0.1", port=8000, log_level="info")

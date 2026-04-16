import os
import base64
import logging
import tempfile
import warnings
from io import BytesIO
from typing import Any

from PIL import Image
import runpod

# Quiet Paddle / OCR logs
os.environ["FLAGS_enable_pir_api"] = "1"
os.environ["FLAGS_enable_pir_in_executor"] = "1"
os.environ["FLAGS_use_mkldnn"] = "0"
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
os.environ["PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK"] = "1"
os.environ["GLOG_minloglevel"] = "3"

logging.getLogger("ppocr").setLevel(logging.ERROR)
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=Warning)

# Import torch before paddle to avoid DLL conflicts in some environments
try:
    import torch  # noqa: F401
except ImportError:
    pass

import paddle
from paddleocr import PaddleOCRVL

if paddle.device.is_compiled_with_cuda():
    paddle.set_device("gpu")
    print("✅ Using GPU")
else:
    paddle.set_device("cpu")
    print("⚠️ Using CPU")

print("🚀 Initializing PaddleOCR-VL 1.5 Worker...")
MODEL = PaddleOCRVL(
    pipeline_version="v1.5",
    use_doc_orientation_classify=False,
    use_doc_unwarping=False,
)


def _decode_image(image_b64: str) -> Image.Image:
    if image_b64.startswith("data:") and "," in image_b64:
        image_b64 = image_b64.split(",", 1)[1]
    image_bytes = base64.b64decode(image_b64)
    return Image.open(BytesIO(image_bytes)).convert("RGB")


def _block_value(block: Any, *keys: str, default: Any = None) -> Any:
    if isinstance(block, dict):
        for key in keys:
            if key in block:
                return block[key]
    for key in keys:
        value = getattr(block, key, None)
        if value is not None:
            return value
    return default


def _coerce_bbox(raw: Any) -> list[float] | None:
    if isinstance(raw, dict):
        if {"x1", "y1", "x2", "y2"}.issubset(raw):
            return [
                float(raw["x1"]),
                float(raw["y1"]),
                float(raw["x2"]),
                float(raw["y2"]),
            ]
        if {"x", "y", "w", "h"}.issubset(raw):
            x = float(raw["x"])
            y = float(raw["y"])
            w = float(raw["w"])
            h = float(raw["h"])
            return [x, y, x + w, y + h]

    if isinstance(raw, list):
        if len(raw) == 4 and all(isinstance(v, (int, float)) for v in raw):
            return [float(v) for v in raw]

        if len(raw) == 4 and all(isinstance(v, list) and len(v) == 2 for v in raw):
            xs = [float(v[0]) for v in raw]
            ys = [float(v[1]) for v in raw]
            return [min(xs), min(ys), max(xs), max(ys)]

        if len(raw) == 8 and all(isinstance(v, (int, float)) for v in raw):
            xs = [float(raw[i]) for i in range(0, 8, 2)]
            ys = [float(raw[i]) for i in range(1, 8, 2)]
            return [min(xs), min(ys), max(xs), max(ys)]

    return None


def _extract_parsing_list(page_result: Any) -> list[Any]:
    if isinstance(page_result, dict):
        value = page_result.get("parsing_res_list")
        return value if isinstance(value, list) else []
    try:
        value = page_result["parsing_res_list"]
        return value if isinstance(value, list) else []
    except Exception:
        return []


def _run_layout_ocr(image: Image.Image, job_input: dict[str, Any]) -> dict[str, Any]:
    layout_threshold = float(job_input.get("layoutThreshold", 0.1))
    use_doc_orientation_classify = bool(
        job_input.get("useDocOrientationClassify", False)
    )
    use_doc_unwarping = bool(job_input.get("useDocUnwarping", False))

    with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as temp_file:
        temp_path = temp_file.name
        image.save(temp_file, format="PNG")

    try:
        preds = MODEL.predict(
            temp_path,
            use_layout_detection=True,
            layout_threshold=layout_threshold,
            use_doc_orientation_classify=use_doc_orientation_classify,
            use_doc_unwarping=use_doc_unwarping,
        )

        detections: list[dict[str, Any]] = []

        for page_result in preds:
            for block in _extract_parsing_list(page_result):
                text = _block_value(block, "content", "text", "transcription", default="")
                bbox = _coerce_bbox(
                    _block_value(
                        block,
                        "bbox",
                        "coordinate",
                        "coordinates",
                        "box",
                        "polygon",
                        default=None,
                    )
                )
                label = _block_value(block, "label", "type", default="text")
                score = _block_value(block, "score", "confidence", default=None)

                if not isinstance(text, str) or not text.strip():
                    continue
                if bbox is None:
                    continue

                item = {
                    "text": text.strip(),
                    "bbox": bbox,
                    "label": str(label) if label is not None else "text",
                }
                if isinstance(score, (int, float)):
                    item["score"] = float(score)

                detections.append(item)

        return {
            "status": "success",
            "detections": detections,
            "total_found": len(detections),
        }
    finally:
        try:
            os.unlink(temp_path)
        except OSError:
            pass


def handler(job: dict[str, Any]) -> dict[str, Any]:
    job_input = job.get("input", {}) or {}

    # RunPod warmup path used by bp-scanner
    if job_input.get("warm"):
        return {
            "status": "warm",
            "detections": [],
            "total_found": 0,
        }

    image_b64 = job_input.get("image_base64") or job_input.get("file")
    if not image_b64:
        return {"error": "Missing 'image_base64' or 'file' in input"}

    try:
        image = _decode_image(image_b64)
        return _run_layout_ocr(image, job_input)
    except Exception as exc:
        return {"error": str(exc)}


runpod.serverless.start({"handler": handler})

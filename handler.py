import os
import base64
import tempfile
from io import BytesIO
from PIL import Image
import runpod

# Suppress noisy logs
os.environ['FLAGS_enable_pir_api'] = '1'
os.environ['FLAGS_use_mkldnn'] = '0'
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
import logging
logging.getLogger("ppocr").setLevel(logging.ERROR)

import paddle
if paddle.device.is_compiled_with_cuda():
    paddle.set_device('gpu')
else:
    paddle.set_device('cpu')

from paddleocr import PaddleOCRVL

# --- GLOBAL WARMUP ---
# Initializes when the Pod boots up, keeping the model loaded in VRAM
print("🚀 Initializing PaddleOCR-VL 1.5 Worker...")
model = PaddleOCRVL(
    pipeline_version="v1.5",
    use_doc_orientation_classify=False,
    use_doc_unwarping=False,
)

def tile_image(img, tile_size=1088, overlap=450):
    w, h = img.size
    tiles = []
    step = tile_size - overlap
    if step <= 0: raise ValueError("tile_size must be larger than overlap")

    x_starts = [0] if w <= tile_size else list(range(0, w - tile_size + 1, step))
    if w > tile_size and x_starts[-1] != w - tile_size: x_starts.append(w - tile_size)

    y_starts = [0] if h <= tile_size else list(range(0, h - tile_size + 1, step))
    if h > tile_size and y_starts[-1] != h - tile_size: y_starts.append(h - tile_size)

    for y in y_starts:
        for x in x_starts:
            x2, y2 = min(x + tile_size, w), min(y + tile_size, h)
            tiles.append((img.crop((x, y, x2, y2)), x, y))
    return tiles

def dedupe_results(results, iou_thresh=0.3):
    if not results: return results
    def iou(b1, b2):
        x1, y1 = max(b1[0], b2[0]), max(b1[1], b2[1])
        x2, y2 = min(b1[2], b2[2]), min(b1[3], b2[3])
        inter = max(0, x2 - x1) * max(0, y2 - y1)
        union = (b1[2]-b1[0])*(b1[3]-b1[1]) + (b2[2]-b2[0])*(b2[3]-b2[1]) - inter
        return inter / union if union > 0 else 0

    keep, suppressed = [], set()
    for i in range(len(results)):
        if i in suppressed: continue
        keep.append(results[i])
        for j in range(i + 1, len(results)):
            if j not in suppressed and iou(results[i][1], results[j][1]) > iou_thresh:
                suppressed.add(j)
    return keep

def handler(job):
    """
    The main RunPod serverless handler. 
    Expects job['input'] to contain a base64 encoded image.
    """
    job_input = job['input']
    
    # Extract parameters from the incoming API request
    image_b64 = job_input.get("image_base64")
    tile_size = job_input.get("tile", 1088)
    overlap = job_input.get("overlap", 450)
    
    if not image_b64:
        return {"error": "Missing 'image_base64' in input"}

    try:
        # Decode image
        image_data = base64.b64decode(image_b64)
        img = Image.open(BytesIO(image_data)).convert("RGB")
        
        # 1. Tile Image
        tiles = tile_image(img, tile_size=tile_size, overlap=overlap)
        
        all_results = []
        # 2. Run Inference on Tiles
        for idx, (tile_img, x_off, y_off) in enumerate(tiles, 1):
            with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as f:
                tile_path = f.name
                tile_img.save(f, format="PNG")
                
            try:
                preds = model.predict(tile_path, use_layout_detection=True, layout_threshold=0.1)
                for page_result in preds:
                    parsing_list = page_result.get("parsing_res_list", [])
                    for block in parsing_list:
                        label, bbox, content = getattr(block, 'label', ''), getattr(block, 'bbox', []), getattr(block, 'content', '')
                        if content and content.strip() and bbox and len(bbox) >= 4:
                            # Shift to global coordinates
                            shifted_bbox = [bbox[0]+x_off, bbox[1]+y_off, bbox[2]+x_off, bbox[3]+y_off]
                            all_results.append((content.strip(), shifted_bbox, label))
            finally:
                os.unlink(tile_path)

        # 3. Deduplicate
        final_results = dedupe_results(all_results)
        
        # 4. Format JSON response for the swarm
        formatted_output = [{"text": res[0], "bbox": res[1], "label": res[2]} for res in final_results]
        
        return {"status": "success", "detections": formatted_output, "total_found": len(formatted_output)}

    except Exception as e:
        return {"error": str(e)}

# Start the RunPod Serverless Worker
runpod.serverless.start({"handler": handler})
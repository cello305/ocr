import os

import modal


APP_NAME = "ocr-test-rapidocr"
OCR_CACHE_DIR = "/cache/rapidocr"
CPU_THREADS = 2
MAX_IMAGE_EDGE = 1800
MIN_CONFIDENCE = 0.40

image = (
    modal.Image.debian_slim(python_version="3.12")
    .apt_install(
        "libgl1",
        "libglib2.0-0",
    )
    .pip_install(
        "fastapi==0.115.12",
        "gradio==6.9.0",
        "numpy",
        "onnxruntime==1.20.1",
        "Pillow",
        "rapidocr-onnxruntime==1.4.4",
    )
    .env(
        {
            "OMP_NUM_THREADS": str(CPU_THREADS),
            "MKL_NUM_THREADS": str(CPU_THREADS),
            "ORT_NUM_THREADS": str(CPU_THREADS),
            "RAPIDOCR_MODEL_DIR": OCR_CACHE_DIR,
            "GRADIO_TEMP_DIR": "/tmp/gradio",
        }
    )
)

ocr_cache = modal.Volume.from_name("ocr-test-rapidocr-cache", create_if_missing=True)
app = modal.App(APP_NAME)

_ocr = None


def get_ocr():
    global _ocr
    if _ocr is not None:
        return _ocr

    from rapidocr_onnxruntime import RapidOCR

    _ocr = RapidOCR(
        det_db_box_thresh=0.3,
        det_db_unclip_ratio=1.5,
    )
    return _ocr


def normalize_image(image):
    image = image.convert("RGB")
    image.thumbnail((MAX_IMAGE_EDGE, MAX_IMAGE_EDGE))
    return image


def extract_text(result):
    if not result:
        return ""

    # Collect detections with full spatial info
    detections = []
    for item in result:
        if len(item) < 3:
            continue
        box, text, score = item[0], (item[1] or "").strip(), item[2]
        if not text or score is None or score < MIN_CONFIDENCE:
            continue
        ys = [pt[1] for pt in box]
        xs = [pt[0] for pt in box]
        center_y = sum(ys) / len(ys)
        min_x = min(xs)
        max_x = max(xs)
        height = max(ys) - min(ys)
        width = max_x - min_x
        detections.append({
            "center_y": center_y,
            "min_x": min_x,
            "max_x": max_x,
            "height": height,
            "width": width,
            "text": text,
        })

    if not detections:
        return ""

    # Line grouping threshold: half the median text height
    heights = sorted(d["height"] for d in detections)
    median_h = heights[len(heights) // 2] if heights else 20
    line_thresh = max(median_h * 0.5, 5)

    # Sort all detections top-to-bottom, then left-to-right
    detections.sort(key=lambda d: (d["center_y"], d["min_x"]))

    # Group into lines by vertical proximity
    lines = []
    current_line = [detections[0]]
    for det in detections[1:]:
        if abs(det["center_y"] - current_line[0]["center_y"]) <= line_thresh:
            current_line.append(det)
        else:
            lines.append(current_line)
            current_line = [det]
    lines.append(current_line)

    # Build output with gap-based spacing
    output = []
    prev_center_y = None
    for line in lines:
        line.sort(key=lambda d: d["min_x"])
        line_center_y = sum(d["center_y"] for d in line) / len(line)

        # Insert blank line for large vertical gaps (paragraph break)
        if prev_center_y is not None:
            gap = line_center_y - prev_center_y
            if gap > median_h * 1.8:
                output.append("")

        # Join words on this line using horizontal gap analysis
        if len(line) == 1:
            line_text = line[0]["text"]
        else:
            # Compute typical char width for this line
            total_chars = sum(len(d["text"]) for d in line)
            total_width = sum(d["width"] for d in line)
            avg_char_w = total_width / max(total_chars, 1)

            parts = [line[0]["text"]]
            for i in range(1, len(line)):
                gap = line[i]["min_x"] - line[i - 1]["max_x"]
                # If gap is wider than ~0.3 of an average char, insert a space
                if gap > avg_char_w * 0.3:
                    parts.append(" ")
                parts.append(line[i]["text"])
            line_text = "".join(parts)

        output.append(line_text)
        prev_center_y = line_center_y

    return "\n".join(output).strip()


@app.function(
    image=image,
    volumes={OCR_CACHE_DIR: ocr_cache},
    cpu=2,
    memory=3072,
    max_containers=1,
    scaledown_window=600,
)
@modal.asgi_app(label="ocr")
def ui():
    import gradio as gr
    import numpy as np
    from fastapi import FastAPI
    from gradio.routes import mount_gradio_app

    web_app = FastAPI()
    get_ocr()

    def process_document(image):
        if image is None:
            return "Please upload an image."

        image = normalize_image(image)
        ocr = get_ocr()
        result, _ = ocr(np.array(image))
        text = extract_text(result)
        return text or "No text detected."

    blocks = gr.Interface(
        fn=process_document,
        inputs=gr.Image(type="pil", label="Upload scanned page / receipt / document"),
        outputs=gr.Textbox(label="Extracted text", lines=24),
        title="Fast Document OCR - RapidOCR",
        description="CPU-friendly OCR for standard scanned documents on Modal. Tuned for text, numbers, receipts, and printed pages.",
    )
    blocks.enable_queue = False

    return mount_gradio_app(
        app=web_app, blocks=blocks, path="/", allowed_paths=["/tmp"]
    )

import os

import modal


APP_NAME = "ocr-test-rapidocr"
OCR_CACHE_DIR = "/cache/rapidocr"
CPU_THREADS = 2
MAX_IMAGE_EDGE = 1800
MIN_CONFIDENCE = 0.45

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

    _ocr = RapidOCR()
    return _ocr


def normalize_image(image):
    image = image.convert("RGB")
    image.thumbnail((MAX_IMAGE_EDGE, MAX_IMAGE_EDGE))
    return image


def extract_text(result):
    if not result:
        return ""

    # Collect detections with spatial info
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
        height = max(ys) - min(ys)
        detections.append((center_y, min_x, height, text))

    if not detections:
        return ""

    # Line grouping threshold: half the median text height
    heights = sorted(d[2] for d in detections)
    median_h = heights[len(heights) // 2] if heights else 20
    line_thresh = max(median_h * 0.5, 5)

    # Sort all detections top-to-bottom, then left-to-right
    detections.sort(key=lambda d: (d[0], d[1]))

    # Group into lines by vertical proximity
    lines = []
    current_line = [detections[0]]
    for det in detections[1:]:
        if abs(det[0] - current_line[0][0]) <= line_thresh:
            current_line.append(det)
        else:
            lines.append(current_line)
            current_line = [det]
    lines.append(current_line)

    # Sort each line left-to-right and build output
    output = []
    prev_center_y = None
    for line in lines:
        line.sort(key=lambda d: d[1])
        line_center_y = sum(d[0] for d in line) / len(line)

        # Insert blank line for large vertical gaps (paragraph break)
        if prev_center_y is not None:
            gap = line_center_y - prev_center_y
            if gap > median_h * 1.8:
                output.append("")

        line_text = "  ".join(d[3] for d in line)
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

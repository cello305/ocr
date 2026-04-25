import re

import modal


APP_NAME = "ocr-test-rapidocr"
OCR_CACHE_DIR = "/cache/rapidocr"
CPU_THREADS = 2
MAX_IMAGE_EDGE = 1800
TARGET_MIN_IMAGE_EDGE = 1400
MAX_UPSCALE_FACTOR = 2.0
MIN_CONFIDENCE = 0.40
SHORT_TEXT_MIN_CONFIDENCE = 0.55
MAX_SKEW_CORRECTION_DEGREES = 4.0

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
        "opencv-python-headless==4.10.0.84",
        "Pillow",
        "rapidocr-onnxruntime==1.4.4",
        "wordninja",
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
        det_limit_side_len=1280,
        det_db_thresh=0.2,
        det_db_box_thresh=0.1,
        det_db_unclip_ratio=1.6,
        det_use_dilation=True,
    )
    return _ocr


def resize_image_for_ocr(image):
    from PIL import Image

    width, height = image.size
    longest_edge = max(width, height)
    if longest_edge <= 0:
        return image

    if longest_edge > MAX_IMAGE_EDGE:
        scale = MAX_IMAGE_EDGE / longest_edge
    elif longest_edge < TARGET_MIN_IMAGE_EDGE:
        scale = min(TARGET_MIN_IMAGE_EDGE / longest_edge, MAX_UPSCALE_FACTOR)
    else:
        scale = 1.0

    if abs(scale - 1.0) < 0.01:
        return image

    return image.resize(
        (max(int(round(width * scale)), 1), max(int(round(height * scale)), 1)),
        Image.Resampling.LANCZOS,
    )


def rotate_array(image_array, angle, border_value=255, interpolation=None):
    import cv2
    import numpy as np

    if interpolation is None:
        interpolation = cv2.INTER_CUBIC if image_array.ndim == 2 else cv2.INTER_LINEAR

    height, width = image_array.shape[:2]
    center = (width / 2.0, height / 2.0)
    matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
    rotated = cv2.warpAffine(
        image_array,
        matrix,
        (width, height),
        flags=interpolation,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=border_value,
    )
    return np.clip(rotated, 0, 255).astype(image_array.dtype)


def estimate_skew_angle(image_array):
    import cv2
    import numpy as np

    binary = cv2.threshold(
        image_array,
        0,
        255,
        cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU,
    )[1]
    if np.count_nonzero(binary) < 250:
        return 0.0

    def projection_score(angle):
        rotated = rotate_array(
            binary,
            angle,
            border_value=0,
            interpolation=cv2.INTER_NEAREST,
        )
        histogram = (rotated > 0).sum(axis=1).astype("float32")
        if not histogram.size or float(histogram.max()) == 0.0:
            return float("-inf")
        return float(histogram.var())

    coarse_angles = range(
        -int(MAX_SKEW_CORRECTION_DEGREES),
        int(MAX_SKEW_CORRECTION_DEGREES) + 1,
    )
    best_angle = max(coarse_angles, key=projection_score)
    fine_angles = np.arange(best_angle - 0.75, best_angle + 0.76, 0.25)
    best_angle = max(fine_angles, key=projection_score)

    return float(best_angle) if abs(best_angle) >= 0.25 else 0.0


def build_ocr_variants(image):
    import cv2
    import numpy as np
    from PIL import Image, ImageFilter, ImageOps

    image = ImageOps.exif_transpose(image).convert("L")
    image = resize_image_for_ocr(image)

    # Phone photos of paper often have weak contrast, shadows, and soft edges.
    image = ImageOps.autocontrast(image, cutoff=1)
    image = image.filter(ImageFilter.MedianFilter(size=3))

    base = np.array(image)
    skew_angle = estimate_skew_angle(base)
    if skew_angle:
        base = rotate_array(base, skew_angle, border_value=255)

    clahe = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(8, 8))
    contrast = clahe.apply(base)
    softened = cv2.GaussianBlur(contrast, (0, 0), 1.1)
    sharpened = cv2.addWeighted(contrast, 1.5, softened, -0.5, 0)

    adaptive = cv2.adaptiveThreshold(
        sharpened,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY,
        35,
        11,
    )
    otsu = cv2.threshold(
        sharpened,
        0,
        255,
        cv2.THRESH_BINARY + cv2.THRESH_OTSU,
    )[1]

    return [
        {
            "name": "grayscale",
            "image": Image.fromarray(cv2.cvtColor(sharpened, cv2.COLOR_GRAY2RGB)),
        },
        {
            "name": "adaptive",
            "image": Image.fromarray(cv2.cvtColor(adaptive, cv2.COLOR_GRAY2RGB)),
        },
        {
            "name": "otsu",
            "image": Image.fromarray(cv2.cvtColor(otsu, cv2.COLOR_GRAY2RGB)),
        },
    ]


def normalize_image(image):
    return build_ocr_variants(image)[0]["image"]


def fix_word_spacing(text):
    """Split concatenated words using statistical word segmentation.

    RapidOCR's recognition model often omits spaces between English words
    (e.g. "Savemoneyon" instead of "Save money on"). This function uses
    wordninja to split long tokens that look like concatenated words,
    while preserving the original casing from the OCR output.
    """
    try:
        import wordninja
    except ImportError:
        return text

    tokens = text.split()
    fixed = []
    for token in tokens:
        # Only attempt to split tokens that are long enough to plausibly
        # be multiple concatenated words, and are mostly alphabetic
        alpha_count = sum(1 for c in token if c.isalpha())
        if len(token) >= 8 and alpha_count / max(len(token), 1) > 0.8:
            splits = wordninja.split(token)
            if len(splits) > 1:
                # Map original casing back onto the split words
                pos = 0
                fixed_token = ""
                for word in splits:
                    for ch in word:
                        while pos < len(token) and token[pos].lower() != ch.lower():
                            fixed_token += token[pos]
                            pos += 1
                        if pos < len(token):
                            fixed_token += token[pos]
                            pos += 1
                    fixed_token += " "
                while pos < len(token):
                    fixed_token += token[pos]
                    pos += 1
                fixed.append(fixed_token.strip())
                continue
        fixed.append(token)
    return " ".join(fixed)


def cleanup_extracted_text(text):
    if not text:
        return ""

    cleaned_lines = []
    for raw_line in text.splitlines():
        line = " ".join(raw_line.split())
        if not line:
            cleaned_lines.append("")
            continue

        line = re.sub(r"^(\d+\.\s+)['`]\s*(?=[A-Z][a-z])", r"\1", line)
        line = re.sub(r"^['`]\s*(?=[A-Z][a-z])", "", line)
        line = re.sub(r"\s+([,.;:!?])", r"\1", line)
        line = re.sub(r"([(\[])\s+", r"\1", line)
        line = re.sub(r"\s+([)\]])", r"\1", line)
        line = re.sub(r"\b([A-Za-z]{2,})\s*-\s*([A-Za-z]{2,})\b", r"\1-\2", line)

        if (
            line.endswith(("'", "`"))
            and re.search(r"[A-Za-z]['`]$", line)
            and line.count("'") + line.count("`") == 1
            and not line.lower().endswith(("s'", "s`"))
        ):
            line = line[:-1]

        if re.fullmatch(r"[_=\-~]{3,}", line):
            continue

        cleaned_lines.append(line)

    cleaned = "\n".join(cleaned_lines)
    cleaned = re.sub(r"\n{3,}", "\n\n", cleaned)
    return cleaned.strip()


def _looks_like_heading(text):
    if not text:
        return False

    stripped = text.strip()
    if re.match(r"^\d+\.\s", stripped):
        return False

    lowered = stripped.lower()
    heading_phrases = (
        "technical responsibilities",
        "analytical and problem-solving skills",
        "collaboration and communication",
        "epic team staff levels",
        "expectations by staff level",
        "effective june",
    )
    if any(phrase in lowered for phrase in heading_phrases):
        return True

    words = stripped.split()
    if len(words) <= 8 and stripped[0].isupper():
        capitalized_words = sum(1 for word in words if word[:1].isupper())
        return capitalized_words >= max(len(words) - 1, 1)

    return False


def _should_append_continuation(previous_text, current_text, current_min_x, left_margin, median_h):
    if not previous_text or not current_text:
        return False

    if _looks_like_heading(previous_text):
        return False

    stripped = current_text.strip()
    if re.match(r"^\d+\.\s", stripped):
        return False
    if _looks_like_heading(stripped):
        return False

    continuation_markers = (
        stripped[:1].islower()
        or stripped.startswith(("(", "["))
        or current_min_x > left_margin + median_h * 0.9
    )
    if not continuation_markers:
        return False

    previous_ends_like_open_clause = previous_text.rstrip().endswith(
        (":", ",", ";", "(", "/", "with", "and", "or")
    )
    previous_is_numbered_item = bool(re.match(r"^\d+\.\s", previous_text.strip()))
    previous_is_short_label = len(previous_text.split()) <= 3 and previous_text[:1].isupper()

    return previous_ends_like_open_clause or previous_is_numbered_item or not previous_is_short_label


def extract_text(result):
    if not result:
        return ""

    # Collect detections with full spatial info
    detections = []
    for item in result:
        if len(item) < 3:
            continue
        box, raw_text, score = item[0], (item[1] or "").strip(), item[2]
        text = fix_word_spacing(raw_text)
        if not text or score is None:
            continue
        min_confidence = (
            SHORT_TEXT_MIN_CONFIDENCE if len(text.replace(" ", "")) <= 3 else MIN_CONFIDENCE
        )
        if score < min_confidence:
            continue
        ys = [pt[1] for pt in box]
        xs = [pt[0] for pt in box]
        center_y = sum(ys) / len(ys)
        min_y = min(ys)
        max_y = max(ys)
        min_x = min(xs)
        max_x = max(xs)
        height = max_y - min_y
        width = max_x - min_x
        detections.append({
            "center_y": center_y,
            "min_y": min_y,
            "max_y": max_y,
            "min_x": min_x,
            "max_x": max_x,
            "height": height,
            "width": width,
            "text": text,
        })

    if not detections:
        return ""

    # Use very conservative line clustering so wrapped rows remain separate.
    heights = sorted(d["height"] for d in detections)
    median_h = heights[len(heights) // 2] if heights else 20
    line_thresh = max(median_h * 0.28, 4)

    # Sort all detections top-to-bottom, then left-to-right
    detections.sort(key=lambda d: (d["center_y"], d["min_x"]))

    def line_stats(line):
        centers = sorted(d["center_y"] for d in line)
        heights_local = sorted(d["height"] for d in line)
        return centers[len(centers) // 2], max(heights_local[len(heights_local) // 2], 1)

    lines = []
    current_line = [detections[0]]
    for det in detections[1:]:
        current_center, current_height = line_stats(current_line)
        vertical_overlap = min(det["max_y"], max(d["max_y"] for d in current_line)) - max(
            det["min_y"], min(d["min_y"] for d in current_line)
        )
        overlap_ratio = vertical_overlap / max(min(det["height"], current_height), 1)
        same_line = (
            abs(det["center_y"] - current_center) <= max(line_thresh, current_height * 0.22)
            and overlap_ratio >= 0.15
        )
        if same_line:
            current_line.append(det)
        else:
            lines.append(current_line)
            current_line = [det]
    lines.append(current_line)

    # Build output with gap-based spacing
    rendered_lines = []
    prev_center_y = None
    for line in lines:
        line.sort(key=lambda d: d["min_x"])
        line_center_y = sum(d["center_y"] for d in line) / len(line)

        # Insert blank line for large vertical gaps (paragraph break)
        if prev_center_y is not None:
            gap = line_center_y - prev_center_y
            if gap > median_h * 2.0:
                rendered_lines.append({"text": "", "min_x": 0, "center_y": line_center_y})

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
                # If gap is wider than ~0.6 of an average char, insert a space
                if gap > avg_char_w * 0.6:
                    parts.append(" ")
                parts.append(line[i]["text"])
            line_text = "".join(parts)

        rendered_lines.append(
            {
                "text": " ".join(line_text.split()),
                "min_x": min(d["min_x"] for d in line),
                "center_y": line_center_y,
            }
        )
        prev_center_y = line_center_y

    left_margin = min((line["min_x"] for line in rendered_lines if line["text"]), default=0)
    merged_output = []
    for line in rendered_lines:
        text = line["text"]
        if not text:
            merged_output.append("")
            continue

        previous_text = next((item for item in reversed(merged_output) if item), "")
        if merged_output and _should_append_continuation(
            previous_text,
            text,
            line["min_x"],
            left_margin,
            median_h,
        ):
            merged_output[-1] = f"{merged_output[-1]} {text}".strip()
        else:
            merged_output.append(text)

    return cleanup_extracted_text("\n".join(merged_output).strip())


def score_ocr_candidate(text, result):
    if not text:
        return float("-inf")

    confidences = [
        item[2]
        for item in result or []
        if len(item) >= 3 and item[2] is not None
    ]
    lines = [line for line in text.splitlines() if line.strip()]
    alnum_chars = sum(1 for ch in text if ch.isalnum())
    short_lines = sum(1 for line in lines if len(line) <= 2)
    orphan_lowercase_lines = sum(
        1
        for line in lines
        if line[:1].islower() and not re.match(r"^\d+\.\s", line)
    )
    dangling_lines = sum(
        1
        for line in lines
        if len(line.split()) <= 2 and line[:1].islower()
    )

    return (
        (sum(confidences) / max(len(confidences), 1)) * 4.0
        + min(alnum_chars, 800) * 0.025
        + len(lines) * 0.2
        - short_lines * 0.25
        - orphan_lowercase_lines * 0.6
        - dangling_lines * 0.8
    )


def choose_best_ocr_candidate(candidates):
    best_candidate = None
    best_score = float("-inf")

    for candidate in candidates:
        score = score_ocr_candidate(candidate.get("text", ""), candidate.get("result") or [])
        if score > best_score:
            best_score = score
            best_candidate = {**candidate, "score": score}

    return best_candidate


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

        ocr = get_ocr()
        candidates = []
        for variant in build_ocr_variants(image):
            result, _ = ocr(np.array(variant["image"]))
            candidates.append(
                {
                    "name": variant["name"],
                    "result": result or [],
                    "text": extract_text(result),
                }
            )

        best_candidate = choose_best_ocr_candidate(candidates)
        return best_candidate["text"] if best_candidate and best_candidate["text"] else "No text detected."

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

import re
import shutil
import subprocess
import tempfile
from pathlib import Path

import modal


APP_NAME = "ocr-test-surya"
CPU_THREADS = 2
MAX_IMAGE_EDGE = 1800
TARGET_MIN_IMAGE_EDGE = 1400
MAX_UPSCALE_FACTOR = 2.0
MIN_CONFIDENCE = 0.40
SHORT_TEXT_MIN_CONFIDENCE = 0.55
MAX_SKEW_CORRECTION_DEGREES = 4.0
TILE_OVERLAP_RATIO = 0.18
MIN_TILE_HEIGHT = 900
TESSERACT_PSMS = (3, 4, 6)

image = (
    modal.Image.debian_slim(python_version="3.12")
    .apt_install(
        "libgl1",
        "libglib2.0-0",
        "tesseract-ocr",
    )
    .pip_install(
        "fastapi==0.115.12",
        "gradio==6.9.0",
        "numpy",
        "opencv-python-headless==4.10.0.84",
        "Pillow",
        "pytesseract",
        "surya-ocr",
        "torch==2.7.1",
        "wordninja",
    )
    .env(
        {
            "OMP_NUM_THREADS": str(CPU_THREADS),
            "MKL_NUM_THREADS": str(CPU_THREADS),
            "GRADIO_TEMP_DIR": "/tmp/gradio",
        }
    )
)

app = modal.App(APP_NAME)


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

    background = cv2.GaussianBlur(base, (0, 0), 31)
    normalized = cv2.divide(base, background, scale=255)
    normalized = cv2.normalize(normalized, None, 0, 255, cv2.NORM_MINMAX)

    clahe = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(8, 8))
    contrast = clahe.apply(normalized)
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
            "name": "shadow_reduced",
            "image": Image.fromarray(cv2.cvtColor(normalized, cv2.COLOR_GRAY2RGB)),
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


def _box_to_polygon(box):
    if not box or len(box) != 4:
        return None
    x1, y1, x2, y2 = box
    return [[x1, y1], [x2, y1], [x2, y2], [x1, y2]]


def surya_line_to_result(line):
    text = getattr(line, "text", None)
    confidence = getattr(line, "confidence", None)
    polygon = getattr(line, "polygon", None)
    bbox = getattr(line, "bbox", None)
    if isinstance(line, dict):
        text = line.get("text", text)
        confidence = line.get("confidence", confidence)
        polygon = line.get("polygon", polygon)
        bbox = line.get("bbox", bbox)

    if not text:
        return None
    if polygon is None:
        polygon = _box_to_polygon(bbox)
    if polygon is None:
        return None
    score = float(confidence) if confidence is not None else 0.9
    return (polygon, text, score)


def parse_surya_predictions(predictions):
    parsed = []
    for prediction in predictions or []:
        text_lines = getattr(prediction, "text_lines", None)
        if isinstance(prediction, dict):
            text_lines = prediction.get("text_lines", text_lines)
        for line in text_lines or []:
            item = surya_line_to_result(line)
            if item is not None:
                parsed.append(item)
    return parsed


def build_surya_candidates(image):
    variants = [
        variant for variant in build_ocr_variants(image)
        if variant["name"] in {"shadow_reduced", "grayscale"}
    ]
    candidates = []
    for variant in variants:
        predictions = run_surya_cli(variant["image"])
        result = parse_surya_predictions(predictions)
        candidates.append(
            {
                "name": f"surya:{variant['name']}",
                "group": f"surya:{variant['name']}",
                "result": result,
                "text": extract_text(result),
            }
        )
    return candidates


def run_surya_cli(image):
    import json

    with tempfile.TemporaryDirectory(prefix="surya-ocr-") as temp_dir:
        temp_path = Path(temp_dir)
        input_path = temp_path / "input.png"
        output_dir = temp_path / "output"
        image.save(input_path)

        command = [
            shutil.which("surya_ocr") or "surya_ocr",
            str(input_path),
            "--output_dir",
            str(output_dir),
            "--task_name",
            "ocr_with_boxes",
            "--disable_math",
        ]
        subprocess.run(command, check=True, capture_output=True, text=True)

        results_path = output_dir / "results.json"
        with results_path.open("r", encoding="utf-8") as handle:
            payload = json.load(handle)

        page_results = payload.get("input") or payload.get(input_path.stem) or []
        return page_results


def build_ocr_inputs(image):
    variants = build_ocr_variants(image)
    inputs = []
    for variant in variants:
        inputs.append(
            {
                "name": variant["name"],
                "image": variant["image"],
                "offset_x": 0,
                "offset_y": 0,
                "group": variant["name"],
            }
        )
        inputs.extend(build_tiled_ocr_inputs(variant["image"], variant["name"]))
    return inputs


def build_tiled_ocr_inputs(image, base_name):
    width, height = image.size
    if height < max(width * 1.2, MIN_TILE_HEIGHT * 1.2):
        return []

    tile_count = 3 if height >= width * 1.35 else 2
    tile_height = max(int(round(height / tile_count)), MIN_TILE_HEIGHT)
    tile_height = min(tile_height, height)
    overlap = int(round(tile_height * TILE_OVERLAP_RATIO))
    step = max(tile_height - overlap, 1)

    inputs = []
    top = 0
    tile_index = 0
    while top < height:
        bottom = min(top + tile_height, height)
        crop = image.crop((0, top, width, bottom))
        inputs.append(
            {
                "name": f"{base_name}_tile_{tile_index + 1}",
                "image": crop,
                "offset_x": 0,
                "offset_y": top,
                "group": f"{base_name}_tiles",
            }
        )
        if bottom >= height:
            break
        top += step
        tile_index += 1
    return inputs


def offset_box(box, offset_x=0, offset_y=0):
    return [[pt[0] + offset_x, pt[1] + offset_y] for pt in box]


def offset_ocr_result(result, offset_x=0, offset_y=0):
    adjusted = []
    for item in result or []:
        if len(item) < 3:
            continue
        box, text, score = item[0], item[1], item[2]
        adjusted.append((offset_box(box, offset_x, offset_y), text, score))
    return adjusted


def _box_bounds(box):
    xs = [pt[0] for pt in box]
    ys = [pt[1] for pt in box]
    return min(xs), min(ys), max(xs), max(ys)


def _intersection_over_union(box_a, box_b):
    ax1, ay1, ax2, ay2 = _box_bounds(box_a)
    bx1, by1, bx2, by2 = _box_bounds(box_b)

    inter_w = max(0, min(ax2, bx2) - max(ax1, bx1))
    inter_h = max(0, min(ay2, by2) - max(ay1, by1))
    intersection = inter_w * inter_h
    if intersection <= 0:
        return 0.0

    area_a = max(ax2 - ax1, 0) * max(ay2 - ay1, 0)
    area_b = max(bx2 - bx1, 0) * max(by2 - by1, 0)
    union = area_a + area_b - intersection
    return intersection / union if union > 0 else 0.0


def dedupe_ocr_results(result):
    deduped = []
    for item in sorted(
        result or [],
        key=lambda entry: (-(entry[2] or 0), entry[0][0][1], entry[0][0][0]),
    ):
        if len(item) < 3:
            continue
        box, raw_text, score = item[0], (item[1] or "").strip(), item[2]
        if not raw_text:
            continue
        normalized_text = re.sub(r"\s+", " ", raw_text).strip().lower()
        duplicate_found = False
        for kept in deduped:
            kept_text = re.sub(r"\s+", " ", (kept[1] or "").strip()).lower()
            if normalized_text != kept_text:
                continue
            if _intersection_over_union(box, kept[0]) >= 0.5:
                duplicate_found = True
                break
        if not duplicate_found:
            deduped.append((box, raw_text, score))
    return deduped


def parse_tesseract_data(data):
    result = []
    count = len(data.get("text", []))
    for i in range(count):
        raw_text = (data["text"][i] or "").strip()
        if not raw_text:
            continue
        try:
            conf = float(data["conf"][i])
        except (TypeError, ValueError):
            continue
        if conf < 0:
            continue
        left = int(data["left"][i])
        top = int(data["top"][i])
        width = int(data["width"][i])
        height = int(data["height"][i])
        result.append(
            (
                [[left, top], [left + width, top], [left + width, top + height], [left, top + height]],
                raw_text,
                max(min(conf / 100.0, 1.0), 0.0),
            )
        )
    return result


def run_tesseract(image):
    import pytesseract

    data = pytesseract.image_to_data(
        image,
        output_type=pytesseract.Output.DICT,
        config="--oem 3 --psm 6 -c preserve_interword_spaces=1",
    )
    return parse_tesseract_data(data)


def run_tesseract_text(image, psm):
    import pytesseract

    return pytesseract.image_to_string(
        image,
        config=f"--oem 3 --psm {psm} -c preserve_interword_spaces=1",
    )


def build_tesseract_text_candidates(image):
    candidates = []
    for variant in build_ocr_variants(image):
        for psm in TESSERACT_PSMS:
            raw_text = run_tesseract_text(variant["image"], psm)
            candidates.append(
                {
                    "name": f"tesseract_text:{variant['name']}:psm{psm}",
                    "group": f"tesseract_text:{variant['name']}:psm{psm}",
                    "result": [],
                    "text": cleanup_extracted_text(raw_text),
                }
            )
    return candidates


def group_ocr_candidates(raw_candidates):
    grouped = {}
    for candidate in raw_candidates:
        group_name = candidate["group"]
        grouped.setdefault(group_name, []).extend(candidate["result"])

    candidates = []
    for group_name, result in grouped.items():
        deduped_result = dedupe_ocr_results(result)
        candidates.append(
            {
                "name": group_name,
                "group": group_name,
                "result": deduped_result,
                "text": extract_text(deduped_result),
            }
        )
    return candidates


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
        line = re.sub(r"\b([A-Za-z]{1,4})\s+ware\b", r"\1ware", line, flags=re.IGNORECASE)

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
    cleaned = re.sub(r"(?<=\d)\.(?=[A-Z])", ". ", cleaned)
    cleaned = re.sub(r"(?<!\n)(?=(?:\d{1,2}\.)\s*[A-Z])", "\n", cleaned)
    cleaned = re.sub(r"\n{3,}", "\n\n", cleaned)
    normalized_lines = []
    for raw_line in cleaned.splitlines():
        if not raw_line.strip():
            if normalized_lines and normalized_lines[-1] != "":
                normalized_lines.append("")
            continue
        for line in split_inline_numbered_items(raw_line):
            if is_suspicious_numeric_line(line):
                continue
            if normalized_lines and lines_look_duplicated(normalized_lines[-1], line):
                continue
            normalized_lines.append(line)
    cleaned = "\n".join(normalized_lines)
    cleaned = re.sub(r"\n{3,}", "\n\n", cleaned)
    return cleaned.strip()


def split_inline_numbered_items(line):
    parts = re.split(r"(?=(?:^|\s)(?:\d{1,2}\.)\s*[A-Z])", line)
    return [part.strip() for part in parts if part and part.strip()]


def is_suspicious_numeric_line(text):
    stripped = text.strip()
    return bool(re.fullmatch(r"\d{2,}\.\d{4,}", stripped))


def _normalize_compare_text(text):
    return re.sub(r"[^a-z0-9]+", " ", text.lower()).strip()


def lines_look_duplicated(left, right):
    normalized_left = _normalize_compare_text(left)
    normalized_right = _normalize_compare_text(right)
    if not normalized_left or not normalized_right:
        return False
    if normalized_left == normalized_right:
        return True
    if len(normalized_left) >= 24 and normalized_left in normalized_right:
        return True
    if len(normalized_right) >= 24 and normalized_right in normalized_left:
        return True

    left_tokens = set(normalized_left.split())
    right_tokens = set(normalized_right.split())
    if not left_tokens or not right_tokens:
        return False
    overlap = len(left_tokens & right_tokens) / max(min(len(left_tokens), len(right_tokens)), 1)
    return overlap >= 0.85 and min(len(left_tokens), len(right_tokens)) >= 5


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
    suspicious_numeric_lines = sum(1 for line in lines if is_suspicious_numeric_line(line))
    duplicate_lines = sum(
        1
        for i in range(1, len(lines))
        if lines_look_duplicated(lines[i - 1], lines[i])
    )
    heading_hits = sum(
        1
        for line in lines
        if _looks_like_heading(line)
    )
    numbered_lines = sum(1 for line in lines if re.match(r"^\d+\.\s", line))
    likely_bad_title = 1 if any("rapidocr" in line.lower() for line in lines[:3]) else 0

    return (
        (sum(confidences) / max(len(confidences), 1)) * 4.0
        + min(alnum_chars, 800) * 0.025
        + len(lines) * 0.2
        + heading_hits * 0.4
        + min(numbered_lines, 8) * 0.25
        - short_lines * 0.25
        - orphan_lowercase_lines * 0.6
        - dangling_lines * 0.8
        - suspicious_numeric_lines * 1.5
        - duplicate_lines * 1.0
        - likely_bad_title * 1.5
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
    cpu=2,
    memory=3072,
    max_containers=1,
    scaledown_window=600,
)
@modal.asgi_app(label="surya-ocr")
def ui():
    import gradio as gr
    from fastapi import FastAPI
    from gradio.routes import mount_gradio_app

    web_app = FastAPI()

    def process_document(image, engine):
        if image is None:
            return "Please upload an image."

        if engine == "Surya":
            candidates = build_surya_candidates(image)
            best_candidate = choose_best_ocr_candidate(candidates)
            if best_candidate and best_candidate["text"]:
                return best_candidate["text"]

        text_candidates = build_tesseract_text_candidates(image)
        raw_candidates = []
        for ocr_input in build_ocr_inputs(image):
            raw_candidates.append(
                {
                    "name": f"tesseract:{ocr_input['name']}",
                    "group": f"tesseract:{ocr_input['group']}",
                    "result": offset_ocr_result(
                        run_tesseract(ocr_input["image"]),
                        offset_x=ocr_input["offset_x"],
                        offset_y=ocr_input["offset_y"],
                    ),
                }
            )

        candidates = text_candidates + group_ocr_candidates(raw_candidates)
        best_candidate = choose_best_ocr_candidate(candidates)
        return best_candidate["text"] if best_candidate and best_candidate["text"] else "No text detected."

    blocks = gr.Interface(
        fn=process_document,
        inputs=[
            gr.Image(type="pil", label="Upload scanned page / receipt / document"),
            gr.Radio(
                choices=["Surya", "Tesseract"],
                value="Surya",
                label="Engine",
            ),
        ],
        outputs=gr.Textbox(label="Extracted text", lines=24),
        title="Fast Document OCR",
        description="Surya document OCR with Tesseract fallback.",
    )
    blocks.enable_queue = False

    return mount_gradio_app(
        app=web_app, blocks=blocks, path="/", allowed_paths=["/tmp"]
    )

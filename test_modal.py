import modal
app = modal.App("test-rapid")
image = modal.Image.debian_slim(python_version="3.12").apt_install("libgl1", "libglib2.0-0").pip_install("rapidocr-onnxruntime==1.4.4", "numpy", "Pillow")

@app.function(image=image)
def test_ocr():
    from rapidocr_onnxruntime import RapidOCR
    try:
        ocr = RapidOCR(det_limit_side_len=1800)
        return "SUCCESS det_limit_side_len"
    except Exception as e:
        return f"FAILED: {e}"

@app.local_entrypoint()
def main():
    print(test_ocr.remote())

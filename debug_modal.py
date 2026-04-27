import modal
import base64

app = modal.App('debug-ocr-remote')
image_deps = modal.Image.debian_slim().apt_install('tesseract-ocr', 'libgl1-mesa-glx', 'libglib2.0-0').pip_install('rapidocr_onnxruntime', 'pytesseract', 'Pillow', 'numpy')

@app.function(image=image_deps)
def debug_remote(img_bytes):
    import modal_app
    from PIL import Image
    import io
    
    image = Image.open(io.BytesIO(img_bytes))
    
    rapid_candidates = modal_app.build_rapidocr_candidates(image)
    tess_candidates = modal_app.build_tesseract_text_candidates(image)
    
    candidates = rapid_candidates + tess_candidates
    
    best = modal_app.choose_best_ocr_candidate(candidates)
    return best['name'], best['text']

@app.local_entrypoint()
def main():
    image_path = r'C:\Users\Emanuel\.gemini\antigravity\brain\898029f9-68ca-461b-8564-0429b2e6fbdc\media__1777252634861.jpg'
    with open(image_path, 'rb') as f:
        img_bytes = f.read()
    
    name, text = debug_remote.remote(img_bytes)
    print(f'BEST IS: {name}')
    print(text)

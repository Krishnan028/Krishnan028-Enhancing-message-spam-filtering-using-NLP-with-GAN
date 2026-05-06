import pytesseract
from PIL import Image, ImageEnhance
import numpy as np
import cv2
import os
import sys

# ---------------------------------------------------------------------------
# Auto-detect Tesseract installation on Windows
# ---------------------------------------------------------------------------
_TESSERACT_PATHS = [
    r"C:\Program Files\Tesseract-OCR\tesseract.exe",
    r"C:\Program Files (x86)\Tesseract-OCR\tesseract.exe",
    os.path.join(os.environ.get("LOCALAPPDATA", ""), "Tesseract-OCR", "tesseract.exe"),
]

_tesseract_available = False
for _p in _TESSERACT_PATHS:
    if os.path.isfile(_p):
        pytesseract.pytesseract.tesseract_cmd = _p
        _tesseract_available = True
        break

if not _tesseract_available:
    # Last resort: check if it's on the system PATH
    import shutil
    if shutil.which("tesseract"):
        _tesseract_available = True

# ---------------------------------------------------------------------------
# EasyOCR fallback (loaded lazily only when needed)
# ---------------------------------------------------------------------------
_easyocr_reader = None

def _get_easyocr_reader():
    """Lazily initialise EasyOCR reader (heavy import, only done once)."""
    global _easyocr_reader
    if _easyocr_reader is None:
        try:
            import easyocr
            _easyocr_reader = easyocr.Reader(['en'], gpu=False, verbose=False)
        except ImportError:
            _easyocr_reader = None
    return _easyocr_reader


class OCRProcessor:
    """Extract text from images using OCR with multi-strategy handwriting support.
    
    Primary engine : Tesseract (if installed)
    Fallback engine: EasyOCR   (pure-Python, no external binary needed)
    """

    def __init__(self):
        self.confidence_threshold = 20
        self.tesseract_available = _tesseract_available

    # ------------------------------------------------------------------
    # Image preprocessing helpers
    # ------------------------------------------------------------------
    def _upscale(self, img_array):
        h, w = img_array.shape[:2]
        if h < 600 or w < 600:
            scale = max(600 / max(h, 1), 600 / max(w, 1), 1.5)
            img_array = cv2.resize(img_array, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_CUBIC)
        return img_array

    def _to_gray(self, img_array):
        if len(img_array.shape) == 3:
            return cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
        return img_array

    def _prep_otsu(self, img_array):
        arr = self._upscale(img_array)
        gray = self._to_gray(arr)
        _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        return Image.fromarray(cv2.fastNlMeansDenoising(binary, h=10))

    def _prep_adaptive(self, img_array):
        arr = self._upscale(img_array)
        gray = self._to_gray(arr)
        blurred = cv2.GaussianBlur(gray, (3, 3), 0)
        adaptive = cv2.adaptiveThreshold(blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                         cv2.THRESH_BINARY, 31, 10)
        return Image.fromarray(adaptive)

    def _prep_contrast(self, img_array):
        arr = self._upscale(img_array)
        pil = Image.fromarray(arr).convert('L')
        pil = ImageEnhance.Contrast(pil).enhance(2.5)
        pil = ImageEnhance.Sharpness(pil).enhance(3.0)
        a = np.array(pil)
        _, binary = cv2.threshold(a, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        return Image.fromarray(binary)

    def _prep_dilate(self, img_array):
        arr = self._upscale(img_array)
        gray = self._to_gray(arr)
        _, inv = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        dilated = cv2.dilate(inv, np.ones((2, 2), np.uint8), iterations=1)
        return Image.fromarray(cv2.bitwise_not(dilated))

    # ------------------------------------------------------------------
    # Tesseract OCR runner
    # ------------------------------------------------------------------
    def _run_ocr(self, pil_img, psm):
        try:
            config = f'--oem 1 --psm {psm}'
            data = pytesseract.image_to_data(pil_img, config=config, output_type=pytesseract.Output.DICT)
            words, confs = [], []
            for i, conf in enumerate(data['conf']):
                try:
                    c = int(conf)
                except Exception:
                    continue
                if c > self.confidence_threshold and str(data['text'][i]).strip():
                    words.append(data['text'][i])
                    confs.append(c)
            text = pytesseract.image_to_string(pil_img, config=config).strip()
            avg_conf = float(np.mean(confs)) if confs else 0.0
            return text, words, avg_conf
        except Exception:
            return '', [], 0.0

    # ------------------------------------------------------------------
    # EasyOCR runner (fallback)
    # ------------------------------------------------------------------
    def _run_easyocr(self, img_array):
        """Use EasyOCR as fallback when Tesseract is unavailable or fails."""
        reader = _get_easyocr_reader()
        if reader is None:
            return '', [], 0.0
        try:
            results = reader.readtext(img_array, detail=1)
            words = []
            confs = []
            texts = []
            for (bbox, text, conf) in results:
                text = text.strip()
                if text and conf > 0.20:
                    words.append(text)
                    confs.append(conf * 100)  # normalise to 0-100 scale
                    texts.append(text)
            full_text = ' '.join(texts)
            avg_conf = float(np.mean(confs)) if confs else 0.0
            return full_text, words, avg_conf
        except Exception:
            return '', [], 0.0

    # ------------------------------------------------------------------
    # Main extraction (tries Tesseract first, then EasyOCR)
    # ------------------------------------------------------------------
    def extract_text(self, image):
        """Try multiple preprocessing strategies and OCR modes; return best result.
        
        Strategy:
          1. If Tesseract is available → try all preprocessing + PSM combos
          2. If Tesseract fails or is missing → fall back to EasyOCR
          3. Return the result with the most detected words
        """
        try:
            img_array = np.array(image.convert('RGB'))

            best_text, best_words, best_conf = '', [], 0.0

            # ----- Strategy 1: Tesseract -----
            if self.tesseract_available:
                strategies = [
                    (self._prep_otsu(img_array), 6),
                    (self._prep_adaptive(img_array), 6),
                    (self._prep_contrast(img_array), 6),
                    (self._prep_dilate(img_array), 6),
                    (self._prep_otsu(img_array), 11),
                    (self._prep_adaptive(img_array), 11),
                    (self._prep_contrast(img_array), 3),
                ]
                for pil_img, psm in strategies:
                    text, words, conf = self._run_ocr(pil_img, psm)
                    if len(words) > len(best_words) or (len(words) == len(best_words) and conf > best_conf):
                        best_text, best_words, best_conf = text, words, conf

            # ----- Strategy 2: EasyOCR fallback -----
            # Use EasyOCR if Tesseract didn't extract anything useful, or isn't installed
            if len(best_words) < 3:
                easy_text, easy_words, easy_conf = self._run_easyocr(img_array)
                if len(easy_words) > len(best_words):
                    best_text, best_words, best_conf = easy_text, easy_words, easy_conf

            return {
                'text': best_text,
                'confident_words': best_words,
                'word_count': len(best_words),
                'confidence': best_conf
            }
        except Exception as e:
            return {'text': '', 'confident_words': [], 'word_count': 0, 'confidence': 0, 'error': str(e)}

    def has_text(self, image):
        return self.extract_text(image)['word_count'] > 2


def get_ocr_processor():
    return OCRProcessor()

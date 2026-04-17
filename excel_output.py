"""
ocr.py — Tesseract-based Devanagari OCR with image preprocessing.

Uses Sanskrit (san) and Hindi (hin) language packs with the combined 'san+hin'
mode for maximum accuracy. Applies preprocessing (grayscale, resize, threshold)
to improve OCR on typical phone photos of book pages.
"""

from __future__ import annotations

from io import BytesIO
from pathlib import Path

import pytesseract
from PIL import Image, ImageEnhance, ImageOps


def preprocess_for_ocr(img: Image.Image) -> Image.Image:
    """
    Basic preprocessing to improve Tesseract results on Devanagari:
      - Convert to grayscale
      - Upscale small images (Tesseract prefers ≥ 300 DPI)
      - Boost contrast
    """
    # Convert to RGB first (in case of RGBA/CMYK)
    if img.mode != 'RGB':
        img = img.convert('RGB')

    # Upscale if the image is small (typical phone photos are already big enough)
    w, h = img.size
    if w < 1000 or h < 1000:
        scale = max(1000 / w, 1000 / h, 1.5)
        new_size = (int(w * scale), int(h * scale))
        img = img.resize(new_size, Image.LANCZOS)

    # Grayscale + contrast boost
    gray = ImageOps.grayscale(img)
    enhanced = ImageEnhance.Contrast(gray).enhance(1.5)

    return enhanced


def extract_text_from_image(
    img: Image.Image,
    lang: str = 'san+hin',
    preprocess: bool = True,
) -> str:
    """
    Run Tesseract OCR on an image. Returns the extracted Devanagari text.

    lang options:
      - 'san'     — Sanskrit only (best for classical text)
      - 'hin'     — Hindi only (often better on modern typography)
      - 'san+hin' — combined (usually best overall)
    """
    if preprocess:
        img = preprocess_for_ocr(img)

    try:
        text = pytesseract.image_to_string(img, lang=lang)
    except pytesseract.TesseractError as e:
        # Fallback to Sanskrit only if combined isn't available
        if '+' in lang:
            text = pytesseract.image_to_string(img, lang=lang.split('+')[0])
        else:
            raise e

    return text.strip()


def get_available_languages() -> list[str]:
    """Return list of Tesseract language packs currently installed."""
    try:
        return pytesseract.get_languages(config='')
    except Exception:
        return []

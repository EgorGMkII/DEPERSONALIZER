"""
OCR Extraction Module using PaddleOCR (Stage 1).
Extracts LineContainers with child Word Tokens.
"""

import os
import re
from typing import List, Optional
import numpy as np
from PIL import Image, ImageDraw
from paddleocr import PaddleOCR
from .config import Token, LineContainer


def get_char_weight(ch: str) -> float:
    """Returns approximate relative pixel width of a character."""
    if ch in " .,:;!|iIl1'\"-`()[]/\t":
        return 0.45
    if ch in "mMWЖШЩ":
        return 1.35
    return 1.0


def split_line_into_word_tokens(
    poly_raw: list,
    text_str: str,
    page_num: int,
    line_idx: int
) -> List[Token]:
    """Splits a line polygon into child Word Tokens with character offset tracking."""
    text_clean = str(text_str).strip()
    if not text_clean or poly_raw is None or len(poly_raw) < 4:
        return []

    try:
        x1, y1 = float(poly_raw[0][0]), float(poly_raw[0][1])
        x2, y2 = float(poly_raw[1][0]), float(poly_raw[1][1])
        x3, y3 = float(poly_raw[2][0]), float(poly_raw[2][1])
        x4, y4 = float(poly_raw[3][0]), float(poly_raw[3][1])
    except Exception:
        return []

    full_weight = sum(get_char_weight(c) for c in text_str)
    if full_weight <= 0:
        return []

    word_matches = list(re.finditer(r'\S+', text_str))
    if not word_matches:
        poly_int = [
            [int(round(x1)), int(round(y1))],
            [int(round(x2)), int(round(y2))],
            [int(round(x3)), int(round(y3))],
            [int(round(x4)), int(round(y4))]
        ]
        return [Token(
            text=text_clean,
            polygon=poly_int,
            page_num=page_num,
            line_idx=line_idx,
            char_start=0,
            char_end=len(text_str)
        )]

    tokens = []
    for match in word_matches:
        word_text = match.group(0)
        c_start = match.start()
        c_end = match.end()

        t_start = sum(get_char_weight(c) for c in text_str[:c_start]) / full_weight
        t_end = sum(get_char_weight(c) for c in text_str[:c_end]) / full_weight

        p1_x = x1 + t_start * (x2 - x1)
        p1_y = y1 + t_start * (y2 - y1)
        p2_x = x1 + t_end * (x2 - x1)
        p2_y = y1 + t_end * (y2 - y1)

        p3_x = x4 + t_end * (x3 - x4)
        p3_y = y4 + t_end * (y3 - y4)
        p4_x = x4 + t_start * (x3 - x4)
        p4_y = y4 + t_start * (y3 - y4)

        w_poly = [
            [int(round(p1_x)), int(round(p1_y))],
            [int(round(p2_x)), int(round(p2_y))],
            [int(round(p3_x)), int(round(p3_y))],
            [int(round(p4_x)), int(round(p4_y))]
        ]
        tokens.append(Token(
            text=word_text,
            polygon=w_poly,
            page_num=page_num,
            line_idx=line_idx,
            char_start=c_start,
            char_end=c_end
        ))

    return tokens


class OCRProcessor:
    def __init__(self):
        print("Initializing PaddleOCR model (lang='ru', without UVDoc warping)...")
        self.ocr = PaddleOCR(
            use_angle_cls=False,
            use_doc_orientation_classify=False,
            use_doc_unwarping=False,
            lang='ru',
            enable_mkldnn=False
        )

    def process_ocr_page(self, image: Image.Image, page_num: int, verbose: bool = False) -> List[LineContainer]:
        """Runs PaddleOCR on a single page image and returns 2-level LineContainers."""
        img_np = np.array(image)
        ocr_result = self.ocr.ocr(img_np)

        if not ocr_result:
            print(f"Warning: OCR found no result on page {page_num}.")
            return []

        res = ocr_result[0]
        if res is None:
            print(f"Warning: OCR found no text on page {page_num}.")
            return []

        lines: List[LineContainer] = []

        def get_field(obj, key):
            if isinstance(obj, dict):
                return obj.get(key)
            elif hasattr(obj, key):
                return getattr(obj, key)
            elif hasattr(obj, '__getitem__'):
                try:
                    return obj[key]
                except (KeyError, TypeError, IndexError):
                    return None
            return None

        # PaddleX Dict / PredictResult format
        polys = get_field(res, "dt_polys")
        if polys is None or (hasattr(polys, "__len__") and len(polys) == 0):
            polys = get_field(res, "rec_polys")
        if polys is None or (hasattr(polys, "__len__") and len(polys) == 0):
            polys = get_field(res, "points")

        texts = get_field(res, "rec_texts")
        if texts is None or (hasattr(texts, "__len__") and len(texts) == 0):
            texts = get_field(res, "rec_text")
        if texts is None or (hasattr(texts, "__len__") and len(texts) == 0):
            texts = get_field(res, "transcription")

        if polys is not None and texts is not None:
            for line_idx, (poly_raw, text) in enumerate(zip(polys, texts), start=1):
                text_str = str(text).strip()
                if not text_str:
                    continue
                try:
                    poly_int = [[int(pt[0]), int(pt[1])] for pt in poly_raw]
                except Exception:
                    continue

                word_tokens = split_line_into_word_tokens(poly_raw, text_str, page_num, line_idx)
                line_obj = LineContainer(
                    text=text_str,
                    polygon=poly_int,
                    page_num=page_num,
                    line_idx=line_idx,
                    words=word_tokens
                )
                lines.append(line_obj)
            return lines

        # PaddleOCR 2.x nested list format
        if isinstance(res, (list, tuple)):
            for line_idx, line in enumerate(res, start=1):
                if not line:
                    continue

                poly_raw = None
                text = ""

                if isinstance(line, dict):
                    poly_raw = line.get("points") if line.get("points") is not None else line.get("dt_polys")
                    text = line.get("transcription") or line.get("rec_text") or line.get("text", "")
                elif isinstance(line, (list, tuple)) and len(line) >= 2:
                    poly_raw = line[0]
                    text_info = line[1]
                    if isinstance(text_info, (list, tuple)):
                        text = text_info[0] if len(text_info) > 0 else ""
                    else:
                        text = str(text_info)

                text_str = str(text).strip()
                if not text_str or poly_raw is None:
                    continue

                try:
                    poly_int = [[int(pt[0]), int(pt[1])] for pt in poly_raw]
                except Exception:
                    continue

                word_tokens = split_line_into_word_tokens(poly_raw, text_str, page_num, line_idx)
                line_obj = LineContainer(
                    text=text_str,
                    polygon=poly_int,
                    page_num=page_num,
                    line_idx=line_idx,
                    words=word_tokens
                )
                lines.append(line_obj)

        return lines

    def save_stage1_debug(self, image: Image.Image, lines: List[LineContainer], page_num: int, debug_dir: str) -> None:
        """Saves Stage 1 debug artifacts (OCR polygons image and plain text dump)."""
        os.makedirs(debug_dir, exist_ok=True)

        # 1. Text dump
        txt_path = os.path.join(debug_dir, f"01_ocr_tokens_page_{page_num}.txt")
        all_words = [w for line in lines for w in line.words]
        with open(txt_path, "w", encoding="utf-8") as f:
            for idx, tok in enumerate(all_words, start=1):
                f.write(f"[{idx:03d}] (Line {tok.line_idx}) {tok.text}\n")

        # 2. OCR Polygons image
        debug_img = image.copy()
        draw = ImageDraw.Draw(debug_img)
        for idx, tok in enumerate(all_words, start=1):
            poly_tuples = [(p[0], p[1]) for p in tok.polygon]
            draw.polygon(poly_tuples, outline="green", width=2)
            min_x = min(p[0] for p in tok.polygon)
            min_y = min(p[1] for p in tok.polygon)
            draw.text((min_x, max(0, min_y - 10)), str(idx), fill="blue")

        debug_img.save(os.path.join(debug_dir, f"01_ocr_raw_page_{page_num}.png"))
        print(f"Saved Stage 1 OCR debug artifacts in '{debug_dir}/'.")

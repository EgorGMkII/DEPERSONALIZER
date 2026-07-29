"""
Masking and Polygon Expansion Module (Stage 3).
Operates on 2-level LineContainers and child Word Tokens.
"""

import os
from typing import List, Tuple
from PIL import Image, ImageDraw
from .config import Token, LineContainer


def expand_polygon(poly: List[List[int]], padding_px: int) -> List[Tuple[int, int]]:
    """Expands polygon vertices outward relative to its centroid by padding_px."""
    if padding_px <= 0 or not poly:
        return [(int(p[0]), int(p[1])) for p in poly]

    cx = sum(p[0] for p in poly) / len(poly)
    cy = sum(p[1] for p in poly) / len(poly)

    expanded = []
    for x, y in poly:
        dx = x - cx
        dy = y - cy
        dist = (dx * dx + dy * dy) ** 0.5
        if dist > 0:
            nx = x + (dx / dist) * padding_px
            ny = y + (dy / dist) * padding_px
        else:
            nx, ny = x, y
        expanded.append((int(round(nx)), int(round(ny))))

    return expanded


class PageMasker:
    def mask_page(self, image: Image.Image, lines: List[LineContainer], padding_px: int = 2) -> Image.Image:
        """Draws black polygons over child Word Tokens marked as PII."""
        masked_img = image.copy()
        draw = ImageDraw.Draw(masked_img)

        all_words = [w for line in lines for w in line.words]
        masked_count = 0

        for tok in all_words:
            if tok.is_pii:
                padded_poly = expand_polygon(tok.polygon, padding_px)
                draw.polygon(padded_poly, fill="black")
                masked_count += 1

        print(f"Masked {masked_count} / {len(all_words)} word tokens on page.")
        return masked_img

    def save_stage3_debug(self, masked_img: Image.Image, page_num: int, debug_dir: str) -> None:
        """Saves Stage 3 debug image."""
        os.makedirs(debug_dir, exist_ok=True)
        masked_img.save(os.path.join(debug_dir, f"03_masked_page_{page_num}.png"))
        print(f"Saved Stage 3 masked page image in '{debug_dir}/'.")

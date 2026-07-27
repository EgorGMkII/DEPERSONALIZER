"""
PII Detection Module using RegEx and Natasha NER (Stage 2).
"""

import json
import os
import re
from dataclasses import asdict
from typing import List, Set
from PIL import Image, ImageDraw
from natasha import (
    Segmenter,
    NewsEmbedding,
    NewsMorphTagger,
    NewsNERTagger,
    Doc
)
from .config import Token, REGEX_PATTERNS

HOMOGLYPH_MAP = str.maketrans({
    'a': 'а', 'c': 'с', 'e': 'е', 'o': 'о', 'p': 'р', 'x': 'х', 'y': 'у',
    'A': 'А', 'B': 'В', 'C': 'С', 'E': 'Е', 'H': 'Н', 'K': 'К', 'M': 'М',
    'O': 'О', 'P': 'Р', 'T': 'Т', 'X': 'Х'
})

EXCLUDED_WORDS = {
    'россия', 'россии', 'российская', 'российской', 'федерация', 'федерации',
    'область', 'области', 'край', 'края', 'район', 'района', 'республика', 'республики',
    'новосибирская', 'новосибирской', 'новосибирск',
    'всоответствии', 'соответствии', 'соответствие', 'статьи', 'закона', 'порядке',
    'рассмотрения', 'обращений', 'обращение', 'граждан', 'отделение', 'фонда',
    'пенсионного', 'социального', 'страхования', 'управление', 'администрация',
    'губернатора', 'правительства', 'приемная', 'общественная', 'действителен', 'действитепен'
}


def normalize_text(text: str) -> str:
    """Converts OCR Latin homoglyph lookalikes back to Cyrillic."""
    return text.translate(HOMOGLYPH_MAP)


class PIIDetector:
    def __init__(self):
        print("Initializing Natasha NER models...")
        self.segmenter = Segmenter()
        self.emb = NewsEmbedding()
        self.morph_tagger = NewsMorphTagger(self.emb)
        self.ner_tagger = NewsNERTagger(self.emb)

    def detect_pii(self, tokens: List[Token]) -> None:
        """Detects PII in page tokens using RegEx matching and Natasha NER entity word matching."""
        if not tokens:
            return

        # 1. Full page text normalized to Cyrillic for Natasha NER & Multi-word RegEx
        raw_words = [tok.text for tok in tokens]
        norm_words = [normalize_text(t) for t in raw_words]
        full_norm_text = " ".join(norm_words)

        # 2. Natasha NER (PER & LOC entities)
        doc = Doc(full_norm_text)
        doc.segment(self.segmenter)
        doc.tag_morph(self.morph_tagger)
        doc.tag_ner(self.ner_tagger)

        ner_words: Set[str] = set()
        for span in doc.spans:
            if span.type == 'PER':
                words = re.findall(r'\w+', span.text.lower())
                for w in words:
                    if len(w) > 1 and w not in EXCLUDED_WORDS:
                        ner_words.add(w)
            elif span.type == 'LOC':
                words = re.findall(r'\w+', span.text.lower())
                for w in words:
                    if len(w) > 1 and w not in EXCLUDED_WORDS:
                        ner_words.add(w)

        # 3. Multi-word RegEx matching on full text spans
        regex_matched_spans = []
        for pattern in REGEX_PATTERNS:
            for match in pattern.finditer(full_norm_text):
                regex_matched_spans.append((match.start(), match.end(), pattern.pattern[:30]))

        # Calculate character start/end offset for each token in full_norm_text
        token_char_spans = []
        curr_pos = 0
        for norm_w in norm_words:
            start_pos = curr_pos
            end_pos = curr_pos + len(norm_w)
            token_char_spans.append((start_pos, end_pos))
            curr_pos = end_pos + 1  # plus space

        # 4. Evaluate each token
        for idx, tok in enumerate(tokens):
            norm_w = norm_words[idx]
            tok_start, tok_end = token_char_spans[idx]
            lower_w = norm_w.lower()
            word_chars = set(re.findall(r'\w+', lower_w))

            is_match = False
            match_reason = ""

            # A. Check multi-word RegEx span overlap
            for r_start, r_end, r_pat in regex_matched_spans:
                if max(tok_start, r_start) < min(tok_end, r_end):
                    is_match = True
                    match_reason = f"RegEx ({r_pat})"
                    break

            # B. Single-token RegEx check fallback
            if not is_match:
                for pattern in REGEX_PATTERNS:
                    if pattern.search(tok.text) or pattern.search(norm_w):
                        is_match = True
                        match_reason = f"RegEx ({pattern.pattern[:30]})"
                        break

            # C. Natasha NER matching (excluding generic words)
            if not is_match and word_chars:
                filtered_chars = {w for w in word_chars if w not in EXCLUDED_WORDS}
                matched = filtered_chars.intersection(ner_words)
                if matched:
                    is_match = True
                    match_reason = f"Natasha NER ({', '.join(matched)})"

            if is_match:
                tok.is_pii = True
                tok.pii_reason = match_reason

    def save_stage2_debug(self, image: Image.Image, tokens: List[Token], page_num: int, debug_dir: str) -> None:
        """Saves Stage 2 debug artifacts (PII JSON dump and highlight image)."""
        os.makedirs(debug_dir, exist_ok=True)

        # 1. Save JSON dump
        json_path = os.path.join(debug_dir, f"02_pii_tokens_page_{page_num}.json")
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump([asdict(tok) for tok in tokens], f, ensure_ascii=False, indent=2)

        # 2. Save PII highlight debug image
        pii_debug_img = image.copy()
        draw = ImageDraw.Draw(pii_debug_img)
        for tok in tokens:
            poly_tuples = [(p[0], p[1]) for p in tok.polygon]
            if tok.is_pii:
                draw.polygon(poly_tuples, outline="red", width=3)
                min_x = min(p[0] for p in tok.polygon)
                min_y = min(p[1] for p in tok.polygon)
                draw.text((min_x, max(0, min_y - 12)), tok.pii_reason or "PII", fill="red")
            else:
                draw.polygon(poly_tuples, outline="green", width=1)

        pii_debug_img.save(os.path.join(debug_dir, f"02_pii_highlight_page_{page_num}.png"))
        print(f"Saved Stage 2 PII debug artifacts in '{debug_dir}/'.")

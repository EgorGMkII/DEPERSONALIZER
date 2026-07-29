"""
PII Detection Module using RegEx and Natasha NER (Stage 2).
Contextual 2-Level Detection (Line-level analysis -> Child Word Token flagging).
Includes POS protection and Header Institution Safeguards.
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
from .config import Token, LineContainer, REGEX_PATTERNS

HOMOGLYPH_MAP = str.maketrans({
    'a': 'а', 'c': 'с', 'e': 'е', 'o': 'о', 'p': 'р', 'x': 'х', 'y': 'у',
    'A': 'А', 'B': 'В', 'C': 'С', 'E': 'Е', 'H': 'Н', 'K': 'К', 'M': 'М',
    'O': 'О', 'P': 'Р', 'T': 'Т', 'X': 'Х'
})

# Words that identify official state headers/institutions
INSTITUTION_HEADER_KEYWORDS = {
    'приемная', 'приемкая', 'общественная', 'управление', 'отделение', 'фонд',
    'фонда', 'правительство', 'правительства', 'администрация', 'губернатор',
    'губернатора', 'губернау', 'департамент', 'министерство', 'социального',
    'пенсионного', 'страхования', 'социальный'
}

# Generic geography & administrative terms that should NEVER be marked as PII
EXCLUDED_GENERIC_WORDS = {
    'россия', 'россии', 'российская', 'российской', 'федерация', 'федерации',
    'область', 'области', 'край', 'края', 'район', 'района', 'республика', 'республики',
    'новосибирская', 'новосибирской', 'новосибирск', 'нсо', 'рф', 'обл', 'обл.',
    'всоответствии', 'соответствии', 'соответствие', 'статьи', 'закона', 'порядке',
    'рассмотрения', 'обращений', 'обращение', 'граждан', 'отделение', 'фонда',
    'пенсионного', 'социального', 'страхования', 'управление', 'администрация',
    'губернатора', 'правительства', 'приемная', 'общественная', 'действителен', 'действитепен',
    'поступившее', 'направляем', 'просим', 'проинформировать', 'автора', 'субъект'
}


def normalize_text(text: str) -> str:
    """Converts OCR Latin homoglyph lookalikes back to Cyrillic."""
    return text.translate(HOMOGLYPH_MAP)


def is_hex_hash(text: str) -> bool:
    """Detects if a string is a long hex certificate hash (e.g. 00F551A9931C...)."""
    clean = text.strip()
    if len(clean) >= 20 and re.fullmatch(r'[0-9A-Fa-f]+', clean):
        return True
    return False


def is_institution_header(norm_text: str) -> bool:
    """Checks if a line contains official institution/header terms."""
    lower = norm_text.lower()
    return any(kw in lower for kw in INSTITUTION_HEADER_KEYWORDS)


class PIIDetector:
    def __init__(self):
        print("Initializing Natasha NER models...")
        self.segmenter = Segmenter()
        self.emb = NewsEmbedding()
        self.morph_tagger = NewsMorphTagger(self.emb)
        self.ner_tagger = NewsNERTagger(self.emb)

    def detect_pii(self, lines: List[LineContainer]) -> None:
        """Detects PII on full sentence/line context and flags child Word Tokens."""
        if not lines:
            return

        for i, line in enumerate(lines):
            if not line.words:
                continue

            raw_line_text = line.text
            norm_line_text = normalize_text(raw_line_text)
            line_is_header = is_institution_header(norm_line_text)

            # Structured Postal Address detection (e.g. "Почтовый адрес: 6301 16, ул Космонавтов...")
            if re.search(r'почтовый\s+адрес', norm_line_text, re.IGNORECASE):
                found_colon = False
                for w_tok in line.words:
                    if 'адрес' in normalize_text(w_tok.text).lower() or ':' in w_tok.text:
                        found_colon = True
                        continue
                    if found_colon:
                        w_tok.is_pii = True
                        w_tok.pii_reason = "Structured PII (Почтовый адрес)"
                if i + 1 < len(lines) and lines[i + 1].words:
                    next_line = lines[i + 1]
                    if not any(k in normalize_text(next_line.text).lower() for k in ['e-mail', 'телефон', 'текст']):
                        for w_tok in next_line.words:
                            w_tok.is_pii = True
                            w_tok.pii_reason = "Structured PII (Почтовый адрес)"

            # 1. Natasha NER on full line context
            doc = Doc(norm_line_text)
            doc.segment(self.segmenter)
            doc.tag_morph(self.morph_tagger)
            doc.tag_ner(self.ner_tagger)

            entity_spans = []

            # Suppress Natasha NER triggers on official institution headers
            if not line_is_header:
                for span in doc.spans:
                    if span.type in ('PER', 'LOC'):
                        span_words = re.findall(r'\w+', span.text.lower())
                        valid_words = [w for w in span_words if len(w) > 1 and w not in EXCLUDED_GENERIC_WORDS and not any(kw in w for kw in INSTITUTION_HEADER_KEYWORDS)]
                        if valid_words:
                            entity_spans.append((span.start, span.stop, f"Natasha NER ({span.type}: {', '.join(valid_words)})"))

            # 2. RegEx patterns on full line text
            regex_spans = []
            for pattern in REGEX_PATTERNS:
                for match in pattern.finditer(raw_line_text):
                    regex_spans.append((match.start(), match.end(), f"RegEx ({pattern.pattern[:30]})"))
                for match in pattern.finditer(norm_line_text):
                    regex_spans.append((match.start(), match.end(), f"RegEx ({pattern.pattern[:30]})"))

            all_matched_spans = entity_spans + regex_spans

            # Map POS tags to tokens for verb/preposition protection
            pos_tags = {token.text.lower(): token.pos for token in doc.tokens}

            # 3. Map line-level matched spans directly to child Word Tokens
            for word_tok in line.words:
                if word_tok.is_pii:
                    continue  # Already flagged by structured rules

                # Skip long hex certificate hashes
                if is_hex_hash(word_tok.text):
                    continue

                w_start = word_tok.char_start
                w_end = word_tok.char_end
                clean_w = normalize_text(word_tok.text).lower().strip('.,():;')

                # Never mark generic meaning or generic geography words as PII
                if clean_w in EXCLUDED_GENERIC_WORDS or any(kw in clean_w for kw in INSTITUTION_HEADER_KEYWORDS):
                    continue

                # Protect Verbs, Prepositions, Adverbs from accidental RegEx over-matching unless part of PER
                pos = pos_tags.get(clean_w, '')
                if pos in ('VERB', 'ADP', 'CCONJ', 'SCONJ', 'ADV') and clean_w not in EXCLUDED_GENERIC_WORDS:
                    # Check if token is matched by Natasha PER (Person)
                    is_per = any('PER' in r for _, _, r in entity_spans)
                    if not is_per:
                        continue

                is_match = False
                match_reason = ""

                for span_start, span_end, reason in all_matched_spans:
                    if max(w_start, span_start) < min(w_end, span_end):
                        is_match = True
                        match_reason = reason
                        break

                if is_match:
                    word_tok.is_pii = True
                    word_tok.pii_reason = match_reason

    def save_stage2_debug(self, image: Image.Image, lines: List[LineContainer], page_num: int, debug_dir: str) -> None:
        """Saves Stage 2 debug artifacts (PII JSON dump and highlight image)."""
        os.makedirs(debug_dir, exist_ok=True)

        all_words = [w for line in lines for w in line.words]

        # 1. Save JSON dump
        json_path = os.path.join(debug_dir, f"02_pii_tokens_page_{page_num}.json")
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump([asdict(w) for w in all_words], f, ensure_ascii=False, indent=2)

        # 2. Save PII highlight debug image
        pii_debug_img = image.copy()
        draw = ImageDraw.Draw(pii_debug_img)
        for tok in all_words:
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

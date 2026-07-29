"""
Configuration, Data Containers, and RegEx Patterns for Depersonalizer.
"""

import re
from dataclasses import dataclass, field
from typing import List, Optional


@dataclass
class Token:
    text: str
    polygon: List[List[int]]  # [[x1, y1], [x2, y2], [x3, y3], [x4, y4]]
    page_num: int
    line_idx: int = 0
    char_start: int = 0
    char_end: int = 0
    confidence: float = 1.0
    is_pii: bool = False
    pii_reason: Optional[str] = None


@dataclass
class LineContainer:
    text: str
    polygon: List[List[int]]
    page_num: int
    line_idx: int
    confidence: float = 1.0
    words: List[Token] = field(default_factory=list)


# Pre-compiled RegEx patterns for Russian PII detection
REGEX_PATTERNS = [
    # IPv4 Address (e.g. 5.130.32.223 or 192.168.1.1)
    re.compile(r'\b\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}\b'),
    # RF Passport: 4 digits series + 6 digits number (e.g., "45 10 123456" or "4510 123456")
    re.compile(r'\b\d{2}\s?\d{2}\s?\d{6}\b'),
    # Phone numbers (+7/8 11 digits or local 7-digit numbers XXX-XX-XX)
    re.compile(r'(?:\+7|8)[\s\-\(]*\d{3}[\s\-\)]*\d{3}[\s\-]*\d{2}[\s\-]*\d{2}|\b\d{3}[\-\s]\d{2}[\-\s]\d{2}\b'),
    # SNILS (11 digits: XXX-XXX-XXX XX or XXXXXXXXXXX)
    re.compile(r'\b\d{3}[\-\s]?\d{3}[\-\s]?\d{3}[\-\s]?\d{2}\b'),
    # INN / KPP / OGRN / OKPO (9-15 digit numeric sequences)
    re.compile(r'\b\d{9,15}\b|(?<=/)\d{9,12}(?=/)|(?<=[А-Яа-яA-Za-z/])\d{9,12}'),
    # Email
    re.compile(r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b'),
    # Dates (including glued prefixes like "по15.052027" or "19.022026")
    re.compile(r'(?:с|по)?\b\d{2}[\.\/]?\d{2}[\.\/]?\d{4}\b|\b\d{2}[\.\/]\d{2}[\.\/]\d{2}\b'),
    # Russian/Latin Initials (e.g. "Г.В. Козина", "A.A", "D.U.") - 1 or 2 dots supported
    re.compile(r'\b[А-ЯЁA-Z]\.\s?[А-ЯЁA-Z]\.?[\)\.\,]*\s?[А-ЯЁA-Z][а-яёA-Za-z\-]+\b'),
    # Surname + Initials (e.g., "Некрасову М.В." or "(Некрасов М.В.).")
    re.compile(r'\b[А-ЯЁA-Z][а-яёA-Za-z\-]+\s+[А-ЯЁA-Z]\.\s?[А-ЯЁA-Z]\.?[\)\.\,]*'),
    # Standalone Initials (1 or 2 dots, Cyrillic/Latin, e.g. "A.A", "М.В.")
    re.compile(r'\b[А-ЯЁA-Z]\.\s?[А-ЯЁA-Z]\.?[\)\.\,]*|\b[А-ЯЁA-Z][А-ЯЁA-Z]\.[\)\.\,]*'),
]

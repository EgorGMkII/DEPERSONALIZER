"""
Configuration, Data Containers, and RegEx Patterns for Depersonalizer.
"""

import re
from dataclasses import dataclass
from typing import List, Optional


@dataclass
class Token:
    text: str
    polygon: List[List[int]]  # [[x1, y1], [x2, y2], [x3, y3], [x4, y4]]
    page_num: int
    is_pii: bool = False
    pii_reason: Optional[str] = None


# Pre-compiled RegEx patterns for Russian PII detection
REGEX_PATTERNS = [
    # RF Passport: 4 digits series + 6 digits number (e.g., "45 10 123456" or "4510 123456")
    re.compile(r'\b\d{2}\s?\d{2}\s?\d{6}\b'),
    # Phone numbers (+7/8 with various separators)
    re.compile(r'(?:\+7|8)[\s\-\(]*\d{3}[\s\-\)]*\d{3}[\s\-]*\d{2}[\s\-]*\d{2}'),
    # SNILS (11 digits: XXX-XXX-XXX XX or XXXXXXXXXXX)
    re.compile(r'\b\d{3}[\-\s]?\d{3}[\-\s]?\d{3}[\-\s]?\d{2}\b'),
    # INN / KPP / OGRN / OKPO (9-15 digit sequences or codes embedded after slashes/letters)
    re.compile(r'\b\d{9,15}\b|(?<=/)\d{9,12}(?=/)|(?<=[А-Яа-яA-Za-z/])\d{9,12}'),
    # Email
    re.compile(r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b'),
    # Dates (DD.MM.YYYY or DD.MM.YY)
    re.compile(r'\b\d{2}[\.\/]\d{2}[\.\/]\d{4}\b|\b\d{2}[\.\/]\d{2}[\.\/]\d{2}\b'),
    # Russian Initials + Surname (e.g., "Г.В. Козина" or "Г. В. Козина")
    re.compile(r'\b[А-ЯЁA-Z]\.\s?[А-ЯЁA-Z]\.\s?[А-ЯЁа-яёA-Za-z\-]+\b', re.IGNORECASE),
    # Surname + Russian Initials (e.g., "Некрасову М.В." or "Козина Г. В.")
    re.compile(r'\b[А-ЯЁа-яёA-Za-z\-]+\s+[А-ЯЁA-Z]\.\s?[А-ЯЁA-Z]\.\b', re.IGNORECASE),
]

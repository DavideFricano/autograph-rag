from __future__ import annotations

import re


class Cleaner:
    """Normalizes raw text before chunking or embedding."""

    @staticmethod
    def make_plain(text: str) -> str:
        """Collapses all whitespace sequences to a single space."""
        text = re.sub(r"\s+", " ", text)
        return text.strip()

"""
src/optional/keyword_spotter.py
=================================
Arabic + multilingual keyword spotting in transcripts (FIXED VERSION)
"""

import re
import logging
from typing import List, Dict, Optional, Set, Tuple
from dataclasses import dataclass

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────
# Arabic normalization
# ─────────────────────────────────────────────
def normalize_arabic(text: str) -> str:
    """Normalize Arabic text for robust matching."""

    if not text:
        return ""

    # Remove diacritics + tatweel
    text = re.sub(r"[\u064b-\u065f\u0670\u0640]", "", text)

    # Normalize Alef variants
    text = re.sub(r"[أإآٱ]", "ا", text)

    # Normalize Teh Marbuta → Heh
    text = text.replace("ة", "ه")

    # Normalize Hamza forms
    text = text.replace("ؤ", "و").replace("ئ", "ي")

    # Normalize Alef Maqsura
    text = text.replace("ى", "ي")

    # Clean punctuation (keep Arabic/English letters + spaces)
    text = re.sub(r"[^\w\s\u0600-\u06FF]", " ", text)

    # Normalize spaces
    text = re.sub(r"\s+", " ", text)

    return text.strip().lower()


# ─────────────────────────────────────────────
# Default keywords
# ─────────────────────────────────────────────
DEFAULT_KEYWORDS = [
    "طوارئ", "موعد نهائي", "امتحان", "اجتماع", "مهم", "ضروري",
    "عاجل", "تحذير", "خطر", "نتيجة", "قرار", "مشكلة", "حل",
    "تقرير", "مشروع", "ميزانية", "إلغاء", "تأجيل",
    "emergency", "deadline", "exam", "meeting", "important",
    "urgent", "cancel", "postpone", "budget", "project",
]


# ─────────────────────────────────────────────
# Data structure
# ─────────────────────────────────────────────
@dataclass
class KeywordMatch:
    keyword: str
    matched_text: str
    match_type: str
    confidence: float
    segment_index: int
    start_time: float
    end_time: float

    def to_dict(self) -> Dict:
        return {
            "keyword": self.keyword,
            "matched_text": self.matched_text,
            "match_type": self.match_type,
            "confidence": round(self.confidence, 3),
            "start_time": round(self.start_time, 2),
            "end_time": round(self.end_time, 2),
        }


# ─────────────────────────────────────────────
# Main Spotter
# ─────────────────────────────────────────────
class KeywordSpotter:

    def __init__(
        self,
        keywords: Optional[List[str]] = None,
        fuzzy_threshold: float = 0.80,
        semantic_threshold: float = 0.72,
        embedder=None,
    ):
        self.keywords = keywords or DEFAULT_KEYWORDS
        self.fuzzy_threshold = fuzzy_threshold
        self.semantic_threshold = semantic_threshold
        self.embedder = embedder

        self._norm_kws = {
            kw: normalize_arabic(kw) for kw in self.keywords
        }

        self._kw_embeddings = None
        if self.embedder:
            self._precompute_embeddings()

    def _precompute_embeddings(self):
        try:
            self._kw_embeddings = self.embedder.embed(self.keywords)
        except Exception as e:
            logger.warning(f"Embedding failed: {e}")
            self._kw_embeddings = None

    def spot(
        self,
        segments: List[Dict],
        methods: List[str] = None,
    ) -> List[KeywordMatch]:

        if methods is None:
            methods = ["exact", "fuzzy"]

        matches: List[KeywordMatch] = []

        for i, seg in enumerate(segments):
            text = seg.get("text", "")
            if not text:
                continue

            norm_text = normalize_arabic(text)

            start = float(seg.get("start", 0.0))
            end = float(seg.get("end", 0.0))

            exact_found: Set[str] = set()

            # ── EXACT ─────────────────────────────
            if "exact" in methods:
                for kw, norm_kw in self._norm_kws.items():
                    if norm_kw and norm_kw in norm_text:
                        matches.append(KeywordMatch(
                            keyword=kw,
                            matched_text=text,
                            match_type="exact",
                            confidence=1.0,
                            segment_index=i,
                            start_time=start,
                            end_time=end,
                        ))
                        exact_found.add(kw)

            # ── FUZZY ─────────────────────────────
            if "fuzzy" in methods:
                from difflib import SequenceMatcher

                words = norm_text.split()

                for kw, norm_kw in self._norm_kws.items():
                    if kw in exact_found:
                        continue

                    kw_words = norm_kw.split()
                    if not kw_words:
                        continue

                    k = len(kw_words)

                    for j in range(len(words) - k + 1):
                        window = " ".join(words[j:j + k])
                        score = SequenceMatcher(None, norm_kw, window).ratio()

                        if self.fuzzy_threshold <= score < 1.0:
                            matches.append(KeywordMatch(
                                keyword=kw,
                                matched_text=window,
                                match_type="fuzzy",
                                confidence=score,
                                segment_index=i,
                                start_time=start,
                                end_time=end,
                            ))
                            break

            # ── SEMANTIC ─────────────────────────
            if (
                "semantic" in methods
                and self.embedder
                and self._kw_embeddings is not None
            ):
                try:
                    import numpy as np

                    seg_emb = self.embedder.embed([text])[0]
                    sims = np.dot(self._kw_embeddings, seg_emb)

                    for idx, sim in enumerate(sims):
                        if float(sim) >= self.semantic_threshold:
                            matches.append(KeywordMatch(
                                keyword=self.keywords[idx],
                                matched_text=text[:80],
                                match_type="semantic",
                                confidence=float(sim),
                                segment_index=i,
                                start_time=start,
                                end_time=end,
                            ))
                except Exception as e:
                    logger.warning(f"Semantic matching failed: {e}")

        # ── DEDUP ─────────────────────────────
        dedup: Dict[Tuple[str, int], KeywordMatch] = {}

        for m in matches:
            key = (m.keyword, m.segment_index)
            if key not in dedup or m.confidence > dedup[key].confidence:
                dedup[key] = m

        return sorted(dedup.values(), key=lambda x: x.start_time)

    def timeline(self, matches: List[KeywordMatch]) -> Dict[str, List[float]]:
        result: Dict[str, List[float]] = {}
        for m in matches:
            result.setdefault(m.keyword, []).append(m.start_time)
        return result

    def format_matches(self, matches: List[KeywordMatch]) -> str:
        if not matches:
            return "No keywords detected."

        lines = [
            "| Keyword | Time | Type | Confidence |",
            "|---------|------|------|------------|"
        ]

        for m in matches:
            lines.append(
                f"| {m.keyword} | {m.start_time:.1f}s | {m.match_type} | {m.confidence:.2f} |"
            )

        return "\n".join(lines)

    def highlight_transcript(self, text: str, matches: List[KeywordMatch]) -> str:
        if not text or not matches:
            return text

        result = text
        for kw in sorted({m.keyword for m in matches}, key=len, reverse=True):
            result = result.replace(kw, f"**{kw}**")
        return result


# ─────────────────────────────────────────────
# BACKWARD COMPATIBILITY (IMPORTANT FIX)
# ─────────────────────────────────────────────
ArabicKeywordSpotter = KeywordSpotter
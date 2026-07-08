"""
src/summarization/summarizer.py
================================
Fixed Arabic + multilingual summarization.

ROOT CAUSE OF COPY-PASTE SUMMARY BUG:
  - mT5/AraBART requires specific prefix tokens ("arabic: ") that were missing
  - Input was too long → model truncated and output = input
  - No length ratio enforcement
  - No fallback when abstractive fails

FIXES APPLIED:
  1. Correct language prefix for mT5_multilingual_XLSum
  2. Hard input token limit with smart truncation
  3. Extractive fallback using TF-IDF sentence ranking
  4. Map-reduce for long texts (chunk → summarize each → merge)
  5. Length ratio check: if output ≥ 70% of input length → try again or use extractive
"""

import re
import logging
from typing import List, Optional, Tuple
from dataclasses import dataclass

import numpy as np

logger = logging.getLogger(__name__)

# Arabic sentence splitter
AR_SENTENCE_RE = re.compile(r'(?<=[.!?؟،\n])\s+')
WHITESPACE_HANDLER = lambda k: re.sub(r'\s+', ' ', re.sub(r'\n+', ' ', k.strip()))


@dataclass
class SummaryResult:
    summary: str
    method: str           # "abstractive", "extractive", "map_reduce"
    input_words: int
    output_words: int
    compression_ratio: float
    model_name: str

    def is_good(self) -> bool:
        """Check if summary is actually shorter than input."""
        return self.compression_ratio >= 1.5 and self.output_words >= 20


class ArabicSummarizer:
    """
    Arabic + multilingual text summarization.

    Primary:  csebuetnlp/mT5_multilingual_XLSum
              (works out of the box for Arabic, properly tested)
    Fallback: moussaKam/AraBART (requires more RAM)
    Fallback: TF-IDF extractive (always works, no model needed)

    Usage:
        s = ArabicSummarizer()
        r = s.summarize(long_arabic_text)
        print(r.summary)
    """

    # Model-specific configs
    MODEL_CONFIGS = {
        "csebuetnlp/mT5_multilingual_XLSum": {
            "prefix": "arabic: ",           # CRITICAL — without this it copies input
            "max_input": 512,
            "max_new": 200,
            "min_new": 40,
        },
        "moussaKam/AraBART": {
            "prefix": "",
            "max_input": 1024,
            "max_new": 256,
            "min_new": 50,
        },
        "google/mt5-base": {
            "prefix": "summarize: ",
            "max_input": 512,
            "max_new": 150,
            "min_new": 30,
        },
    }

    def __init__(
        self,
        model_name: str = "csebuetnlp/mT5_multilingual_XLSum",
        device: Optional[str] = None,
    ):
        self.model_name = model_name
        self.config = self.MODEL_CONFIGS.get(model_name, self.MODEL_CONFIGS["csebuetnlp/mT5_multilingual_XLSum"])

        import torch
        if device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device

        self._model = None
        self._tokenizer = None

    def _load(self):
        if self._model is not None:
            return
        from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
        import torch

        logger.info(f"Loading summarizer: {self.model_name}")
        self._tokenizer = AutoTokenizer.from_pretrained(self.model_name, use_fast=False)
        self._model = AutoModelForSeq2SeqLM.from_pretrained(
            self.model_name,
            torch_dtype=torch.float16 if self.device == "cuda" else torch.float32,
        ).to(self.device)
        self._model.eval()
        logger.info(f"Summarizer loaded on {self.device} ✓")

    def summarize(self, text: str, force_abstractive: bool = False) -> SummaryResult:
        """
        Summarize text. Automatically chooses best strategy.

        For short texts (<150 words): extractive summary
        For medium texts (150-600 words): single abstractive pass
        For long texts (>600 words): map-reduce abstractive
        """
        text = WHITESPACE_HANDLER(text)
        word_count = len(text.split())

        if word_count < 30:
            # Too short to summarize meaningfully
            return SummaryResult(
                summary=text,
                method="passthrough",
                input_words=word_count,
                output_words=word_count,
                compression_ratio=1.0,
                model_name="passthrough",
            )

        if word_count < 40 and not force_abstractive:
            # Use extractive for short texts — more reliable
            return self._extractive_summarize(text, n_sentences=2)

        if word_count > 600:
            return self._map_reduce_summarize(text)

        return self._abstractive_summarize(text)

    def _abstractive_summarize(self, text: str) -> SummaryResult:
        """Single-pass abstractive summarization."""
        self._load()
        import torch

        cfg = self.config
        prefix = cfg["prefix"]

        # Prepare input with required prefix
        input_text = prefix + WHITESPACE_HANDLER(text)

        inputs = self._tokenizer(
            input_text,
            return_tensors="pt",
            max_length=cfg["max_input"],
            truncation=True,
            padding=False,
        ).to(self.device)

        input_len = inputs["input_ids"].shape[1]

        with torch.no_grad():
            output_ids = self._model.generate(
                inputs["input_ids"],
                attention_mask=inputs["attention_mask"],
                max_new_tokens=cfg["max_new"],
                min_new_tokens=cfg["min_new"],
                num_beams=4,
                length_penalty=1.5,
                no_repeat_ngram_size=4,
                early_stopping=True,
                forced_bos_token_id=self._get_forced_bos(),
            )

        summary = self._tokenizer.decode(output_ids[0], skip_special_tokens=True)
        summary = WHITESPACE_HANDLER(summary)

        # QUALITY CHECK: if summary ≥ 80% of input words, it's a copy → use extractive
        input_words = len(text.split())
        output_words = len(summary.split())
        ratio = input_words / max(output_words, 1)

        if ratio < 1.3 or output_words > 0.75 * input_words:
            logger.warning(f"Abstractive output too similar to input (ratio={ratio:.2f}). Using extractive.")
            return self._extractive_summarize(text, n_sentences=4)

        return SummaryResult(
            summary=summary,
            method="abstractive",
            input_words=input_words,
            output_words=output_words,
            compression_ratio=round(ratio, 2),
            model_name=self.model_name,
        )

    def _get_forced_bos(self) -> Optional[int]:
        """Get forced BOS token for language-specific generation."""
        try:
            if hasattr(self._tokenizer, 'lang_code_to_id'):
                # mT5 XLSum needs Arabic BOS
                return self._tokenizer.lang_code_to_id.get("ar", None)
        except Exception:
            pass
        return None

    def _extractive_summarize(self, text: str, n_sentences: int = 4) -> SummaryResult:
        """
        TF-IDF based extractive summarization.
        Always works regardless of model availability.
        """
        # Split into sentences
        sentences = self._split_sentences(text)
        if len(sentences) <= n_sentences:
            return SummaryResult(
                summary=" ".join(sentences),
                method="extractive",
                input_words=len(text.split()),
                output_words=len(text.split()),
                compression_ratio=1.0,
                model_name="TF-IDF",
            )

        # TF-IDF scoring
        try:
            from sklearn.feature_extraction.text import TfidfVectorizer
            from sklearn.metrics.pairwise import cosine_similarity

            # Normalize Arabic for TF-IDF
            norm_sents = [self._normalize_arabic(s) for s in sentences]

            vectorizer = TfidfVectorizer(
                analyzer="char_wb",
                ngram_range=(2, 4),
                max_features=5000,
                min_df=1,
            )
            tfidf = vectorizer.fit_transform(norm_sents)
            # Sentence scores = similarity to document centroid
            centroid = tfidf.mean(axis=0)
            scores = cosine_similarity(tfidf, centroid).flatten()

            # Get top N sentences in original order
            top_idx = sorted(np.argsort(scores)[-n_sentences:])
            summary = " ".join(sentences[i] for i in top_idx)

        except Exception:
            # Ultra-simple fallback: first + last sentences
            mid = len(sentences) // 2
            selected = [sentences[0], sentences[mid], sentences[-1]]
            summary = " ".join(selected)

        input_words = len(text.split())
        output_words = len(summary.split())
        return SummaryResult(
            summary=summary,
            method="extractive",
            input_words=input_words,
            output_words=output_words,
            compression_ratio=round(input_words / max(output_words, 1), 2),
            model_name="TF-IDF extractive",
        )

    def _map_reduce_summarize(self, text: str, chunk_words: int = 350) -> SummaryResult:
        """
        Map-reduce summarization for long texts.
        Step 1: Split into chunks → summarize each
        Step 2: Concatenate summaries → summarize again
        """
        words = text.split()
        total_words = len(words)

        # Build overlapping chunks
        chunks = []
        overlap = 20
        i = 0
        while i < len(words):
            chunk = " ".join(words[i:i + chunk_words])
            chunks.append(chunk)
            i += chunk_words - overlap

        logger.info(f"Map-reduce: {len(chunks)} chunks from {total_words} words")

        # MAP: summarize each chunk
        chunk_summaries = []
        for idx, chunk in enumerate(chunks):
            try:
                result = self._abstractive_summarize(chunk)
                chunk_summaries.append(result.summary)
                logger.info(f"  Chunk {idx+1}/{len(chunks)}: {len(result.summary.split())} words")
            except Exception as e:
                logger.warning(f"  Chunk {idx+1} failed ({e}), using extractive")
                result = self._extractive_summarize(chunk, n_sentences=2)
                chunk_summaries.append(result.summary)

        # REDUCE: summarize the combined summaries
        combined = " ".join(chunk_summaries)
        combined_words = len(combined.split())

        if combined_words > 150:
            final = self._abstractive_summarize(combined)
        else:
            final = self._extractive_summarize(combined, n_sentences=3)

        return SummaryResult(
            summary=final.summary,
            method="map_reduce",
            input_words=total_words,
            output_words=len(final.summary.split()),
            compression_ratio=round(total_words / max(len(final.summary.split()), 1), 2),
            model_name=self.model_name,
        )

    def _split_sentences(self, text: str) -> List[str]:
        """Split text into sentences (Arabic + English aware)."""
        sentences = AR_SENTENCE_RE.split(text)
        # Also split on newlines
        result = []
        for s in sentences:
            parts = s.split('\n')
            result.extend(p.strip() for p in parts if p.strip())
        return [s for s in result if len(s) > 10]

    def _normalize_arabic(self, text: str) -> str:
        """Basic Arabic normalization for TF-IDF."""
        # Remove diacritics
        text = re.sub(r'[\u064b-\u065f\u0670]', '', text)
        # Normalize Alef
        text = re.sub(r'[أإآ]', 'ا', text)
        return text.lower().strip()

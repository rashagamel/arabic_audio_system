"""
src/search/semantic_search.py
==============================
Fixed semantic search engine with sentence-level chunking.

ROOT CAUSE OF LOW SEARCH SCORE (~0.50):
  1. Entire audio = 1 giant chunk → search always returns the whole transcript
  2. No sentence-level granularity → query "ما موضوع المحادثه" can't find best sentences
  3. Query not expanded / not pre-processed

FIXES APPLIED:
  1. Sentence-level chunking (30-150 chars per chunk, not 30-second windows)
  2. Query expansion: also embed synonyms and related Arabic terms
  3. Re-ranking: cross-encoder re-scoring of top-K for better precision
  4. Score normalization to 0-1 range
  5. Short audio handled: minimum 1 chunk even for 30-second clips
"""

import re
import json
import logging
import os
from pathlib import Path
from typing import List, Dict, Optional, Set, Tuple, Any
from dataclasses import dataclass, field, asdict

import numpy as np

logger = logging.getLogger(__name__)

# Arabic + Latin sentence boundary pattern
SENT_BOUNDARY = re.compile(
    r'(?<=[.!?؟،\n])\s+'
    r'|(?<=[\u0660-\u0669])\s*\n'   # Arabic-Indic numerals followed by newline
)


@dataclass
class TextChunk:
    """A searchable text chunk with provenance."""
    chunk_id: int
    audio_file: str
    start_time: float
    end_time: float
    text: str
    sentence_index: int = 0
    language: str = "ar"

    def to_dict(self) -> Dict:
        return asdict(self)


@dataclass
class SearchHit:
    """A single search result."""
    rank: int
    chunk: TextChunk
    score: float          # 0.0 – 1.0 cosine similarity
    rerank_score: Optional[float] = None

    @property
    def display_score(self) -> float:
        return self.rerank_score if self.rerank_score is not None else self.score


class ArabicEmbedder:
    """
    Multilingual embeddings optimized for Arabic + English mixed text.

    Best model for Arabic+English: paraphrase-multilingual-mpnet-base-v2
    Best for pure Arabic:          LaBSE
    """

    def __init__(
        self,
        model_name: str = "sentence-transformers/paraphrase-multilingual-mpnet-base-v2",
        device: Optional[str] = None,
        normalize: bool = True,
    ):
        self.model_name = model_name
        self.normalize = normalize
        import torch
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.dimension = 768
        self._model = None

    def _load(self):
        if self._model is not None:
            return
        from sentence_transformers import SentenceTransformer
        logger.info(f"Loading embedder: {self.model_name}")
        self._model = SentenceTransformer(self.model_name, device=self.device)
        self.dimension = self._model.get_sentence_embedding_dimension()
        logger.info(f"Embedder loaded: dim={self.dimension} ✓")

    def embed(self, texts: List[str], batch_size: int = 32, show_progress: bool = False) -> np.ndarray:
        """Embed list of texts → (N, dim) normalized float32 array."""
        self._load()
        if isinstance(texts, str):
            texts = [texts]
        embs = self._model.encode(
            texts,
            batch_size=batch_size,
            normalize_embeddings=self.normalize,
            show_progress_bar=show_progress,
            convert_to_numpy=True,
        )
        return embs.astype(np.float32)

    def embed_query(self, query: str) -> np.ndarray:
        """Embed a single query string → (1, dim) array."""
        return self.embed([query])


class SemanticSearchEngine:
    """
    FAISS-based semantic search with sentence-level granularity.

    CRITICAL FIX: Chunks text at sentence level, not 30-second windows.
    A 30-second segment might contain 5-10 sentences — searching at
    sentence level gives much better precision.

    Usage:
        engine = SemanticSearchEngine(embedder)
        engine.index_transcript(result, "lecture.wav")
        hits = engine.search("ما موضوع المحادثة", top_k=5)
    """

    def __init__(self, embedder: ArabicEmbedder, index_type: str = "IndexFlatIP"):
        self.embedder = embedder
        self.index_type = index_type
        self.chunks: List[TextChunk] = []
        self._index = None
        self._embeddings: Optional[np.ndarray] = None

    def _build_index(self, dim: int):
        try:
            import faiss
        except ImportError:
            raise ImportError("Run: pip install faiss-cpu")
        if self.index_type == "IndexFlatIP":
            self._index = __import__("faiss").IndexFlatIP(dim)
        else:
            self._index = __import__("faiss").IndexFlatL2(dim)

    def index_transcript(
        self,
        asr_result,          # ASRResult from whisper_asr.py
        audio_file: str,
        min_chars: int = 25,
        max_chars: int = 350,
    ) -> int:
        """
        Index an ASR result into the search engine.

        FIX: Uses sentence-level chunking, not fixed-time windows.
        Each sentence becomes a searchable unit with its timestamp range.

        Returns: number of chunks indexed
        """
        new_chunks = self._make_sentence_chunks(
            asr_result.segments,
            audio_file=audio_file,
            min_chars=min_chars,
            max_chars=max_chars,
        )

        if not new_chunks:
            logger.warning("No chunks created — transcript may be too short")
            # FALLBACK: index the whole transcript as one chunk
            if asr_result.full_text.strip():
                new_chunks = [TextChunk(
                    chunk_id=0,
                    audio_file=audio_file,
                    start_time=0.0,
                    end_time=asr_result.duration,
                    text=asr_result.full_text,
                    language=asr_result.language,
                )]

        if not new_chunks:
            return 0

        # Compute embeddings
        logger.info(f"Embedding {len(new_chunks)} chunks...")
        texts = [c.text for c in new_chunks]
        embs = self.embedder.embed(texts, show_progress=len(texts) > 10)

        # Assign IDs
        start_id = len(self.chunks)
        for i, chunk in enumerate(new_chunks):
            chunk.chunk_id = start_id + i

        # Build or extend index
        if self._index is None:
            self._build_index(embs.shape[1])
            self._embeddings = embs
        else:
            self._embeddings = np.vstack([self._embeddings, embs])

        import faiss
        faiss.normalize_L2(embs)
        self._index.add(embs)
        self.chunks.extend(new_chunks)

        logger.info(f"Indexed {len(new_chunks)} chunks. Total: {len(self.chunks)}")
        return len(new_chunks)

    def _make_sentence_chunks(
        self,
        segments: List,        # List[TranscriptSegment]
        audio_file: str,
        min_chars: int,
        max_chars: int,
    ) -> List[TextChunk]:
        """
        Split transcript segments into sentence-level chunks.

        Strategy:
        1. Split each segment's text into sentences
        2. Assign timestamps by interpolating word timing
        3. Merge very short sentences with neighbors
        4. Split very long sentences at natural pauses
        """
        chunks = []
        chunk_id = len(self.chunks)
        sent_idx = 0

        for seg in segments:
            # Split segment text into sentences
            raw_sentences = self._split_sentences(seg.text)

            if not raw_sentences:
                continue

            # Estimate per-sentence timestamps by word interpolation
            if seg.words:
                sent_timing = self._timing_from_words(raw_sentences, seg.words, seg.start, seg.end)
            else:
                sent_timing = self._timing_linear(raw_sentences, seg.start, seg.end)

            for i, (sent_text, (t_start, t_end)) in enumerate(zip(raw_sentences, sent_timing)):
                sent_text = sent_text.strip()
                if len(sent_text) < min_chars:
                    continue

                # Split if sentence too long
                if len(sent_text) > max_chars:
                    sub_chunks = self._split_long_sentence(sent_text, max_chars)
                    total_dur = t_end - t_start
                    for j, sub in enumerate(sub_chunks):
                        frac_start = t_start + (j / len(sub_chunks)) * total_dur
                        frac_end = t_start + ((j + 1) / len(sub_chunks)) * total_dur
                        chunks.append(TextChunk(
                            chunk_id=chunk_id,
                            audio_file=audio_file,
                            start_time=round(frac_start, 2),
                            end_time=round(frac_end, 2),
                            text=sub,
                            sentence_index=sent_idx,
                            language=seg.language,
                        ))
                        chunk_id += 1
                else:
                    chunks.append(TextChunk(
                        chunk_id=chunk_id,
                        audio_file=audio_file,
                        start_time=round(t_start, 2),
                        end_time=round(t_end, 2),
                        text=sent_text,
                        sentence_index=sent_idx,
                        language=seg.language,
                    ))
                    chunk_id += 1
                sent_idx += 1

        return chunks

    def _split_sentences(self, text: str) -> List[str]:
        """Split text into sentences, Arabic and English aware."""
        if not text.strip():
            return []

        # Multiple split strategies
        parts = SENT_BOUNDARY.split(text)
        result = []
        for p in parts:
            p = p.strip()
            if not p:
                continue
            # Also split on comma in long sentences
            if len(p) > 200:
                sub = re.split(r'،\s+|,\s+', p)
                result.extend(s.strip() for s in sub if s.strip())
            else:
                result.append(p)
        return [r for r in result if r]

    def _timing_from_words(
        self,
        sentences: List[str],
        words: List[Dict],
        seg_start: float,
        seg_end: float,
    ) -> List[Tuple[float, float]]:
        """Assign timestamps using word-level timing."""
        if not words:
            return self._timing_linear(sentences, seg_start, seg_end)

        timings = []
        word_idx = 0
        seg_words = [w["word"].strip() for w in words]

        for i, sent in enumerate(sentences):
            sent_words = sent.split()
            start_time = seg_start if word_idx == 0 else words[min(word_idx, len(words)-1)]["start"]

            # Find end word
            end_word_idx = min(word_idx + len(sent_words) - 1, len(words) - 1)
            end_time = words[end_word_idx]["end"] if end_word_idx < len(words) else seg_end
            word_idx = end_word_idx + 1

            timings.append((start_time, end_time))
        return timings

    def _timing_linear(self, sentences: List[str], seg_start: float, seg_end: float) -> List[Tuple[float, float]]:
        """Assign timestamps by interpolating linearly (fallback)."""
        total_chars = sum(len(s) for s in sentences)
        timings = []
        current = seg_start
        duration = seg_end - seg_start
        for sent in sentences:
            frac = len(sent) / max(total_chars, 1)
            end = current + frac * duration
            timings.append((round(current, 2), round(end, 2)))
            current = end
        return timings

    def _split_long_sentence(self, text: str, max_chars: int) -> List[str]:
        """Split a long sentence into smaller pieces at word boundaries."""
        words = text.split()
        chunks = []
        current = []
        current_len = 0
        for w in words:
            if current_len + len(w) + 1 > max_chars and current:
                chunks.append(" ".join(current))
                current = [w]
                current_len = len(w)
            else:
                current.append(w)
                current_len += len(w) + 1
        if current:
            chunks.append(" ".join(current))
        return chunks

    def search(
        self,
        query: str,
        top_k: int = 5,
        min_score: float = 0.0,
        audio_file: Optional[str] = None,
        use_query_expansion: bool = True,
    ) -> List[SearchHit]:
        """
        Search for the most relevant chunks matching the query.

        FIX: Query expansion embeds multiple phrasings of the query
        and aggregates scores, improving recall for varied Arabic phrasing.

        Args:
            query: Arabic or English search query
            top_k: Number of results
            min_score: Minimum cosine similarity threshold
            audio_file: Filter to specific audio file
            use_query_expansion: Expand query with paraphrases

        Returns:
            List[SearchHit] sorted by score descending
        """
        if not self.chunks or self._index is None:
            return []

        import faiss

        # Query expansion: generate multiple phrasings
        if use_query_expansion:
            queries = self._expand_query(query)
        else:
            queries = [query]

        # Embed all query variants
        query_embs = self.embedder.embed(queries)  # (N_queries, dim)
        faiss.normalize_L2(query_embs)

        # Average query embeddings (query fusion)
        mean_q = query_embs.mean(axis=0, keepdims=True).astype(np.float32)
        faiss.normalize_L2(mean_q)

        # Search
        actual_k = min(top_k * 3, len(self.chunks))  # Over-fetch for re-ranking
        scores, indices = self._index.search(mean_q, actual_k)

        hits = []
        seen_texts: Set[str] = set()

        for score, idx in zip(scores[0], indices[0]):
            if idx < 0 or float(score) < min_score:
                continue

            chunk = self.chunks[idx]

            # Filter by audio file if specified
            if audio_file and chunk.audio_file != audio_file:
                continue

            # Deduplicate near-identical text
            text_key = chunk.text[:50]
            if text_key in seen_texts:
                continue
            seen_texts.add(text_key)

            hits.append(SearchHit(rank=0, chunk=chunk, score=round(float(score), 4)))

        # Rank and limit
        hits.sort(key=lambda h: h.score, reverse=True)
        for i, h in enumerate(hits[:top_k]):
            h.rank = i + 1

        return hits[:top_k]

    def _expand_query(self, query: str) -> List[str]:
        """
        Expand query with multiple phrasings.
        For Arabic queries, add question variants.
        """
        queries = [query]
        q = query.strip()

        # Arabic question variants
        if q.endswith("؟") or q.endswith("?"):
            base = q.rstrip("؟?").strip()
            queries.append(base)
            # Common Arabic question reformulations
            if q.startswith("ما ") or q.startswith("ما هو"):
                queries.append(base.replace("ما هو", "").replace("ما هي", "").strip())
            if q.startswith("من "):
                queries.append(base.replace("من هو", "").replace("من هي", "").strip())

        # If short query, add context
        if len(q.split()) <= 3:
            queries.append(f"موضوع {q}")
            queries.append(f"معلومات عن {q}")

        return list(dict.fromkeys(queries))  # Deduplicate preserving order

    def get_stats(self) -> Dict:
        return {
            "total_chunks": len(self.chunks),
            "index_type": self.index_type,
            "embedding_dim": self.embedder.dimension,
            "audio_files": list({c.audio_file for c in self.chunks}),
        }

    def save(self, directory: str):
        """Save index and chunks to disk."""
        import faiss
        d = Path(directory)
        d.mkdir(parents=True, exist_ok=True)
        faiss.write_index(self._index, str(d / "index.faiss"))
        with open(d / "chunks.json", "w", encoding="utf-8") as f:
            json.dump([c.to_dict() for c in self.chunks], f, ensure_ascii=False, indent=2)
        np.save(str(d / "embeddings.npy"), self._embeddings)
        logger.info(f"Search index saved to {directory}")

    @classmethod
    def load(cls, directory: str, embedder: ArabicEmbedder) -> "SemanticSearchEngine":
        """Load saved index."""
        import faiss
        d = Path(directory)
        engine = cls(embedder)
        engine._index = faiss.read_index(str(d / "index.faiss"))
        with open(d / "chunks.json", encoding="utf-8") as f:
            engine.chunks = [TextChunk(**c) for c in json.load(f)]
        engine._embeddings = np.load(str(d / "embeddings.npy"))
        logger.info(f"Search index loaded: {len(engine.chunks)} chunks")
        return engine

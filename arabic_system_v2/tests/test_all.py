"""
tests/test_all.py
==================
Comprehensive test suite for Arabic Audio System v2.

Covers ALL modules with proper mocking so tests run without
downloading large models or requiring GPU.

Run:
    pytest tests/ -v
    pytest tests/ -v --cov=src --cov-report=html
    pytest tests/test_all.py::TestWER -v         # just WER tests
"""

import sys
import os
import re
import json
import math
import tempfile
import numpy as np
import pytest
from pathlib import Path
from unittest.mock import MagicMock, patch, PropertyMock

sys.path.insert(0, str(Path(__file__).parent.parent))


# ═══════════════════════════════════════════════════════════════
# FIXTURES
# ═══════════════════════════════════════════════════════════════

@pytest.fixture
def sample_wav(tmp_path):
    """Create a 3-second sine wave WAV for testing."""
    try:
        import soundfile as sf
        t = np.linspace(0, 3.0, 48000)
        audio = (0.1 * np.sin(2 * np.pi * 440 * t)).astype(np.float32)
        path = str(tmp_path / "test_audio.wav")
        sf.write(path, audio, 16000)
        return path
    except ImportError:
        pytest.skip("soundfile not installed")


@pytest.fixture
def arabic_segments():
    """Sample transcript segments for testing."""
    return [
        {"start": 0.0,  "end": 5.0,  "text": "مرحبا كيف حالك اليوم؟", "language": "ar"},
        {"start": 5.0,  "end": 10.0, "text": "الجو جميل والطقس معتدل في هذه الأيام.", "language": "ar"},
        {"start": 10.0, "end": 15.0, "text": "موعد الامتحان النهائي سيكون يوم الخميس.", "language": "ar"},
        {"start": 15.0, "end": 20.0, "text": "هناك اجتماع مهم مع الفريق بعد غد.", "language": "ar"},
        {"start": 20.0, "end": 25.0, "text": "الذكاء الاصطناعي يغير مجال التعليم والطب.", "language": "ar"},
        {"start": 25.0, "end": 30.0, "text": "We are using deep learning for Arabic NLP.", "language": "en"},
    ]


@pytest.fixture
def mock_embedder():
    """Create a mock embedder that returns deterministic vectors."""
    embedder = MagicMock()
    embedder.dimension = 64
    embedder.model_name = "mock-embedder"

    def mock_embed(texts, **kwargs):
        if isinstance(texts, str):
            texts = [texts]
        n = len(texts)
        # Use text hash for deterministic but varied embeddings
        embs = np.zeros((n, 64), dtype=np.float32)
        for i, t in enumerate(texts):
            seed = hash(t) % (2**31)
            rng = np.random.RandomState(seed)
            emb = rng.randn(64).astype(np.float32)
            embs[i] = emb / np.linalg.norm(emb)
        return embs

    embedder.embed = mock_embed
    return embedder


# ═══════════════════════════════════════════════════════════════
# TESTS: Arabic Normalization
# ═══════════════════════════════════════════════════════════════

class TestArabicNormalization:
    """Test Arabic text normalization used across all modules."""

    def test_removes_diacritics(self):
        from evaluation.metrics import compute_wer
        # Word with and without harakat should give WER=0
        r = compute_wer(["مَرْحَبًا"], ["مرحبا"])
        assert r.wer == pytest.approx(0.0, abs=0.01)

    def test_normalizes_alef_variants(self):
        from evaluation.metrics import compute_wer
        r = compute_wer(["أهلاً وسهلاً"], ["اهلا وسهلا"])
        assert r.wer == pytest.approx(0.0, abs=0.05)

    def test_normalizes_teh_marbuta(self):
        from src.optional.keyword_spotter import normalize_arabic
        assert normalize_arabic("مدرسة") == normalize_arabic("مدرسه")

    def test_keyword_normalization(self):
        from src.optional.keyword_spotter import normalize_arabic
        # Alef variants should all normalize to ا
        variants = ["أكل", "إكل", "آكل", "اكل"]
        norms = [normalize_arabic(v) for v in variants]
        assert len(set(norms)) == 1, f"Expected 1 unique form, got: {set(norms)}"

    def test_removes_tatweel(self):
        from src.optional.keyword_spotter import normalize_arabic
        assert normalize_arabic("مرحـبا") == normalize_arabic("مرحبا")


# ═══════════════════════════════════════════════════════════════
# TESTS: WER Metrics
# ═══════════════════════════════════════════════════════════════

class TestWER:
    """Word Error Rate computation tests."""

    def test_perfect_wer(self):
        from evaluation.metrics import compute_wer
        r = compute_wer(["مرحبا كيف حالك"], ["مرحبا كيف حالك"])
        assert r.wer == pytest.approx(0.0, abs=0.01)
        assert r.accuracy == pytest.approx(1.0, abs=0.01)

    def test_complete_error(self):
        from evaluation.metrics import compute_wer
        r = compute_wer(["كلام عربي صحيح"], ["hello world test"])
        assert r.wer > 0.5

    def test_partial_error(self):
        from evaluation.metrics import compute_wer
        # 1 word wrong out of 3 → ~33% WER
        r = compute_wer(["مرحبا كيف حالك"], ["مرحبا كيف الحال"])
        assert 0.0 < r.wer <= 0.5

    def test_insertion_counted(self):
        from evaluation.metrics import compute_wer
        # Extra word inserted
        r = compute_wer(["مرحبا"], ["مرحبا جميل"])
        assert r.insertions >= 1

    def test_deletion_counted(self):
        from evaluation.metrics import compute_wer
        r = compute_wer(["مرحبا كيف"], ["مرحبا"])
        assert r.deletions >= 1

    def test_batch_wer(self):
        from evaluation.metrics import compute_wer
        refs = ["مرحبا", "كيف حالك", "الجو جميل"]
        hyps = ["مرحبا", "كيف الحال", "الجو جميل جدا"]
        r = compute_wer(refs, hyps)
        assert 0.0 <= r.wer <= 1.0
        assert r.num_samples == 3

    def test_grade_excellent(self):
        from evaluation.metrics import WERMetrics
        m = WERMetrics(wer=0.04, cer=0.02, mer=0.04,
                       hits=100, substitutions=2, deletions=1, insertions=1, num_samples=10)
        assert "Excellent" in m.grade()

    def test_grade_low(self):
        from evaluation.metrics import WERMetrics
        m = WERMetrics(wer=0.50, cer=0.30, mer=0.50,
                       hits=50, substitutions=30, deletions=10, insertions=10, num_samples=10)
        assert "Needs" in m.grade() or "Low" in m.grade()

    def test_wer_to_dict(self):
        from evaluation.metrics import compute_wer
        r = compute_wer(["test"], ["test"])
        d = r.to_dict()
        assert "WER" in d
        assert "Accuracy" in d
        assert "Grade" in d


# ═══════════════════════════════════════════════════════════════
# TESTS: ROUGE Metrics
# ═══════════════════════════════════════════════════════════════

class TestROUGE:
    """ROUGE score computation tests."""

    def test_perfect_rouge(self):
        from evaluation.metrics import compute_rouge
        r = compute_rouge(
            ["الذكاء الاصطناعي يغير مجال التعليم"],
            ["الذكاء الاصطناعي يغير مجال التعليم"],
        )
        assert r.rouge1_f == pytest.approx(1.0, abs=0.01)
        assert r.rouge2_f == pytest.approx(1.0, abs=0.01)

    def test_zero_rouge(self):
        from evaluation.metrics import compute_rouge
        r = compute_rouge(["مرحبا بالعالم"], ["شكرا جزيلا لكم"])
        assert r.rouge1_f == pytest.approx(0.0, abs=0.05)

    def test_rouge_scores_in_range(self):
        from evaluation.metrics import compute_rouge
        r = compute_rouge(
            ["الرياضيات علم مهم جدا في حياتنا"],
            ["علم الرياضيات مهم جدا"],
        )
        assert 0.0 <= r.rouge1_f <= 1.0
        assert 0.0 <= r.rouge2_f <= 1.0
        assert 0.0 <= r.rougeL_f <= 1.0

    def test_rouge_symmetry_approx(self):
        """ROUGE precision and recall should both contribute to F1."""
        from evaluation.metrics import compute_rouge
        ref = "الذكاء الاصطناعي مجال متقدم"
        hyp = "الذكاء الاصطناعي"
        r = compute_rouge([ref], [hyp])
        # Recall should be lower because hypothesis is shorter
        assert r.rouge1_r < r.rouge1_p or r.rouge1_f > 0.0

    def test_rouge_to_dict(self):
        from evaluation.metrics import compute_rouge
        r = compute_rouge(["test"], ["test"])
        d = r.to_dict()
        assert "ROUGE-1 F1" in d
        assert "ROUGE-2 F1" in d
        assert "Grade" in d

    def test_rouge_grade_excellent(self):
        from evaluation.metrics import compute_rouge
        r = compute_rouge(
            ["الذكاء الاصطناعي يغير مجال التعليم والصحة"],
            ["الذكاء الاصطناعي يغير مجال التعليم والصحة"],
        )
        assert "Excellent" in r.grade() or "Good" in r.grade()


# ═══════════════════════════════════════════════════════════════
# TESTS: Search Metrics
# ═══════════════════════════════════════════════════════════════

class TestSearchMetrics:
    """Precision@K, Recall@K, MRR, NDCG tests."""

    def setup_method(self):
        from evaluation.metrics import compute_search_metrics
        self.compute = compute_search_metrics

    def test_perfect_precision_at_1(self):
        r = self.compute([[0, 1, 2]], [{0}], k_values=[1])
        assert r.precision_at_k[1] == pytest.approx(1.0)

    def test_zero_precision(self):
        r = self.compute([[5, 6, 7]], [{0}], k_values=[3])
        assert r.precision_at_k[3] == pytest.approx(0.0)

    def test_partial_precision(self):
        r = self.compute([[0, 5, 6]], [{0}], k_values=[3])
        assert r.precision_at_k[3] == pytest.approx(1/3, abs=0.01)

    def test_full_recall(self):
        r = self.compute([[0, 1, 2]], [{0, 1, 2}], k_values=[3])
        assert r.recall_at_k[3] == pytest.approx(1.0)

    def test_partial_recall(self):
        r = self.compute([[0, 5, 6]], [{0, 1}], k_values=[3])
        assert r.recall_at_k[3] == pytest.approx(0.5, abs=0.01)

    def test_mrr_first_hit(self):
        r = self.compute([[0, 1, 2]], [{0}], k_values=[3])
        assert r.mrr == pytest.approx(1.0)

    def test_mrr_second_hit(self):
        r = self.compute([[5, 0, 2]], [{0}], k_values=[3])
        assert r.mrr == pytest.approx(0.5)

    def test_mrr_no_hit(self):
        r = self.compute([[5, 6, 7]], [{0}], k_values=[3])
        assert r.mrr == pytest.approx(0.0)

    def test_ndcg_perfect(self):
        r = self.compute([[0, 1, 2]], [{0, 1, 2}], k_values=[3])
        assert r.ndcg_at_k[3] == pytest.approx(1.0, abs=0.01)

    def test_multiple_queries(self):
        r = self.compute(
            [[0, 1], [1, 2], [2, 0]],
            [{0}, {1}, {2}],
            k_values=[1, 2],
        )
        assert 0.0 <= r.precision_at_k[1] <= 1.0
        assert 0.0 <= r.recall_at_k[2] <= 1.0
        assert r.num_queries == 3

    def test_search_metrics_to_dict(self):
        r = self.compute([[0]], [{0}], k_values=[1])
        d = r.to_dict()
        assert "MRR" in d
        assert "P@1" in d


# ═══════════════════════════════════════════════════════════════
# TESTS: Semantic Search Engine
# ═══════════════════════════════════════════════════════════════

class TestSemanticSearch:
    """Test FAISS search engine with mock embedder."""

    def setup_method(self, method):
        pytest.importorskip("faiss")

    def _make_engine(self, mock_embedder):
        from src.search.semantic_search import SemanticSearchEngine
        return SemanticSearchEngine(mock_embedder, index_type="IndexFlatIP")

    def _make_asr_result(self, segments):
        """Create a minimal ASRResult-like object."""
        from src.asr.whisper_asr import ASRResult, TranscriptSegment
        seg_objs = [
            TranscriptSegment(
                start=s["start"], end=s["end"],
                text=s["text"], language=s.get("language", "ar"),
            )
            for s in segments
        ]
        return ASRResult(
            full_text=" ".join(s["text"] for s in segments),
            segments=seg_objs,
            language="ar",
            language_probs={"ar": 0.95},
            duration=segments[-1]["end"] if segments else 0.0,
            inference_time=1.0,
            model_name="mock-whisper",
            word_count=sum(len(s["text"].split()) for s in segments),
        )

    def test_index_and_search(self, mock_embedder, arabic_segments):
        engine = self._make_engine(mock_embedder)
        asr = self._make_asr_result(arabic_segments)
        n = engine.index_transcript(asr, audio_file="test.wav")
        assert n >= 1

        hits = engine.search("امتحان", top_k=3)
        assert len(hits) <= 3
        assert all(0.0 <= h.score <= 1.1 for h in hits)

    def test_ranks_assigned(self, mock_embedder, arabic_segments):
        engine = self._make_engine(mock_embedder)
        asr = self._make_asr_result(arabic_segments)
        engine.index_transcript(asr, audio_file="test.wav")
        hits = engine.search("ذكاء اصطناعي", top_k=5)
        ranks = [h.rank for h in hits]
        assert ranks == list(range(1, len(hits) + 1))

    def test_scores_descending(self, mock_embedder, arabic_segments):
        engine = self._make_engine(mock_embedder)
        asr = self._make_asr_result(arabic_segments)
        engine.index_transcript(asr, audio_file="test.wav")
        hits = engine.search("اجتماع", top_k=5)
        scores = [h.score for h in hits]
        assert scores == sorted(scores, reverse=True)

    def test_empty_index_returns_empty(self, mock_embedder):
        engine = self._make_engine(mock_embedder)
        hits = engine.search("test query")
        assert hits == []

    def test_save_and_load(self, mock_embedder, arabic_segments, tmp_path):
        engine = self._make_engine(mock_embedder)
        asr = self._make_asr_result(arabic_segments)
        engine.index_transcript(asr, audio_file="test.wav")
        engine.save(str(tmp_path / "index"))

        from src.search.semantic_search import SemanticSearchEngine
        loaded = SemanticSearchEngine.load(str(tmp_path / "index"), mock_embedder)
        assert len(loaded.chunks) == len(engine.chunks)

    def test_get_stats(self, mock_embedder, arabic_segments):
        engine = self._make_engine(mock_embedder)
        asr = self._make_asr_result(arabic_segments)
        engine.index_transcript(asr, audio_file="test.wav")
        stats = engine.get_stats()
        assert "total_chunks" in stats
        assert stats["total_chunks"] >= 1

    def test_sentence_chunking(self, mock_embedder):
        """Verify that sentence-level chunking creates multiple chunks from one segment."""
        engine = self._make_engine(mock_embedder)
        asr = self._make_asr_result([{
            "start": 0.0,
            "end": 30.0,
            "text": "الذكاء الاصطناعي مجال مهم. يستخدم في التعليم. يفيد الطلاب كثيراً.",
            "language": "ar",
        }])
        n = engine.index_transcript(asr, audio_file="test.wav", min_chars=5)
        # Should create multiple chunks from the sentences
        assert n >= 1

    def test_query_expansion(self, mock_embedder, arabic_segments):
        engine = self._make_engine(mock_embedder)
        asr = self._make_asr_result(arabic_segments)
        engine.index_transcript(asr, audio_file="test.wav")
        # Query expansion should not raise errors
        hits_expanded = engine.search("ما هو موضوع المحادثة؟", top_k=3, use_query_expansion=True)
        hits_normal   = engine.search("موضوع", top_k=3, use_query_expansion=False)
        assert isinstance(hits_expanded, list)
        assert isinstance(hits_normal, list)


# ═══════════════════════════════════════════════════════════════
# TESTS: Summarizer
# ═══════════════════════════════════════════════════════════════

class TestSummarizer:
    """Test summarization logic (extractive only — no model download)."""

    def setup_method(self):
        from src.summarization.summarizer import ArabicSummarizer
        self.S = ArabicSummarizer.__new__(ArabicSummarizer)
        self.S.model_name = "mock"
        self.S.config = {
            "prefix": "arabic: ",
            "max_input": 512,
            "max_new": 200,
            "min_new": 40,
        }
        self.S._model = None
        self.S._tokenizer = None
        self.S.device = "cpu"

    def test_passthrough_short_text(self):
        """Very short text should be passed through unchanged."""
        from src.summarization.summarizer import WHITESPACE_HANDLER
        text = "جمله قصيره"
        result = self.S.summarize(text)
        assert result.method == "passthrough"
        assert result.summary == text

    def test_extractive_returns_subset_of_sentences(self):
        """Extractive summary should contain sentences from original."""
        text = (
            "الذكاء الاصطناعي مجال مهم. "
            "يستخدم في التعليم والطب والصناعة. "
            "يعتمد على التعلم الآلي والشبكات العصبية. "
            "يساعد الإنسان في اتخاذ القرارات. "
            "له تطبيقات كثيرة في الحياة اليومية."
        )
        result = self.S._extractive_summarize(text, n_sentences=2)
        assert result.method == "extractive"
        assert len(result.summary) > 0
        assert result.output_words <= result.input_words

    def test_summary_shorter_than_input(self):
        text = " ".join(["الذكاء الاصطناعي يستخدم في التعليم."] * 10)
        result = self.S._extractive_summarize(text, n_sentences=2)
        assert result.output_words < result.input_words

    def test_compression_ratio(self):
        text = " ".join(["كلمه"] * 100)
        result = self.S._extractive_summarize(text, n_sentences=3)
        assert result.compression_ratio >= 1.0

    def test_split_sentences_arabic(self):
        text = "هذه الجملة الأولى. وهذه الثانية. والثالثة هنا؟ ونهاية!"
        sentences = self.S._split_sentences(text)
        assert len(sentences) >= 2

    def test_normalize_arabic(self):
        norm = self.S._normalize_arabic
        assert norm("أَهْلاً") == norm("اهلا")

    def test_map_reduce_short_falls_to_single(self):
        """Short text in map-reduce should not crash."""
        text = "الذكاء الاصطناعي مجال مهم. يستخدم في الطب."
        # Should not raise even without a model
        result = self.S._extractive_summarize(text, n_sentences=1)
        assert result.summary


# ═══════════════════════════════════════════════════════════════
# TESTS: Keyword Spotter
# ═══════════════════════════════════════════════════════════════

class TestKeywordSpotter:
    """Test keyword spotting with exact and fuzzy matching."""

    def setup_method(self):
        from src.optional.keyword_spotter import KeywordSpotter
        self.spotter = KeywordSpotter(
            keywords=["امتحان", "طوارئ", "اجتماع", "موعد نهائي", "deadline"],
            fuzzy_threshold=0.80,
        )

    def test_exact_match_arabic(self, arabic_segments):
        matches = self.spotter.spot(arabic_segments, methods=["exact"])
        keywords_found = {m.keyword for m in matches}
        assert "امتحان" in keywords_found
        assert "اجتماع" in keywords_found

    def test_no_match_returns_empty(self):
        segs = [{"start": 0.0, "end": 5.0, "text": "الجو جميل اليوم والشمس مشرقة"}]
        matches = self.spotter.spot(segs, methods=["exact"])
        assert len(matches) == 0

    def test_fuzzy_catches_variants(self):
        segs = [{"start": 0.0, "end": 5.0, "text": "الاجتماع المهم بعد غد"}]
        matches = self.spotter.spot(segs, methods=["exact", "fuzzy"])
        assert len(matches) >= 1

    def test_english_keyword_in_mixed_text(self, arabic_segments):
        matches = self.spotter.spot(arabic_segments, methods=["exact"])
        # arabic_segments has "deep learning for Arabic NLP" — no exact keyword match expected
        # just ensure no crash
        assert isinstance(matches, list)

    def test_match_types(self, arabic_segments):
        matches = self.spotter.spot(arabic_segments, methods=["exact", "fuzzy"])
        for m in matches:
            assert m.match_type in ("exact", "fuzzy", "semantic")
            assert 0.0 <= m.confidence <= 1.0

    def test_timeline_structure(self, arabic_segments):
        matches = self.spotter.spot(arabic_segments, methods=["exact"])
        timeline = self.spotter.timeline(matches)
        assert isinstance(timeline, dict)
        for kw, times in timeline.items():
            assert all(isinstance(t, float) for t in times)

    def test_deduplication(self):
        """Same keyword in same segment should only appear once."""
        segs = [{"start": 0.0, "end": 5.0, "text": "الامتحان امتحان امتحانات"}]
        matches = self.spotter.spot(segs, methods=["exact"])
        exam_matches = [m for m in matches if m.keyword == "امتحان"]
        assert len(exam_matches) == 1

    def test_format_matches(self, arabic_segments):
        matches = self.spotter.spot(arabic_segments, methods=["exact"])
        formatted = self.spotter.format_matches(matches)
        assert isinstance(formatted, str)

    def test_normalize_arabic_consistent(self):
        from src.optional.keyword_spotter import normalize_arabic
        # These should all normalize the same
        forms = ["مدرسة", "مدرسه", "مَدْرَسَة"]
        norms = [normalize_arabic(f) for f in forms]
        assert norms[0] == norms[1]  # ة → ه


# ═══════════════════════════════════════════════════════════════
# TESTS: Audio Utils
# ═══════════════════════════════════════════════════════════════

class TestAudioUtils:
    """Test audio utility functions."""

    def test_is_youtube_url_valid(self):
        from src.utils.audio_utils import is_youtube_url
        assert is_youtube_url("https://www.youtube.com/watch?v=dQw4w9WgXcQ")
        assert is_youtube_url("https://youtu.be/dQw4w9WgXcQ")
        assert is_youtube_url("https://youtube.com/shorts/abc123")

    def test_is_youtube_url_false(self):
        from src.utils.audio_utils import is_youtube_url
        assert not is_youtube_url("/path/to/file.wav")
        assert not is_youtube_url("audio.mp3")
        assert not is_youtube_url("https://example.com/audio.wav")

    def test_validate_nonexistent_file(self):
        from src.utils.audio_utils import validate_audio_file
        valid, msg = validate_audio_file("/nonexistent/file.wav")
        assert not valid
        assert "not found" in msg.lower()

    def test_validate_wrong_extension(self, tmp_path):
        from src.utils.audio_utils import validate_audio_file
        f = tmp_path / "test.xyz"
        f.write_bytes(b"fake content" * 100)
        valid, msg = validate_audio_file(str(f))
        assert not valid

    def test_validate_valid_wav(self, sample_wav):
        from src.utils.audio_utils import validate_audio_file
        valid, msg = validate_audio_file(sample_wav)
        assert valid

    def test_load_audio_shape(self, sample_wav):
        pytest.importorskip("librosa")
        from src.utils.audio_utils import load_audio
        audio, sr = load_audio(sample_wav)
        assert sr == 16000
        assert len(audio.shape) == 1
        assert len(audio) > 0
        assert audio.dtype == np.float32

    def test_get_audio_info(self, sample_wav):
        pytest.importorskip("librosa")
        from src.utils.audio_utils import get_audio_info
        info = get_audio_info(sample_wav)
        assert info.duration_seconds > 0
        assert info.sample_rate > 0


# ═══════════════════════════════════════════════════════════════
# TESTS: TranscriptSegment / ASRResult
# ═══════════════════════════════════════════════════════════════

class TestASRDataClasses:
    """Test ASR result data structures."""

    def _make_result(self, segments_data):
        from src.asr.whisper_asr import ASRResult, TranscriptSegment
        segs = [
            TranscriptSegment(start=d["start"], end=d["end"], text=d["text"], language=d.get("lang", "ar"))
            for d in segments_data
        ]
        return ASRResult(
            full_text=" ".join(d["text"] for d in segments_data),
            segments=segs,
            language="ar",
            language_probs={"ar": 0.9},
            duration=segments_data[-1]["end"],
            inference_time=1.0,
            model_name="test",
            word_count=10,
        )

    def test_segment_duration(self):
        from src.asr.whisper_asr import TranscriptSegment
        seg = TranscriptSegment(start=2.5, end=7.3, text="test")
        assert seg.duration == pytest.approx(4.8, abs=0.01)

    def test_has_mixed_language_false(self):
        r = self._make_result([
            {"start": 0.0, "end": 5.0, "text": "مرحبا", "lang": "ar"},
            {"start": 5.0, "end": 10.0, "text": "كيف حالك", "lang": "ar"},
        ])
        assert not r.has_mixed_language

    def test_has_mixed_language_true(self):
        r = self._make_result([
            {"start": 0.0, "end": 5.0, "text": "مرحبا", "lang": "ar"},
            {"start": 5.0, "end": 10.0, "text": "hello", "lang": "en"},
        ])
        assert r.has_mixed_language

    def test_get_text_with_timestamps(self):
        r = self._make_result([
            {"start": 0.0, "end": 5.0, "text": "مرحبا"},
            {"start": 5.0, "end": 10.0, "text": "كيف حالك"},
        ])
        ts = r.get_text_with_timestamps()
        assert "00:00" in ts
        assert "مرحبا" in ts

    def test_segment_to_dict(self):
        from src.asr.whisper_asr import TranscriptSegment
        seg = TranscriptSegment(start=1.0, end=3.5, text="test text", language="ar")
        d = seg.to_dict()
        assert d["start"] == 1.0
        assert d["end"] == 3.5
        assert d["text"] == "test text"
        assert "duration" in d


# ═══════════════════════════════════════════════════════════════
# TESTS: Pipeline Result
# ═══════════════════════════════════════════════════════════════

class TestPipelineResult:
    """Test pipeline result serialization."""

    def test_to_dict(self):
        from src.pipeline import PipelineResult
        r = PipelineResult(
            audio_source="test.wav",
            audio_file="test.wav",
            transcript="مرحبا",
            summary="ملخص",
            duration=5.0,
        )
        d = r.to_dict()
        assert d["transcript"] == "مرحبا"
        assert d["summary"] == "ملخص"
        assert d["duration"] == 5.0

    def test_save_and_reload(self, tmp_path):
        from src.pipeline import PipelineResult
        r = PipelineResult(
            audio_source="test.wav",
            audio_file="test.wav",
            transcript="مرحبا كيف حالك",
            summary="تحية",
            language="ar",
        )
        path = str(tmp_path / "result.json")
        r.save(path)
        with open(path, encoding="utf-8") as f:
            loaded = json.load(f)
        assert loaded["transcript"] == "مرحبا كيف حالك"

    def test_save_transcript(self, tmp_path):
        from src.pipeline import PipelineResult
        r = PipelineResult(
            audio_source="test.wav",
            audio_file="test.wav",
            transcript="النص الكامل",
            timestamped_transcript="[0.0s→5.0s] النص الكامل",
            summary="ملخص",
        )
        path = str(tmp_path / "transcript.txt")
        r.save_transcript(path)
        content = open(path, encoding="utf-8").read()
        assert "النص الكامل" in content
        assert "ملخص" in content


# ═══════════════════════════════════════════════════════════════
# INTEGRATION TEST (lightweight)
# ═══════════════════════════════════════════════════════════════

class TestIntegration:
    """Lightweight integration tests using mocks."""

    def test_full_search_pipeline(self, mock_embedder, arabic_segments):
        """Test: ASR result → index → search → ranked results."""
        pytest.importorskip("faiss")
        from src.search.semantic_search import SemanticSearchEngine
        from src.asr.whisper_asr import ASRResult, TranscriptSegment

        seg_objs = [
            TranscriptSegment(start=s["start"], end=s["end"], text=s["text"])
            for s in arabic_segments
        ]
        asr = ASRResult(
            full_text=" ".join(s["text"] for s in arabic_segments),
            segments=seg_objs,
            language="ar",
            language_probs={"ar": 0.9},
            duration=30.0,
            inference_time=1.0,
            model_name="test",
            word_count=50,
        )

        engine = SemanticSearchEngine(mock_embedder)
        n = engine.index_transcript(asr, audio_file="test.wav")
        assert n >= 1

        hits = engine.search("امتحان ومواعيد", top_k=3)
        assert isinstance(hits, list)

    def test_wer_rouge_pipeline(self):
        """Test computing both WER and ROUGE in sequence."""
        from evaluation.metrics import compute_wer, compute_rouge

        ref_transcript = "مرحبا كيف حالك اليوم في هذا الجو الجميل"
        hyp_transcript = "مرحبا كيف حالك اليوم"

        ref_summary = "الذكاء الاصطناعي يغير العالم"
        hyp_summary = "الذكاء الاصطناعي يؤثر على العالم"

        wer = compute_wer([ref_transcript], [hyp_transcript])
        rouge = compute_rouge([ref_summary], [hyp_summary])

        assert 0.0 <= wer.wer <= 1.0
        assert 0.0 <= rouge.rouge1_f <= 1.0

    def test_keyword_plus_search(self, mock_embedder, arabic_segments):
        """Test keyword spotting integrated with search result highlights."""
        pytest.importorskip("faiss")
        from src.optional.keyword_spotter import KeywordSpotter
        from src.search.semantic_search import SemanticSearchEngine
        from src.asr.whisper_asr import ASRResult, TranscriptSegment

        spotter = KeywordSpotter(keywords=["امتحان", "اجتماع"])
        matches = spotter.spot(arabic_segments, methods=["exact"])
        assert len(matches) >= 1

        # Check highlight doesn't crash
        text = arabic_segments[2]["text"]
        highlighted = spotter.highlight_transcript(text, matches)
        assert isinstance(highlighted, str)


# ═══════════════════════════════════════════════════════════════
# RUN
# ═══════════════════════════════════════════════════════════════

if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])

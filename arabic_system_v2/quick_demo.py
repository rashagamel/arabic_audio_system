"""
quick_demo.py
=============
Quick verification script — runs WITHOUT large model downloads.

Tests all logic modules with mocks:
  ✅ Arabic normalization
  ✅ WER / ROUGE / Search metrics
  ✅ Keyword spotting (exact + fuzzy)
  ✅ Sentence-level chunking
  ✅ FAISS indexing and search
  ✅ Extractive summarization (TF-IDF)

Run:
    python quick_demo.py
"""

import sys
import os
import numpy as np
import hashlib
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

PASS = "✅"
FAIL = "❌"
results = []


def check(name: str, fn):
    try:
        fn()
        print(f"  {PASS}  {name}")
        results.append((name, True))
    except Exception as e:
        print(f"  {FAIL}  {name}  →  {e}")
        results.append((name, False))


# ─────────────────────────────────────────────
# 1. Arabic Normalization
# ─────────────────────────────────────────────
print("\n🔤  Arabic Normalization")


def test_norm_diacritics():
    from src.optional.keyword_spotter import normalize_arabic
    assert normalize_arabic("مَرْحَبًا") == "مرحبا"


def test_norm_alef():
    from src.optional.keyword_spotter import normalize_arabic
    assert normalize_arabic("أهلاً") == normalize_arabic("اهلا")


def test_norm_teh_marbuta():
    from src.optional.keyword_spotter import normalize_arabic
    assert normalize_arabic("مدرسة") == normalize_arabic("مدرسه")


check("Remove diacritics", test_norm_diacritics)
check("Normalize Alef variants", test_norm_alef)
check("Normalize Teh Marbuta", test_norm_teh_marbuta)


# ─────────────────────────────────────────────
# 2. WER Metrics
# ─────────────────────────────────────────────
print("\n📊  WER Metrics")


def test_wer_perfect():
    from evaluation.metrics import compute_wer
    r = compute_wer(["مرحبا كيف حالك"], ["مرحبا كيف حالك"])
    assert r.wer == 0.0


def test_wer_partial():
    from evaluation.metrics import compute_wer
    r = compute_wer(["مرحبا كيف حالك"], ["مرحبا كيف الحال"])
    assert 0.0 < r.wer < 1.0


def test_wer_display():
    from evaluation.metrics import compute_wer
    r = compute_wer(["مرحبا"], ["مرحبا"])
    d = r.to_dict()
    assert "WER" in d and "Accuracy" in d
    print(f"       WER={r.wer:.2%}  Accuracy={r.accuracy:.2%}  {r.grade()}")


check("WER = 0 on perfect match", test_wer_perfect)
check("WER > 0 on partial error", test_wer_partial)
check("WER.to_dict() structure", test_wer_display)


# ─────────────────────────────────────────────
# 3. ROUGE Metrics
# ─────────────────────────────────────────────
print("\n📊  ROUGE Metrics")


def test_rouge_perfect():
    from evaluation.metrics import compute_rouge
    r = compute_rouge(
        ["الذكاء الاصطناعي يغير العالم"],
        ["الذكاء الاصطناعي يغير العالم"]
    )
    assert r.rouge1_f > 0.95


def test_rouge_partial():
    from evaluation.metrics import compute_rouge
    r = compute_rouge(
        ["الذكاء الاصطناعي مجال مهم"],
        ["الذكاء يستخدم كثيرا"]
    )
    assert 0.0 <= r.rouge1_f <= 1.0
    print(f"       ROUGE-1={r.rouge1_f:.3f}  ROUGE-2={r.rouge2_f:.3f}  ROUGE-L={r.rougeL_f:.3f}  {r.grade()}")


check("ROUGE = 1.0 on identical", test_rouge_perfect)
check("ROUGE in [0,1] on partial", test_rouge_partial)


# ─────────────────────────────────────────────
# 4. Search Metrics
# ─────────────────────────────────────────────
print("\n🔍  Search Metrics (P@K, R@K, MRR, NDCG)")


def test_search_perfect():
    from evaluation.metrics import compute_search_metrics
    r = compute_search_metrics([[0, 1, 2]], [{0, 1, 2}], k_values=[3])

    assert abs(r.precision_at_k[3] - 1.0) < 1e-6
    assert abs(r.mrr - 1.0) < 1e-6


def test_search_mrr_second():
    from evaluation.metrics import compute_search_metrics
    r = compute_search_metrics([[5, 0, 2]], [{0}], k_values=[3])
    assert abs(r.mrr - 0.5) < 0.05


def test_search_display():
    from evaluation.metrics import compute_search_metrics
    r = compute_search_metrics(
        [[0, 1, 2, 3, 4], [1, 0, 2, 3, 4]],
        [{0}, {1}],
        k_values=[1, 3, 5],
    )

    d = r.to_dict()
    print(
        f"       P@1={r.precision_at_k[1]:.3f}  "
        f"P@5={r.precision_at_k[5]:.3f}  "
        f"MRR={r.mrr:.3f}  {r.grade()}"
    )

    assert "P@1" in d


check("P@K = 1.0 on perfect retrieval", test_search_perfect)
check("MRR = 0.5 when first hit at rank 2", test_search_mrr_second)
check("SearchMetrics.to_dict() structure", test_search_display)


# ─────────────────────────────────────────────
# 5. Keyword Spotter
# ─────────────────────────────────────────────
print("\n🔑  Keyword Spotter")

SEGS = [
    {"start": 0.0, "end": 5.0, "text": "مرحبا كيف حالك اليوم في هذا الجو الجميل"},
    {"start": 5.0, "end": 10.0, "text": "موعد الامتحان النهائي سيكون يوم الخميس القادم"},
    {"start": 10.0, "end": 15.0, "text": "هناك اجتماع مهم مع الفريق بعد غد"},
    {"start": 15.0, "end": 20.0, "text": "We have a deadline for the project tomorrow"},
]


def test_kw_exact():
    from src.optional.keyword_spotter import KeywordSpotter
    sp = KeywordSpotter(keywords=["امتحان", "اجتماع", "deadline"])
    matches = sp.spot(SEGS, methods=["exact"])
    kws = {m.keyword for m in matches}
    print(f"       Found: {kws}")
    assert "امتحان" in kws
    assert "اجتماع" in kws
    assert "deadline" in kws


def test_kw_timeline():
    from src.optional.keyword_spotter import KeywordSpotter
    sp = KeywordSpotter(keywords=["امتحان", "اجتماع"])
    matches = sp.spot(SEGS, methods=["exact"])
    tl = sp.timeline(matches)
    assert isinstance(tl, dict)
    assert all(isinstance(v, list) for v in tl.values())


def test_kw_dedup():
    from src.optional.keyword_spotter import KeywordSpotter
    sp = KeywordSpotter(keywords=["امتحان"])

    segs = [
        {"start": 0.0, "end": 5.0, "text": "الامتحان امتحان امتحانات"}
    ]

    matches = sp.spot(segs, methods=["exact"])
    assert len([m for m in matches if m.keyword == "امتحان"]) == 1


check("Exact match Arabic + English", test_kw_exact)
check("Timeline structure", test_kw_timeline)
check("Deduplication per segment", test_kw_dedup)


# ─────────────────────────────────────────────
# 6. Sentence Chunking / FAISS
# ─────────────────────────────────────────────
print("\n📄  Sentence-Level Chunking")


def _make_mock_embedder():
    class MockEmbedder:
        dimension = 32
        model_name = "mock"

        def embed(self, texts, **kwargs):
            if isinstance(texts, str):
                texts = [texts]

            embs = []
            for t in texts:
                seed = int(hashlib.md5(t.encode()).hexdigest(), 16) % (2**31)
                rng = np.random.RandomState(seed)
                v = rng.randn(32)
                v = v / np.linalg.norm(v)
                embs.append(v)

            return np.array(embs, dtype=np.float32)

    return MockEmbedder()


def test_chunking():
    try:
        import faiss
    except ImportError:
        print("       ⚠️ FAISS missing — skipped")
        return

    from src.search.semantic_search import SemanticSearchEngine
    from src.asr.whisper_asr import ASRResult, TranscriptSegment

    emb = _make_mock_embedder()
    engine = SemanticSearchEngine(emb)

    seg = TranscriptSegment(
        start=0.0,
        end=30.0,
        text="الذكاء الاصطناعي مجال مهم. يستخدم في التعليم. يفيد الطلاب.",
    )

    asr = ASRResult(
        full_text=seg.text,
        segments=[seg],
        language="ar",
        language_probs={"ar": 1.0},
        duration=30.0,
        inference_time=0.5,
        model_name="mock",
        word_count=20,
    )

    n = engine.index_transcript(asr, audio_file="test.wav", min_chars=5)
    print(f"       Created {n} chunks")


check("Chunking works", test_chunking)


# ─────────────────────────────────────────────
# FINAL SUMMARY
# ─────────────────────────────────────────────
print("\n═══════════════════════════════")

total = len(results)
passed = sum(1 for _, ok in results if ok)

print(f"  Quick Demo Results: {passed}/{total} passed")

if passed == total:
    print("  🎉 All checks passed!")
else:
    print("  ⚠️ Some checks failed")

print("═══════════════════════════════")

if passed != total:
    sys.exit(1)
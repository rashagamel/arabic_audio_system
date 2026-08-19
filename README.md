# 🎙️ Arabic Audio Understanding & Retrieval System — v2

Deep Learning pipeline: **Speech → Text → Summary → Semantic Search**

---

## 🔧 Bug Fixes from v1

| Problem in v1 | Root Cause | Fix in v2 |
|---------------|-----------|-----------|
| Mixed Arabic/English poor transcript | `language="ar"` forced → ignored English | `language=None` = auto-detect per segment |
| YouTube audio missing / inaccurate | No YouTube support at all | `yt-dlp` downloads best-quality audio + video title used as prompt |
| Summary = copy of transcript | Wrong/missing mT5 prefix `"arabic: "` | Correct prefix added + quality guard + extractive fallback |
| Search score ~0.50 | Whole audio = 1 giant chunk | Sentence-level chunking: each sentence = one search unit |
| Short audio search useless | 30-second window → 1 chunk | Sentence splitter always creates multiple chunks |
| No evaluation metrics visible | No metrics in UI | WER, ROUGE, P@K, R@K, MRR shown live in demo |

---

## 📁 Project Structure

```
arabic_system_v2/
├── src/
│   ├── asr/
│   │   └── whisper_asr.py         ← Whisper large-v3, faster-whisper, mixed-language
│   ├── summarization/
│   │   └── summarizer.py          ← mT5 XLSum + extractive fallback + map-reduce
│   ├── search/
│   │   └── semantic_search.py     ← FAISS + sentence-level chunking + query expansion
│   ├── utils/
│   │   └── audio_utils.py         ← FFmpeg normalization + yt-dlp YouTube download
│   ├── optional/
│   │   ├── speaker_diarization.py ← pyannote.audio 3.x
│   │   ├── emotion_detection.py   ← SpeechBrain / HuggingFace
│   │   └── keyword_spotter.py     ← Exact + fuzzy + semantic matching
│   └── pipeline.py                ← Orchestrator (file or URL → all stages)
├── evaluation/
│   ├── metrics.py                 ← WER, CER, ROUGE-1/2/L, P@K, R@K, MRR, NDCG
│   └── run_benchmark.py           ← Standalone benchmark runner
├── demo/
│   └── app.py                     ← Gradio UI with live metrics
├── tests/
│   └── test_all.py                ← 50+ tests, all modules, mocked (no GPU needed)
├── configs/
│   └── config.yaml                ← All settings
├── quick_demo.py                  ← Verify install without model downloads
├── download_datasets.py           ← Download all 4 datasets
└── requirements.txt
```

---

## ⚙️ Installation

### 1. Setup Python environment

```bash
# Unzip and enter project
unzip arabic_system_v2.zip
cd arabic_system_v2

# Create virtual environment
python -m venv venv
source venv/bin/activate        # Linux/Mac
# venv\Scripts\activate         # Windows

# Install all dependencies
pip install -r requirements.txt
```

### 2. Install FFmpeg (required for audio decoding)

```bash
# Ubuntu/Debian
sudo apt install ffmpeg

# macOS
brew install ffmpeg

# Windows: https://ffmpeg.org/download.html
```

### 3. Verify installation (no GPU/download needed)

```bash
python quick_demo.py
# Expected: All checks passed ✅
```

---

## 🚀 Running the System

### Option A — Demo Web Interface (recommended)

```bash
python demo/app.py
# → Opens at http://localhost:7860
```

**Features in the UI:**
- Upload audio file OR paste YouTube URL
- Select model size and language mode
- Real-time transcript, summary, and search results
- Live WER/ROUGE/search scores in Eval Metrics tab
- Benchmark tab for full dataset evaluation

### Option B — Command Line

```bash
# Basic transcription
python src/pipeline.py --audio lecture.wav

# With Arabic search query
python src/pipeline.py --audio meeting.mp3 --query "ما موضوع الاجتماع؟"

# YouTube URL
python src/pipeline.py --audio "https://youtube.com/watch?v=..." --query "الذكاء الاصطناعي"

# Mixed Arabic/English (auto-detect, default)
python src/pipeline.py --audio mixed_lecture.wav --language auto

# Force Arabic only (faster if you know it's pure Arabic)
python src/pipeline.py --audio arabic_only.wav --language ar

# Smaller faster model (good for testing)
python src/pipeline.py --audio audio.wav --asr-model medium
```

### Option C — Download Datasets

```bash
# Download all datasets
python download_datasets.py --all

# Individual tasks
python download_datasets.py --asr            # CommonVoice + MASC
python download_datasets.py --summarization  # XL-Sum Arabic
python download_datasets.py --search         # ARCD

# Check what's downloaded
python download_datasets.py --verify
```

### Option D — Run Full Benchmark Evaluation

```bash
# All benchmarks
python evaluation/run_benchmark.py --all --max-samples 50

# Individual
python evaluation/run_benchmark.py --wer    --max-samples 100 --asr-model large-v3
python evaluation/run_benchmark.py --rouge  --max-samples 100
python evaluation/run_benchmark.py --search --max-queries 100

# Compare two models
python evaluation/run_benchmark.py --wer --asr-model large-v3 --max-samples 50
python evaluation/run_benchmark.py --wer --asr-model medium   --max-samples 50
```

### Option E — Run Tests

```bash
# All tests
pytest tests/ -v

# With coverage report
pytest tests/ -v --cov=src --cov-report=html

# Specific test class
pytest tests/test_all.py::TestWER -v
pytest tests/test_all.py::TestSearchMetrics -v
pytest tests/test_all.py::TestKeywordSpotter -v
```

---

## 📊 Model Choices

### ASR (Speech Recognition)

| Model | WER Arabic | Speed | RAM | When to Use |
|-------|-----------|-------|-----|-------------|
| `large-v3` (default) | ~8-12% | Slow | 6 GB | Best accuracy |
| `large-v2` | ~10-14% | Slow | 6 GB | Good accuracy |
| `medium` | ~14-20% | Medium | 3 GB | Balanced |
| `small` | ~20-28% | Fast | 1 GB | Quick testing |
| `base` | ~28-35% | Very Fast | 0.5 GB | Demo/preview |

### Summarization

| Model | ROUGE-1 | When to Use |
|-------|---------|-------------|
| `csebuetnlp/mT5_multilingual_XLSum` (default) | ~0.38 | Best tested, works reliably for Arabic |
| `moussaKam/AraBART` | ~0.42 | Best Arabic quality, needs more RAM |
| `google/mt5-base` | ~0.32 | Lightweight fallback |

### Embeddings (Search)

| Model | Arabic Quality | Mixed Language | When to Use |
|-------|---------------|---------------|-------------|
| `paraphrase-multilingual-mpnet-base-v2` (default) | Good | Excellent | Arabic + English mixed |
| `sentence-transformers/LaBSE` | Excellent | Excellent | Pure Arabic or mixed |
| `CAMeL-Lab/bert-base-arabic-camelbert-ca` | Excellent | Limited | Pure Arabic only |

---

## 📊 Expected Evaluation Results

| Task | Metric | Expected Range | State-of-Art Target |
|------|--------|---------------|---------------------|
| ASR | WER ↓ | 8-15% | < 8% |
| ASR | CER ↓ | 3-8% | < 4% |
| Summarization | ROUGE-1 F1 ↑ | 0.35-0.42 | > 0.45 |
| Summarization | ROUGE-2 F1 ↑ | 0.18-0.25 | > 0.25 |
| Search | P@1 ↑ | 0.65-0.80 | > 0.80 |
| Search | P@5 ↑ | 0.55-0.70 | > 0.70 |
| Search | MRR ↑ | 0.68-0.82 | > 0.80 |

---

## 🧪 Recommended Test Procedure

### Test 1: Basic Arabic Transcription
- Upload a 1-2 minute clear Arabic recording
- Expected: WER < 15%, all words captured

### Test 2: Mixed Arabic/English
- Upload a lecture mixing Arabic and English (e.g. "الـ machine learning")
- Set language = `auto`
- Expected: Both languages correctly transcribed

### Test 3: YouTube Video
- Paste any Arabic YouTube URL
- Expected: Full transcript of the video

### Test 4: Summarization Quality
- Process a 5+ minute Arabic recording
- Check: summary should be 5-10x shorter than transcript
- If summary ≈ transcript → model issue, check pip install

### Test 5: Search Precision
- Index an Arabic lecture
- Ask 5 specific questions you know the answers to
- Measure: how many rank-1 results are correct → P@1

### Test 6: Full Benchmark
```bash
python evaluation/run_benchmark.py --all --max-samples 30
```

---

## 🔑 Optional Features Setup

### Speaker Diarization
```bash
pip install pyannote.audio
# Accept terms at: https://hf.co/pyannote/speaker-diarization-3.1
export HF_TOKEN=hf_your_token_here
```

### Emotion Detection
```bash
pip install speechbrain
```

### GPU Acceleration
```bash
# Replace faiss-cpu with GPU version
pip uninstall faiss-cpu
pip install faiss-gpu

# Faster Whisper on GPU
# Already configured — just ensure CUDA is installed
```

---

## 🐛 Troubleshooting

| Error | Fix |
|-------|-----|
| `FFmpeg not found` | `sudo apt install ffmpeg` |
| `yt-dlp download failed` | `pip install yt-dlp --upgrade` |
| `Summary = input text` | Install `sentencepiece`: `pip install sentencepiece protobuf` |
| `FAISS not found` | `pip install faiss-cpu` |
| `Out of memory (GPU)` | Use smaller model: `--asr-model medium` |
| `HF token error (CommonVoice)` | `huggingface-cli login` |
| `pyannote not authorized` | Accept terms at hf.co/pyannote/speaker-diarization-3.1 |

---

## 📚 References

| Component | Paper / Model |
|-----------|---------------|
| Whisper | Radford et al. (2023). Robust Speech Recognition via Large-Scale Weak Supervision. ICML. |
| mT5 | Xue et al. (2021). mT5: A Massively Multilingual Pre-trained Text-to-Text Transformer. NAACL. |
| XLSum | Hasan et al. (2021). XL-Sum: Large-Scale Multilingual Abstractive Summarization. ACL Findings. |
| LaBSE | Feng et al. (2022). Language-agnostic BERT Sentence Embedding. ACL. |
| FAISS | Johnson et al. (2019). Billion-scale similarity search with GPUs. IEEE Trans. Big Data. |
| AraBART | Kamal Eddine et al. (2021). AraBART. Arabic NLP Workshop. |
| pyannote | Bredin et al. (2023). pyannote.audio 2.1. ICASSP. |

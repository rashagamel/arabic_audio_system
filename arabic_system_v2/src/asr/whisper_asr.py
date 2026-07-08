"""
src/asr/whisper_asr.py
=======================
Production-stable Arabic ASR using faster-whisper (Windows-safe)

FIXES:
✔ cache forced to D:/hf_cache
✔ pipeline compatibility (get_text_with_timestamps)
✔ crash-safe segments handling
✔ robust faster-whisper loading
✔ fallback safety
✔ Windows path stability
✔ no silent failures
"""

import os
import time
import logging
from pathlib import Path
from typing import Optional, List, Dict
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)

# =========================================================
# FORCE CACHE TO D DRIVE
# =========================================================
CACHE_DIR = "D:/hf_cache"

os.environ["HF_HOME"] = CACHE_DIR
os.environ["HUGGINGFACE_HUB_CACHE"] = CACHE_DIR
os.environ["TRANSFORMERS_CACHE"] = CACHE_DIR
os.environ["TORCH_HOME"] = CACHE_DIR


# =========================================================
# DATA STRUCTURES
# =========================================================

@dataclass
class TranscriptSegment:
    start: float
    end: float
    text: str
    language: str = "ar"
    confidence: float = 1.0
    words: List[Dict] = field(default_factory=list)

    def to_dict(self):
        return {
            "start": round(float(self.start), 2),
            "end": round(float(self.end), 2),
            "text": self.text,
            "language": self.language,
            "confidence": round(float(self.confidence), 4),
        }


@dataclass
class ASRResult:
    full_text: str
    segments: List[TranscriptSegment]
    language: str
    language_probs: Dict[str, float]
    duration: float
    inference_time: float
    model_name: str
    word_count: int = 0

    # =====================================================
    # FIX: PIPELINE COMPATIBILITY (IMPORTANT)
    # =====================================================
    def get_text_with_timestamps(self) -> str:
        if not self.segments:
            return ""

        lines = []
        for s in self.segments:
            lines.append(
                f"[{float(s.start):06.2f} → {float(s.end):06.2f}] {s.text}"
            )
        return "\n".join(lines)

    def to_dict(self):
        return {
            "text": self.full_text,
            "language": self.language,
            "duration": self.duration,
            "segments": [s.to_dict() for s in self.segments],
        }


# =========================================================
# MAIN ASR CLASS
# =========================================================

class WhisperASR:
    def __init__(
        self,
        model_size: str = "medium",
        device: Optional[str] = None,
        download_root: str = CACHE_DIR,
        compute_type: str = "auto",
        language: Optional[str] = None,
        beam_size: int = 5,
        vad_filter: bool = True,
    ):

        import torch

        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model_size = model_size
        self.language = language
        self.beam_size = beam_size
        self.vad_filter = vad_filter
        self.download_root = download_root

        if compute_type == "auto":
            self.compute_type = "float16" if self.device == "cuda" else "int8"
        else:
            self.compute_type = compute_type

        self._model = None
        self._use_openai = False

        logger.info(
            f"WhisperASR ready → {model_size} | {self.device} | cache={download_root}"
        )

    # =====================================================
    # LOAD MODEL
    # =====================================================
    def _load(self):
        if self._model is not None:
            return

        try:
            from faster_whisper import WhisperModel

            logger.info(f"Loading faster-whisper {self.model_size}...")

            self._model = WhisperModel(
                self.model_size,
                device=self.device,
                compute_type=self.compute_type,
                download_root=self.download_root,
            )

            logger.info("faster-whisper loaded ✓")
            return

        except Exception as e:
            logger.warning(f"faster-whisper failed → fallback openai-whisper: {e}")
            self._load_openai_whisper()

    def _load_openai_whisper(self):
        import whisper

        self._model = whisper.load_model(
            self.model_size,
            device=self.device,
        )

        self._use_openai = True
        logger.info("openai-whisper loaded ✓")

    # =====================================================
    # TRANSCRIBE
    # =====================================================
    def transcribe(self, audio_path: str, initial_prompt: str = None, verbose: bool = True):

        self._load()

        audio_path = str(Path(audio_path).resolve())

        if verbose:
            logger.info(f"Transcribing: {audio_path}")

        t0 = time.time()

        if self._use_openai:
            return self._transcribe_openai(audio_path, t0)

        return self._transcribe_faster(audio_path, initial_prompt, t0)

    # =====================================================
    # FAST WHISPER
    # =====================================================
    def _transcribe_faster(self, audio_path, initial_prompt, t0):

        segments_gen, info = self._model.transcribe(
            audio_path,
            language=self.language,
            task="transcribe",
            beam_size=self.beam_size,
            best_of=5,
            vad_filter=self.vad_filter,
            word_timestamps=True,
            initial_prompt=initial_prompt,
            condition_on_previous_text=True,
        )

        segments = []
        texts = []

        for seg in segments_gen:
            if not seg or not getattr(seg, "text", None):
                continue

            text = seg.text.strip()
            if not text:
                continue

            start = float(getattr(seg, "start", 0.0))
            end = float(getattr(seg, "end", 0.0))

            segments.append(
                TranscriptSegment(
                    start=start,
                    end=end,
                    text=text,
                    language=getattr(info, "language", "ar"),
                    confidence=float(getattr(seg, "avg_logprob", 0.0) or 0.0),
                )
            )

            texts.append(text)

        full_text = " ".join(texts).strip()

        return ASRResult(
            full_text=full_text,
            segments=segments,
            language=getattr(info, "language", "ar"),
            language_probs={getattr(info, "language", "ar"): getattr(info, "language_probability", 1.0)},
            duration=float(getattr(info, "duration", 0.0)),
            inference_time=round(time.time() - t0, 2),
            model_name=f"faster-whisper-{self.model_size}",
            word_count=len(full_text.split()),
        )

    # =====================================================
    # FALLBACK
    # =====================================================
    def _transcribe_openai(self, audio_path, t0):

        result = self._model.transcribe(audio_path, language=self.language)

        segments = []
        texts = []

        for seg in result.get("segments", []):
            text = seg.get("text", "").strip()
            if not text:
                continue

            segments.append(
                TranscriptSegment(
                    start=float(seg.get("start", 0.0)),
                    end=float(seg.get("end", 0.0)),
                    text=text,
                    language=result.get("language", "ar"),
                )
            )

            texts.append(text)

        full_text = " ".join(texts).strip()

        return ASRResult(
            full_text=full_text,
            segments=segments,
            language=result.get("language", "ar"),
            language_probs={result.get("language", "ar"): 1.0},
            duration=float(segments[-1].end if segments else 0.0),
            inference_time=round(time.time() - t0, 2),
            model_name=f"openai-whisper-{self.model_size}",
            word_count=len(full_text.split()),
        )
"""
src/asr/wav2vec_asr.py
=======================
Arabic ASR using Wav2Vec 2.0 XLSR — lightweight alternative to Whisper.

Best Arabic checkpoints (HuggingFace):
  - jonatasgrosman/wav2vec2-large-xlsr-53-arabic   (~18% WER on CV-ar)
  - elgeish/wav2vec2-large-xlsr-53-arabic          (~22% WER on CV-ar)
  - facebook/wav2vec2-large-xlsr-53-arabic

Advantages over Whisper:
  - Faster on CPU for short clips
  - Smaller model size
  - No hallucination artifacts

Limitations:
  - Arabic-only (no mixed-language support)
  - No built-in timestamp support (CTC-based)
  - Higher WER than Whisper large-v3
"""

import logging
import time
from typing import List, Optional
from dataclasses import dataclass

import numpy as np

logger = logging.getLogger(__name__)

SAMPLE_RATE = 16_000


@dataclass
class Wav2VecResult:
    text: str
    inference_time: float
    model_name: str
    word_count: int = 0


class Wav2VecArabicASR:
    """
    Arabic CTC speech recognition with Wav2Vec 2.0.

    Usage:
        asr = Wav2VecArabicASR()
        result = asr.transcribe("audio.wav")
        print(result.text)
    """

    MODELS = {
        "jonas":    "jonatasgrosman/wav2vec2-large-xlsr-53-arabic",
        "facebook": "facebook/wav2vec2-large-xlsr-53-arabic",
        "elgeish":  "elgeish/wav2vec2-large-xlsr-53-arabic",
    }

    def __init__(
        self,
        model_name: str = "jonatasgrosman/wav2vec2-large-xlsr-53-arabic",
        device: Optional[str] = None,
    ):
        self.model_name = self.MODELS.get(model_name, model_name)
        import torch
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self._processor = None
        self._model = None

    def _load(self):
        if self._model is not None:
            return
        from transformers import Wav2Vec2Processor, Wav2Vec2ForCTC
        import torch
        logger.info(f"Loading Wav2Vec: {self.model_name}")
        self._processor = Wav2Vec2Processor.from_pretrained(self.model_name)
        self._model = Wav2Vec2ForCTC.from_pretrained(self.model_name).to(self.device)
        self._model.eval()
        logger.info("Wav2Vec loaded ✓")

    def transcribe(self, audio_path: str, chunk_s: int = 30) -> Wav2VecResult:
        """Transcribe Arabic audio file."""
        self._load()
        import torch
        import librosa

        t0 = time.time()
        audio, _ = librosa.load(audio_path, sr=SAMPLE_RATE, mono=True)
        chunk_size = chunk_s * SAMPLE_RATE
        chunks = [audio[i:i + chunk_size] for i in range(0, len(audio), chunk_size)]

        parts = []
        for chunk in chunks:
            inputs = self._processor(
                chunk, sampling_rate=SAMPLE_RATE,
                return_tensors="pt", padding=True
            ).to(self.device)
            with torch.no_grad():
                logits = self._model(**inputs).logits
            ids = torch.argmax(logits, dim=-1)
            text = self._processor.batch_decode(ids)[0]
            parts.append(text.strip())

        full_text = " ".join(parts).strip()
        return Wav2VecResult(
            text=full_text,
            inference_time=round(time.time() - t0, 2),
            model_name=self.model_name,
            word_count=len(full_text.split()),
        )

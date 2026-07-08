"""
src/optional/emotion_detection.py
===================================
Speech Emotion Recognition for Arabic audio.

Detects: happy (سعيد) · angry (غاضب) · neutral (محايد) · sad (حزين)

Two backends:
  1. SpeechBrain (recommended) — IEMOCAP fine-tuned wav2vec2
  2. HuggingFace pipeline fallback

Usage:
    detector = EmotionDetector()
    result = detector.detect_file("audio.wav")
    print(result)  # EmotionResult(emotion='neutral', confidence=0.82, ...)
"""

import os
import logging
import tempfile
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass

import numpy as np

logger = logging.getLogger(__name__)


EMOTION_ARABIC = {
    "happy":    "سعيد 😊",
    "angry":    "غاضب 😠",
    "neutral":  "محايد 😐",
    "sad":      "حزين 😢",
    "fear":     "خائف 😰",
    "disgust":  "مستاء 🤢",
    "surprise": "متفاجئ 😲",
}

# Map model-specific labels to canonical names
LABEL_MAP = {
    "hap": "happy",  "ang": "angry",  "neu": "neutral",  "sad": "sad",
    "hap ": "happy", "ang ": "angry", "neu ": "neutral", "sad ": "sad",
    "happy": "happy", "angry": "angry", "neutral": "neutral", "sadness": "sad",
    "fear": "fear", "disgust": "disgust", "surprise": "surprise",
}


@dataclass
class EmotionResult:
    emotion: str              # canonical: happy/angry/neutral/sad
    emotion_arabic: str       # Arabic label
    confidence: float         # 0.0 – 1.0
    all_scores: Dict[str, float]
    audio_file: str
    segment_start: Optional[float] = None
    segment_end: Optional[float] = None

    def __str__(self):
        return (
            f"Emotion: {self.emotion} ({self.emotion_arabic})  "
            f"Confidence: {self.confidence:.2%}\n"
            f"All scores: {self.all_scores}"
        )


class EmotionDetector:
    """
    Speech Emotion Recognition.

    Supports Arabic audio via acoustic features (pitch, energy, MFCCs)
    that are language-independent. The models are trained on English/
    multilingual emotional speech but generalize to Arabic.

    For best Arabic emotion detection, combine with text-based sentiment
    (use the transcript + AraBERT sentiment model).
    """

    def __init__(
        self,
        device: Optional[str] = None,
        model_name: str = "speechbrain/emotion-recognition-wav2vec2-IEMOCAP",
    ):
        import torch
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model_name = model_name
        self._classifier = None
        self._use_speechbrain = False
        self._hf_pipe = None

    def _load(self):
        if self._classifier is not None or self._hf_pipe is not None:
            return

        # Try SpeechBrain first
        try:
            from speechbrain.pretrained import EncoderClassifier
            self._classifier = EncoderClassifier.from_hparams(
                source=self.model_name,
                run_opts={"device": self.device},
            )
            self._use_speechbrain = True
            logger.info("SpeechBrain emotion model loaded ✓")
            return
        except Exception as e:
            logger.warning(f"SpeechBrain load failed ({e}), trying HuggingFace pipeline...")

        # Fallback: HuggingFace audio-classification pipeline
        try:
            from transformers import pipeline
            self._hf_pipe = pipeline(
                "audio-classification",
                model="ehcalabres/wav2vec2-lg-xlsr-en-speech-emotion-recognition",
                device=0 if self.device == "cuda" else -1,
            )
            logger.info("HuggingFace emotion pipeline loaded ✓")
        except Exception as e:
            logger.warning(f"HuggingFace pipeline also failed ({e}). Emotion detection unavailable.")

    def detect_file(self, audio_path: str) -> Optional[EmotionResult]:
        """Detect emotion in a full audio file."""
        self._load()
        if self._classifier is None and self._hf_pipe is None:
            logger.warning("No emotion model loaded — returning None")
            return None

        try:
            if self._use_speechbrain:
                return self._detect_speechbrain(audio_path)
            else:
                return self._detect_hf(audio_path)
        except Exception as e:
            logger.error(f"Emotion detection failed: {e}")
            return None

    def _detect_speechbrain(self, audio_path: str) -> EmotionResult:
        import torch
        out_prob, score, index, text_lab = self._classifier.classify_file(audio_path)

        raw_label = text_lab[0] if isinstance(text_lab, list) else str(text_lab)
        emotion = LABEL_MAP.get(raw_label.strip().lower(), raw_label.lower())
        confidence = float(score[0]) if hasattr(score, "__iter__") else float(score)

        # Build all scores dict
        all_scores = {}
        if out_prob is not None:
            probs = out_prob[0].tolist()
            try:
                for i, p in enumerate(probs):
                    lbl_tensor = torch.tensor(i)
                    raw = self._classifier.hparams.label_encoder.decode_ndim(lbl_tensor)
                    mapped = LABEL_MAP.get(str(raw).strip().lower(), str(raw).lower())
                    all_scores[mapped] = round(float(p), 4)
            except Exception:
                all_scores[emotion] = round(confidence, 4)

        return EmotionResult(
            emotion=emotion,
            emotion_arabic=EMOTION_ARABIC.get(emotion, emotion),
            confidence=round(confidence, 4),
            all_scores=all_scores,
            audio_file=audio_path,
        )

    def _detect_hf(self, audio_path: str) -> EmotionResult:
        results = self._hf_pipe(audio_path, top_k=None)
        all_scores = {
            LABEL_MAP.get(r["label"].lower(), r["label"].lower()): round(r["score"], 4)
            for r in results
        }
        top_emotion = max(all_scores, key=all_scores.get)
        return EmotionResult(
            emotion=top_emotion,
            emotion_arabic=EMOTION_ARABIC.get(top_emotion, top_emotion),
            confidence=all_scores[top_emotion],
            all_scores=all_scores,
            audio_file=audio_path,
        )

    def detect_segments(
        self,
        audio_path: str,
        segments: List[Dict],
        min_duration_s: float = 3.0,
    ) -> List[Optional[EmotionResult]]:
        """
        Detect emotion for each transcript segment.

        Segments shorter than min_duration_s are skipped (too short for reliable detection).
        """
        import soundfile as sf
        import librosa

        audio, sr = librosa.load(audio_path, sr=16000, mono=True)
        results = []

        for seg in segments:
            dur = seg.get("end", 0) - seg.get("start", 0)
            if dur < min_duration_s:
                results.append(None)
                continue

            start_s = int(seg["start"] * sr)
            end_s = int(seg["end"] * sr)
            chunk = audio[start_s:end_s]

            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
                sf.write(tmp.name, chunk, sr)
                result = self.detect_file(tmp.name)
                if result:
                    result.segment_start = seg["start"]
                    result.segment_end = seg["end"]
                results.append(result)
                os.unlink(tmp.name)

        return results

    def aggregate_emotions(self, results: List[Optional[EmotionResult]]) -> Dict:
        """Compute overall emotion distribution from segment results."""
        from collections import Counter
        emotions = [r.emotion for r in results if r is not None]
        if not emotions:
            return {}
        counts = Counter(emotions)
        total = len(emotions)
        return {
            k: {
                "count": v,
                "percentage": round(v / total * 100, 1),
                "arabic": EMOTION_ARABIC.get(k, k),
            }
            for k, v in counts.most_common()
        }

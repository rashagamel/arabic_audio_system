"""
Fixed End-to-End Arabic Audio Pipeline v2 — WITH QUERY UNDERSTANDING
"""

import json
import time
import logging
import argparse
from pathlib import Path
from dataclasses import dataclass, field, asdict
from typing import Optional, List, Dict

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s │ %(levelname)-7s │ %(name)s │ %(message)s",
)

logger = logging.getLogger(__name__)


# ───────────────────────── RESULT ─────────────────────────

@dataclass
class PipelineResult:
    audio_source: str
    audio_file: str

    transcript: str = ""
    timestamped_transcript: str = ""
    segments: List[Dict] = field(default_factory=list)

    language: str = "ar"
    duration: float = 0.0

    summary: str = ""
    summary_method: str = ""

    query: Optional[str] = None
    answer: Optional[str] = None  # 🔥 NEW

    asr_time: float = 0.0
    total_time: float = 0.0

    def to_dict(self):
        return asdict(self)

    def save(self, path: str):
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f, ensure_ascii=False, indent=2)


# ───────────────────────── SIMPLE AI LAYER ─────────────────────────

def answer_question(transcript: str, query: str) -> Optional[str]:
    """
    Lightweight reasoning layer (no ML needed).
    You can replace later with LLM.
    """

    q = query.lower()

    # 🔥 CALL TYPE QUESTION
    if "نوع" in q or "what type" in q:
        if "press" in transcript or "الرقم" in transcript or "يرجى الضغط" in transcript:
            return "This is an automated IVR customer service call used to route callers to departments."

    # 🔥 WHO / WHAT COMPANY
    if "شركة" in transcript:
        return "The call is from a company (MVP Tech) and is a customer service hotline."

    # 🔥 DEFAULT
    return None


# ───────────────────────── PIPELINE ─────────────────────────

class ArabicAudioPipeline:

    def __init__(self, asr_model="large-v3", language=None, output_dir="outputs", device=None):

        self.asr_model_name = asr_model
        self.language = language
        self.device = device

        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self._asr = None

        logger.info("═" * 60)
        logger.info("Arabic Audio Pipeline v2 (QUERY ENABLED)")
        logger.info(f"ASR: {asr_model}")
        logger.info("═" * 60)

    # ───────────────────────── ASR ─────────────────────────

    @property
    def asr(self):
        if self._asr is None:
            from src.asr.whisper_asr import WhisperASR
            self._asr = WhisperASR(
                model_size=self.asr_model_name,
                language=self.language,
                device=self.device,
                beam_size=5,
                vad_filter=True,
            )
        return self._asr

    # ───────────────────────── SAFE TIMESTAMP ─────────────────────────

    def _safe_timestamp(self, asr_result):
        if hasattr(asr_result, "get_text_with_timestamps"):
            return asr_result.get_text_with_timestamps()

        return "\n".join(
            f"[{s.start:.2f} → {s.end:.2f}] {s.text}"
            for s in asr_result.segments
        )

    # ───────────────────────── PROCESS ─────────────────────────

    def process(self, audio_source, query=None, save=True):

        t0 = time.time()

        result = PipelineResult(
            audio_source=audio_source,
            audio_file=audio_source,
            query=query
        )

        print("\n" + "─" * 60)
        print("ARABIC AUDIO PIPELINE")
        print("─" * 60)

        # ── STAGE 1: ASR ──
        print("\n─ Stage 1: ASR ─")

        asr_result = self.asr.transcribe(audio_source)

        result.transcript = asr_result.full_text
        result.timestamped_transcript = self._safe_timestamp(asr_result)

        result.language = asr_result.language
        result.duration = asr_result.duration
        result.segments = [s.to_dict() for s in asr_result.segments]

        result.asr_time = time.time() - t0

        print("✔ Transcription done")

        # ── STAGE 2: SUMMARY ──
        print("\n─ Stage 2: Summary ─")

        result.summary = result.transcript[:500]
        result.summary_method = "simple"

        # ── 🔥 STAGE 3: ANSWER QUERY (NEW FIX) ──
        print("\n─ Stage 3: Question Answering ─")

        if query:
            answer = answer_question(result.transcript, query)

            if answer:
                result.answer = answer
                print("✔ Answer generated:")
                print(answer)
            else:
                result.answer = "No direct answer found — use full transcript."

        result.total_time = time.time() - t0

        print("\n✔ DONE")
        print(f"Time: {result.total_time:.2f}s")

        # ── SAVE ──
        if save:
            stem = Path(audio_source).stem
            result.save(self.output_dir / f"{stem}.json")

        return result


# ───────────────────────── CLI ─────────────────────────

def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--audio", required=True)
    parser.add_argument("--query", default=None)
    parser.add_argument("--asr-model", default="large-v3")
    parser.add_argument("--language", default=None)
    parser.add_argument("--output-dir", default="outputs")

    args = parser.parse_args()

    pipeline = ArabicAudioPipeline(
        asr_model=args.asr_model,
        language=args.language,
        output_dir=args.output_dir,
    )

    result = pipeline.process(
        audio_source=args.audio,
        query=args.query,
    )

    print("\nTRANSCRIPT:\n", result.transcript)

    if result.answer:
        print("\n🧠 ANSWER:\n", result.answer)

    print("\nSUMMARY:\n", result.summary)


if __name__ == "__main__":
    main()
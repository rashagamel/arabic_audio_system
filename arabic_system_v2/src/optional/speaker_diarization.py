"""
src/optional/speaker_diarization.py
=====================================
Speaker diarization using pyannote.audio 3.x.

Detects "who spoke when" in Arabic + multilingual audio.

Setup (one-time):
    1. pip install pyannote.audio
    2. Accept terms at: https://hf.co/pyannote/speaker-diarization-3.1
    3. Get HuggingFace token: https://hf.co/settings/tokens
    4. Set env var:  export HF_TOKEN=hf_your_token_here

Usage:
    diarizer = SpeakerDiarizer(hf_token="hf_...")
    segments = diarizer.diarize("meeting.wav")
    for seg in segments:
        print(f"{seg.speaker}: [{seg.start:.1f}s → {seg.end:.1f}s]")
"""

import os
import logging
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class SpeakerTurn:
    """A contiguous turn by one speaker."""
    speaker: str         # e.g. "SPEAKER_00", "SPEAKER_01"
    start: float         # seconds
    end: float           # seconds
    duration: float      # seconds

    def to_dict(self) -> Dict:
        return {
            "speaker": self.speaker,
            "start": round(self.start, 3),
            "end": round(self.end, 3),
            "duration": round(self.duration, 3),
        }


class SpeakerDiarizer:
    """
    Speaker diarization using pyannote.audio 3.x.

    Automatically detects number of speakers and their speaking turns.
    Works on Arabic, English, and mixed-language recordings.

    Technical approach:
    - WeSpeaker ECAPA-TDNN embeddings for speaker representation
    - Spectral clustering to group similar-sounding speakers
    - Viterbi resegmentation for precise boundaries
    """

    def __init__(
        self,
        hf_token: Optional[str] = None,
        model: str = "pyannote/speaker-diarization-3.1",
        device: str = "cpu",
        min_speakers: int = 1,
        max_speakers: int = 10,
    ):
        self.hf_token = hf_token or os.environ.get("HF_TOKEN", "")
        self.model_name = model
        self.device = device
        self.min_speakers = min_speakers
        self.max_speakers = max_speakers
        self._pipeline = None

    def _load(self):
        if self._pipeline is not None:
            return
        try:
            import torch
            from pyannote.audio import Pipeline

            if not self.hf_token:
                raise ValueError(
                    "HuggingFace token required. Set HF_TOKEN env var or pass hf_token=..."
                )

            self._pipeline = Pipeline.from_pretrained(
                self.model_name,
                use_auth_token=self.hf_token,
            )
            if self.device == "cuda" and torch.cuda.is_available():
                self._pipeline = self._pipeline.to(torch.device("cuda"))
            logger.info("pyannote speaker diarization pipeline loaded ✓")
        except ImportError:
            raise ImportError(
                "Install pyannote.audio:\n"
                "  pip install pyannote.audio\n"
                "Then accept model terms at: https://hf.co/pyannote/speaker-diarization-3.1"
            )

    def diarize(self, audio_path: str) -> List[SpeakerTurn]:
        """
        Run speaker diarization on audio file.

        Args:
            audio_path: Path to WAV file (16 kHz mono recommended)

        Returns:
            List of SpeakerTurn sorted by start time
        """
        self._load()

        diarization = self._pipeline(
            audio_path,
            min_speakers=self.min_speakers,
            max_speakers=self.max_speakers,
        )

        turns = []
        for turn, _, speaker in diarization.itertracks(yield_label=True):
            turns.append(SpeakerTurn(
                speaker=speaker,
                start=round(turn.start, 3),
                end=round(turn.end, 3),
                duration=round(turn.end - turn.start, 3),
            ))

        logger.info(f"Diarization complete: {len(turns)} turns, "
                    f"{len({t.speaker for t in turns})} speakers")
        return turns

    def assign_to_segments(
        self,
        transcript_segments: List[Dict],
        diarization: List[SpeakerTurn],
    ) -> List[Dict]:
        """
        Add speaker labels to ASR transcript segments.

        Strategy: for each transcript segment, find the speaker with the
        most overlap in time and assign that label.

        Args:
            transcript_segments: [{"start": ..., "end": ..., "text": ...}]
            diarization: List of SpeakerTurn from diarize()

        Returns:
            Same list with "speaker" key added to each segment
        """
        enriched = []
        for seg in transcript_segments:
            seg = dict(seg)
            seg_start = seg.get("start", 0.0)
            seg_end = seg.get("end", 0.0)

            # Find speaker with maximum overlap in this segment
            best_speaker = "UNKNOWN"
            best_overlap = 0.0
            for turn in diarization:
                overlap = max(
                    0.0,
                    min(seg_end, turn.end) - max(seg_start, turn.start)
                )
                if overlap > best_overlap:
                    best_overlap = overlap
                    best_speaker = turn.speaker

            seg["speaker"] = best_speaker
            enriched.append(seg)

        return enriched

    def get_speaker_stats(self, turns: List[SpeakerTurn]) -> Dict:
        """Return speaking time statistics per speaker."""
        from collections import defaultdict
        stats = defaultdict(lambda: {"total_time_s": 0.0, "turns": 0})
        for t in turns:
            stats[t.speaker]["total_time_s"] += t.duration
            stats[t.speaker]["turns"] += 1
        return {
            k: {
                "total_time_s": round(v["total_time_s"], 2),
                "turns": v["turns"],
                "avg_turn_s": round(v["total_time_s"] / v["turns"], 2),
            }
            for k, v in sorted(stats.items())
        }

    def format_transcript_with_speakers(
        self,
        segments: List[Dict],
    ) -> str:
        """Format diarized transcript as readable dialogue."""
        lines = []
        prev_speaker = None
        for seg in segments:
            speaker = seg.get("speaker", "UNKNOWN")
            start = seg.get("start", 0.0)
            end = seg.get("end", 0.0)
            text = seg.get("text", "")

            if speaker != prev_speaker:
                lines.append(f"\n[{speaker}]  ({start:.1f}s → {end:.1f}s)")
                prev_speaker = speaker

            lines.append(f"  {text}")
        return "\n".join(lines).strip()

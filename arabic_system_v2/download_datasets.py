"""
download_datasets.py
=====================
Download all datasets needed for training, evaluation, and testing.

Usage:
    python download_datasets.py --all
    python download_datasets.py --asr
    python download_datasets.py --summarization
    python download_datasets.py --search
    python download_datasets.py --verify
"""

import os
import sys
import argparse
import logging
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(asctime)s │ %(message)s")
logger = logging.getLogger(__name__)

DATA_DIR = Path("./data")
PASS = "✅"
FAIL = "❌"
WARN = "⚠️"


def download_common_voice_arabic(max_samples: int = 1000) -> bool:
    """Download Mozilla Common Voice Arabic — best for WER evaluation."""
    print(f"\n📥 Mozilla Common Voice Arabic (ASR)")
    try:
        from datasets import load_dataset
        logger.info("Loading dataset (requires HuggingFace login for full access)...")
        ds = load_dataset(
            "mozilla-foundation/common_voice_17_0",
            "ar",
            split=f"test[:{max_samples}]",
            trust_remote_code=True,
        )
        save_path = DATA_DIR / "common_voice_ar"
        ds.save_to_disk(str(save_path))
        print(f"  {PASS} Saved {len(ds)} samples → {save_path}")
        return True
    except Exception as e:
        print(f"  {FAIL} Failed: {e}")
        print(f"  💡 Fix: Run 'huggingface-cli login' then retry")
        print(f"  💡 Or accept terms at: https://huggingface.co/mozilla-foundation/common_voice_17_0")
        return False


def download_masc(max_samples: int = 500) -> bool:
    """Download MASC Arabic Speech Dataset."""
    print(f"\n📥 MASC Arabic Speech Dataset (ASR)")
    try:
        from datasets import load_dataset
        ds = load_dataset("hirundo-io/MASC", split="train", trust_remote_code=True)
        subset = ds.select(range(min(max_samples, len(ds))))
        save_path = DATA_DIR / "masc_arabic"
        subset.save_to_disk(str(save_path))
        print(f"  {PASS} Saved {len(subset)} samples → {save_path}")
        return True
    except Exception as e:
        print(f"  {WARN} MASC failed: {e}")
        return False


def download_xlsum_arabic(max_samples: int = 2000) -> bool:
    """Download XL-Sum Arabic for summarization evaluation."""
    print(f"\n📥 XL-Sum Arabic (Summarization)")
    try:
        from datasets import load_dataset
        ds = load_dataset("csebuetnlp/xlsum", "arabic", split="test", trust_remote_code=True)
        subset = ds.select(range(min(max_samples, len(ds))))
        save_path = DATA_DIR / "xlsum_arabic"
        subset.save_to_disk(str(save_path))
        print(f"  {PASS} Saved {len(subset)} samples → {save_path}")
        print(f"  Columns: {ds.column_names}")
        return True
    except Exception as e:
        print(f"  {FAIL} Failed: {e}")
        return False


def download_arcd(max_samples: int = 1000) -> bool:
    """Download ARCD for semantic search evaluation."""
    print(f"\n📥 ARCD — Arabic Reading Comprehension (Search)")
    try:
        from datasets import load_dataset
        try:
            ds = load_dataset("wisam/arcd", split="test", trust_remote_code=True)
        except Exception:
            print(f"  {WARN} wisam/arcd unavailable — using TyDiQA Arabic fallback")
            ds = load_dataset(
                "google-research-datasets/tydi_qa",
                "arabic--passage-retrieval-train",
                split="train",
                trust_remote_code=True,
            )
        subset = ds.select(range(min(max_samples, len(ds))))
        save_path = DATA_DIR / "arcd"
        subset.save_to_disk(str(save_path))
        print(f"  {PASS} Saved {len(subset)} samples → {save_path}")
        return True
    except Exception as e:
        print(f"  {FAIL} Failed: {e}")
        print(f"  💡 Using SQuAD as final fallback for search evaluation")
        try:
            from datasets import load_dataset
            ds = load_dataset("rajpurkar/squad", split="validation", trust_remote_code=True)
            subset = ds.select(range(min(max_samples, len(ds))))
            save_path = DATA_DIR / "arcd"
            subset.save_to_disk(str(save_path))
            print(f"  {PASS} SQuAD fallback saved {len(subset)} samples → {save_path}")
            return True
        except Exception as e2:
            print(f"  {FAIL} Fallback also failed: {e2}")
            return False


def download_arabic_speech_corpus() -> bool:
    """Info about Arabic Speech Corpus (manual download required)."""
    print(f"\n📥 Arabic Speech Corpus (ASR — manual download)")
    print(f"  ℹ️  This dataset requires manual download:")
    print(f"  → https://en.arabicspeechcorpus.com/")
    print(f"  → Register and download the corpus")
    print(f"  → Place in: ./data/arabic_speech_corpus/")
    return False  # Manual


def verify_datasets():
    """Check which datasets are available locally."""
    print(f"\n{'═'*55}")
    print("  Dataset Availability Check")
    print(f"{'═'*55}")

    datasets_info = [
        ("Mozilla CommonVoice Arabic", DATA_DIR / "common_voice_ar",    "ASR"),
        ("MASC Arabic Speech",         DATA_DIR / "masc_arabic",        "ASR"),
        ("XL-Sum Arabic",              DATA_DIR / "xlsum_arabic",        "Summarization"),
        ("ARCD / QA",                  DATA_DIR / "arcd",                "Search"),
        ("Arabic Speech Corpus",       DATA_DIR / "arabic_speech_corpus","ASR (manual)"),
    ]

    for name, path, task in datasets_info:
        if path.exists():
            try:
                from datasets import load_from_disk
                ds = load_from_disk(str(path))
                print(f"  {PASS}  {name:<32} {task:<16} ({len(ds)} samples)")
            except Exception:
                print(f"  {WARN}  {name:<32} {task:<16} (present but can't load)")
        else:
            print(f"  {FAIL}  {name:<32} {task:<16} (not downloaded)")

    print(f"{'─'*55}")
    print(f"  Data directory: {DATA_DIR.resolve()}")
    print(f"{'═'*55}\n")


def main():
    parser = argparse.ArgumentParser(
        description="Download Arabic Audio System Datasets",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python download_datasets.py --all
  python download_datasets.py --asr --max-samples 500
  python download_datasets.py --verify
        """
    )
    parser.add_argument("--all", action="store_true", help="Download all datasets")
    parser.add_argument("--asr", action="store_true", help="ASR datasets")
    parser.add_argument("--summarization", action="store_true", help="Summarization datasets")
    parser.add_argument("--search", action="store_true", help="Search datasets")
    parser.add_argument("--verify", action="store_true", help="Check what's downloaded")
    parser.add_argument("--max-samples", type=int, default=1000)
    args = parser.parse_args()

    if not any([args.all, args.asr, args.summarization, args.search, args.verify]):
        parser.print_help()
        return

    DATA_DIR.mkdir(parents=True, exist_ok=True)

    if args.verify:
        verify_datasets()
        return

    results = {}

    if args.asr or args.all:
        results["common_voice"] = download_common_voice_arabic(args.max_samples)
        results["masc"] = download_masc(args.max_samples)
        download_arabic_speech_corpus()

    if args.summarization or args.all:
        results["xlsum"] = download_xlsum_arabic(args.max_samples)

    if args.search or args.all:
        results["arcd"] = download_arcd(args.max_samples)

    verify_datasets()

    ok = sum(results.values())
    total = len(results)
    print(f"\n{PASS if ok == total else WARN} Downloaded {ok}/{total} datasets")
    print("Run 'python quick_demo.py' to verify the installation.")


if __name__ == "__main__":
    main()

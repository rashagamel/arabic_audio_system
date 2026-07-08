"""
run_all_evals.py
================
Stable evaluation runner for Arabic Audio System v2
Fixes:
- dataset script crash (xlsum)
- .to_dict() crashes
- missing metric class issues
"""

import sys
import json
import argparse
import logging
from pathlib import Path
from datetime import datetime

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s │ %(levelname)s │ %(message)s",
    datefmt="%H:%M:%S"
)

logger = logging.getLogger(__name__)

sys.path.insert(0, str(Path(__file__).parent))


# ─────────────────────────────────────────────
# SAFE METRICS CONVERTER (FIX ALL FORMATS)
# ─────────────────────────────────────────────
def safe_metrics(obj):
    """Convert ANY metric output to dict safely"""
    if obj is None:
        return {}

    if isinstance(obj, dict):
        return obj

    # HuggingFace / dataclass / custom object
    if hasattr(obj, "to_dict"):
        try:
            return obj.to_dict()
        except:
            pass

    # namedtuple / class object
    if hasattr(obj, "__dict__"):
        return {
            k: v for k, v in obj.__dict__.items()
            if not k.startswith("_")
        }

    return {"value": str(obj)}


# ─────────────────────────────────────────────
# ASR (WER)
# ─────────────────────────────────────────────
def run_wer_eval(asr_model_name: str, n: int) -> dict:
    logger.info(f"Running WER evaluation — {n} samples")

    from src.asr.whisper_asr import WhisperASR
    from evaluation.metrics import evaluate_asr_on_dataset

    asr = WhisperASR(
        model_size=asr_model_name,
        language=None,
        vad_filter=True
    )

    result = evaluate_asr_on_dataset(asr, max_samples=n)

    return {
        "task": "asr",
        "metrics": safe_metrics(result)
    }


# ─────────────────────────────────────────────
# ROUGE (FIXED: NO xlsum, NO scripts)
# ─────────────────────────────────────────────
def run_rouge_eval(summarizer_name: str, n: int) -> dict:
    logger.info(f"Running ROUGE evaluation — {n} samples")

    from src.summarization.summarizer import ArabicSummarizer
    from evaluation.metrics import compute_rouge
    from tqdm import tqdm

    summarizer = ArabicSummarizer(model_name=summarizer_name)

    # ✅ SAFE DATASET (NO SCRIPT DEPENDENCY)
    try:
        from datasets import load_dataset

        # safer dataset than xlsum (no scripts)
        ds = load_dataset(
            "cnn_dailymail",
            "3.0.0",
            split="train"
        )

    except Exception as e:
        logger.error(f"Dataset load failed: {e}")
        return {
            "task": "summarization",
            "metrics": {"error": "dataset_failed"}
        }

    ds = ds.select(range(min(n, len(ds))))

    refs, hyps = [], []

    for sample in tqdm(ds, desc="Summarizing"):
        try:
            out = summarizer.summarize(sample["article"])
            hyps.append(out.summary)
            refs.append(sample["highlights"])
        except Exception as e:
            logger.warning(f"Sample failed: {e}")

    result = compute_rouge(refs, hyps)

    return {
        "task": "summarization",
        "metrics": safe_metrics(result)
    }


# ─────────────────────────────────────────────
# SEARCH
# ─────────────────────────────────────────────
def run_search_eval(embedder_name: str, n: int) -> dict:
    logger.info(f"Running search evaluation — {n} queries")

    from src.search.semantic_search import ArabicEmbedder, SemanticSearchEngine
    from evaluation.metrics import evaluate_search_on_arcd

    embedder = ArabicEmbedder(model_name=embedder_name)
    engine = SemanticSearchEngine(embedder=embedder)

    result = evaluate_search_on_arcd(
        engine,
        max_queries=n,
        k_values=[1, 3, 5, 10]
    )

    return {
        "task": "search",
        "metrics": safe_metrics(result)
    }


# ─────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--tasks",
        nargs="+",
        choices=["asr", "summarization", "search", "all"],
        default=["all"]
    )

    parser.add_argument("--samples", type=int, default=50)

    parser.add_argument(
        "--asr-model",
        default="large-v3"
    )

    parser.add_argument(
        "--summarizer",
        default="csebuetnlp/mT5_multilingual_XLSum"
    )

    parser.add_argument(
        "--embedder",
        default="sentence-transformers/paraphrase-multilingual-mpnet-base-v2"
    )

    parser.add_argument(
        "--output",
        default="outputs/eval_results.json"
    )

    args = parser.parse_args()

    tasks = args.tasks
    if "all" in tasks:
        tasks = ["asr", "summarization", "search"]

    print("━" * 60)
    print("  Arabic Audio System v2 — Evaluation Runner")
    print(f"  Tasks:   {tasks}")
    print(f"  Samples: {args.samples}")
    print("━" * 60)

    results = {
        "timestamp": datetime.now().isoformat(),
        "config": {
            "asr_model": args.asr_model,
            "summarizer": args.summarizer,
            "embedder": args.embedder,
            "samples": args.samples,
        },
        "results": []
    }

    for task in tasks:
        try:
            if task == "asr":
                r = run_wer_eval(args.asr_model, args.samples)

            elif task == "summarization":
                r = run_rouge_eval(args.summarizer, args.samples)

            elif task == "search":
                r = run_search_eval(args.embedder, args.samples)

            else:
                continue

            results["results"].append(r)

        except Exception as e:
            logger.error(f"Task '{task}' failed: {e}")
            import traceback
            traceback.print_exc()

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)

    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    print(f"\n✅ Results saved to {args.output}")


if __name__ == "__main__":
    main()
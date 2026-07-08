"""
demo/app.py
===========

FIXED VERSION:
- Fixed Gradio "str has no _id" crash
- Ensured all inputs are proper Components
- Fixed top_k handling safely
"""

import sys
from pathlib import Path
import logging

sys.path.insert(0, str(Path(__file__).parent.parent))

logging.basicConfig(level=logging.INFO, format="%(asctime)s │ %(message)s")
logger = logging.getLogger(__name__)

# ── CSS ─────────────────────────────────────────────────────────────
CSS = """
.container { max-width: 1200px; margin: auto; }
.arabic-text { direction: rtl; font-size: 16px; line-height: 1.8; }
"""

# ── Pipeline ────────────────────────────────────────────────────────
_pipeline = None


def get_pipeline(asr_model="large-v3", summarizer=None,
                 embedder=None, language=None):
    global _pipeline
    try:
        from src.pipeline import ArabicAudioPipeline

        _pipeline = ArabicAudioPipeline(
            asr_model=asr_model,
            language=language if language != "auto" else None,
            output_dir="outputs",
        )
        return _pipeline, None
    except Exception as e:
        return None, str(e)


# ── FIXED HANDLER ───────────────────────────────────────────────────
def process_audio_handler(
    audio_file,
    youtube_url,
    search_query,
    reference_transcript,
    asr_model_choice,
    summarizer_choice,
    language_choice,
    top_k,
):

    # ✅ SAFE TOP_K CONVERSION (FIX)
    try:
        top_k = int(top_k)
    except Exception:
        top_k = 5

    source = None
    if youtube_url and youtube_url.strip():
        source = youtube_url.strip()
    elif audio_file:
        source = audio_file
    else:
        return ["⚠️ Upload audio or YouTube URL", "", "", "", "", {}, "", ""]

    try:
        pipeline, err = get_pipeline(
            asr_model=asr_model_choice,
            summarizer=summarizer_choice,
            language=language_choice,
        )

        if err:
            return [f"❌ Error: {err}", "", "", "", "", {}, "", ""]

        result = pipeline.process(
            audio_source=source,
            query=search_query or None,
            top_k=top_k,
            save=True,
        )

        transcript_md = result.transcript
        summary_md = result.summary
        search_md = str(result.search_results)

        return [
            transcript_md,
            result.timestamped_transcript,
            summary_md,
            search_md,
            "OK",
            {},
            result.transcript,
            result.summary,
        ]

    except Exception as e:
        import traceback
        return [f"❌ {e}", "", "", "", "", {}, "", ""]


# ── UI ───────────────────────────────────────────────────────────────
def build_ui():
    import gradio as gr

    with gr.Blocks(css=CSS) as demo:

        gr.Markdown("# 🎙️ Arabic Audio System v2")

        with gr.Tab("Process"):

            # ── INPUTS (IMPORTANT FIX: ALL ARE COMPONENTS) ──
            audio_input = gr.Audio(type="filepath")
            yt_input = gr.Textbox()
            search_q = gr.Textbox()

            asr_model_dd = gr.Dropdown(
                ["large-v3", "medium", "small"],
                value="large-v3"
            )

            summarizer_dd = gr.Dropdown(
                ["mT5", "AraBART"],
                value="mT5"
            )

            language_dd = gr.Dropdown(
                ["auto", "ar", "en"],
                value="auto"
            )

            top_k_sl = gr.Slider(1, 20, value=5)

            ref_transcript = gr.Textbox()

            # ── OUTPUTS ──
            transcript_out = gr.Textbox()
            timestamped_out = gr.Textbox()
            summary_out = gr.Textbox()
            search_out = gr.Textbox()
            stats_out = gr.JSON()
            raw1 = gr.Textbox()
            raw2 = gr.Textbox()

            run_btn = gr.Button("Run")

            # ── FIXED CLICK (NO STRINGS INSIDE INPUTS) ──
            run_btn.click(
                fn=process_audio_handler,
                inputs=[
                    audio_input,
                    yt_input,
                    search_q,
                    ref_transcript,
                    asr_model_dd,
                    summarizer_dd,
                    language_dd,
                    top_k_sl,
                ],
                outputs=[
                    transcript_out,
                    timestamped_out,
                    summary_out,
                    search_out,
                    stats_out,
                    raw1,
                    raw2,
                ],
            )

    return demo


# ── MAIN ─────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=7860)
    args = parser.parse_args()

    print("Running...")
    demo = build_ui()
    demo.launch(server_port=args.port)
"""
setup.py — Arabic Audio Understanding & Retrieval System v2
"""
from setuptools import setup, find_packages
from pathlib import Path

long_description = (Path(__file__).parent / "README.md").read_text(encoding="utf-8")

setup(
    name="arabic-audio-system",
    version="2.0.0",
    description="Deep Learning Based Arabic Audio Understanding and Retrieval System",
    long_description=long_description,
    long_description_content_type="text/markdown",
    python_requires=">=3.9",
    packages=find_packages(where="."),
    install_requires=[
        "torch>=2.1.0",
        "torchaudio>=2.1.0",
        "transformers>=4.40.0",
        "datasets>=2.18.0",
        "openai-whisper>=20231117",
        "faster-whisper>=1.0.0",
        "sentence-transformers>=2.6.0",
        "faiss-cpu>=1.8.0",
        "gradio>=4.26.0",
        "yt-dlp>=2024.3.10",
        "librosa>=0.10.1",
        "soundfile>=0.12.1",
        "pydub>=0.25.1",
        "jiwer>=3.0.4",
        "rouge-score>=0.1.2",
        "scikit-learn>=1.4.0",
        "pyarabic>=0.6.15",
        "tqdm>=4.66.2",
        "pyyaml>=6.0.1",
        "numpy>=1.26.0",
    ],
    extras_require={
        "advanced": ["pyannote.audio>=3.2.0", "speechbrain>=1.0.0"],
        "dev":      ["pytest>=8.1.0", "pytest-cov>=5.0.0"],
        "gpu":      ["faiss-gpu>=1.8.0"],
        "bertscore": ["bert-score>=0.3.13"],
    },
    entry_points={
        "console_scripts": [
            "arabic-audio=src.pipeline:main",
            "arabic-demo=demo.app:main",
            "arabic-benchmark=evaluation.run_benchmark:main",
        ],
    },
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
    ],
)

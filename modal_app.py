"""
modal_app.py
────────────
Decoupled API serverless backend for VocalMirror running on Modal.com.
Provides a high-speed, GPU-accelerated REST API endpoint `/analyze` that
processes audio uploads on an A10G GPU and returns structured JSON analysis results.

Usage:
    modal deploy modal_app.py
"""

from __future__ import annotations

import os
import sys
import tempfile
import shutil
import logging
import modal

logger = logging.getLogger(__name__)

# Define the Modal App
app = modal.App("vocal-mirror")

# Configure the Docker container image with baked-in models & dependencies
image = (
    modal.Image.debian_slim(python_version="3.10")
    .apt_install("libsndfile1", "ffmpeg")
    .pip_install_from_requirements("requirements.txt")
    .run_commands("python -m spacy download en_core_web_sm")
    # Bake in HF pipelines at build time so runtime startup is instant
    .run_commands(
        "python -c 'from transformers import pipeline; "
        "pipeline(\"text-classification\", model=\"cardiffnlp/twitter-roberta-base-sentiment-latest\", top_k=None); "
        "pipeline(\"text-classification\", model=\"j-hartmann/emotion-english-distilroberta-base\", top_k=None)'"
    )
    # Bake in Whisper model weights at build time
    .run_commands(
        "python -c 'import whisper; whisper.load_model(\"tiny\"); whisper.load_model(\"small\")'"
    )
    # Mount files from current dir to /root in the container
    .add_local_dir(".", remote_path="/root")
)


@app.function(
    image=image,
    gpu="A10G",
)
@modal.concurrent(max_inputs=100)
@modal.asgi_app()
def serve():
    from fastapi import FastAPI, Request, Query, HTTPException
    from fastapi.middleware.cors import CORSMiddleware
    from starlette.datastructures import UploadFile as StarletteUploadFile
    import gc
    import torch

    # Ensure python knows where to import from
    sys.path.append("/root")
    os.chdir("/root")

    from modules.speech import transcribe_audio, extract_acoustic_features, compute_acoustic_nervousness
    from modules.nlp import score_sentiment, score_confidence, detect_fillers, analyze_vocabulary
    from modules.fusion import fuse_scores, aggregate_nlp_outputs

    # Create FastAPI host
    fastapi_app = FastAPI(title="VocalMirror Backend API")

    # Add robust CORS middleware for wildcard origins (allow_credentials must be False for wildcard)
    fastapi_app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=False,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    @fastapi_app.post("/analyze")
    async def analyze(
        request: Request,
        model_size: str = Query("small", regex="^(tiny|small)$"),
    ):
        """
        Stateless endpoint to analyze audio file.
        Uses Request form data to bypass Pydantic v2 UploadFile forward reference schemas.
        """
        # Parse form data dynamically
        form = await request.form()
        file = form.get("file")

        if not file or not isinstance(file, StarletteUploadFile):
            raise HTTPException(
                status_code=400,
                detail="No audio file package found in multipart parameter 'file'.",
            )

        # Create temp file to save the uploaded stream
        suffix = os.path.splitext(file.filename or ".wav")[1]
        if not suffix:
            suffix = ".wav"
        
        tmp = tempfile.NamedTemporaryFile(suffix=suffix, delete=False)
        tmp_path = tmp.name
        
        try:
            # Write uploaded data to temp file
            shutil.copyfileobj(file.file, tmp)
            tmp.close()

            # Execute pipeline
            # 1. Speech Transcription
            transcription = transcribe_audio(tmp_path, model_size=model_size)
            text = transcription.get("text", "").strip()
            
            if not text:
                raise HTTPException(
                    status_code=400,
                    detail="Audio could not be transcribed. Please check recording quality or duration.",
                )

            # 2. Acoustic Feature Extraction
            acoustic_features = extract_acoustic_features(tmp_path)
            acoustic_scores = compute_acoustic_nervousness(acoustic_features)

            # 3. NLP Scoring
            sentiment = score_sentiment(text)
            confidence = score_confidence(text)
            fillers = detect_fillers(text)
            vocab = analyze_vocabulary(text)

            # 4. Fusion Scoring
            nlp_aggregated = aggregate_nlp_outputs(sentiment, confidence, fillers, vocab)
            fusion = fuse_scores(acoustic_scores, nlp_aggregated)

            # 5. Build structured response
            response_payload = {
                "text": text,
                "found_fillers": fillers.get("found_fillers", []),
                "filler_count": fillers.get("filler_count", 0),
                "filler_density": fillers.get("filler_density", 0.0),
                "severity": fusion.get("severity", "medium"),
                "interpretation": fusion.get("interpretation", ""),
                "radar": fusion.get("radar", {}),
                "metrics": {
                    "pitch_variance_norm": float(acoustic_scores.get("pitch_variance_norm", 0.0)),
                    "speech_rate_norm": float(acoustic_scores.get("speech_rate_norm", 0.0)),
                    "pause_freq_norm": float(acoustic_scores.get("pause_freq_norm", 0.0)),
                    "positive_sentiment": float(sentiment.get("positive", 0.0)),
                    "neutral_sentiment": float(sentiment.get("neutral", 0.0)),
                    "negative_sentiment": float(sentiment.get("negative", 0.0)),
                    "emotion_confidence": float(confidence.get("emotion_confidence", 0.0)),
                    "lexicon_confidence": float(confidence.get("lexicon_confidence", 0.0)),
                    "vocab_richness": float(vocab.get("vocab_richness", 0.0)),
                    "syntactic_fluency": float(vocab.get("syntactic_fluency", 0.0)),
                }
            }
            return response_payload

        except HTTPException:
            raise
        except Exception as exc:
            logger.error("API analysis failed: %s", exc, exc_info=True)
            raise HTTPException(status_code=500, detail=f"Speech analysis failed: {str(exc)}")
        
        finally:
            # Clean up temp audio file
            if os.path.exists(tmp_path):
                try:
                    os.unlink(tmp_path)
                except Exception:
                    pass
            # Trigger garbage collection and purge GPU cache
            gc.collect()
            if torch.backends.mps.is_available():
                torch.mps.empty_cache()

    return fastapi_app

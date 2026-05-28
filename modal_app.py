"""
modal_app.py
────────────
Modal deployment definition for VocalMirror.
Bakes HF/Whisper models directly into the container image to eliminate startup latency,
runs the speech modules on an A10G cloud GPU, and exposes the Gradio app via FastAPI.

Usage:
    modal deploy modal_app.py
"""

from __future__ import annotations

import os
import sys
import modal

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
    from fastapi import FastAPI
    from fastapi.middleware.cors import CORSMiddleware
    from gradio.routes import mount_gradio_app

    # Ensure python knows where to import from
    sys.path.append("/root")
    os.chdir("/root")

    from app import build_ui

    # Create FastAPI host
    fastapi_app = FastAPI(title="VocalMirror Backend")

    # Add robust CORS middleware to allow cross-origin requests from Vercel
    # Note: allow_credentials MUST be False when using wildcard '*' in allow_origins
    fastapi_app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=False,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # Security middleware to strip SAMEORIGIN and allow iframe embedding everywhere
    @fastapi_app.middleware("http")
    async def bypass_iframe_restrictions(request, call_next):
        response = await call_next(request)
        # Safe deletion from MutableHeaders in Starlette/FastAPI using 'del' instead of '.pop()'
        if "X-Frame-Options" in response.headers:
            del response.headers["X-Frame-Options"]
        # Add modern Content Security Policy for cross-domain embedding
        response.headers["Content-Security-Policy"] = "frame-ancestors *"
        return response

    # Build Gradio Block UI
    demo, theme, css = build_ui()

    # Mount Gradio block to FastAPI root
    return mount_gradio_app(fastapi_app, demo, path="/")

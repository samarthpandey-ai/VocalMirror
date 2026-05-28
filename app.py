"""
app.py — VocalMirror Gradio Application
────────────────────────────────────────
Entry point for the VocalMirror multimodal AI public speaking coach.
Structured for HuggingFace Spaces deployment + iframe embed in Vercel portfolio.

Usage:
    python app.py

Environment variables (optional):
    VM_SERVER_NAME  : Custom server name for HF Spaces (default: "vocalmirror")
    VM_SHARE        : Set to "true" to enable Gradio share link (default: false)
"""

from __future__ import annotations

import os
import tempfile
import traceback
from pathlib import Path
from typing import Optional

import gradio as gr
import numpy as np

# ── Module imports (lazy-load heavy models inside each module) ────────────────
from modules.speech import transcribe_audio, extract_acoustic_features, compute_acoustic_nervousness
from modules.nlp import score_sentiment, score_confidence, detect_fillers, analyze_vocabulary
from modules.fusion import fuse_scores, aggregate_nlp_outputs

# ── UI helpers ────────────────────────────────────────────────────────────────
from ui.radar_chart import build_radar_chart, build_empty_radar
from ui.theme import CUSTOM_CSS, get_theme


# ──────────────────────────────────────────────────────────────────────────────
# Core analysis pipeline
# ──────────────────────────────────────────────────────────────────────────────

def analyze_speech(
    audio_input,
    model_size: str = "small",
) -> tuple:
    """
    Full VocalMirror pipeline: audio → radar chart + incongruence score.

    Args:
        audio_input: Audio data from gr.Audio — can be (sample_rate, np.ndarray)
                     or a filepath string depending on Gradio version.
        model_size:  Whisper model size ("tiny" or "small").

    Returns:
        Tuple of (transcript_html, radar_figure, score_html, metrics_html)
        for Gradio output components.
    """
    if audio_input is None:
        return (
            "<p style='color:#8b949e'>⬆ Upload or record audio to begin analysis.</p>",
            build_empty_radar(),
            _render_score_html(None, None, None),
            _render_metrics_html(None),
        )

    try:
        # ── Step 1: Save audio to temp file if needed ──────────────────────
        audio_path = _ensure_audio_file(audio_input)

        # ── Step 2: Speech Module ──────────────────────────────────────────
        transcription = transcribe_audio(audio_path, model_size=model_size)
        acoustic_features = extract_acoustic_features(audio_path)
        acoustic_scores = compute_acoustic_nervousness(acoustic_features)

        text = transcription.get("text", "").strip()
        if not text:
            return (
                "<p style='color:#ff5733'>⚠ Could not transcribe audio. Please check the recording quality.</p>",
                build_empty_radar(),
                _render_score_html(None, None, None),
                _render_metrics_html(None),
            )

        # ── Step 3: NLP Module ─────────────────────────────────────────────
        sentiment  = score_sentiment(text)
        confidence = score_confidence(text)
        fillers    = detect_fillers(text)
        vocab      = analyze_vocabulary(text)

        # ── Step 4: Fusion ─────────────────────────────────────────────────
        nlp_aggregated = aggregate_nlp_outputs(sentiment, confidence, fillers, vocab)
        fusion = fuse_scores(acoustic_scores, nlp_aggregated)

        radar_dims = fusion["radar"]
        severity   = fusion["severity"]
        interp     = fusion["interpretation"]

        # ── Step 5: Render outputs ─────────────────────────────────────────
        transcript_html = _render_transcript(text, fillers.get("found_fillers", []))
        radar_fig       = build_radar_chart(radar_dims, severity=severity)
        score_html      = _render_score_html(radar_dims["incongruence_score"], severity, interp)
        metrics_html    = _render_metrics_html({
            "acoustic_scores": acoustic_scores,
            "sentiment":       sentiment,
            "confidence":      confidence,
            "fillers":         fillers,
            "vocab":           vocab,
            "radar":           radar_dims,
        })

        return transcript_html, radar_fig, score_html, metrics_html

    except Exception as exc:  # noqa: BLE001
        tb = traceback.format_exc()
        error_html = (
            f"<div style='color:#ff5733; font-family: JetBrains Mono, monospace; "
            f"font-size:0.82rem; white-space:pre-wrap; padding:1rem; "
            f"background:rgba(255,87,51,0.08); border-radius:8px; border:1px solid rgba(255,87,51,0.2)'>"
            f"<b>Analysis Error</b>\n\n{str(exc)}\n\n{tb}</div>"
        )
        return error_html, build_empty_radar(), _render_score_html(None, None, str(exc)), _render_metrics_html(None)


# ──────────────────────────────────────────────────────────────────────────────
# HTML rendering helpers
# ──────────────────────────────────────────────────────────────────────────────

def _ensure_audio_file(audio_input) -> str:
    """
    Ensure we have a filepath string. Gradio may pass either a filepath or
    a (sample_rate, ndarray) tuple depending on version.
    """
    if isinstance(audio_input, str):
        return audio_input
    if isinstance(audio_input, (list, tuple)) and len(audio_input) == 2:
        import soundfile as sf
        sr, data = audio_input
        if isinstance(data, np.ndarray):
            # Convert to float32 if needed
            if data.dtype != np.float32:
                data = data.astype(np.float32)
                if data.max() > 1.0:
                    data = data / 32768.0
            tmp = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
            sf.write(tmp.name, data, sr)
            return tmp.name
    # Gradio 4+ filepath case
    if hasattr(audio_input, "name"):
        return audio_input.name
    raise ValueError(f"Unsupported audio input type: {type(audio_input)}")


def _highlight_fillers(text: str, fillers_found: list[str]) -> str:
    """Wrap filler words in styled <mark> tags for transcript display."""
    import re
    if not fillers_found:
        return text
    # Sort by length descending to match multi-word first
    sorted_fillers = sorted(set(fillers_found), key=len, reverse=True)
    for filler in sorted_fillers:
        pattern = re.compile(r'\b' + re.escape(filler) + r'\b', re.IGNORECASE)
        text = pattern.sub(
            f'<mark style="background:rgba(255,195,0,0.25);color:#ffc300;'
            f'border-radius:3px;padding:0 2px;">{filler}</mark>',
            text
        )
    return text


def _render_transcript(text: str, fillers_found: list[str]) -> str:
    """Render transcript as HTML with highlighted filler words."""
    highlighted = _highlight_fillers(text, fillers_found)
    filler_count = len(fillers_found)
    filler_note = (
        f"<div style='margin-top:0.75rem;font-size:0.8rem;color:#8b949e;'>"
        f"<span style='color:#ffc300'>●</span> "
        f"{filler_count} filler word{'s' if filler_count != 1 else ''} highlighted in amber</div>"
        if filler_count > 0
        else "<div style='margin-top:0.75rem;font-size:0.8rem;color:#38ef7d;'>✓ No filler words detected</div>"
    )
    return (
        f"<div style='font-family: JetBrains Mono, monospace; font-size:0.88rem; "
        f"line-height:1.7; color:#c9d1d9; padding:0.5rem;'>{highlighted}</div>"
        f"{filler_note}"
    )


def _render_score_html(
    score: Optional[float],
    severity: Optional[str],
    interpretation: Optional[str],
) -> str:
    """Render the Incongruence Score badge + interpretation."""
    if score is None:
        return (
            "<div style='text-align:center;padding:2rem;color:#8b949e;'>"
            "<div style='font-size:1.5rem;margin-bottom:0.5rem'>🎙</div>"
            "<p style='margin:0'>Incongruence score will appear here after analysis.</p></div>"
        )

    pct = int(round(score * 100))
    color_map = {
        "low":    ("#38ef7d", "rgba(56,239,125,0.12)",  "rgba(56,239,125,0.3)"),
        "medium": ("#ffc300", "rgba(255,195,0,0.12)",   "rgba(255,195,0,0.3)"),
        "high":   ("#ff5733", "rgba(255,87,51,0.12)",   "rgba(255,87,51,0.3)"),
    }
    color, bg, border = color_map.get(severity or "medium", color_map["medium"])

    return f"""
<div style='text-align:center;padding:1.5rem;'>
  <div style='font-size:0.75rem;font-weight:600;text-transform:uppercase;
              letter-spacing:0.1em;color:#8b949e;margin-bottom:0.75rem;
              font-family:Inter,sans-serif;'>
    Incongruence Score
  </div>
  <div style='display:inline-block;font-size:3rem;font-weight:700;
              font-family:Inter,sans-serif;padding:0.5rem 1.5rem;
              border-radius:999px;letter-spacing:-1px;
              color:{color};background:{bg};border:1px solid {border};
              margin-bottom:1rem;'>
    {pct}%
  </div>
  <div style='font-size:0.92rem;line-height:1.6;color:#c9d1d9;
              font-family:Inter,sans-serif;max-width:400px;margin:0 auto;
              padding:0.75rem 1rem;background:rgba(255,255,255,0.03);
              border-radius:10px;border:1px solid rgba(255,255,255,0.07);'>
    {interpretation or ""}
  </div>
</div>"""


def _render_metrics_html(data: Optional[dict]) -> str:
    """Render a compact metrics breakdown table."""
    if data is None:
        return "<div style='color:#8b949e;font-size:0.88rem;padding:0.5rem;'>Metrics will appear after analysis.</div>"

    radar = data.get("radar", {})
    acoustic = data.get("acoustic_scores", {})
    fillers = data.get("fillers", {})

    def bar(value: float, color: str = "#38ef7d") -> str:
        pct = int(value * 100)
        return (
            f"<div style='display:flex;align-items:center;gap:0.5rem;'>"
            f"<div style='flex:1;height:6px;background:rgba(255,255,255,0.08);border-radius:999px;overflow:hidden;'>"
            f"<div style='width:{pct}%;height:100%;background:{color};border-radius:999px;transition:width 0.5s ease;'></div>"
            f"</div><span style='font-size:0.78rem;color:#8b949e;min-width:32px;text-align:right;'>{pct}%</span></div>"
        )

    rows = [
        ("Vocal Confidence",      radar.get("vocal_confidence", 0),      "#38ef7d"),
        ("Clarity",               radar.get("clarity", 0),               "#a8edea"),
        ("Filler Control",        radar.get("filler_word_density", 0),   "#ffc300"),
        ("Linguistic Sentiment",  radar.get("linguistic_sentiment", 0),  "#7c83fd"),
        ("Pitch Variance (norm)", acoustic.get("pitch_variance_norm", 0),"#ff9a9e"),
        ("Filler Words Found",    min(fillers.get("filler_count", 0) / 20, 1.0), "#ffc300"),
    ]

    rows_html = "".join(
        f"<tr>"
        f"<td style='padding:0.4rem 0.5rem;color:#8b949e;font-size:0.8rem;"
        f"white-space:nowrap;font-family:Inter,sans-serif;'>{label}</td>"
        f"<td style='padding:0.4rem 0.5rem;min-width:160px;'>{bar(val, col)}</td>"
        f"</tr>"
        for label, val, col in rows
    )

    return (
        f"<table style='width:100%;border-collapse:collapse;'>"
        f"<tbody>{rows_html}</tbody></table>"
    )


# ──────────────────────────────────────────────────────────────────────────────
# Gradio UI
# ──────────────────────────────────────────────────────────────────────────────

def build_ui() -> tuple[gr.Blocks, gr.themes.Base, str]:
    theme = get_theme()

    with gr.Blocks(
        title="VocalMirror — AI Speaking Coach",
    ) as demo:

        # ── Header ──────────────────────────────────────────────────────────
        gr.HTML("""
        <div id="vm-header">
          <h1>🎙 VocalMirror</h1>
          <p>AI-powered speaking coach · Discover the gap between your words and your voice</p>
        </div>
        """)

        # ── Main layout ──────────────────────────────────────────────────────
        with gr.Row(equal_height=False):

            # Left column — Input
            with gr.Column(scale=1, min_width=300):
                gr.HTML("<div style='font-size:0.75rem;text-transform:uppercase;letter-spacing:0.1em;color:#8b949e;font-weight:600;margin-bottom:0.5rem;'>Audio Input</div>")

                audio_input = gr.Audio(
                    sources=["microphone", "upload"],
                    type="filepath",
                    label="Record or Upload Speech",
                    elem_id="vm-audio-input",
                )

                model_toggle = gr.Radio(
                    choices=["tiny", "small"],
                    value="small",
                    label="Whisper Model",
                    info="tiny = faster (~5s) · small = more accurate (~20s)",
                    elem_id="vm-model-toggle",
                )

                analyze_btn = gr.Button(
                    "✦ Analyze My Speech",
                    variant="primary",
                    elem_id="vm-analyze-btn",
                    size="lg",
                )

                gr.HTML("<hr class='vm-divider'>")
                gr.HTML("""
                <div style='font-size:0.78rem;color:#8b949e;line-height:1.6;padding:0.25rem;'>
                  <b style='color:#c9d1d9;'>How it works</b><br>
                  1. Record or upload a 15–120 second speech clip<br>
                  2. Whisper transcribes → librosa extracts acoustic patterns<br>
                  3. RoBERTa scores linguistic confidence + sentiment<br>
                  4. Fusion engine computes your <b style='color:#38ef7d'>Incongruence Score</b>
                </div>
                """)

            # Center column — Radar + Score
            with gr.Column(scale=2, min_width=420):
                radar_plot = gr.Plot(
                    value=build_empty_radar(),
                    label="Speaking Profile Radar",
                    elem_id="vm-radar-plot",
                    show_label=False,
                )
                score_html = gr.HTML(
                    value=_render_score_html(None, None, None),
                    elem_id="vm-incongruence-box",
                )

            # Right column — Transcript + Metrics
            with gr.Column(scale=1, min_width=280):
                gr.HTML("<div style='font-size:0.75rem;text-transform:uppercase;letter-spacing:0.1em;color:#8b949e;font-weight:600;margin-bottom:0.5rem;'>Transcript</div>")
                transcript_html = gr.HTML(
                    value="<p style='color:#8b949e;font-size:0.88rem;'>Transcript will appear here after analysis.</p>",
                    elem_id="vm-transcript",
                )
                gr.HTML("<hr class='vm-divider'>")
                gr.HTML("<div style='font-size:0.75rem;text-transform:uppercase;letter-spacing:0.1em;color:#8b949e;font-weight:600;margin-bottom:0.5rem;'>Dimension Breakdown</div>")
                metrics_html = gr.HTML(
                    value=_render_metrics_html(None),
                    elem_id="vm-metrics",
                )

        # ── Footer ───────────────────────────────────────────────────────────
        gr.HTML("""
        <div style='text-align:center;padding:1.5rem 0 0.5rem;color:#8b949e;font-size:0.78rem;'>
          Built with Whisper · librosa · RoBERTa · spaCy · Gradio · Plotly<br>
          <span style='color:rgba(255,255,255,0.15);'>VocalMirror v1.0 · For portfolio embedding, host on HuggingFace Spaces</span>
        </div>
        """)

        # ── Event wiring ─────────────────────────────────────────────────────
        analyze_btn.click(
            fn=analyze_speech,
            inputs=[audio_input, model_toggle],
            outputs=[transcript_html, radar_plot, score_html, metrics_html],
            api_name="analyze",
        )

    return demo, theme, CUSTOM_CSS


# ──────────────────────────────────────────────────────────────────────────────
# Launch
# ──────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    demo, theme, css = build_ui()
    demo.launch(
        server_name=os.getenv("VM_SERVER_NAME", "0.0.0.0"),
        server_port=int(os.getenv("VM_PORT", "7860")),
        share=os.getenv("VM_SHARE", "false").lower() == "true",
        favicon_path=None,
        # Gradio 6: theme and css are passed to launch()
        theme=theme,
        css=css,
        # iframe embed config — safe for Vercel portfolio
        allowed_paths=[str(Path(__file__).parent)],
    )

"""
ui/radar_chart.py
──────────────────
Builds the 5-dimension Plotly radar chart for VocalMirror.
Returns a plotly.graph_objects.Figure — no Gradio dependency here.
"""

from __future__ import annotations
import plotly.graph_objects as go
from typing import Optional


# ──────────────────────────────────────────────────────────────────────────────
# Constants
# ──────────────────────────────────────────────────────────────────────────────

RADAR_LABELS = [
    "Vocal Confidence",
    "Clarity",
    "Filler Control",      # filler_word_density (inverted, so higher=better)
    "Linguistic Sentiment",
    "Congruence",          # 1 - incongruence_score (inverted so higher=better)
]

SEVERITY_COLORS = {
    "low":    {"fill": "rgba(56, 239, 125, 0.25)", "line": "#38ef7d"},
    "medium": {"fill": "rgba(255, 195, 0,   0.25)", "line": "#ffc300"},
    "high":   {"fill": "rgba(255, 87,  51,  0.25)", "line": "#ff5733"},
}

BACKGROUND_COLOR = "#0d1117"
GRIDLINE_COLOR   = "rgba(255,255,255,0.12)"
LABEL_COLOR      = "#c9d1d9"
TICK_COLOR       = "rgba(255,255,255,0.35)"


# ──────────────────────────────────────────────────────────────────────────────
# Chart builder
# ──────────────────────────────────────────────────────────────────────────────

def build_radar_chart(
    radar_dimensions: dict,
    severity: str = "medium",
    title: Optional[str] = None,
) -> go.Figure:
    """
    Build a styled 5-dimension radar (spider) chart.

    Args:
        radar_dimensions: Dict with keys:
            vocal_confidence, clarity, filler_word_density,
            linguistic_sentiment, incongruence_score — all in [0, 1].
        severity: One of "low", "medium", "high" — controls chart color.
        title: Optional chart title override.

    Returns:
        A Plotly Figure object ready for gr.Plot() rendering.
    """
    # Map dimensions to radar order — invert incongruence so higher = better
    values = [
        radar_dimensions.get("vocal_confidence",    0.5),
        radar_dimensions.get("clarity",             0.5),
        radar_dimensions.get("filler_word_density", 0.5),
        radar_dimensions.get("linguistic_sentiment",0.5),
        1.0 - radar_dimensions.get("incongruence_score", 0.5),  # inverted
    ]
    # Close the polygon by repeating first value
    values_closed  = values + [values[0]]
    labels_closed  = RADAR_LABELS + [RADAR_LABELS[0]]

    colors = SEVERITY_COLORS.get(severity, SEVERITY_COLORS["medium"])

    # ── Trace ────────────────────────────────────────────────────────────────
    trace = go.Scatterpolar(
        r=values_closed,
        theta=labels_closed,
        fill="toself",
        fillcolor=colors["fill"],
        line=dict(color=colors["line"], width=2.5),
        marker=dict(size=6, color=colors["line"], opacity=0.9),
        name="Your Score",
        hovertemplate="<b>%{theta}</b><br>Score: %{r:.2f}<extra></extra>",
    )

    # ── Benchmark ghost trace (ideal speaker = 0.85 on all dims) ────────────
    ideal = [0.85] * len(RADAR_LABELS)
    ideal_closed = ideal + [ideal[0]]

    ghost_trace = go.Scatterpolar(
        r=ideal_closed,
        theta=labels_closed,
        fill="toself",
        fillcolor="rgba(255,255,255,0.04)",
        line=dict(color="rgba(255,255,255,0.2)", width=1, dash="dot"),
        marker=dict(size=0),
        name="Ideal Speaker",
        hovertemplate="<b>%{theta}</b><br>Ideal: %{r:.2f}<extra></extra>",
    )

    # ── Layout ───────────────────────────────────────────────────────────────
    chart_title = title or "VocalMirror — Speaking Profile"

    layout = go.Layout(
        title=dict(
            text=chart_title,
            font=dict(family="Inter, sans-serif", size=18, color="#e6edf3"),
            x=0.5,
            xanchor="center",
            pad=dict(b=10),
        ),
        polar=dict(
            bgcolor=BACKGROUND_COLOR,
            radialaxis=dict(
                visible=True,
                range=[0, 1],
                tickvals=[0.25, 0.5, 0.75, 1.0],
                ticktext=["25%", "50%", "75%", "100%"],
                tickfont=dict(color=TICK_COLOR, size=10),
                gridcolor=GRIDLINE_COLOR,
                linecolor=GRIDLINE_COLOR,
                angle=90,
            ),
            angularaxis=dict(
                tickfont=dict(
                    family="Inter, sans-serif",
                    size=13,
                    color=LABEL_COLOR,
                ),
                linecolor=GRIDLINE_COLOR,
                gridcolor=GRIDLINE_COLOR,
            ),
        ),
        paper_bgcolor=BACKGROUND_COLOR,
        plot_bgcolor=BACKGROUND_COLOR,
        showlegend=True,
        legend=dict(
            font=dict(color=LABEL_COLOR, size=11, family="Inter, sans-serif"),
            bgcolor="rgba(255,255,255,0.05)",
            bordercolor="rgba(255,255,255,0.1)",
            borderwidth=1,
            x=0.82,
            y=0.02,
        ),
        margin=dict(l=60, r=60, t=80, b=40),
        height=420,
    )

    fig = go.Figure(data=[ghost_trace, trace], layout=layout)
    return fig


def build_empty_radar() -> go.Figure:
    """Return a placeholder radar chart with all zeros — shown before analysis."""
    empty_dims = {
        "vocal_confidence":    0.0,
        "clarity":             0.0,
        "filler_word_density": 0.0,
        "linguistic_sentiment":0.0,
        "incongruence_score":  0.0,
    }
    fig = build_radar_chart(empty_dims, severity="medium", title="Awaiting Analysis…")
    return fig

"""
ui/theme.py
───────────
VocalMirror Gradio theming — dark glass aesthetic.
Exports:
  - CUSTOM_CSS   : raw CSS string injected into Gradio app
  - get_theme()  : returns a gr.themes.Base() subclass
"""

import gradio as gr

# ──────────────────────────────────────────────────────────────────────────────
# Custom CSS — glassmorphism dark design
# ──────────────────────────────────────────────────────────────────────────────

CUSTOM_CSS = """
/* ── Google Fonts ─────────────────────────────────────────── */
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&family=JetBrains+Mono:wght@400;500&display=swap');

/* ── Root tokens ──────────────────────────────────────────── */
:root {
  --vm-bg-deep:      #060b14;
  --vm-bg-card:      rgba(13, 17, 23, 0.85);
  --vm-glass:        rgba(255, 255, 255, 0.04);
  --vm-glass-border: rgba(255, 255, 255, 0.09);
  --vm-accent:       #38ef7d;
  --vm-accent-2:     #11998e;
  --vm-warn:         #ffc300;
  --vm-danger:       #ff5733;
  --vm-text-primary: #e6edf3;
  --vm-text-muted:   #8b949e;
  --vm-radius:       14px;
  --vm-radius-sm:    8px;
}

/* ── Body & overall background ───────────────────────────── */
body, .gradio-container, .main {
  background: radial-gradient(ellipse at 20% 30%, rgba(17, 153, 142, 0.08) 0%, transparent 60%),
              radial-gradient(ellipse at 80% 70%, rgba(56, 239, 125, 0.05) 0%, transparent 60%),
              var(--vm-bg-deep) !important;
  font-family: 'Inter', sans-serif !important;
  color: var(--vm-text-primary) !important;
  min-height: 100vh;
}

/* ── Header / hero ───────────────────────────────────────── */
#vm-header {
  text-align: center;
  padding: 2.5rem 1rem 1.5rem;
}

#vm-header h1 {
  font-size: clamp(2rem, 5vw, 3.2rem);
  font-weight: 700;
  background: linear-gradient(135deg, #38ef7d 0%, #11998e 50%, #a8edea 100%);
  -webkit-background-clip: text;
  -webkit-text-fill-color: transparent;
  background-clip: text;
  letter-spacing: -0.5px;
  margin: 0;
}

#vm-header p {
  color: var(--vm-text-muted);
  font-size: 1.05rem;
  margin-top: 0.5rem;
  font-weight: 300;
}

/* ── Glass card panels ───────────────────────────────────── */
.vm-card {
  background: var(--vm-glass);
  backdrop-filter: blur(20px);
  -webkit-backdrop-filter: blur(20px);
  border: 1px solid var(--vm-glass-border);
  border-radius: var(--vm-radius);
  padding: 1.25rem;
  transition: border-color 0.3s ease, box-shadow 0.3s ease;
}

.vm-card:hover {
  border-color: rgba(56, 239, 125, 0.2);
  box-shadow: 0 0 24px rgba(56, 239, 125, 0.06);
}

/* ── Gradio component overrides ──────────────────────────── */
.gr-form, .gr-box {
  background: var(--vm-glass) !important;
  border: 1px solid var(--vm-glass-border) !important;
  border-radius: var(--vm-radius) !important;
}

/* Labels */
.gr-form label, label.svelte-1b6s6xi, .label-wrap span {
  color: var(--vm-text-muted) !important;
  font-size: 0.8rem !important;
  font-weight: 500 !important;
  text-transform: uppercase !important;
  letter-spacing: 0.08em !important;
}

/* Buttons */
button.primary, #vm-analyze-btn {
  background: linear-gradient(135deg, #38ef7d, #11998e) !important;
  color: #060b14 !important;
  font-weight: 700 !important;
  font-family: 'Inter', sans-serif !important;
  font-size: 1rem !important;
  border: none !important;
  border-radius: var(--vm-radius) !important;
  padding: 0.75rem 2rem !important;
  transition: transform 0.2s ease, box-shadow 0.2s ease !important;
  cursor: pointer !important;
}

button.primary:hover, #vm-analyze-btn:hover {
  transform: translateY(-2px) !important;
  box-shadow: 0 8px 25px rgba(56, 239, 125, 0.3) !important;
}

button.secondary {
  background: var(--vm-glass) !important;
  border: 1px solid var(--vm-glass-border) !important;
  color: var(--vm-text-primary) !important;
  border-radius: var(--vm-radius) !important;
}

/* Radio buttons (model toggle) */
.gr-radio-group label {
  background: var(--vm-glass) !important;
  border: 1px solid var(--vm-glass-border) !important;
  border-radius: var(--vm-radius-sm) !important;
  padding: 0.4rem 0.9rem !important;
  color: var(--vm-text-primary) !important;
  transition: all 0.2s ease !important;
}

.gr-radio-group label:has(input:checked) {
  background: rgba(56, 239, 125, 0.15) !important;
  border-color: var(--vm-accent) !important;
  color: var(--vm-accent) !important;
}

/* Audio input */
.gr-audio {
  border-radius: var(--vm-radius) !important;
  background: var(--vm-glass) !important;
}

/* Textbox (transcript) */
textarea, .gr-text-input {
  background: rgba(0, 0, 0, 0.3) !important;
  color: var(--vm-text-primary) !important;
  border-color: var(--vm-glass-border) !important;
  border-radius: var(--vm-radius-sm) !important;
  font-family: 'JetBrains Mono', monospace !important;
  font-size: 0.88rem !important;
}

/* ── Incongruence badge ───────────────────────────────────── */
#vm-incongruence-box {
  text-align: center;
  padding: 1.5rem;
}

.vm-score-badge {
  display: inline-block;
  font-size: 3rem;
  font-weight: 700;
  font-family: 'Inter', sans-serif;
  padding: 0.5rem 1.5rem;
  border-radius: 999px;
  margin-bottom: 0.75rem;
  letter-spacing: -1px;
}

.vm-score-low    { color: #38ef7d; background: rgba(56, 239, 125, 0.12); border: 1px solid rgba(56, 239, 125, 0.3); }
.vm-score-medium { color: #ffc300; background: rgba(255, 195, 0,   0.12); border: 1px solid rgba(255, 195, 0,   0.3); }
.vm-score-high   { color: #ff5733; background: rgba(255, 87,  51,  0.12); border: 1px solid rgba(255, 87,  51,  0.3); }

/* ── Dividers ─────────────────────────────────────────────── */
.vm-divider {
  border: none;
  border-top: 1px solid var(--vm-glass-border);
  margin: 1rem 0;
}

/* ── Scrollbars ───────────────────────────────────────────── */
::-webkit-scrollbar { width: 6px; }
::-webkit-scrollbar-track { background: transparent; }
::-webkit-scrollbar-thumb { background: rgba(255,255,255,0.1); border-radius: 999px; }
::-webkit-scrollbar-thumb:hover { background: rgba(56, 239, 125, 0.3); }

/* ── Loading animation overlay ───────────────────────────── */
.vm-analyzing {
  display: flex;
  align-items: center;
  gap: 0.75rem;
  color: var(--vm-accent);
  font-size: 0.9rem;
  font-weight: 500;
}

.vm-dot {
  width: 8px;
  height: 8px;
  background: var(--vm-accent);
  border-radius: 50%;
  animation: vm-pulse 1.2s ease-in-out infinite;
}

.vm-dot:nth-child(2) { animation-delay: 0.2s; }
.vm-dot:nth-child(3) { animation-delay: 0.4s; }

@keyframes vm-pulse {
  0%, 100% { opacity: 0.3; transform: scale(0.8); }
  50%       { opacity: 1;   transform: scale(1.2); }
}
"""


# ──────────────────────────────────────────────────────────────────────────────
# Gradio theme
# ──────────────────────────────────────────────────────────────────────────────

def get_theme() -> gr.themes.Base:
    """Return a custom Gradio theme matching the VocalMirror dark glass aesthetic."""
    return gr.themes.Base(
        primary_hue=gr.themes.colors.emerald,
        secondary_hue=gr.themes.colors.slate,
        neutral_hue=gr.themes.colors.gray,
        font=[gr.themes.GoogleFont("Inter"), "sans-serif"],
        font_mono=[gr.themes.GoogleFont("JetBrains Mono"), "monospace"],
    ).set(
        body_background_fill="#060b14",
        body_background_fill_dark="#060b14",
        block_background_fill="rgba(13,17,23,0.85)",
        block_background_fill_dark="rgba(13,17,23,0.85)",
        block_border_color="rgba(255,255,255,0.09)",
        block_border_color_dark="rgba(255,255,255,0.09)",
        block_label_text_color="#8b949e",
        block_label_text_color_dark="#8b949e",
        button_primary_background_fill="linear-gradient(135deg, #38ef7d, #11998e)",
        button_primary_background_fill_dark="linear-gradient(135deg, #38ef7d, #11998e)",
        button_primary_text_color="#060b14",
        button_primary_text_color_dark="#060b14",
        input_background_fill="rgba(0,0,0,0.3)",
        input_background_fill_dark="rgba(0,0,0,0.3)",
        input_border_color="rgba(255,255,255,0.09)",
        input_border_color_dark="rgba(255,255,255,0.09)",
        input_placeholder_color="#8b949e",
        input_placeholder_color_dark="#8b949e",
    )

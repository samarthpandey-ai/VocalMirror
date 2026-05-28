# VocalMirror

> A multimodal AI public speaking coach that computes an **Incongruence Score** — the gap between the confidence of your words and the nervousness of your voice.

## Features

| Feature | Tech |
|---|---|
| Speech-to-text | OpenAI Whisper (tiny / small toggle) |
| Acoustic analysis | librosa — pitch variance, speech rate, pause frequency |
| Sentiment scoring | RoBERTa (`cardiffnlp/twitter-roberta-base-sentiment-latest`) |
| Confidence scoring | DistilRoBERTa (`j-hartmann/emotion-english-distilroberta-base`) |
| Filler word detection | spaCy + regex |
| Vocabulary richness | spaCy type-token ratio + syntactic fluency |
| Fusion | Weighted arithmetic — no extra ML |
| UI | Gradio 4 + Plotly radar chart |

## Architecture

```
Audio → Whisper STT → transcript
Audio → librosa    → acoustic features → acoustic nervousness score
transcript → RoBERTa  → linguistic confidence + sentiment
transcript → spaCy    → filler density + vocab richness
↓
Fusion Engine → 5D Radar Chart + Incongruence Score
```

## Setup

```bash
# Create & activate a virtual environment
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Download spaCy model (done automatically on first run, but can be done manually)
python -m spacy download en_core_web_sm
```

## Run locally

```bash
python app.py
```

Open [http://localhost:7860](http://localhost:7860) in your browser.

## Whisper Model Toggle

- **tiny** (~5-8s/clip): Faster, less accurate — good for quick demos
- **small** (~15-30s/clip): More accurate — recommended for real analysis

## Radar Chart Dimensions

| Dimension | Description |
|---|---|
| Vocal Confidence | `1 - acoustic_nervousness` |
| Clarity | `vocab_richness × syntactic_fluency` |
| Filler Control | `1 - filler_density` (higher = fewer fillers) |
| Linguistic Sentiment | Positive sentiment probability |
| Congruence | `1 - incongruence_score` (higher = more authentic) |

## Incongruence Score

```
incongruence = |vocal_confidence − linguistic_confidence|
```

- **< 15%** — Low: your voice and words are aligned ✅
- **15–30%** — Mild: subtle nervous cues detected 🟡
- **30–50%** — Moderate: noticeable gap 🟠
- **> 50%** — High: significant mismatch 🔴

## Deploy to HuggingFace Spaces (for Vercel portfolio embed)

1. Create a HuggingFace Space (Gradio SDK)
2. Push this repo to the Space repository
3. In your Vercel portfolio, add an iframe:

```html
<iframe
  src="https://YOUR-USERNAME-vocalmirror.hf.space"
  allow="microphone"
  width="100%"
  height="700px"
  style="border:none; border-radius:16px;"
></iframe>
```

## Project Structure

```
VOcal_antigravity/
├── app.py                        # Gradio entrypoint
├── requirements.txt
├── modules/
│   ├── speech/
│   │   ├── transcriber.py        # Whisper STT
│   │   ├── acoustic_analyzer.py  # librosa feature extraction
│   │   ├── acoustic_scorer.py    # nervousness computation
│   │   └── baselines.json        # RAVDESS-derived baselines
│   ├── nlp/
│   │   ├── sentiment_scorer.py   # RoBERTa sentiment
│   │   ├── confidence_scorer.py  # DistilRoBERTa emotion
│   │   ├── filler_detector.py    # spaCy + regex
│   │   ├── vocab_analyzer.py     # TTR + syntactic fluency
│   │   └── confidence_lexicon.json
│   └── fusion.py                 # Radar score computation
└── ui/
    ├── radar_chart.py            # Plotly radar chart builder
    └── theme.py                  # Custom Gradio CSS/theming
```

# Adaptive Reading Assistant

> **An AI-powered browser extension that simplifies English vocabulary in real-time, calibrated to your reading level.**

The Adaptive Reading Assistant rewrites complex paragraphs on any webpage to match a chosen CEFR proficiency level — Beginner, Intermediate, or Advanced — while preserving every fact, name, and number in the original. Words that are slightly above your current level are annotated with amber underlines; hovering any underlined word reveals its CEFR grade and an English definition pulled from WordNet. The system runs entirely on your own machine: no cloud API keys, no data sent to third parties.

---

## Table of Contents

1. [How It Works](#how-it-works)
2. [System Architecture](#system-architecture)
3. [Repository Structure](#repository-structure)
4. [Prerequisites](#prerequisites)
5. [Installation](#installation)
6. [Running the Backend Server](#running-the-backend-server)
7. [Loading the Browser Extension](#loading-the-browser-extension)
8. [Using the Extension](#using-the-extension)
9. [CEFR Difficulty Tiers Explained](#cefr-difficulty-tiers-explained)
10. [Fine-tuning Your Own Model](#fine-tuning-your-own-model)
11. [API Reference](#api-reference)
12. [Troubleshooting](#troubleshooting)
13. [Project Roadmap](#project-roadmap)

---

## How It Works

When you click **Simplify This Page**, the extension collects all readable block-level elements on the page (paragraphs, headings, list items, table cells) and sends them in small batches to a local Flask server. The server runs two analytical passes on each paragraph.

The first pass is a **complexity gate**: the NLP engine tokenises the text with spaCy, lemmatises each content word, and looks it up in a 10,000-word CEFR vocabulary. If fewer than two words exceed your chosen level, the paragraph is already readable and is returned unchanged — no model inference occurs, keeping latency low.

The second pass calls the **AI simplifier**: a `google/flan-t5-large` model (or your own fine-tuned checkpoint if one is present) rewrites the paragraph using a structured prompt that constrains the output to preserve sentence count, all named entities, and all numerical facts. A quality gate compares input and output lengths; if the model dropped more than half the content (a summarisation failure mode), the original text is returned instead.

Alongside simplification, the engine also identifies **medium-difficulty words** — those one CEFR tier above your comfort zone but not hard enough to warrant replacement — and returns their character positions. The extension injects `<span>` elements at those exact positions, enabling the amber hover-definition feature without touching the surrounding DOM structure.

---

## System Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                     Browser (Chrome)                        │
│                                                             │
│  ┌──────────────┐    messages    ┌───────────────────────┐  │
│  │  popup.html  │◄──────────────►│    content.js         │  │
│  │  popup.js    │                │  (injected per-tab)   │  │
│  └──────────────┘                └──────────┬────────────┘  │
│                                             │ fetch         │
└─────────────────────────────────────────────┼───────────────┘
                                              │ HTTP
                              ┌───────────────▼───────────────┐
                              │    Flask API  (app.py)        │
                              │    localhost:5000             │
                              │                               │
                              │  ┌──────────────────────────┐ │
                              │  │  ComplexityAnalyzer       │ │
                              │  │  (complexity.py)          │ │
                              │  │  spaCy · CEFR CSV ·       │ │
                              │  │  Word Frequency CSV       │ │
                              │  └──────────────────────────┘ │
                              │                               │
                              │  ┌──────────────────────────┐ │
                              │  │  Simplifier               │ │
                              │  │  (simplifier.py)          │ │
                              │  │  flan-t5-large (lazy)     │ │
                              │  └──────────────────────────┘ │
                              └───────────────────────────────┘
```

The popup and content script communicate via `chrome.tabs.sendMessage`. Progress state (percentage, running flag, completion status) is written to `chrome.storage.local` by the content script on every batch, and the popup's `storage.onChanged` listener updates the progress bar in real time — even if the popup is reopened mid-run after being closed.

---

## Repository Structure

```
adaptive-reading-assistant/
│
├── nlp_engine/                   # Python backend — run this first
│   ├── app.py                    # Flask API server (4 endpoints)
│   ├── complexity.py             # CEFR-based word difficulty analyser
│   ├── simplifier.py             # flan-T5 inference wrapper
│   ├── requirements.txt          # Python dependencies
│   └── models/
│       └── t5-simplifier/        # Optional: place your fine-tuned
│                                 # checkpoint here to use it instead
│                                 # of the public flan-t5-large model
│
├── browser_extension/            # Chrome extension — load via chrome://extensions
│   ├── manifest.json             # MV3 manifest (permissions, CSP)
│   ├── background.js             # Service worker — registers context menu
│   ├── content.js                # Injected per-tab: DOM rewriting + hover popups
│   ├── popup.html                # Extension popup UI (two-screen: onboard + app)
│   ├── popup.js                  # Popup logic: state management, server health
│   ├── styles.css                # Page-injected styles (modified paragraphs,
│   │                             #   amber word highlights, definition card)
│   ├── library_image.avif        # Onboarding background image
│   └── functionality_page_background.jpg  # App screen background
│
└── README.md                     # This file
```

---

## Prerequisites

Before installing, ensure the following are available on your system.

**Python 3.10 or later** is required. You can verify with `python --version`. If you need to install Python, download it from [python.org](https://www.python.org/downloads/).

**A modern GPU is strongly recommended** for acceptable inference speed. The flan-t5-large model has 770M parameters; on CPU, each paragraph takes 8–15 seconds. On a mid-range GPU (RTX 3060 or better), it drops to under a second. The system will run on CPU if no GPU is detected, but page simplification will be slow for content-heavy articles.

**Google Chrome** version 88 or later is required for Manifest V3 support.

**PyTorch with CUDA** must be installed separately before `requirements.txt` because pip cannot automatically select the correct CUDA build. Follow the instruction in Step 1 below.

---

## Installation

### Step 1 — Install PyTorch with CUDA support

Open a terminal and run the following command. This installs PyTorch built against CUDA 12.6, which is required for GPU acceleration. If you are on CPU only, visit [pytorch.org](https://pytorch.org/get-started/locally/) and select your configuration to get the appropriate install command.

```bash
pip3 install torch torchvision --index-url https://download.pytorch.org/whl/cu126
```

### Step 2 — Install remaining Python dependencies

Navigate to the `nlp_engine/` directory and install the packages listed in `requirements.txt`.

```bash
cd nlp_engine
pip install -r requirements.txt
```

### Step 3 — Download the spaCy English language model

The complexity analyser uses spaCy's small English model for tokenisation, lemmatisation, and POS tagging. Download it with:

```bash
python -m spacy download en_core_web_sm
```

This is a one-time download of approximately 12 MB. spaCy caches it permanently, so you will not need to repeat this step on subsequent runs.

### Step 4 — Vocabulary datasets (automatic on first run)

The CEFR vocabulary list and word-frequency corpus are downloaded automatically the first time `app.py` starts, via `kagglehub`. The download requires an internet connection on first startup only; both datasets are cached locally and all subsequent startups are offline. The total download is approximately 15 MB.

---

## Running the Backend Server

From inside the `nlp_engine/` directory, start the Flask server:

```bash
python app.py
```

You should see output similar to the following in your terminal. The CEFR and frequency datasets download on the very first run; this adds roughly 20–30 seconds to startup but never happens again.

```
[Complexity] CEFR vocabulary loaded: 8,657 entries.
[Complexity] Frequency list loaded: top 50,000 words.
 * Running on http://0.0.0.0:5000
```

The server remains running in the background while you use the extension. You can leave this terminal window open. To stop the server, press `Ctrl + C`.

> **Important:** The AI model itself (`flan-t5-large`) is not loaded at startup. It loads lazily the first time a simplification request arrives, which takes 15–30 seconds as it downloads approximately 1.2 GB of weights. Subsequent requests within the same session are instant because the model stays in memory. If you have placed a fine-tuned checkpoint in `models/t5-simplifier/`, that will be loaded instead and the public model will not be downloaded.

---

## Loading the Browser Extension

1. Open Chrome and navigate to `chrome://extensions`.
2. Enable **Developer mode** using the toggle in the top-right corner.
3. Click **Load unpacked**.
4. In the file picker, navigate to and select the `browser_extension/` folder.

The extension icon (📖) will appear in your Chrome toolbar. Chrome remembers unpacked extensions across restarts, so you only need to load it once. If you modify any extension files (popup, content script, styles), click the refresh icon on the extension's card at `chrome://extensions` to reload the changes.

---

## Using the Extension

**First launch — Onboarding screen.** The first time you open the extension, you will see the onboarding screen with an overview of features. Click **Get Started** to proceed to the main interface. The extension remembers that you have completed onboarding, so this screen only appears once.

**Selecting a reading level.** Three levels are available, each corresponding to a range of the Common European Framework of Reference for Languages (CEFR):

| Level | CEFR Range | Audience |
|---|---|---|
| Beginner | A1 – B1 | New readers, language learners, young students |
| Intermediate | A2 – B2 | High-school level, general adult readers |
| Advanced | B2 – C1 | University level, professionals |

Your selection is persisted across browser sessions so you never need to re-select it.

**Simplifying a page.** Navigate to any article or webpage with substantial text content, open the extension popup, confirm your level is selected, and click **Simplify This Page**. The progress bar fills as batches of paragraphs are processed. Simplified paragraphs are marked with a subtle blue left border; hovering them shows a tooltip with the original text.

**Hover definitions.** Words with an amber dotted underline are one CEFR tier above your chosen level — difficult enough to be worth knowing, but not so difficult that the simplifier replaced them. Hovering any such word for 260 milliseconds opens a definition card showing the word, its CEFR grade badge, its part of speech, a WordNet definition, and an example sentence where available.

**Restoring the original.** Click **Restore Original** to undo all simplifications and remove all amber highlights from the current page. The page returns exactly to its pre-simplification state.

**Revisiting the onboarding screen.** Click **ⓘ About** in the footer to return to the onboarding screen at any time.

---

## CEFR Difficulty Tiers Explained

The extension classifies each content word into one of three tiers relative to your chosen level. Understanding this helps explain what gets simplified, what gets highlighted, and what gets left alone.

The **easy tier** covers words at or below your current level. These words are untouched and receive no visual treatment.

The **medium tier** covers words exactly one step above your level (e.g. B1 words for a Beginner user). These words receive the amber dotted underline and hover definition. They are close enough to your level that encountering them in context — with a definition available on demand — is genuinely educational. Replacing them would deprive you of a learning opportunity.

The **hard tier** covers words two or more steps above your level (e.g. B2, C1, C2 for a Beginner user). These words are sent to the AI model for replacement. The model has full sentence context so it can choose a semantically appropriate simpler alternative rather than a dictionary synonym.

Words absent from the CEFR list are treated as hard for Beginner and Intermediate users unless they appear in the top 50,000 most frequent English words, in which case they are treated as easy. This heuristic correctly handles modern technical terms (e.g. "website", "download") that predate the CEFR wordlists.

---

## Fine-tuning Your Own Model

The simplifier is designed to work out of the box with the public `google/flan-t5-large` model, but it automatically upgrades to a fine-tuned checkpoint if you place one in the right location. This is the recommended path for production use because a model fine-tuned on parallel simplification data (such as the Newsela corpus) produces substantially better output than the general instruction-tuned model.

To use a fine-tuned model, place the checkpoint directory at `nlp_engine/models/t5-simplifier/`. The directory must contain at minimum `config.json`, `tokenizer.json` (or `tokenizer_config.json`), and the model weights file (`model.safetensors` or `pytorch_model.bin`). When `app.py` starts, it checks whether `models/t5-simplifier/` exists; if it does, that model is loaded instead of downloading flan-t5-large.

For fine-tuning instructions using the Newsela Article Corpus with LoRA (Low-Rank Adaptation), gradient checkpointing, and proper checkpoint management on Kaggle or Colab, refer to the separate fine-tuning notebook included with this project.

---

## API Reference

The Flask server exposes four endpoints, all served at `http://localhost:5000`.

**`GET /ping`** — Health check. Returns `{"status": "ok", "model": "<model label>"}`. The popup polls this endpoint on open to determine the server status indicator colour (green = online, red = offline).

**`POST /rewrite`** — Main simplification endpoint. Accepts `{"paragraphs": [...], "user_level": "beginner"}` and returns `{"results": [{original, rewritten, changed, annotations}]}`. The `annotations` array contains the character positions of medium-difficulty words in the displayed text, enabling the content script to inject amber highlight spans without making a second server call.

**`GET /define?word=<lemma>&pos=<POS>`** — Word definition lookup. Returns `{"word", "level", "pos", "definition", "example"}`. Called by the content script on hover, with a client-side cache so each word is fetched at most once per page session.

**`POST /split`** — Legacy sentence splitter. Accepts `{"text": "..."}` and returns `{"sentences": [...]}`. Retained for compatibility with earlier extension versions.

---

## Troubleshooting

**The extension says "Server offline" even though `app.py` is running.** Verify that the server is actually listening on port 5000 by opening `http://localhost:5000/ping` in a browser tab. If you see a JSON response, the server is running but the extension cannot reach it — this is usually a browser or antivirus firewall rule blocking `localhost` fetch requests. Check your system firewall settings and ensure that `http://localhost:5000/*` is not blocked.

**The first simplification request takes 30+ seconds.** This is expected on the very first call in a session. The AI model loads lazily on the first request. All subsequent requests in the same session will be fast because the model stays in GPU memory. On a CPU-only machine, inference will be slow for every request — consider using a machine with a GPU for a better experience.

**Simplified paragraphs look identical to the originals.** This means the complexity gate decided the paragraphs did not contain enough hard words for your chosen level. Try switching to **Beginner** level, which has the broadest definition of "hard" words, to see more aggressive simplification.

**Amber word highlights appear in the wrong positions after simplification.** This can happen if the page's JavaScript re-renders the DOM after the content script has injected spans. Single-page applications (news sites, wikis) are particularly prone to this because they may replace DOM nodes on scroll or navigation. In this case, use **Restore Original** and try again — the extension will re-process the freshly rendered DOM.

**The `kagglehub` download fails on startup.** Ensure you have an internet connection on first run. If you are behind a corporate proxy, set the `HTTPS_PROXY` environment variable before starting the server. Once the datasets are cached locally (at `~/.cache/kagglehub/`), internet access is no longer required.

---

## Project Roadmap

The following improvements are planned for future versions of the project.

**Fine-tuned model integration.** Training a dedicated simplification model on the Newsela Article Corpus using LoRA on the attention QKV and FFN weight blocks will substantially improve simplification quality over the general-purpose flan-t5-large baseline.

**PDF support.** The current version only operates on webpages. Extending the pipeline to extract text from PDF files opened in the browser (via PDF.js) would cover academic papers, reports, and textbooks.

**Vocabulary learning mode.** Rather than simply defining amber words on hover, a future mode could track which words a user has looked up, build a personal vocabulary profile, and adjust the medium/hard thresholds dynamically as the user's demonstrated vocabulary grows.

**Offline model serving.** The current architecture requires model weights to be loaded into the server process. A future version could serve the model via a persistent background process (e.g. using `uvicorn` with model pre-loading) so that the model stays warm even when the server is restarted between sessions.

**Multi-language support.** The complexity analyser and simplifier are English-only. The architecture is language-agnostic; swapping the spaCy model and CEFR vocabulary for German, French, or Spanish equivalents would enable simplification in those languages with minimal code changes.

---

## Licence

This project is released for personal and academic use. The CEFR vocabulary dataset is licensed under Apache 2.0 (Kaggle: `nezahatkk/10-000-english-words-cerf-labelled`). The word frequency corpus is derived from the Google Web Trillion Word Corpus and distributed under MIT licence (Kaggle: `rtatman/english-word-frequency`). The `google/flan-t5-large` model weights are released under the Apache 2.0 licence by Google. The Newsela Article Corpus, if used for fine-tuning, is subject to Newsela's data sharing terms and is not redistributable.

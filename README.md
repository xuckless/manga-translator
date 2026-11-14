# Manga Translator

Turns raw Japanese manga pages into lettered English pages:

```
pages ──► detect ──► translate ──► clean ──► typeset ──► translated/
          RT-DETR     Claude        fill /     shape-aware
          bubbles +   vision        LaMa       line breaking
          text boxes  (per region)  inpaint
```

## How it works

| Stage | What it does | Why this way |
|---|---|---|
| **Detect** (`detection.py`) | [`ogkalu/comic-text-and-bubble-detector`](https://huggingface.co/ogkalu/comic-text-and-bubble-detector) (RT-DETR-v2, Apache-2.0) finds *balloons*, *text inside balloons* and *text over artwork*. ~0.3 s/page. | Vision LLMs are bad at pixel coordinates (the old Gemini boxes were loose, and the renderer wiped artwork). A dedicated detector gets the geometry right. |
| **Masks** (`masks.py`) | Traces each balloon's real interior from the image (ovals, spikes and clouds keep their shape), and builds a glyph mask (including the white halo around narration text). | Lettering follows the balloon outline, and only the text pixels get erased. |
| **Translate** (`translation/`) | Each region is drawn on the page as a numbered box ("set-of-marks"). Claude gets the marked page plus the story so far and the glossary, and returns structured JSON per region: transcription, translation, a condensed fallback, a category and the reading order. | The model only reads and translates, which it is good at. Context across pages keeps names and voices consistent. |
| **Clean** (`render.py`, `inpaint.py`) | Balloon text is painted with the balloon colour. Text over artwork is removed with LaMa (ONNX, CPU). | Plain fills look perfect on flat paper. LaMa rebuilds screentone and hatching. |
| **Typeset** (`typeset.py`) | Picks the largest font size whose layout fits the balloon *shape*: each line gets the width the balloon has at that height. A small DP balances line lengths and avoids breaks like "…the / cat". If the text won't fit, it uses the short translation, then flags an overflow. Sizes are normalised across the page. | Gives the centred lozenge look of professional lettering, without hand-tuned wrapping. |

SFX are left as drawn art by default (`--sfx` replaces them too). Regions the model marks as `noise` are never touched.

## Setup

```bash
python -m venv venv && source venv/bin/activate
pip install -r requirements.txt
cp .env.example .env                 # optional: ANTHROPIC_API_KEY for the CLI / a server-wide key
```

The detector (~170 MB) and LaMa (~200 MB) download from Hugging Face on first use.

## Web UI (self-hosted)

```bash
python -m manga_translator serve                  # http://127.0.0.1:8000
```

1. **Library**: create a chapter.
2. **Add pages…**: pick the page images in the file picker (or drop them on the page). They are sorted
   by file name (`page2` before `page10`). Drag to reorder, then **Add & translate**. Later uploads are
   appended after the existing pages, and translation keeps the story context across them.
3. **Settings**: paste your Anthropic API key. It stays in your browser and is sent only when a
   translation starts. The server keeps it in memory for that job and never writes it to disk.
4. Click any page to **review**: the numbered boxes match the list of translations. Edit the text,
   change a category (`noise` = leave it alone), then **Save & re-letter page**.
5. **Read ▸** opens the reader: paged (right-to-left or left-to-right) or vertical scroll, fit height or
   width, `O` to peek at the original, and your place is remembered per chapter.

Lettering options (font, ALL CAPS, sound effects, LaMa/OpenCV) are per chapter. **Re-letter** applies
them again without any API calls.

On a server or LAN, set an access token:

```bash
python -m manga_translator serve --host 0.0.0.0 --token "$(openssl rand -hex 16)"
```

Or with Docker (CPU image; chapters live in `./data/chapters`):

```bash
MANGA_TRANSLATOR_TOKEN=change-me docker compose up -d --build
```

The UI and the CLI share the same folders (`--library`, default `data/chapters`), so either can pick
up where the other stopped.

## CLI

```bash
# everything in one go -> data/chapters/chapter-9/translated/
python -m manga_translator run data/chapters/chapter-9

# or stage by stage (each one resumes where it stopped)
python -m manga_translator detect    data/chapters/chapter-9
python -m manga_translator translate data/chapters/chapter-9 --effort medium
python -m manga_translator render    data/chapters/chapter-9 --debug
```

A chapter is a folder of page images (`page1.jpg`, `page2.jpg`, …).

Useful flags: `--pages page5.jpg page6.jpg` (only these pages), `--retranslate`, `--model claude-sonnet-5-5`,
`--honorifics localize`, `--uppercase`, `--font path/to/font.ttf`, `--inpaint opencv` (no model download),
`--sfx`.

### Without an API key: the manual backend

```bash
python -m manga_translator translate data/chapters/chapter-9 --backend manual
```

This writes `work/manual/<page>.marked.jpg` and `<page>.request.md` (the prompt plus the JSON schema). Any
model or person can answer by writing `<page>.response.json`. Re-run the same command to import the
answers. This is also how to hand-correct a page: edit its translations in `work/chapter.json` and
re-run `render`.

## Layout

```
manga_translator/      the package (cli, pipeline, detection, masks, inpaint, typeset, render)
  web/                 self-hosted UI (FastAPI + plain JS, no build step)
  translation/         prompts, response schema, Claude + manual backends
assets/fonts/          Comic Neue Bold (SIL OFL)
tests/                 typesetting tests (pytest)
legacy/                the original prototype (Surya OCR, Argos/Gemini translators, contour renderer)
data/chapters/<ch>/    pages (git-ignored)
  work/chapter.json    detections + translations: single source of truth, safe to edit by hand
  work/debug/          mask overlays: green = traced balloon, orange = ellipse fallback, red = erased
  translated/          output
```

## Fonts

The default font is Comic Neue Bold (OFL). Scanlation groups often use CC Wild Words or Anime Ace. If you
have a licence for one, pass `--font` (and usually `--uppercase`).

## License

Licensed under either of

- Apache License, Version 2.0 ([LICENSE-APACHE](LICENSE-APACHE))
- MIT license ([LICENSE-MIT](LICENSE-MIT))

at your option. Unless you explicitly state otherwise, any contribution you intentionally submit for
inclusion in this work, as defined in the Apache-2.0 license, is dual licensed as above, without any
additional terms or conditions.

Third-party parts keep their own licenses: Comic Neue (`assets/fonts/`, SIL OFL 1.1, see
`assets/fonts/OFL.txt`); the detector and LaMa weights are downloaded at runtime from Hugging Face
(both Apache-2.0) and are not part of this repository. Manga pages you process are not covered by
this license.

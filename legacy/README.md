# Legacy prototype

The original scripts, kept for reference. They are not used by `manga_translator/`.

- `extractor.py`: Surya OCR + proximity box merging (pixel `[x1,y1,x2,y2]` boxes)
- `translator.py`: Argos / Google / deep-translator text-only translation
- `gtranslator.py`: Gemini one-shot OCR + translation (0-1000 `[ymin,xmin,ymax,xmax]` boxes)
- `insertion.py`: Canny-contour renderer + `BoundingBoxTester`

Known issues that led to the rewrite:
- `insertion.py` inverts its own flag: when it *does* find a balloon contour it fills that contour's
  whole bounding rectangle, wiping artwork. When it doesn't find one, it draws a rounded box.
- Gemini boxes are approximate, so contour matching often fails or grabs the wrong shape.
- `extractor.py` and `gtranslator.py` write incompatible box formats into the same `imagedata.json`.
  Some Gemini entries have malformed boxes, which the renderer silently skips.
- `rglob` + plain sorting orders pages `page1, page10, page11, …, page2`, which breaks context.
- `textwrap` wraps by an estimated character count, not measured width, so text overflows or shrinks.

# Graph Digitizer v4 — pixel-tracing hybrid

Numbers come from pixels, not from an LLM:
1. pdftoppm renders each page at 300 DPI
2. OpenCV detects every plot box
3. Axes calibrated from gridlines + per-label OCR with candidate-expansion RANSAC (exact ticks, log/linear auto)
4. Curves color-segmented and traced pixel-by-pixel (cliffs, dashed, black curves handled)
5. Claude labels titles/axes/legends only (one small image call per graph); engine works without a key too
6. Per-curve confidence included in the JSON

Measured on the reference datasheet: 13/13 graphs, curve values within ~0.5–1% of axis range on verified curves.

## Run (Docker — recommended, includes tesseract & poppler)
cp .env.example .env   # add ANTHROPIC_API_KEY
docker compose up --build   # http://localhost:8000

## Run locally on Windows (without Docker)
Install Tesseract (UB Mannheim build) and Poppler for Windows, add both to PATH, then:
pip install -r requirements.txt
set ANTHROPIC_API_KEY=sk-ant-...
uvicorn main:app --host 0.0.0.0 --port 8000

Known limits: dense multi-curve log-log plots (e.g. SOA) may merge similar-hue curves — flagged by low confidence; fix in the drag editor. Scanned/noisy PDFs reduce OCR reliability.

---
title: Graph Digitizer
emoji: 📈
colorFrom: green
colorTo: blue
sdk: docker
app_port: 8000
pinned: false
---

# Graph Digitizer — pixel-tracing hybrid

Upload a datasheet PDF → every graph is traced from pixels (not guessed by an LLM) → edit inline → export JSON / SVG / CSV / XLSX.

## Pipeline
1. pdftoppm renders each page at 300 DPI
2. OpenCV detects plot boxes
3. Axes calibrated from gridlines + per-label OCR (RANSAC, log/linear auto)
4. Curves color-segmented and traced pixel-by-pixel
5. Claude labels titles / axes / legends only (one small image call per graph)

## Editor features
Ghost overlay (compare trace vs original), drag points / drag whole line, add/delete points,
merge / split / extend curves, point table, zoom / pan / cursor readout, manual box drawing
for missed graphs, undo / reset. Exports: JSON, plain-line SVG, CSV, XLSX.

## Configuration (Space → Settings → Secrets/Variables)
- **ANTHROPIC_API_KEY** (secret, required for labels — tracing works without it)
- **ACCESS_CODE** (secret, recommended) — gates the app so a public URL can't burn your API credits
- **MODEL** (variable, optional) — default `claude-sonnet-4-6`

Numbers come from pixels; accuracy is verifiable via the ghost overlay. Dense log-log plots
(e.g. SOA) may need the merge/split tools — low-confidence curves are flagged in red.

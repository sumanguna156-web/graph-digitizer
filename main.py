"""
Datasheet Graph Digitizer — backend v4 (pixel-tracing hybrid)
-------------------------------------------------------------
Numbers come from PIXELS (OpenCV trace + OCR-calibrated axes) — not from an LLM.
Claude is used ONLY for semantics: graph title, axis label text, curve legend
names, annotations. If the LLM call fails, graphs still stream with generic
labels — the data is unaffected.

Requires system packages: poppler-utils (pdftoppm), tesseract-ocr.

Env:
  ANTHROPIC_API_KEY   required for labels (engine works without it)
  MODEL               default claude-sonnet-4-6
  DPI                 render resolution, default 300

Run: uvicorn main:app --host 0.0.0.0 --port 8000
"""

import base64, json, os, re, subprocess, tempfile, time, uuid

import cv2
import numpy as np
import anthropic
from fastapi import FastAPI, HTTPException, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse, FileResponse
from fastapi.staticfiles import StaticFiles

import digitizer as D

MODEL = os.environ.get("MODEL", "claude-sonnet-4-6")
DPI = int(os.environ.get("DPI", "300"))
MAX_PDF_MB = 20
UPLOAD_TTL_SEC = 900

app = FastAPI(title="Datasheet Graph Digitizer v4")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

try:
    client = anthropic.Anthropic()
except Exception:  # noqa: BLE001
    client = None

UPLOADS: dict = {}


def _purge():
    now = time.time()
    for k in [k for k, v in UPLOADS.items() if now - v["ts"] > UPLOAD_TTL_SEC]:
        UPLOADS.pop(k, None)


# ---------------- semantics via Claude (labels only) ----------------

def semantics(img, box, curves):
    """Ask Claude for title / axis labels / legend names. Data untouched."""
    if client is None:
        return None
    x, y, w, h = box
    H, W = img.shape[:2]
    pad = 140
    crop = img[max(y - pad, 0):min(y + h + pad, H), max(x - pad, 0):min(x + w + pad, W)]
    scale = 820.0 / max(crop.shape[:2])
    if scale < 1:
        crop = cv2.resize(crop, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
    png = cv2.imencode(".png", crop)[1].tobytes()
    desc = [{"color": c["color"],
             "x_start": c["x"][0], "x_end": c["x"][-1],
             "y_start": c["y"][0], "y_end": c["y"][-1]} for c in curves]
    prompt = f"""This image is one graph from a datasheet. Curves were already traced from pixels;
their colors and endpoints are: {json.dumps(desc)}
Respond ONLY with JSON, no markdown:
{{"title":"...","x_label":"...","y_label":"...","annotations":["test conditions in the plot box"],
"curves":[{{"color":"#the hex from the list","label":"legend/curve name from the image","linestyle":"solid|dashed"}}]}}
Match each listed color to the correct curve name shown in the image (e.g. "VGS=6V", "CISS", "1ms").
If two listed colors are the same physical curve band, give them the same label."""
    try:
        msg = client.messages.create(
            model=MODEL, max_tokens=800,
            messages=[{"role": "user", "content": [
                {"type": "image", "source": {"type": "base64", "media_type": "image/png",
                                             "data": base64.b64encode(png).decode()}},
                {"type": "text", "text": prompt}]}])
        text = "".join(b.text for b in msg.content if b.type == "text")
        return json.loads(re.sub(r"```json|```", "", text).strip())
    except Exception:  # noqa: BLE001
        return None


def confidence(curve):
    """Simple per-curve confidence: point density across its x-span."""
    xs = curve["x"]
    if len(xs) < 8:
        return 0.3
    span = max(xs) - min(xs)
    return round(min(1.0, len(xs) / 40.0) * (0.6 + 0.4 * (span > 0)), 2)


# ---------------- routes ----------------

@app.post("/api/upload")
async def upload(file: UploadFile = File(...)):
    _purge()
    if file.content_type != "application/pdf":
        raise HTTPException(400, "Only PDF files are supported.")
    raw = await file.read()
    if len(raw) > MAX_PDF_MB * 1024 * 1024:
        raise HTTPException(413, f"PDF larger than {MAX_PDF_MB} MB.")
    uid = uuid.uuid4().hex
    UPLOADS[uid] = {"raw": raw, "name": file.filename, "ts": time.time()}
    return {"id": uid, "name": file.filename, "size": len(raw)}


@app.get("/api/stream/{uid}")
def stream(uid: str):
    item = UPLOADS.get(uid)
    if not item:
        raise HTTPException(404, "Upload not found or expired. Re-upload the PDF.")

    def sse(event, data):
        return f"event: {event}\ndata: {json.dumps(data, default=float)}\n\n"

    def gen():
        with tempfile.TemporaryDirectory() as td:
            pdf = os.path.join(td, "in.pdf")
            open(pdf, "wb").write(item["raw"])
            try:
                subprocess.run(["pdftoppm", "-png", "-r", str(DPI), pdf,
                                os.path.join(td, "pg")], check=True, timeout=120)
            except Exception as e:  # noqa: BLE001
                yield sse("error", {"stage": "render", "message": str(e)}); return
            pages = sorted(f for f in os.listdir(td) if f.startswith("pg"))

            # inventory: detect all plot boxes first (local, instant)
            page_imgs, page_boxes, inv = {}, {}, []
            for pi, f in enumerate(pages, 1):
                img = cv2.imread(os.path.join(td, f))
                boxes = D.find_plot_boxes(img)
                page_imgs[pi], page_boxes[pi] = img, boxes
                for gi in range(len(boxes)):
                    inv.append({"id": f"P{pi}_G{gi+1}", "page": pi})
            per_page = {}
            for g in inv:
                per_page[str(g["page"])] = per_page.get(str(g["page"]), 0) + 1
            yield sse("inventory", {"total": len(inv), "per_page": per_page,
                                    "graphs": inv, "engine": "pixel-trace v4"})

            idx = 0
            for pi in sorted(page_imgs):
                img = page_imgs[pi]
                _, extracted = D.digitize_page_img(img, page_boxes[pi])
                for gi, g in enumerate(extracted):
                    gid = f"P{pi}_G{gi+1}"
                    yield sse("progress", {"index": idx, "total": len(inv),
                                           "id": gid, "title": gid, "stage": "trace"})
                    idx += 1
                    if "error" in g or not g.get("curves"):
                        yield sse("graph_error", {"id": gid, "page": pi, "title": gid,
                                                  "message": g.get("error", "no curves traced")})
                        continue
                    yield sse("progress", {"index": idx - 1, "total": len(inv),
                                           "id": gid, "title": gid, "stage": "verify"})
                    sem = semantics(img, g["box"], g["curves"]) or {}
                    label_by_color = {c.get("color"): c for c in sem.get("curves", [])}
                    curves = []
                    for ci, c in enumerate(g["curves"]):
                        meta = label_by_color.get(c["color"], {})
                        curves.append({"label": meta.get("label", f"curve {ci+1}"),
                                       "color": c["color"],
                                       "linestyle": meta.get("linestyle", "solid"),
                                       "confidence": confidence(c),
                                       "x": c["x"], "y": c["y"]})
                    out = {"id": gid, "page": pi,
                           "title": sem.get("title", gid),
                           "x_axis": {"label": sem.get("x_label", "X"), **g["x_axis"]},
                           "y_axis": {"label": sem.get("y_label", "Y"), **g["y_axis"]},
                           "annotations": sem.get("annotations", []),
                           "curves": curves}
                    yield sse("graph", out)
            yield sse("done", {"total": len(inv)})
        UPLOADS.pop(uid, None)

    return StreamingResponse(gen(), media_type="text/event-stream",
                             headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})


@app.get("/api/health")
def health():
    return {"ok": True, "engine": "pixel-trace v4", "model": MODEL,
            "llm_labels": client is not None}


app.mount("/static", StaticFiles(directory="static"), name="static")


@app.get("/")
def index():
    return FileResponse("static/index.html")

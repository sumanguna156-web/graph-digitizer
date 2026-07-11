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
DPI = int(os.environ.get("DPI", "200"))
ACCESS_CODE = os.environ.get("ACCESS_CODE", "").strip()
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


def crop_b64(img, box, pad=0):
    """PNG of the plot box, for the ghost-overlay verification layer."""
    x, y, w, h = box
    H, W = img.shape[:2]
    c = img[max(y-pad,0):min(y+h+pad,H), max(x-pad,0):min(x+w+pad,W)]
    ok, buf = cv2.imencode(".png", c)
    return base64.b64encode(buf.tobytes()).decode() if ok else None


def page_b64(img, maxw=1400):
    sc = min(1.0, maxw / img.shape[1])
    im = cv2.resize(img, None, fx=sc, fy=sc, interpolation=cv2.INTER_AREA) if sc < 1 else img
    ok, buf = cv2.imencode(".png", im)
    return (base64.b64encode(buf.tobytes()).decode() if ok else None,
            im.shape[1], im.shape[0], sc)


def confidence(curve):
    """Simple per-curve confidence: point density across its x-span."""
    xs = curve["x"]
    if len(xs) < 8:
        return 0.3
    span = max(xs) - min(xs)
    return round(min(1.0, len(xs) / 40.0) * (0.6 + 0.4 * (span > 0)), 2)


# ---------------- routes ----------------

def _check_code(code: str | None):
    """If ACCESS_CODE is set, every entry point requires it."""
    if ACCESS_CODE and (code or "").strip() != ACCESS_CODE:
        raise HTTPException(401, "Invalid or missing access code.")


@app.post("/api/upload")
async def upload(file: UploadFile = File(...), code: str = ""):
    _check_code(code)
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
def stream(uid: str, code: str = ""):
    _check_code(code)
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
            item["pages"] = page_imgs   # cached for manual box drawing
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
                           "box": g["box"],
                           "crop": crop_b64(img, g["box"]),
                           "title": sem.get("title", gid),
                           "x_axis": {"label": sem.get("x_label", "X"), **g["x_axis"]},
                           "y_axis": {"label": sem.get("y_label", "Y"), **g["y_axis"]},
                           "annotations": sem.get("annotations", []),
                           "curves": curves}
                    yield sse("graph", out)
            yield sse("done", {"total": len(inv)})
        item["ts"] = time.time()   # keep pages cached for manual box drawing

    return StreamingResponse(gen(), media_type="text/event-stream",
                             headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})


@app.get("/api/pages/{uid}")
def pages(uid: str, code: str = ""):
    _check_code(code)
    item = UPLOADS.get(uid)
    if not item or "pages" not in item:
        raise HTTPException(404, "No rendered pages for this upload. Run extraction first.")
    out = []
    for pi, img in sorted(item["pages"].items()):
        b64, w, h, sc = page_b64(img)
        out.append({"page": pi, "image": b64, "width": w, "height": h, "scale": sc,
                    "full_width": int(img.shape[1]), "full_height": int(img.shape[0])})
    return {"pages": out}


@app.post("/api/trace_box")
async def trace_box(payload: dict, code: str = ""):
    """Digitize a user-drawn region. Coordinates are in FULL-RESOLUTION page pixels."""
    _check_code(code)
    uid = payload.get("uid")
    item = UPLOADS.get(uid)
    if not item or "pages" not in item:
        raise HTTPException(404, "Upload expired. Re-run extraction.")
    pi = int(payload.get("page", 1))
    img = item["pages"].get(pi)
    if img is None:
        raise HTTPException(400, "No such page.")
    box = [int(payload["x"]), int(payload["y"]), int(payload["w"]), int(payload["h"])]
    if box[2] < 60 or box[3] < 60:
        raise HTTPException(400, "Region too small.")
    g = D.digitize_box(img, box)
    if "error" in g or not g.get("curves"):
        raise HTTPException(422, g.get("error", "no curves found in that region"))
    sem = semantics(img, g["box"], g["curves"]) or {}
    lbl = {c.get("color"): c for c in sem.get("curves", [])}
    curves = [{"label": lbl.get(c["color"], {}).get("label", f"curve {i+1}"),
               "color": c["color"],
               "linestyle": lbl.get(c["color"], {}).get("linestyle", "solid"),
               "confidence": confidence(c), "x": c["x"], "y": c["y"]}
              for i, c in enumerate(g["curves"])]
    gid = payload.get("id") or f"P{pi}_M{int(time.time())%10000}"
    return json.loads(json.dumps({
        "id": gid, "page": pi, "box": g["box"], "crop": crop_b64(img, g["box"]),
        "title": sem.get("title", gid),
        "x_axis": {"label": sem.get("x_label", "X"), **g["x_axis"]},
        "y_axis": {"label": sem.get("y_label", "Y"), **g["y_axis"]},
        "annotations": sem.get("annotations", []), "curves": curves}, default=float))


@app.get("/api/health")
def health():
    return {"ok": True, "engine": "pixel-trace v4", "model": MODEL,
            "llm_labels": client is not None, "auth": bool(ACCESS_CODE)}


app.mount("/static", StaticFiles(directory="static"), name="static")


@app.get("/")
def index():
    return FileResponse("static/index.html")

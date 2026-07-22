"""Pixel-tracing graph digitizer core (v4). No LLM for numbers."""
import cv2, numpy as np, pytesseract, re, json

# ---------- 1. plot box detection ----------
def find_plot_boxes(img):
    g = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    bw = cv2.threshold(g, 200, 255, cv2.THRESH_BINARY_INV)[1]
    scale = img.shape[1] / 2500.0  # 1.0 at ~300 DPI; smaller at lower DPI
    kl = max(int(60 * scale), 24)
    h_k = cv2.getStructuringElement(cv2.MORPH_RECT, (kl, 1))
    v_k = cv2.getStructuringElement(cv2.MORPH_RECT, (1, kl))
    lines = cv2.dilate(cv2.erode(bw, h_k), h_k) | cv2.dilate(cv2.erode(bw, v_k), v_k)
    cnts, _ = cv2.findContours(lines, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    boxes = []
    H, W = g.shape
    for c in cnts:
        x, y, w, h = cv2.boundingRect(c)
        minside = max(int(400 * scale), 150)
        if w > minside and h > minside and w < 0.9*W and h < 0.9*H:
            boxes.append((x, y, w, h))
    # dedupe: drop boxes contained in another with similar size, keep outermost distinct
    boxes = sorted(boxes, key=lambda b: -b[2]*b[3])
    keep = []
    for b in boxes:
        x,y,w,h = b
        dup = False
        for k in keep:
            kx,ky,kw,kh = k
            t1 = max(int(30*scale),12); t2 = max(int(60*scale),24); t3 = max(int(40*scale),16)
            if abs(x-kx)<t1 and abs(y-ky)<t1 and abs(w-kw)<t2 and abs(h-kh)<t2:
                dup = True; break
            if x>kx+t3 and y>ky+t3 and x+w<kx+kw-t3 and y+h<ky+kh-t3:
                dup = True; break
        if not dup: keep.append(b)
    rb = max(img.shape[0]/10.0, 1)
    return sorted(keep, key=lambda b: (round(b[1]/rb), b[0]))  # reading order

# ---------- 2. axis calibration via OCR ----------
def _ocr_numbers(strip):
    if strip.size == 0: return []
    g = cv2.cvtColor(strip, cv2.COLOR_BGR2GRAY) if strip.ndim == 3 else strip
    up = cv2.resize(g, None, fx=3, fy=3, interpolation=cv2.INTER_CUBIC)
    up = cv2.threshold(up, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]
    out = []
    for psm in (11, 6):
        cfg = f'--psm {psm} -c tessedit_char_whitelist=0123456789.-'
        d = pytesseract.image_to_data(up, config=cfg, output_type=pytesseract.Output.DICT)
        for i, t in enumerate(d['text']):
            t = t.strip().rstrip('.').lstrip('-') if t else t
            if not t or not re.fullmatch(r'\d*\.?\d+', t): continue
            if d['width'][i] < 8 or d['height'][i] < 12 or d['height'][i] > 140: continue
            try: v = float(t)
            except ValueError: continue
            out.append((v, (d['left'][i] + d['width'][i]/2)/3.0,
                           (d['top'][i] + d['height'][i]/2)/3.0))
    return out

def _ransac_line(v, px, tol, wt=None):
    """best (a,b,inlier_idx) for px ~ a*v + b using pair seeding.
    wt: optional per-point preference weight; one inlier max per pixel position."""
    n = len(v); best = (None, None, [], -1.0)
    if wt is None: wt = np.ones(n)
    for i in range(n):
        for j in range(i+1, n):
            if v[j] == v[i] or abs(px[j]-px[i]) < 5: continue
            a = (px[j]-px[i])/(v[j]-v[i]); b = px[i]-a*v[i]
            if abs(a) < 1e-9: continue
            bypix = {}
            for k in range(n):
                if abs(a*v[k]+b-px[k]) < tol:
                    key = round(px[k]/6)
                    if key not in bypix or wt[k] > wt[bypix[key]]:
                        bypix[key] = k
            inl = list(bypix.values())
            score = len(inl) + 0.05*sum(wt[k] for k in inl)
            if score > best[3]: best = (a, b, inl, score)
    a, b, idx, _ = best
    if idx and len(idx) >= 2:
        A = np.vstack([v[idx], np.ones(len(idx))]).T
        (a, b), *_ = np.linalg.lstsq(A, px[idx], rcond=None)
        return a, b, idx
    return None, None, []

def _fit_axis(vals_px, span_px=600):  # [(value, pixel)] -> dict
    seen = {}
    for v, p in vals_px:
        key = (v, round(p/5))
        if key not in seen: seen[key] = (v, p)
    vals_px = sorted(seen.values(), key=lambda t: t[1])
    vals = np.array([v for v, p in vals_px], float)
    px = np.array([p for v, p in vals_px], float)
    if len(vals) < 2: return None
    tol = span_px * 0.02
    a_l, b_l, in_l = _ransac_line(vals, px, tol)
    a_g = b_g = None; in_g = []
    if np.all(vals > 0):
        a_g, b_g, in_g = _ransac_line(np.log10(vals), px, tol)
    use_log = False
    if a_g is not None and (len(in_g) > len(in_l) or
        (len(in_g) == len(in_l) and len(in_l) <= 3 and
         np.allclose(np.log10(vals[in_g]), np.round(np.log10(vals[in_g])), atol=0.02)
         and vals[in_g].max()/max(vals[in_g].min(),1e-12) >= 100)):
        use_log = True
    if use_log:
        a, b, inl = a_g, b_g, in_g
    else:
        a, b, inl = a_l, b_l, in_l
    if a is None or len(inl) < 2: return None
    tick_vals = sorted(set(vals[inl].tolist()))
    def px2val(p):
        t = (p - b)/a
        return 10**t if use_log else t
    return {'scale': 'log' if use_log else 'linear', 'ticks': tick_vals,
            'min': float(min(tick_vals)), 'max': float(max(tick_vals)),
            'px2val': px2val, 'n_inl': len(inl)}

def calibrate(img, box):
    x, y, w, h = box
    H, W = img.shape[:2]
    # x-axis: strip below box
    xs = img[y+h+3:min(y+h+58, H), max(x-45,0):min(x+w+45, W)]
    xt = [(v, cx + max(x-45,0)) for v, cx, cy in _ocr_numbers(xs)]
    # y-axis: strip left of box
    ys = img[max(y-20,0):min(y+h+20, H), max(x-130,0):x-2]
    yt = [(v, cy + max(y-20,0)) for v, cx, cy in _ocr_numbers(ys)]
    return _fit_axis(xt, w), _fit_axis(yt, h)


# ---------- 2b. gridline-anchored calibration (primary) ----------
def _gridline_positions(img, box):
    x, y, w, h = box
    crop = img[y:y+h, x:x+w]
    g = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    grayline = ((g > 120) & (g < 225)).astype(np.uint8)
    col_frac = grayline[10:h-10, :].mean(axis=0)
    row_frac = grayline[:, 10:w-10].mean(axis=1)
    def cluster(idxs):
        out = []
        for i in idxs:
            if out and i - out[-1][-1] <= 6: out[-1].append(i)
            else: out.append([i])
        return [int(np.mean(c)) for c in out]
    xs = cluster(list(np.where(col_frac > 0.45)[0]))
    ys = cluster(list(np.where(row_frac > 0.45)[0]))
    xs = sorted(set(xs + [0, w-1]))
    ys = sorted(set(ys + [0, h-1]))
    return xs, ys  # positions relative to box

def _ocr_single(img_crop):
    if img_crop.size == 0: return None
    g = cv2.cvtColor(img_crop, cv2.COLOR_BGR2GRAY) if img_crop.ndim == 3 else img_crop
    up = cv2.resize(g, None, fx=4, fy=4, interpolation=cv2.INTER_CUBIC)
    up = cv2.threshold(up, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]
    for psm in (8, 7):
        t = pytesseract.image_to_string(up, config=f'--psm {psm} -c tessedit_char_whitelist=0123456789.').strip()
        t = t.strip('.')
        if t and re.fullmatch(r'\d*\.?\d+', t):
            try: return float(t)
            except ValueError: pass
    return None

def _fit_pairs(pairs, span):
    """pairs [(value, px_rel)] -> axis dict with log/linear decision + tick reconstruction."""
    if len(pairs) < 2: return None
    pairs = sorted(pairs, key=lambda t: t[1])
    vals = np.array([v for v, p, *r in pairs], float)
    px = np.array([p for v, p, *r in pairs], float)
    wt = np.array([ (r[0] if r else 1.0) for v, p, *r in pairs], float)
    tol = span * 0.025
    a_l, b_l, in_l = _ransac_line(vals, px, tol, wt)
    a_g = None; in_g = []
    pos = vals > 0
    if pos.sum() >= 2:
        a_g, b_g, in_g0 = _ransac_line(np.log10(vals[pos]), px[pos], tol, wt[pos])
        idxmap = np.where(pos)[0]; in_g = [int(idxmap[k]) for k in in_g0] if in_g0 else []
    iv_g = vals[in_g] if in_g else np.array([])
    pow10 = len(iv_g) >= 2 and np.all(np.abs(np.log10(iv_g) - np.round(np.log10(iv_g))) < 0.02)
    use_log = a_g is not None and len(in_g) >= 2 and (len(in_g) > len(in_l) + 1 or
               (pow10 and iv_g.max()/iv_g.min() >= 10 and len(in_g) >= len(in_l)))
    a, b, inl = (a_g, b_g, in_g) if use_log else (a_l, b_l, in_l)
    if a is None or len(inl) < 2 or abs(a) < 1e-9: return None
    def px2val(p):
        t = (p - b) / a
        return 10**t if use_log else t
    v0, v1 = px2val(0), px2val(span)
    lo, hi = (v0, v1) if v0 < v1 else (v1, v0)
    if use_log:
        e0, e1 = round(np.log10(lo)), round(np.log10(hi))
        ticks = [10.0**e for e in range(int(e0), int(e1)+1)]
        lo, hi = 10.0**e0, 10.0**e1
    else:
        iv = np.sort(vals[inl]); steps = np.diff(iv)
        step = float(np.median(steps[steps > 0])) if len(steps) else (hi-lo)/5
        mag = 10**np.floor(np.log10(step))
        step = min([1,2,2.5,5,10], key=lambda m: abs(m*mag-step))*mag
        def snap_edge(v):
            for sub in (step, step/2, step/4):
                m = round(v/sub)*sub
                if abs(m - v) < step*0.18: return round(m, 10)
            return float(f'{v:.4g}')
        lo, hi = snap_edge(lo), snap_edge(hi)
        n = max(int(round((hi-lo)/step)), 1)
        ticks = [round(lo + k*step, 10) for k in range(n+1)]
    return {'scale': 'log' if use_log else 'linear', 'ticks': ticks,
            'min': float(lo), 'max': float(hi), 'px2val': px2val}

def _expand(v, p, w0=1.0):
    """candidate values for an OCR token that may have lost its decimal point."""
    out = [(v, p, w0)]
    if v >= 10 and abs(v - round(v)) < 1e-9:
        out.append((v/10.0, p, w0*0.5))
        if v >= 100: out.append((v/100.0, p, w0*0.25))
    return out

def _majority_grid(positions, ocr_by_idx, min_frac=0.55):
    """Reconstruct tick values for evenly-spaced gridlines by majority vote.
    Only fires when gridlines are uniform AND >= min_frac of readings agree on
    one (v0, step) line, so a few OCR errors can't derail it. Returns list|None."""
    n = len(positions)
    if n < 4:
        return None
    pos = np.array(positions, float)
    gaps = np.diff(pos)
    if gaps.min() <= 0 or gaps.std() / gaps.mean() > 0.12:
        return None  # not a clean uniform grid (likely log or irregular)
    items = [(i, v) for i, v in ocr_by_idx.items() if v is not None and 0 <= v < 1e4]
    if len(items) < 3:
        return None
    best = None
    for a in range(len(items)):
        for b in range(a + 1, len(items)):
            ia, va = items[a]; ib, vb = items[b]
            if ib == ia:
                continue
            step = (vb - va) / (ib - ia)
            if step <= 0:
                continue
            v0 = va - ia * step
            tol = max(abs(step) * 0.10, 0.3)
            hits = [i for i, v in items if abs((v0 + i * step) - v) <= tol]
            # bonus for nice step and v0 near a round number
            nice = any(abs(step - c) < c * 0.03 for c in (0.1,0.2,0.25,0.5,1,2,2.5,5,10,20,25,50,100))
            score = len(hits) + (0.5 if nice else 0)
            if best is None or score > best[0]:
                best = (score, v0, step, hits)
    if best is None:
        return None
    _, v0, step, hits = best
    if len(hits) < max(3, int(n * min_frac)):
        return None  # not enough agreement -> don't trust it
    # snap step + v0 to nice values
    for c in (0.1,0.2,0.25,0.5,1,2,2.5,5,10,20,25,50,100):
        if abs(step - c) < c * 0.05:
            step = c; break
    v0 = round(v0 / step) * step if step >= 1 else round(v0, 4)
    return [round(v0 + i * step, 6) for i in range(n)]


def calibrate_grid(img, box):
    x, y, w, h = box
    H, W = img.shape[:2]
    gx, gy = _gridline_positions(img, box)
    xpairs, ypairs = [], []
    x_ocr_by_idx, y_ocr_by_idx = {}, {}
    for i, px_rel in enumerate(gx):
        cx0 = x + px_rel
        crop = img[y+h+4:min(y+h+52, H), max(cx0-45,0):min(cx0+45, W)]
        v = _ocr_single(crop)
        if v is not None:
            x_ocr_by_idx[i] = v
            xpairs += _expand(v, float(px_rel), 1.2)
    for i, py_rel in enumerate(gy):
        cy0 = y + py_rel
        crop = img[max(cy0-22,0):min(cy0+22, H), max(x-120,0):x-4]
        v = _ocr_single(crop)
        if v is not None:
            y_ocr_by_idx[i] = v
            ypairs += _expand(v, float(py_rel), 1.2)
    # merge strip-based OCR (better decimals) into the same pools, box-relative px
    xs = img[y+h+3:min(y+h+58, H), max(x-45,0):min(x+w+45, W)]
    for v, cx0, cy0 in _ocr_numbers(xs):
        xpairs += _expand(v, cx0 + max(x-45,0) - x, 1.0)
    ys = img[max(y-20,0):min(y+h+20, H), max(x-130,0):x-2]
    for v, cx0, cy0 in _ocr_numbers(ys):
        ypairs += _expand(v, cy0 + max(y-20,0) - y, 1.0)
    fx = _fit_pairs(xpairs, w)
    fy = _fit_pairs(ypairs, h)
    return fx, fy

# ---------- 3. curve tracing ----------
def _color_masks(crop):
    hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
    Hh, S, V = hsv[...,0].astype(int), hsv[...,1].astype(int), hsv[...,2].astype(int)
    colored = (S > 70) & (V > 70)
    pix = np.column_stack([Hh[colored], S[colored], V[colored]])
    masks = []
    if len(pix) > 200:
        # quantize hue into buckets, split buckets by V (light vs dark same hue)
        feats = np.column_stack([pix[:,0], pix[:,2]//1]).astype(float)
        feats[:,0] *= 2.0  # weight hue
        best = None
        from numpy.random import default_rng
        for k in range(1, 7):
            crit = cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER
            comp, lab, cen = cv2.kmeans(feats.astype(np.float32), k, None,
                                        (crit, 50, 0.5), 5, cv2.KMEANS_PP_CENTERS)
            # separation: min pairwise center distance
            if k == 1: sep = 1e9
            else:
                dd = [np.linalg.norm(cen[i]-cen[j]) for i in range(k) for j in range(i+1,k)]
                sep = min(dd)
            score = comp/len(feats)
            if sep < 25: break  # centers too close -> stop growing k
            best = (k, lab, cen)
        k, lab, cen = best
        ys_, xs_ = np.where(colored)
        for ci in range(k):
            m = np.zeros(crop.shape[:2], np.uint8)
            sel = (lab.ravel() == ci)
            if sel.sum() < 150: continue
            m[ys_[sel], xs_[sel]] = 255
            m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, np.ones((5,5), np.uint8))
            mean_bgr = crop[m > 0].mean(axis=0)
            masks.append((m, mean_bgr))
    # dark (black) curves: dark, low saturation, exclude small text blobs
    dark = ((V < 110) & (S < 90)).astype(np.uint8)*255
    dark[:12, :] = 0; dark[-12:, :] = 0; dark[:, :12] = 0; dark[:, -12:] = 0
    ch0, cw0 = dark.shape
    hk = cv2.getStructuringElement(cv2.MORPH_RECT, (max(int(cw0*0.55),80), 1))
    vk = cv2.getStructuringElement(cv2.MORPH_RECT, (1, max(int(ch0*0.55),80)))
    longlines = cv2.dilate(cv2.erode(dark, hk), hk) | cv2.dilate(cv2.erode(dark, vk), vk)
    dark = cv2.subtract(dark, cv2.dilate(longlines, np.ones((3,3), np.uint8)))
    dark = cv2.morphologyEx(dark, cv2.MORPH_CLOSE, np.ones((13,13), np.uint8))
    n, labs, stats, _ = cv2.connectedComponentsWithStats(dark)
    dm = np.zeros_like(dark)
    ch, cw = crop.shape[:2]
    for i in range(1, n):
        x0, y0, ww, hh, area = stats[i]
        if not ((ww > 0.3*cw or hh > 0.3*ch) and area > 800 and hh > 4): continue
        comp0 = (labs == i)
        yy, xx = np.where(comp0)
        near_border = ((xx < 10) | (xx > cw-11) | (yy < 10) | (yy > ch-11)).mean()
        if near_border > 0.6: continue  # frame remnant
        comp = comp0
        cols = comp[:, max(x0,0):x0+ww].sum(axis=0)
        cols = cols[cols > 0]
        if len(cols) == 0: continue
        if np.median(cols) > 18: continue      # text block, not a thin curve
        if len(cols) < 0.35*max(ww, hh): continue
        dm[comp] = 255
    if dm.sum() > 255*400:
        masks.append((dm, np.array([30.,30.,30.])))
    return masks

def _components(mask, min_span=0.25):
    """split a color mask into separate curve components."""
    n, labs, stats, _ = cv2.connectedComponentsWithStats(
        cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((7,7), np.uint8)))
    h, w = mask.shape
    out = []
    for i in range(1, n):
        x0, y0, ww, hh, area = stats[i]
        if area < 250: continue
        if ww < min_span*w and hh < min_span*h: continue
        out.append((labs == i).astype(np.uint8)*255)
    return out or ([mask] if mask.sum() > 255*250 else [])

def _trace_component(comp, inset=6):
    """single-valued pixel path: one y per column (median of largest run)."""
    h, w = comp.shape
    core = comp[inset:h-inset, inset:w-inset]
    pts = []
    for col in range(core.shape[1]):
        ys = np.where(core[:, col] > 0)[0]
        if len(ys) == 0: continue
        # split into contiguous runs, keep the largest
        runs, start = [], ys[0]
        for a, b in zip(ys, ys[1:]):
            if b - a > 3:
                runs.append((start, a)); start = b
        runs.append((start, ys[-1]))
        s, e = max(runs, key=lambda r: r[1]-r[0])
        pts.append((col + inset, (s + e)/2.0 + inset))
    return pts

def trace(mask, box, cal_x, cal_y, inset=6):
    x, y, w, h = box
    data = []
    for comp in _components(mask):
        pts = _trace_component(comp, inset)
        if len(pts) < 8: continue
        d = [(float(cal_x['px2val'](px)), float(cal_y['px2val'](py))) for px, py in pts]
        data.append(d)
    return data  # list of curves

def _dedupe(curves, xr, yr, tol=0.02, xlog=False, ylog=False):
    """drop shade-duplicates: near-identical paths from the same physical curve.
    On log axes the comparison happens in log space, else a small absolute value
    (e.g. 2 pF) looks 'identical' to a large one under a linear tolerance."""
    tx = (lambda v: np.log10(np.maximum(v, 1e-12))) if xlog else (lambda v: v)
    ty = (lambda v: np.log10(np.maximum(v, 1e-12))) if ylog else (lambda v: v)
    keep = []
    span = lambda d: max(a for a, b in d) - min(a for a, b in d)
    for d in sorted(curves, key=lambda d: (span(d), len(d)), reverse=True):
        dup = False
        xs = tx(np.array([a for a, b in d])); ys = ty(np.array([b for a, b in d]))
        for k in keep:
            kx = tx(np.array([a for a, b in k])); ky = ty(np.array([b for a, b in k]))
            lo, hi = max(xs.min(), kx.min()), min(xs.max(), kx.max())
            if hi - lo < 0.3*abs(tx(np.array([xr + 1e-9]))[0] if xlog else xr): continue
            q = np.linspace(lo, hi, 40)
            o1, o2 = np.argsort(xs), np.argsort(kx)
            diff = np.abs(np.interp(q, xs[o1], ys[o1]) - np.interp(q, kx[o2], ky[o2]))
            ref = (ys.max() - ys.min()) + (ky.max() - ky.min())
            if np.median(diff) < tol * max(ref, 1e-6): dup = True; break
        if not dup: keep.append(d)
    return keep

def downsample(data, n=40):
    if len(data) <= n: return data
    idx = np.linspace(0, len(data)-1, n).astype(int)
    return [data[i] for i in sorted(set(idx.tolist()))]

def digitize_page_img(img, boxes=None):
    if boxes is None: boxes = find_plot_boxes(img)
    out = []
    for bi, box in enumerate(boxes):
        cx, cy = calibrate_grid(img, box)
        if not cx or not cy:
            sx, sy = calibrate(img, box)
            x0, y0 = box[0], box[1]
            if cx is None and sx is not None:
                f = sx['px2val']; cx = {**sx, 'px2val': (lambda p, F=f, X=x0: F(p + X))}
            if cy is None and sy is not None:
                f = sy['px2val']; cy = {**sy, 'px2val': (lambda p, F=f, Y=y0: F(p + Y))}
        if not cx or not cy:
            out.append({'box': [int(v) for v in box], 'error': 'axis calibration failed'}); continue
        x, y, w, h = box
        crop = img[y:y+h, x:x+w]
        xr = abs(float(cx['max']) - float(cx['min'])) or 1.0
        yr = abs(float(cy['max']) - float(cy['min'])) or 1.0
        raw = []
        for m, bgr in _color_masks(crop):
            hexc = '#%02x%02x%02x' % (int(bgr[2]), int(bgr[1]), int(bgr[0]))
            for d in trace(m, (0,0,w,h), cx, cy):
                if len(d) >= 8: raw.append((hexc, d))
        kept = _dedupe([d for _, d in raw], xr, yr,
                       xlog=(cx['scale'] == 'log'), ylog=(cy['scale'] == 'log'))
        kept_ids = {id(d) for d in kept}
        curves = []
        for hexc, d in raw:
            if id(d) not in kept_ids: continue
            ds = downsample(sorted(d, key=lambda p: p[0]), 60)
            curves.append({'color': hexc,
                           'x': [round(a, 4) for a, b in ds],
                           'y': [round(b, 4) for a, b in ds]})
        out.append({'box': [int(v) for v in box],
                    'x_axis': {k: float(cx[k]) if isinstance(cx[k], (np.floating, float, int)) else cx[k] for k in ('scale','ticks','min','max')},
                    'y_axis': {k: float(cy[k]) if isinstance(cy[k], (np.floating, float, int)) else cy[k] for k in ('scale','ticks','min','max')},
                    'curves': curves})
    return img, out

def digitize_page(path):
    img = cv2.imread(path)
    return digitize_page_img(img)

def _digitize_page_legacy(path):
    img = cv2.imread(path)
    out = []
    for bi, box in enumerate(find_plot_boxes(img)):
        cx, cy = calibrate_grid(img, box)
        if not cx or not cy:
            sx, sy = calibrate(img, box)
            x0, y0 = box[0], box[1]
            if cx is None and sx is not None:
                f = sx['px2val']; cx = {**sx, 'px2val': (lambda p, F=f, X=x0: F(p + X))}
            if cy is None and sy is not None:
                f = sy['px2val']; cy = {**sy, 'px2val': (lambda p, F=f, Y=y0: F(p + Y))}
        if not cx or not cy:
            out.append({'box': box, 'error': 'axis calibration failed'}); continue
        x, y, w, h = box
        crop = img[y:y+h, x:x+w]
        curves = []
        for m, bgr in _color_masks(crop):
            d = trace(m, (0,0,w,h), cx, cy)
            if not d or len(d) < 8: continue
            d = downsample(d)
            hexc = '#%02x%02x%02x' % (int(bgr[2]), int(bgr[1]), int(bgr[0]))
            curves.append({'color': hexc,
                           'x': [round(a, 4) for a, b in d],
                           'y': [round(b, 4) for a, b in d]})
        out.append({'box': [int(v) for v in box],
                    'x_axis': {k: cx[k] for k in ('scale','ticks','min','max')},
                    'y_axis': {k: cy[k] for k in ('scale','ticks','min','max')},
                    'curves': curves})
    return img, out

if __name__ == '__main__':
    import sys
    img, res = digitize_page(sys.argv[1])
    for i, g in enumerate(res):
        if 'error' in g: print(i, 'ERROR', g['error'], g['box']); continue
        print(i, 'box', g['box'], '| x', g['x_axis']['scale'], g['x_axis']['ticks'],
              '| y', g['y_axis']['scale'], g['y_axis']['ticks'], '| curves', len(g['curves']))


def digitize_box(img, box):
    """Digitize one explicit plot box (used by manual box drawing)."""
    _, res = digitize_page_img(img, [tuple(int(v) for v in box)])
    return res[0] if res else {'error': 'trace failed'}

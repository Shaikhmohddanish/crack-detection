# app.py
import io, os, time, logging
from pathlib import Path
from flask import Flask, request, send_file, jsonify, render_template, abort
from flask_cors import CORS
from ultralytics import YOLO
from PIL import Image
import torch
import cv2

# Be conservative with CPU threads on shared host
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")
os.environ.setdefault("MPLBACKEND", "Agg")  # avoid GUI backends

try:
    torch.set_num_threads(1)
except Exception:
    pass
try:
    cv2.setNumThreads(0)
except Exception:
    pass

logging.basicConfig(level=logging.INFO)
log = logging.getLogger("crack-api")

# ----- config -----
MAX_BYTES = 8 * 1024 * 1024   # 8 MB request cap
MAX_LONG_SIDE = 1280          # server-side resize cap

WEIGHTS_PATH = os.getenv("WEIGHTS_PATH", "best.pt")
DEFAULT_IMGSZ = int(os.getenv("IMGSZ", "640"))
DEFAULT_CONF  = float(os.getenv("CONF", "0.25"))
DEVICE        = os.getenv("DEVICE", "cpu")  # <- Intel Mac: use CPU

# Verify model file exists
if not os.path.exists(WEIGHTS_PATH):
    raise RuntimeError(f"Model file {WEIGHTS_PATH} not found. Make sure it's in your repository.")

app = Flask(__name__, template_folder='templates')
CORS(app)

@app.before_request
def _limit_body():
    cl = request.content_length
    if cl is not None and cl > MAX_BYTES:
        abort(413)  # Payload Too Large

model = YOLO(WEIGHTS_PATH)

def _resize_if_needed(pil_img, max_side=MAX_LONG_SIDE):
    w, h = pil_img.size
    long_side = max(w, h)
    if long_side <= max_side:
        return pil_img
    scale = max_side / float(long_side)
    new_w, new_h = int(w * scale), int(h * scale)
    return pil_img.resize((new_w, new_h))

@app.get("/")
def index():
    return render_template("index.html")

@app.get("/debug")
def debug():
    exists = os.path.exists(WEIGHTS_PATH)
    size = os.path.getsize(WEIGHTS_PATH) if exists else None
    import ultralytics
    import sys
    return jsonify({
        "python": sys.version,
        "torch": torch.__version__,
        "ultralytics": ultralytics.__version__,
        "cv2": cv2.__version__,
        "weights_path": WEIGHTS_PATH,
        "weights_exists": exists,
        "weights_size_bytes": size
    })

@app.get("/health")
def health():
    return jsonify({"status": "ok", "weights": WEIGHTS_PATH})

def _read_image_from_request():
    if "file" not in request.files:
        return None, ("no file uploaded; expected form field 'file'", 400)
    f = request.files["file"]
    if f.filename == "":
        return None, ("empty filename", 400)
    try:
        img = Image.open(f.stream).convert("RGB")
        return img, None
    except Exception as e:
        return None, (f"failed to read image: {e}", 400)

@app.post("/predict")
def predict_image():
    try:
        img, err = _read_image_from_request()
        if err:
            msg, code = err
            log.error("read_image error: %s", msg)
            return jsonify({"error": msg}), code

        conf  = float(request.values.get("conf", DEFAULT_CONF))
        imgsz = int(request.values.get("imgsz", DEFAULT_IMGSZ))

        # server-side safety resize (prevents RAM spikes)
        img = _resize_if_needed(img, MAX_LONG_SIDE)

        t0 = time.time()
        results = model.predict(
            source=img,
            imgsz=imgsz,
            conf=conf,
            device="cpu",      # force CPU on Render
            verbose=False
        )
        dt = time.time() - t0

        r0 = results[0]
        dets = int(len(r0.boxes) if r0.boxes is not None else 0)

        annotated_bgr = r0.plot()
        annotated_rgb = annotated_bgr[:, :, ::-1]
        out = Image.fromarray(annotated_rgb)

        buf = io.BytesIO()
        out.save(buf, format="PNG")
        buf.seek(0)
        response = send_file(buf, mimetype="image/png")
        response.headers["X-Inference-Time"] = f"{dt:.3f}s"
        response.headers["X-Detections"] = str(dets)
        return response
    except Exception as e:
        log.exception("predict failed")
        return jsonify({"error": str(e)}), 500

@app.post("/predict-json")
def predict_json():
    try:
        img, err = _read_image_from_request()
        if err:
            msg, code = err
            log.error("read_image error: %s", msg)
            return jsonify({"error": msg}), code

        conf  = float(request.values.get("conf", DEFAULT_CONF))
        imgsz = int(request.values.get("imgsz", DEFAULT_IMGSZ))

        # server-side safety resize (prevents RAM spikes)
        img = _resize_if_needed(img, MAX_LONG_SIDE)

        results = model.predict(
            source=img,
            imgsz=imgsz,
            conf=conf,
            device="cpu",      # force CPU on Render
            verbose=False
        )
        r0 = results[0]
        names = r0.names

        boxes = []
        if r0.boxes is not None and len(r0.boxes):
            xyxy = r0.boxes.xyxy.cpu().numpy()
            confs = r0.boxes.conf.cpu().numpy()
            clss  = r0.boxes.cls.cpu().numpy()
            for i in range(len(xyxy)):
                boxes.append({
                    "xyxy": [float(x) for x in xyxy[i]],
                    "conf": float(confs[i]),
                    "class_id": int(clss[i]),
                    "class_name": names[int(clss[i])] if names else str(int(clss[i])),
                })

        masks = []
        if getattr(r0, "masks", None) is not None:
            for poly in r0.masks.xy:
                masks.append([[float(x), float(y)] for x, y in poly])

        return jsonify({"boxes": boxes, "masks": masks})
    except Exception as e:
        log.exception("predict_json failed")
        return jsonify({"error": str(e)}), 500

@app.get("/debug")
def debug():
    exists = os.path.exists(WEIGHTS_PATH)
    size = os.path.getsize(WEIGHTS_PATH) if exists else None
    import ultralytics
    import sys
    return jsonify({
        "python": sys.version,
        "torch": torch.__version__,
        "ultralytics": ultralytics.__version__,
        "cv2": cv2.__version__,
        "weights_path": WEIGHTS_PATH,
        "weights_exists": exists,
        "weights_size_bytes": size
    })

if __name__ == "__main__":
    # dev server when run directly
    app.run(host="0.0.0.0", port=int(os.getenv("PORT", 8000)), debug=True)

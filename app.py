# app.py
import io, os, time
from pathlib import Path
from flask import Flask, request, send_file, jsonify, render_template
from flask_cors import CORS
from ultralytics import YOLO
from PIL import Image

WEIGHTS_PATH = os.getenv("WEIGHTS_PATH", "best.pt")
DEFAULT_IMGSZ = int(os.getenv("IMGSZ", "640"))
DEFAULT_CONF  = float(os.getenv("CONF", "0.25"))
DEVICE        = os.getenv("DEVICE", "cpu")  # <- Intel Mac: use CPU

# Ensure model file exists before loading
if not os.path.exists(WEIGHTS_PATH):
    import urllib.request
    url = os.getenv("WEIGHTS_URL")  # set this in Render env if needed
    if url:
        print(f"Downloading model from {url}...")
        urllib.request.urlretrieve(url, WEIGHTS_PATH)
    else:
        print(f"Warning: Model file {WEIGHTS_PATH} not found. Make sure it's in your repository.")

app = Flask(__name__, template_folder='templates')
CORS(app)

model = YOLO(WEIGHTS_PATH)

@app.get("/")
def index():
    return render_template("index.html")

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
    img, err = _read_image_from_request()
    if err:
        msg, code = err
        return jsonify({"error": msg}), code

    conf  = float(request.values.get("conf", DEFAULT_CONF))
    imgsz = int(request.values.get("imgsz", DEFAULT_IMGSZ))

    t0 = time.time()
    results = model.predict(source=img, imgsz=imgsz, conf=conf, device=DEVICE)
    dt = time.time() - t0

    r0 = results[0]
    annotated_bgr = r0.plot()      # numpy BGR
    annotated_rgb = annotated_bgr[:, :, ::-1]
    out = Image.fromarray(annotated_rgb)

    buf = io.BytesIO()
    out.save(buf, format="PNG")
    buf.seek(0)
    # Create response and then add headers
    response = send_file(buf, mimetype="image/png")
    response.headers["X-Inference-Time"] = f"{dt:.3f}s"
    response.headers["X-Detections"] = str(len(r0.boxes) if r0.boxes is not None else 0)
    return response

@app.post("/predict-json")
def predict_json():
    img, err = _read_image_from_request()
    if err:
        msg, code = err
        return jsonify({"error": msg}), code

    conf  = float(request.values.get("conf", DEFAULT_CONF))
    imgsz = int(request.values.get("imgsz", DEFAULT_IMGSZ))

    results = model.predict(source=img, imgsz=imgsz, conf=conf, device=DEVICE)
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

if __name__ == "__main__":
    # dev server when run directly
    app.run(host="0.0.0.0", port=int(os.getenv("PORT", 8000)), debug=True)

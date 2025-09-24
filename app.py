import os
import cv2
import numpy as np
from PIL import Image
from functools import lru_cache

import torch  # for torchvision tensor type
from torchvision import transforms
import mediapipe as mp
import onnxruntime as ort
import gradio as gr

mp_face = mp.solutions.face_detection

# ----------------------------
# 1) Load ONNX model (auto-select provider)
# ----------------------------
@lru_cache(maxsize=1)
def load_onnx_session(onnx_path: str):
    if not os.path.exists(onnx_path):
        raise FileNotFoundError(
            f"ONNX file not found: {onnx_path}. "
            f"Make sure you exported it (e.g., python export_onnx.py)."
        )

    available = ort.get_available_providers()
    preferred = [
        "TensorrtExecutionProvider",  # if available
        "CUDAExecutionProvider",      # GPU (CUDA)
        "CPUExecutionProvider",       # CPU fallback
    ]
    providers = [p for p in preferred if p in available] or ["CPUExecutionProvider"]

    sess = ort.InferenceSession(onnx_path, providers=providers)

    # Try declared names; fall back to actual graph names if needed
    try:
        input_name = "input"
        output_name = "logits"
        _ = sess.run([output_name], {input_name: np.zeros((1, 3, 224, 224), dtype=np.float32)})
    except Exception:
        input_name = sess.get_inputs()[0].name
        output_name = sess.get_outputs()[0].name

    return sess, input_name, output_name, providers


# ----------------------------
# 2) Helpers (face detect + preprocessing + inference)
# ----------------------------
def _to_pil(arr):
    return Image.fromarray(arr.astype(np.uint8))

def detect_and_crop_face(pil_image, min_size=80, margin=0.25, conf_th=0.5):
    """
    Detect the most confident face and return a square, padded crop.
    Falls back to a center crop if detection is weak or tiny.
    """
    img = np.array(pil_image)
    h, w = img.shape[:2]

    with mp_face.FaceDetection(model_selection=0, min_detection_confidence=conf_th) as fd:
        # MediaPipe expects BGR input
        results = fd.process(cv2.cvtColor(img, cv2.COLOR_RGB2BGR))

    if results.detections:
        det = max(results.detections, key=lambda d: d.score[0])
        rel = det.location_data.relative_bounding_box
        x, y, bw, bh = rel.xmin, rel.ymin, rel.width, rel.height

        # convert to absolute
        x1 = max(int(x * w), 0); y1 = max(int(y * h), 0)
        x2 = min(int((x + bw) * w), w - 1); y2 = min(int((y + bh) * h), h - 1)

        # reject tiny boxes
        if (x2 - x1) >= min_size and (y2 - y1) >= min_size:
            # make square with margin
            cx = (x1 + x2) // 2
            cy = (y1 + y2) // 2
            half = int(max(x2 - x1, y2 - y1) * (1 + margin) / 2)
            sx1 = max(cx - half, 0); sy1 = max(cy - half, 0)
            sx2 = min(cx + half, w - 1); sy2 = min(cy + half, h - 1)
            crop = img[sy1:sy2, sx1:sx2]
            return _to_pil(crop), (sx1, sy1, sx2, sy2)

    # fallback: center crop square
    side = min(h, w)
    cx, cy = w // 2, h // 2
    half = side // 2
    sx1 = cx - half; sy1 = cy - half
    sx2 = cx + half; sy2 = cy + half
    crop = img[sy1:sy2, sx1:sx2]
    # fix typo (xs2 -> sx2)
    crop = img[sy1:sy2, sx1:sx2]
    return _to_pil(crop), (sx1, sy1, sx2, sy2)

# ImageNet normalization (required for torchvision-pretrained backbones)
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD  = [0.229, 0.224, 0.225]

def _softmax_np(x, axis=1):
    x = x - np.max(x, axis=axis, keepdims=True)
    e = np.exp(x)
    return e / np.sum(e, axis=axis, keepdims=True)

@lru_cache(maxsize=2)
def build_transform(use_imagenet_norm: bool):
    return transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD) if use_imagenet_norm
        else transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
    ])

def predict(image: Image.Image, tf, sess, input_name, output_name):
    # torch -> numpy float32
    img_tensor = tf(image).unsqueeze(0)                       # (1,3,224,224) torch.FloatTensor
    inp = img_tensor.numpy().astype(np.float32)               # ORT expects numpy float32
    inp = np.ascontiguousarray(inp)                           # safety for some backends

    logits = sess.run([output_name], {input_name: inp})[0]    # (1,2)
    probs = _softmax_np(logits, axis=1)[0]                    # (2,)
    pred = int(np.argmax(probs))
    return pred, probs

# ----------------------------
# 3) Gradio UI logic
# ----------------------------
def process_images(files, skip_crop, use_imagenet, onnx_path):
    """
    Args:
        files: list of tempfile paths (or None)
        skip_crop: bool
        use_imagenet: bool
        onnx_path: str
    Returns:
        providers_text: str
        gallery_detected: list[(PIL.Image, caption)]
        gallery_crops: list[(PIL.Image, caption)]
        table_rows: list of lists for Dataframe
    """
    if not files:
        return ("No files uploaded.",
                [], [], [])

    try:
        sess, ORT_INPUT, ORT_OUTPUT, providers = load_onnx_session(onnx_path)
    except Exception as e:
        return (f"Model error: {e}", [], [], [])

    tfm = build_transform(bool(use_imagenet))

    gallery_detected = []
    gallery_crops = []
    table_rows = []  # [filename, label, real%, spoof%]

    for f in files:
        # Gradio may pass file objects or paths; handle both
        path = f.name if hasattr(f, "name") else str(f)
        try:
            image = Image.open(path).convert("RGB")
        except Exception as e:
            table_rows.append([os.path.basename(path), f"Load error: {e}", "-", "-"])
            continue

        disp_img = np.array(image).copy()

        if not skip_crop:
            cropped_face, box = detect_and_crop_face(image)
            (x1, y1, x2, y2) = box
            cv2.rectangle(disp_img, (x1, y1), (x2, y2), (0, 255, 0), 2)
        else:
            cropped_face = image

        # Prediction
        try:
            label_idx, probs = predict(cropped_face, tfm, sess, ORT_INPUT, ORT_OUTPUT)
            label = "Real" if label_idx == 0 else "Spoof"
            real_pct = f"{probs[0]*100:.2f}%"
            spoof_pct = f"{probs[1]*100:.2f}%"
        except Exception as e:
            label = f"Inference error: {e}"
            real_pct = "-"
            spoof_pct = "-"

        gallery_detected.append((_to_pil(disp_img), f"Uploaded: {os.path.basename(path)}"))
        gallery_crops.append((cropped_face, f"Crop · Pred: {label} · Real {real_pct} · Spoof {spoof_pct}"))
        table_rows.append([os.path.basename(path), label, real_pct, spoof_pct])

    providers_text = "Providers: " + ", ".join(providers) if providers else "Providers: N/A"
    return providers_text, gallery_detected, gallery_crops, table_rows


with gr.Blocks(title="Face Anti-Spoofing Detector (ONNX + Gradio)") as demo:
    gr.Markdown("# Face Anti-Spoofing Detector")
    gr.Markdown("ONNX Runtime · Upload one or more images to check if the face is real or spoofed.")

    with gr.Row():
        files = gr.File(label="Choose images", file_count="multiple", file_types=["image"])
    with gr.Row():
        skip_crop = gr.Checkbox(label="Skip face crop (use full image)", value=False)
        use_imagenet = gr.Checkbox(label="Use ImageNet normalization", value=True)
        onnx_path = gr.Textbox(label="ONNX path", value="face_antispoof.onnx")

    run_btn = gr.Button("Run")

    providers_out = gr.Markdown()
    gr.Markdown("### Detected Faces")
    gallery_detected = gr.Gallery(height="auto", columns=2, preview=True, allow_preview=True)
    gr.Markdown("### Crops & Predictions")
    gallery_crops = gr.Gallery(height="auto", columns=4, preview=True, allow_preview=True)
    results_table = gr.Dataframe(
        headers=["filename", "prediction", "real_conf", "spoof_conf"],
        datatype=["str", "str", "str", "str"],
        interactive=False,
        wrap=True,
        label="Results"
    )

    run_btn.click(
        fn=process_images,
        inputs=[files, skip_crop, use_imagenet, onnx_path],
        outputs=[providers_out, gallery_detected, gallery_crops, results_table]
    )

if __name__ == "__main__":
    demo.launch()

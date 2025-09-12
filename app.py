import streamlit as st
import torch  # needed for torchvision transforms tensor type
from PIL import Image
import numpy as np
import cv2
from torchvision import transforms
import mediapipe as mp
import onnxruntime as ort
import os

mp_face = mp.solutions.face_detection

# ----------------------------
# 1) Load ONNX model (auto-select provider)
# ----------------------------
@st.cache_resource(show_spinner=False)
def load_onnx_session(onnx_path: str):
    if not os.path.exists(onnx_path):
        raise FileNotFoundError(
            f"ONNX file not found: {onnx_path}. "
            f"Make sure you exported it (e.g., python export_onnx.py)."
        )
    available = ort.get_available_providers()
    preferred = [
        "CUDAExecutionProvider",      # GPU (CUDA)
        "TensorrtExecutionProvider",  # GPU (TensorRT)
        "CPUExecutionProvider"        # CPU fallback
    ]
    providers = [p for p in preferred if p in available]
    if not providers:
        providers = ["CPUExecutionProvider"]
    sess = ort.InferenceSession(onnx_path, providers=providers)

    # Try to use declared names; fall back to actual graph names if different
    try:
        input_name = "input"
        output_name = "logits"
        # quick shape check; will throw if name doesn't exist
        _ = sess.get_inputs()
        _ = sess.run([output_name], {input_name: np.zeros((1,3,224,224), dtype=np.float32)})
    except Exception:
        input_name = sess.get_inputs()[0].name
        output_name = sess.get_outputs()[0].name

    return sess, input_name, output_name, providers

try:
    ort_sess, ORT_INPUT, ORT_OUTPUT, USED_PROVIDERS = load_onnx_session("face_antispoof.onnx")
except Exception as e:
    ort_sess, ORT_INPUT, ORT_OUTPUT, USED_PROVIDERS = None, None, None, None
    st.error(str(e))

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
    return _to_pil(crop), (sx1, sy1, sx2, sy2)

# ImageNet normalization (required for torchvision-pretrained backbones)
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD  = [0.229, 0.224, 0.225]

def _softmax_np(x, axis=1):
    x = x - np.max(x, axis=axis, keepdims=True)
    e = np.exp(x)
    return e / np.sum(e, axis=axis, keepdims=True)

@st.cache_data(show_spinner=False)
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
# 3) Streamlit UI
# ----------------------------
st.title("Face Anti-Spoofing Detector")
st.caption("ONNX Runtime · Providers: " + (", ".join(USED_PROVIDERS) if USED_PROVIDERS else "N/A"))

st.write("Upload one or more images to check if the face is real or spoofed.")

uploaded_files = st.file_uploader(
    "Choose images", type=["jpg", "jpeg", "png"], accept_multiple_files=True
)

skip_crop = st.checkbox("Skip face crop (use full image)")
use_imagenet = st.checkbox("Use ImageNet normalization", value=True)

tfm = build_transform(use_imagenet)

if uploaded_files and ort_sess is not None:
    for uploaded_file in uploaded_files:
        image = Image.open(uploaded_file).convert("RGB")

        # Visualize detection bbox
        disp_img = np.array(image).copy()
        if not skip_crop:
            cropped_face, box = detect_and_crop_face(image)
            (x1, y1, x2, y2) = box
            cv2.rectangle(disp_img, (x1, y1), (x2, y2), (0, 255, 0), 2)
        else:
            cropped_face = image

        st.image(disp_img, caption=f"Uploaded: {uploaded_file.name}", use_container_width=True)
        st.image(cropped_face, caption="Detected/Cropped Face", width=224)

        label_idx, probs = predict(cropped_face, tfm, ort_sess, ORT_INPUT, ORT_OUTPUT)
        label = "Real" if label_idx == 0 else "Spoof"
        st.success(f"Prediction: {label}")
        st.info(f"Confidence — Real: {probs[0]*100:.2f}%, Spoof: {probs[1]*100:.2f}%")
elif uploaded_files and ort_sess is None:
    st.stop()
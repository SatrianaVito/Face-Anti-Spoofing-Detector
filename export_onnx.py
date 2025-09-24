import torch
from src.model import ResNet18Binary

MODEL_PTH = "face_antispoof.pth"
ONNX_OUT  = "face_antispoof.onnx"

def main():
    device = torch.device("cpu")  # export on CPU
    model = ResNet18Binary(pretrained=False)
    state = torch.load(MODEL_PTH, map_location=device)
    model.load_state_dict(state)
    model.eval()

    # dummy input matches your eval pipeline: (N,3,224,224)
    dummy = torch.randn(1, 3, 224, 224, device=device)

    torch.onnx.export(
        model, dummy, ONNX_OUT,
        input_names=["input"],
        output_names=["logits"],
        dynamic_axes={"input": {0: "batch"}, "logits": {0: "batch"}},
        opset_version=17,  # 13+ is usually fine; 17 is widely supported
        do_constant_folding=True
    )
    print(f"Exported to {ONNX_OUT}")

if __name__ == "__main__":
    main()

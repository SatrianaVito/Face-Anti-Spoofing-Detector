import torch
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt
from src.dataset import get_dataloaders
from src.model import ResNet18Binary

def _run_eval(model, loader, device, split_name="Validation"):
    model.eval()
    all_labels, all_preds = [], []

    with torch.no_grad():
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs, 1)
            all_labels.extend(labels.cpu().numpy())
            all_preds.extend(predicted.cpu().numpy())

    correct = sum(p == l for p, l in zip(all_preds, all_labels))
    total = len(all_labels) if all_labels else 0
    acc = 100.0 * correct / total if total else 0.0
    print(f"{split_name} Accuracy: {acc:.2f}%")

    if total:
        cm = confusion_matrix(all_labels, all_preds)
        disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["Real", "Spoof"])
        disp.plot(cmap=plt.cm.Blues)
        plt.title(f"{split_name} Confusion Matrix")
        plt.show()

def evaluate_model(data_dir, model_path="face_antispoof.pth"):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Try (train, val, test) else (train, val)
    loaders = get_dataloaders(data_dir, do_face_crop=True)
    if len(loaders) == 3:
        _, val_loader, test_loader = loaders
    else:
        _, val_loader = loaders
        test_loader = None  # no test split yet

    # Instantiate the pretrained model wrapper
    model = ResNet18Binary(pretrained=False).to(device)  # pretrained=False for eval; weights come from the .pth
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    _run_eval(model, val_loader, device, split_name="Validation")
    if test_loader is not None:
        _run_eval(model, test_loader, device, split_name="Test")
    else:
        print("No test loader found. Add a test split (e.g., data/test/...) and update get_dataloaders to return it.")
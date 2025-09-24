import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR
from src.dataset import get_dataloaders
from src.model import ResNet18Binary  # swapped in

def train_model(data_dir, epochs=15, lr=3e-4, weight_decay=1e-4, warmup_epochs=3):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    train_loader, val_loader, _ = get_dataloaders(
        data_dir,
        batch_size=32,
        do_face_crop=True,
        augment=True,
        img_size=224,
        num_workers=4,
    )
    model = ResNet18Binary(pretrained=True).to(device)

    # Freeze backbone for warmup
    for name, p in model.backbone.named_parameters():
        if not name.startswith("fc."):
            p.requires_grad = False

    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    optimizer = optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=lr, weight_decay=weight_decay)
    scheduler = CosineAnnealingLR(optimizer, T_max=epochs - warmup_epochs)

    def evaluate(split_loader):
        model.eval()
        correct, total = 0, 0
        with torch.no_grad():
            for images, labels in split_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                preds = outputs.argmax(1)
                correct += (preds == labels).sum().item()
                total += labels.numel()
        return 100.0 * correct / total if total else 0.0

    best_val = 0.0
    for epoch in range(epochs):
        model.train()
        # Unfreeze after warmup
        if epoch == warmup_epochs:
            for p in model.backbone.parameters():
                p.requires_grad = True
            optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)

        running = 0.0
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=2.0)
            optimizer.step()
            running += loss.item()

        if epoch >= warmup_epochs:
            scheduler.step()

        val_acc = evaluate(val_loader)
        print(f"Epoch {epoch+1}/{epochs}  Loss: {running/len(train_loader):.4f}  Val Acc: {val_acc:.2f}%")
        if val_acc > best_val:
            best_val = val_acc
            torch.save(model.state_dict(), "face_antispoof.pth")
            print(f"  ↳ New best; model saved.")
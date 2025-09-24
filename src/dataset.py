# src/dataset.py
import torch
from torchvision import datasets, transforms
from PIL import Image
import numpy as np

# (A) OPTIONAL: light-weight face crop using the same idea as in app.py
# If you prefer, you can import your exact function from app.py instead.
# dataset.py
try:
    from app import detect_and_crop_face
    _HAS_APP_CROPPER = True
except Exception:
    _HAS_APP_CROPPER = False
    def detect_and_crop_face(pil_image, min_size=80, margin=0.25, conf_th=0.5):
        return pil_image, (0, 0, pil_image.width, pil_image.height)

class FaceCropTransform:
    def __init__(self, enabled: bool = False):
        self.enabled = enabled

    def __call__(self, img: Image.Image):
        if not self.enabled:
            return img
        cropped, _ = detect_and_crop_face(img)  # uses mediapipe path if available
        return cropped

# (B) Standard normalization you already use
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD  = [0.229, 0.224, 0.225]

def get_dataloaders(
    data_dir,
    batch_size=32,
    do_face_crop=False,          # NEW
    augment=True,                # NEW
    img_size=224,                # NEW
    num_workers=2,               # NEW
):
    # Train pipeline
    train_aug = []
    # 1) optional face crop, BEFORE resizing
    train_aug.append(FaceCropTransform(enabled=do_face_crop))

    # 2) resize & (optional) augmentations
    if augment:
        train_aug.extend([
            transforms.Resize((max(256, img_size), max(256, img_size))),
            transforms.RandomResizedCrop(img_size, scale=(0.7, 1.0)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
            transforms.RandomRotation(degrees=10),
        ])
    else:
        train_aug.extend([
            transforms.Resize((img_size, img_size)),
        ])

    # 3) to tensor + normalize (+ optional RandomErasing if augmenting)
    train_aug.extend([
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])
    if augment:
        train_aug.append(
            transforms.RandomErasing(p=0.25, scale=(0.02, 0.12), ratio=(0.3, 3.3))
        )

    train_tf = transforms.Compose(train_aug)

    # Eval/Val pipeline (no heavy augments; optional face crop to match train)
    eval_tf = transforms.Compose([
        FaceCropTransform(enabled=do_face_crop),
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])

    train_data = datasets.ImageFolder(f"{data_dir}/train", transform=train_tf)
    val_data   = datasets.ImageFolder(f"{data_dir}/val",   transform=eval_tf)
    # Optional test split if present
    try:
        test_data  = datasets.ImageFolder(f"{data_dir}/test",  transform=eval_tf)
    except Exception:
        test_data = None

    train_loader = torch.utils.data.DataLoader(
        train_data, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=True
    )
    val_loader   = torch.utils.data.DataLoader(
        val_data, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True
    )
    if test_data is not None:
        test_loader  = torch.utils.data.DataLoader(
            test_data, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True
        )
        return train_loader, val_loader, test_loader
    else:
        return train_loader, val_loader

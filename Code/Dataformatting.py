import torch
import torchvision
from torchvision.models.detection import fasterrcnn_resnet50_fpn
from torchvision.datasets import CocoDetection
from torchvision.transforms import functional as F
from torch.utils.data import DataLoader
import os

# ----------------------------
# CONFIG
# ----------------------------
DATASET_ROOT = "Data sets\Training Data"
IMG_DIR = os.path.join(DATASET_ROOT, "images")
ANN_FILE = os.path.join(DATASET_ROOT, "annotations/instances_train.json")

NUM_CLASSES = 2  # background + item
BATCH_SIZE = 2
EPOCHS = 10
LR = 0.005

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ----------------------------
# DATASET
# ----------------------------
class CocoDataset(CocoDetection):
    def __getitem__(self, idx):
        img, targets = super().__getitem__(idx)

        boxes = []
        labels = []

        for t in targets:
            x, y, w, h = t["bbox"]
            boxes.append([x, y, x + w, y + h])
            labels.append(1)  # single class: item

        boxes = torch.tensor(boxes, dtype=torch.float32)
        labels = torch.tensor(labels, dtype=torch.int64)

        target = {
            "boxes": boxes,
            "labels": labels
        }

        img = F.to_tensor(img)
        return img, target


def collate_fn(batch):
    return tuple(zip(*batch))


dataset = CocoDataset(IMG_DIR, ANN_FILE)
loader = DataLoader(
    dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    collate_fn=collate_fn
)

# ----------------------------
# MODEL
# ----------------------------
model = fasterrcnn_resnet50_fpn(weights="DEFAULT")

# Replace classifier head
in_features = model.roi_heads.box_predictor.cls_score.in_features
model.roi_heads.box_predictor = torchvision.models.detection.faster_rcnn.FastRCNNPredictor(
    in_features, NUM_CLASSES
)

model.to(DEVICE)

# ----------------------------
# OPTIMIZER
# ----------------------------
params = [p for p in model.parameters() if p.requires_grad]
optimizer = torch.optim.SGD(
    params,
    lr=LR,
    momentum=0.9,
    weight_decay=0.0005
)

# ----------------------------
# TRAIN LOOP
# ----------------------------
model.train()

for epoch in range(EPOCHS):
    epoch_loss = 0

    for images, targets in loader:
        images = [img.to(DEVICE) for img in images]
        targets = [{k: v.to(DEVICE) for k, v in t.items()} for t in targets]

        loss_dict = model(images, targets)
        losses = sum(loss for loss in loss_dict.values())

        optimizer.zero_grad()
        losses.backward()
        optimizer.step()

        epoch_loss += losses.item()

    print(f"Epoch {epoch+1}/{EPOCHS} - Loss: {epoch_loss:.3f}")

# ----------------------------
# SAVE MODEL
# ----------------------------
torch.save(model.state_dict(), "fasterrcnn_items.pth")
print("Training complete. Model saved.")

import json
import copy
from pathlib import Path

# === INPUT FILES ===
COCO_FILES = [
    "task1/instances_default.json",
    "task2/instances_default.json",
    "task3/instances_default.json",
]

OUTPUT_FILE = "instances_merged.json"

# ===================

merged = {
    "images": [],
    "annotations": [],
    "categories": []
}

image_id_offset = 0
annotation_id_offset = 0

category_map = None

for coco_path in COCO_FILES:
    with open(coco_path, "r") as f:
        coco = json.load(f)

    # Copy categories once and verify consistency
    if not merged["categories"]:
        merged["categories"] = copy.deepcopy(coco["categories"])
        category_map = {c["id"]: c["id"] for c in coco["categories"]}
    else:
        assert coco["categories"] == merged["categories"], \
            f"Category mismatch in {coco_path}"

    # Map old image IDs to new ones
    image_id_map = {}

    for img in coco["images"]:
        new_img = copy.deepcopy(img)
        new_id = img["id"] + image_id_offset
        image_id_map[img["id"]] = new_id
        new_img["id"] = new_id
        merged["images"].append(new_img)

    # Copy annotations
    for ann in coco["annotations"]:
        new_ann = copy.deepcopy(ann)
        new_ann["id"] = ann["id"] + annotation_id_offset
        new_ann["image_id"] = image_id_map[ann["image_id"]]
        merged["annotations"].append(new_ann)

    image_id_offset += max(img["id"] for img in coco["images"]) + 1
    annotation_id_offset += max(ann["id"] for ann in coco["annotations"]) + 1

# Save merged file
with open(OUTPUT_FILE, "w") as f:
    json.dump(merged, f, indent=2)

print(f"✅ Merged COCO saved to {OUTPUT_FILE}")
print(f"Images: {len(merged['images'])}")
print(f"Annotations: {len(merged['annotations'])}")

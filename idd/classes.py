"""Class vocabulary shared by the Indian Road Driving dataset and the dev benchmark."""

CLASSES = [
    "car", "truck", "bus", "autorickshaw", "motorcycle", "bicycle",
    "rider", "person", "animal", "traffic light", "traffic sign", "vehicle fallback",
]
CLASS_IDX = {c: i for i, c in enumerate(CLASSES)}

# Classes the plan singles out as India-specific and COCO-weak.
INDIA_CLASSES = ["autorickshaw", "animal", "rider"]

# COCO-80 class id -> our class name, used only for the COCO-pretrained baseline.
COCO80_TO_OURS = {
    0: "person", 1: "bicycle", 2: "car", 3: "motorcycle", 5: "bus", 7: "truck",
    9: "traffic light",
    14: "animal", 15: "animal", 16: "animal", 17: "animal", 18: "animal",
    19: "animal", 20: "animal", 21: "animal", 22: "animal", 23: "animal",
}

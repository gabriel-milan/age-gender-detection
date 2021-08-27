import cv2
import pickle
from pathlib import Path

import numpy as np
from tqdm import tqdm

# Configuration
MODES = ["original", "custom"]
DATASET_PATH = Path("pics/age_dataset/")
ORIGINAL_MODEL = "original_models/age_net.caffemodel"
ORIGINAL_PROTO = "original_models/age_deploy.prototxt"
CUSTOM_MODEL = "custom_models/age/optimized_graph.pb"
CUSTOM_PROTO = "custom_models/age/optimized_graph.pbtxt"
ORIGINAL_AGE_RANGES = [
    (0, 2),
    (4, 6),
    (8, 12),
    (15, 20),
    (25, 32),
    (38, 43),
    (48, 53),
    (60, float("inf"))
]
ORIGINAL_AGE_MEANS = [
    1,
    5,
    10,
    17.5,
    28.5,
    40.5,
    50.5,
    80,
]
CUSTOM_AGE_RANGES = [
    (0, 2),
    (3, 5),
    (6, 8),
    (9, 11),
    (12, 14),
    (15, 17),
    (18, 20),
    (21, 23),
    (24, 26),
    (27, 29),
    (30, 32),
    (33, 35),
    (36, 38),
    (39, 41),
    (42, 44),
    (45, 47),
    (48, 50),
    (51, 54),
    (55, 59),
    (60, 64),
    (65, 69),
    (70, 79),
    (80, float('inf')),
]
CUSTOM_AGE_MEANS = [
    1,
    4,
    7,
    10,
    13,
    16,
    19,
    22,
    25,
    28,
    31,
    34,
    37,
    40,
    43,
    46,
    49,
    52.5,
    57,
    62,
    67,
    74.5,
    80,
]

# Load models
original_net = cv2.dnn.readNet(ORIGINAL_MODEL, ORIGINAL_PROTO)
custom_net = cv2.dnn.readNet(CUSTOM_MODEL, CUSTOM_PROTO)

# Detect ages
print(f"Detecting ages on dataset: {DATASET_PATH}")

ages = {}

for subdir in tqdm(list(DATASET_PATH.iterdir())):
    if subdir.is_dir():
        age = int(str(subdir).split("/")[-1])
        original_count = 0
        original_diff = 0
        custom_count = 0
        custom_diff = 0
        total_count = 0
        for image_path in subdir.iterdir():
            if image_path.is_file():
                total_count += 1
                image: np.ndarray = cv2.imread(str(image_path))
                blob_original = cv2.dnn.blobFromImage(
                    image,
                    1,
                    (227, 227),
                    (78.4263377603, 87.7689143744, 114.895847746),
                    swapRB=False
                )
                blob_custom = cv2.dnn.blobFromImage(
                    image,
                    1,
                    (227, 227),
                    (0, 0, 0),
                    swapRB=True,
                ) / 255
                original_net.setInput(blob_original)
                original_preds = original_net.forward()
                original_argmax = np.argmax(original_preds[0])
                original_age_range = ORIGINAL_AGE_RANGES[original_argmax]
                original_age_correct = original_age_range[0] <= age <= original_age_range[1]
                original_age_diff = abs(
                    ORIGINAL_AGE_MEANS[original_argmax] - age)
                if original_age_correct:
                    original_count += 1
                original_diff += original_age_diff
                custom_net.setInput(blob_custom)
                custom_preds = custom_net.forward()
                custom_argmax = np.argmax(custom_preds[0])
                custom_age_range = CUSTOM_AGE_RANGES[custom_argmax]
                custom_age_correct = custom_age_range[0] <= age <= custom_age_range[1]
                custom_age_diff = abs(
                    CUSTOM_AGE_MEANS[custom_argmax] - age)
                if custom_age_correct:
                    custom_count += 1
                custom_diff += custom_age_diff
        ages[age] = {
            "original": {
                "accuracy": original_count / total_count,
                "mae": original_diff / total_count,
                "detected": original_count,
            },
            "custom": {
                "accuracy": custom_count / total_count,
                "mae": custom_diff / total_count,
                "detected": custom_count,
            },
            "total_count": total_count,
        }

# Save results to pickle file
print("Saving results to pickle file...")
with open(f'ages.pickle', 'wb') as f:
    pickle.dump(ages, f)

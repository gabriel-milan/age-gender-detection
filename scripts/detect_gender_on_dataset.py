import cv2
import pickle
from pathlib import Path

import numpy as np
from tqdm import tqdm

# Configuration
MODES = ["original", "custom"]
DATASET_PATH = Path("pics/gender_dataset/")
ORIGINAL_MODEL = "original_models/gender_net.caffemodel"
ORIGINAL_PROTO = "original_models/gender_deploy.prototxt"
CUSTOM_MODEL = "custom_models/gender/optimized_graph.pb"
CUSTOM_PROTO = "custom_models/gender/optimized_graph.pbtxt"
GENDERS = ["female", "male"]
THRESHOLD = 0.5

# Load models
original_net = cv2.dnn.readNet(ORIGINAL_MODEL, ORIGINAL_PROTO)
custom_net = cv2.dnn.readNet(CUSTOM_MODEL, CUSTOM_PROTO)

# Detect genders
print(f"Detecting genders on dataset: {DATASET_PATH}")

genders = {}

for subdir in DATASET_PATH.iterdir():
    if subdir.is_dir():
        gender = str(subdir).split("/")[-1]
        original_count = 0
        original_diff = 0
        custom_count = 0
        custom_diff = 0
        total_count = 0
        for image_path in tqdm(list(subdir.iterdir())):
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
                original_preds = original_net.forward()[0][0]
                original_gender = GENDERS[int(original_preds > THRESHOLD)]
                if original_gender == gender:
                    original_count += 1
                custom_net.setInput(blob_custom)
                custom_preds = custom_net.forward()[0][0]
                custom_gender = GENDERS[int(custom_preds > THRESHOLD)]
                if custom_gender == gender:
                    custom_count += 1
        genders[gender] = {
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
with open(f'genders.pickle', 'wb') as f:
    pickle.dump(genders, f)

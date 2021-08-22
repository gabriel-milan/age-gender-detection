import os

import cv2
import numpy as np
from tqdm import tqdm

DATASET_PATH = 'pics/gender_dataset/'


def parse_gender(gender):
    if gender == "male":
        return [0, 1]
    elif gender == "female":
        return [1, 0]
    else:
        raise ValueError(f"Gender {gender} is unknown")


# Initialize X and y
X = []
y = []

# Iterate over directories in DATASET_PATH
for dir_name in tqdm(list(os.listdir(DATASET_PATH))):
    dir_path = os.path.join(DATASET_PATH, dir_name)
    if not os.path.isdir(dir_path):
        continue
    gender = dir_name

    # Iterate over files in dir_path
    for file_name in os.listdir(dir_path):
        file_path = os.path.join(dir_path, file_name)
        if not os.path.isfile(file_path):
            continue

        # Read image
        image = cv2.imread(file_path)
        blob = cv2.dnn.blobFromImage(image, 1.0, (227, 227), swapRB=False)

        # Append image and gender to X and y
        X.append(blob)
        y.append(parse_gender(gender))

# Convert X and y to numpy arrays
X = np.vstack(X).astype(np.float64)
y = np.array(y).astype(np.float64)

# Normalize X
X = X / 255.0

# Save X and y to disk
np.save('X_gender.npy', X)
np.save('y_gender.npy', y)

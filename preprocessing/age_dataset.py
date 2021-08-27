from pathlib import Path
import pandas as pd
import os

import cv2
import numpy as np
from tqdm import tqdm

AGE_RANGES = [
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
DATASET_PATH = 'pics/age_dataset/'


def parse_age(age):
    """
    Parses age into a one-hot vector.
    """
    age_vector = [0] * len(AGE_RANGES)
    for i, (lower, upper) in enumerate(AGE_RANGES):
        if lower <= age <= upper:
            age_vector[i] = 1
            break
    return age_vector


age_cols = [f"age_{i}" for i in range(len(AGE_RANGES))]
cols = ["image_name"] + age_cols
df = pd.DataFrame(columns=cols)
i = 0

# Iterate over directories in DATASET_PATH
for dir_name in tqdm(list(os.listdir(DATASET_PATH))):
    dir_path = os.path.join(DATASET_PATH, dir_name)
    if not os.path.isdir(dir_path):
        continue
    age = parse_age(int(dir_name))

    # Iterate over files in dir_path
    for file_name in os.listdir(dir_path):
        file_path = os.path.join(dir_path, file_name)
        if not os.path.isfile(file_path):
            continue

        # Add to dataframe
        df.loc[i] = [file_path] + age
        i += 1

df["image_name"] = df["image_name"].apply(
    lambda x: "/home/gabriel-milan/data/age/" + x.split(DATASET_PATH)[-1])

# Save dataframe
df.to_csv("age_dataset.csv", index=False)

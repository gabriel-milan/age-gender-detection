import pandas as pd
from pathlib import Path

DATASET_PATH = "pics/gender_dataset/"
MALE_PATH = Path(DATASET_PATH) / "male"
FEMALE_PATH = Path(DATASET_PATH) / "female"

# Generate dataset, collecting filenames for each gender
# and storing them with gender as target
df = pd.DataFrame(columns=["image_name", "gender"])
i = 0
for file in MALE_PATH.iterdir():
    if file.is_file():
        df.loc[i] = [str(file), "male"]
        i += 1

for file in FEMALE_PATH.iterdir():
    if file.is_file():
        df.loc[i] = [str(file), "female"]
        i += 1

# Convert gender to binary classification
df["gender"] = (df["gender"] == "male").astype(int)
df["image_name"] = df["image_name"].apply(
    lambda x: "/home/gabriel-milan/data/gender/" + x.split(DATASET_PATH)[-1])

# Save dataframe
df.to_csv("gender_dataset.csv", index=False)

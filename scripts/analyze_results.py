import pickle

import numpy as np

PICKLE_FILENAME = "faces_cascade.pickle"
# CONFIDENCE_RANGE = [0.01, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1]
CONFIDENCE_RANGE = [0.7]

# Open pickle file
with open(PICKLE_FILENAME, "rb") as f:
    faces_dict = pickle.load(f)

# Print the number of faces for each key
total = 0
detected = 0
for age in faces_dict:
    total_pictures = faces_dict[age]["faces"]
    confidences = np.array(faces_dict[age]["confidences"])
    for confidence_threshold in CONFIDENCE_RANGE:
        detected += confidences[confidences > confidence_threshold].shape[0]
        total += total_pictures
        print(
            f"{age}-year-old faces with confidence > {confidence_threshold} = {confidences[confidences > confidence_threshold].shape[0]}/{total_pictures}")
print(f"Total faces = {total}")
print(f"Detected faces = {detected}")
print("Detection ratio = {:.2f} %".format(detected / total * 100))

import pickle

import numpy as np

PICKLE_FILENAME = "analysis/ages.pickle"

# Open pickle file
with open(PICKLE_FILENAME, "rb") as f:
    d = pickle.load(f)

# Print the number of faces for each key
total = 0
original_detected = 0
original_mae = 0
custom_detected = 0
custom_mae = 0
for age in d:
    total += d[age]["total_count"]
    original_detected += d[age]["original"]["detected"]
    original_mae += d[age]["original"]["mae"] * d[age]["total_count"]
    custom_detected += d[age]["custom"]["detected"]
    custom_mae += d[age]["custom"]["mae"] * d[age]["total_count"]
    print("#" * 80)
    print(f"Age: {age}: {d[age]['total_count']} faces.")
    print(
        f"Original: {d[age]['original']['accuracy'] * 100}% detected, MAE: {d[age]['original']['mae']}")
    print(
        f"Custom: {d[age]['custom']['accuracy'] * 100}% detected, MAE: {d[age]['custom']['mae']}")
print("#" * 80)
print(f"Total faces = {total}")
print("Detection ratio for ORIGINAL = {:.2f} %".format(
    original_detected / total * 100))
print("MAE for ORIGINAL = {:.2f}".format(original_mae / total))
print("Detection ratio for CUSTOM = {:.2f} %".format(
    custom_detected / total * 100))
print("MAE for CUSTOM = {:.2f}".format(custom_mae / total))

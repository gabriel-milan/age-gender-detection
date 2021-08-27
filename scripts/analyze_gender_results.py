import pickle
import numpy as np
import matplotlib.pyplot as plt

PICKLE_FILENAME = "analysis/genders.pickle"

# Open pickle file
with open(PICKLE_FILENAME, "rb") as f:
    d = pickle.load(f)

# Print the number of faces for each key
total = 0
original_detected = 0
custom_detected = 0
X = [0, 1]
y_original = []
y_custom = []
for gender in d:
    total += d[gender]["total_count"]
    original_detected += d[gender]["original"]["detected"]
    custom_detected += d[gender]["custom"]["detected"]
    y_original.append(d[gender]["original"]["accuracy"])
    y_custom.append(d[gender]["custom"]["accuracy"])
    print("#" * 80)
    print(f"Gender: {gender}: {d[gender]['total_count']} faces.")
    print(
        f"Original: {d[gender]['original']['accuracy'] * 100}% detected")
    print(
        f"Custom: {d[gender]['custom']['accuracy'] * 100}% detected")
print("#" * 80)
print(f"Total faces = {total}")
print("Detection ratio for ORIGINAL = {:.2f} %".format(
    original_detected / total * 100))
print("Detection ratio for CUSTOM = {:.2f} %".format(
    custom_detected / total * 100))

X = np.array(X)
ax = plt.subplot(111)
ax.bar(X-0.5, y_original, width=0.5, color='b', align='center')
ax.bar(X, y_custom, width=0.5, color='r', align='center')
plt.legend(["Original", "Custom"])
plt.show()

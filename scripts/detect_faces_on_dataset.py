import cv2
import pickle
from pathlib import Path

import numpy as np
from tqdm import tqdm

# Configuration
MODE = "cascade"  # "cascade" or "caffe"
DATASET_PATH = Path('pics/dataset')
FACE_PROTO = "opencv_face_detector.pbtxt"
FACE_MODEL = "opencv_face_detector_uint8.pb"
FACE_CASCADE_MODEL = "cascade_models/haarcascade_frontalface_default.xml"
CONFIDENCE_THRESHOLD = 0.7

# Load face detection model
print("Loading face detection model...")
if MODE == "caffe":
    face_detection_model = cv2.dnn.readNet(FACE_MODEL, FACE_PROTO)
elif MODE == "cascade":
    face_detection_model = cv2.CascadeClassifier(str(FACE_CASCADE_MODEL))

# Iterate over subdirectories of dataset, split with subdirectory name
print("Detecting faces...")
n_faces = {}
for subdir in tqdm(DATASET_PATH.iterdir()):
    if subdir.is_dir():
        age = int(subdir.name)
        if age not in n_faces:
            n_faces[age] = {
                "faces": 0,
                "confidences": []
            }
        for image_path in subdir.iterdir():
            if image_path.is_file():
                n_faces[age]["faces"] += 1
                # Read image
                image: np.ndarray = cv2.imread(str(image_path))

                # Detect faces
                if MODE == "caffe":
                    blob = cv2.dnn.blobFromImage(image, 1.0, (300, 300), [
                        104, 117, 123], True, False)
                    face_detection_model.setInput(blob)
                    detections: np.ndarray = face_detection_model.forward()
                    greater_confidence = 0
                    for i in range(detections.shape[2]):
                        confidence = detections[0, 0, i, 2]
                        if confidence > greater_confidence:
                            greater_confidence = confidence
                    n_faces[age]["confidences"].append(greater_confidence)
                elif MODE == "cascade":
                    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
                    faces = face_detection_model.detectMultiScale(
                        gray, 1.03, 2)
                    if (len(faces) > 0):
                        n_faces[age]["confidences"].append(1)

# Save results to pickle file
print("Saving results to pickle file...")
with open(f'faces_{MODE}.pickle', 'wb') as f:
    pickle.dump(n_faces, f)

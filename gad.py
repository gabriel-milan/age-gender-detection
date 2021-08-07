from pathlib import Path

import cv2
import typer

# Age
AGE_MODEL_DEFAULT = "original"
AGE_MODEL_HELP = "Path to exported model or \"original\""
AGE_MODEL_ORIGINAL_MODEL = "original_models/age_net.caffemodel"
AGE_MODEL_ORIGINAL_PROTO = "original_models/age_deploy.prototxt"

# Face
FACE_MODES = ["dnn", "cascade"]
FACE_MODES_DEFAULT = "dnn"
FACE_MODES_HELP = "How to detect faces"
FACE_DNN_MODEL = "original_models/opencv_face_detector_uint8.pb"
FACE_DNN_PROTO = "original_models/opencv_face_detector.pbtxt"
FACE_CASCADE_MODEL = "original_models/haarcascade_frontalface_default.xml"
FACE_CASCADE_SCALE = 1.3
FACE_CASCADE_NEIGHBORS = 5

# Gender
GENDER_MODES = ["original"]
GENDER_MODES_DEFAULT = "original"
GENDER_MODES_HELP = "How to detect genders"
GENDER_MODEL_ORIGINAL_MODEL = "original_models/gender_net.caffemodel"
GENDER_MODEL_ORIGINAL_PROTO = "original_models/gender_deploy.prototxt"
GENDER_LIST = ["Male", "Female"]

# General settings
BOX_PADDING = 20
CONFIDENCE_THRESHOLD = 0.7
IMAGE_NOT_FOUND = "pics/not-found.png"

###############
#
# Typer stuff
#
###############


def validate_age_model(age_model: str) -> str:
    # Check if age_model path exists
    if not Path(age_model).exists() and age_model != "original":
        raise typer.BadParameter(f"Age model not found: {age_model}")
    return age_model


def validate_face_mode(face_mode: str) -> str:
    if face_mode not in FACE_MODES:
        raise typer.BadParameter(f"Invalid face mode: {face_mode}")
    return face_mode


def validate_gender_mode(gender_mode: str) -> str:
    if gender_mode not in GENDER_MODES:
        raise typer.BadParameter(f"Invalid gender mode: {gender_mode}")
    return gender_mode


###############
#
# OpenCV stuff
#
###############

def get_models(age_model: str, face_mode: str, gender_mode: str) -> tuple:
    """Loads models for every classifier"""

    # Age model
    if age_model == "original":
        age_net = cv2.dnn.readNet(
            AGE_MODEL_ORIGINAL_MODEL, AGE_MODEL_ORIGINAL_PROTO)
    else:
        age_model_path = Path(age_model)
        # List files inside the model directory
        age_model_files = list(age_model_path.iterdir())
        # Check if the model directory contains a model file
        if len(age_model_files) == 0:
            raise typer.BadParameter(f"Age model not found: {age_model}")
        # Get pb and pbtxt files
        pb_file: str = [f for f in age_model_files if (
            f.suffix == ".pb" or f.suffix == ".caffemodel")][0]
        pbtxt_file: str = [f for f in age_model_files if (
            f.suffix == ".pbtxt" or f.suffix == ".prototxt")][0]
        # Load model
        age_net = cv2.dnn.readNet(str(pbtxt_file), str(pb_file))

    # Face model
    if face_mode == "dnn":
        face_net = cv2.dnn.readNet(FACE_DNN_MODEL, FACE_DNN_PROTO)
    elif face_mode == "cascade":
        face_net = cv2.CascadeClassifier(FACE_CASCADE_MODEL)

    # Gender model
    if gender_mode == "original":
        gender_net = cv2.dnn.readNet(
            GENDER_MODEL_ORIGINAL_MODEL, GENDER_MODEL_ORIGINAL_PROTO)

    return age_net, face_net, gender_net


def get_age_mean_values(age_model: str):
    """Loads mean values for age model"""
    if age_model == "original":
        return (78.4263377603, 87.7689143744, 114.895847746)
    else:
        return (0, 0, 0)


def get_face_mean_values(face_mode: str):
    """Loads mean values for face model"""
    if face_mode == "dnn":
        return [104, 117, 123]
    elif face_mode == "cascade":
        return (0, 0, 0)


def get_gender_mean_values(gender_mode: str):
    """Loads mean values for gender model"""
    if gender_mode == "original":
        return (78.4263377603, 87.7689143744, 114.895847746)
    else:
        return (0, 0, 0)


def get_age_scale_values(age_model: str):
    """Loads age scale for age model"""
    if age_model == "original":
        return ['(0-2)', '(4-6)', '(8-12)', '(15-20)',
                '(25-32)', '(38-43)', '(48-53)', '(60+)']
    else:
        return ['(0-3)', '(4-7)', '(8-13)', '(14-22)',
                '(23-35)', '(36-45)', '(46-60)', '(60+)']


def highlight_faces(net, frame, face_mode: str, conf_threshold: float = CONFIDENCE_THRESHOLD):
    """Detects faces in an image and highlight them"""
    frameOpencvDnn = frame.copy()
    faceBoxes = []
    frameHeight = frameOpencvDnn.shape[0]
    frameWidth = frameOpencvDnn.shape[1]
    if face_mode == "dnn":
        blob = cv2.dnn.blobFromImage(
            frameOpencvDnn, 1.0, (300, 300), get_face_mean_values(face_mode), True, False)

        net.setInput(blob)
        detections = net.forward()
        for i in range(detections.shape[2]):
            confidence = detections[0, 0, i, 2]
            if confidence > conf_threshold:
                x1 = int(detections[0, 0, i, 3]*frameWidth)
                y1 = int(detections[0, 0, i, 4]*frameHeight)
                x2 = int(detections[0, 0, i, 5]*frameWidth)
                y2 = int(detections[0, 0, i, 6]*frameHeight)
                faceBoxes.append([x1, y1, x2, y2])
                cv2.rectangle(frameOpencvDnn, (x1, y1), (x2, y2),
                              (0, 255, 0), int(round(frameHeight/150)), 8)
    elif face_mode == "cascade":
        gray = cv2.cvtColor(frameOpencvDnn, cv2.COLOR_BGR2GRAY)
        faces = net.detectMultiScale(
            gray, FACE_CASCADE_SCALE, FACE_CASCADE_NEIGHBORS)
        for (x, y, w, h) in faces:
            faceBoxes.append([x, y, x+w, y+h])
            cv2.rectangle(frameOpencvDnn, (x, y), (x+w, y+h),
                          (0, 255, 0), int(round(frameHeight/150)), 8)
    return frameOpencvDnn, faceBoxes


def run_detection(image_source, age_model: str, face_mode: str, gender_mode: str):
    """Runs detection on image source"""

    # Load models
    age_net, face_net, gender_net = get_models(
        age_model, face_mode, gender_mode)

    # Loop waiting for key
    while cv2.waitKey(1) < 0:

        # Read from image or video source
        hasFrame, frame = image_source.read()
        if not hasFrame:
            cv2.waitKey()
            break

        # Detect faces
        resultImg, faceBoxes = highlight_faces(face_net, frame, face_mode)
        if not faceBoxes:
            cv2.putText(resultImg, "No face detected",
                        (150, 150), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            cv2.imshow("Detecting age and gender", resultImg)
            typer.echo("No face detected!")

        # Detect age and gender
        for faceBox in faceBoxes:
            face = frame[max(0, faceBox[1]-BOX_PADDING):min(faceBox[3]+BOX_PADDING, frame.shape[0]-1),
                         max(0, faceBox[0]-BOX_PADDING):min(faceBox[2]+BOX_PADDING, frame.shape[1]-1)]

            blob = cv2.dnn.blobFromImage(
                face, 1.0, (227, 227), get_gender_mean_values(gender_mode), swapRB=False)
            gender_net.setInput(blob)
            gender_preds = gender_net.forward()
            gender = GENDER_LIST[gender_preds[0].argmax()]
            typer.echo(f'Gender: {gender}')

            blob = cv2.dnn.blobFromImage(
                face, 1.0, (227, 227), get_age_mean_values(age_model), swapRB=False)
            if age_model != "original":
                blob /= 255
            age_net.setInput(blob)
            age_preds = age_net.forward()
            age_list = get_age_scale_values(age_model)
            age = age_list[age_preds[0].argmax()]
            confidence = age_preds.max()
            typer.echo(f'Age: {age[1:-1]} years')
            typer.echo("Confidence: {:.4f}%".format(confidence * 100))

            cv2.putText(resultImg, f'{gender}, {age} {str(confidence * 100)[:2]}%', (
                faceBox[0], faceBox[1]-10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2, cv2.LINE_AA)
            cv2.imshow("Detecting age and gender", resultImg)


###############
#
# Main CLI function
#
###############


def main(
    age_model: str = typer.Option(
        AGE_MODEL_DEFAULT, help=AGE_MODEL_HELP, callback=validate_age_model),
    face_mode: str = typer.Option(
        FACE_MODES_DEFAULT, help=FACE_MODES_HELP, callback=validate_face_mode),
    gender_mode: str = typer.Option(
        GENDER_MODES_DEFAULT, help=GENDER_MODES_HELP, callback=validate_gender_mode),
    image: str = typer.Option(
        None, help="If None, will use video camera. If set, will run detection on the image"),
    image_dir: str = typer.Option(
        None, help="Directory with images only. If set, will run detection on all of them."),
):
    if image_dir is not None:
        image_files = Path(image_dir).glob("**/*")
        for image in image_files:
            run_detection(cv2.VideoCapture(str(image)),
                          age_model, face_mode, gender_mode)
    else:
        video = cv2.VideoCapture(image if image else 0)
        run_detection(video, age_model, face_mode, gender_mode)


if __name__ == "__main__":
    typer.run(main)

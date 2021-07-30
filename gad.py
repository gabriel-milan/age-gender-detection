import cv2
import argparse

MODES = ["original", "custom"]
DEFAULT_MODE = "original"


def highlightFace(net, frame, conf_threshold=0.7):
    frameOpencvDnn = frame.copy()
    frameHeight = frameOpencvDnn.shape[0]
    frameWidth = frameOpencvDnn.shape[1]
    blob = cv2.dnn.blobFromImage(frameOpencvDnn, 1.0, (300, 300), [
                                 104, 117, 123], True, False)

    net.setInput(blob)
    detections = net.forward()
    faceBoxes = []
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
    return frameOpencvDnn, faceBoxes


parser = argparse.ArgumentParser()
parser.add_argument('--image')
parser.add_argument('--mode')
parser.add_argument('--model_id')

args = parser.parse_args()

mode = DEFAULT_MODE if args.mode is None else args.mode if args.mode in MODES else DEFAULT_MODE
model_id = 1 if args.model_id is None else args.model_id

faceProto = "opencv_face_detector.pbtxt"
faceModel = "opencv_face_detector_uint8.pb"
if mode == "original":
    ageProto = "age_deploy.prototxt"
    ageModel = "age_net.caffemodel"
    ageList = ['(0-2)', '(4-6)', '(8-12)', '(15-20)',
               '(25-32)', '(38-43)', '(48-53)', '(60-100)']
else:
    ageProto = f"custom_models/{model_id}/optimized_graph.pbtxt"
    ageModel = f"custom_models/{model_id}/optimized_graph.pb"
    ageList = ['(0-3)', '(4-7)', '(8-13)', '(14-22)',
               '(23-35)', '(36-45)', '(46-60)', '(60-100)']
genderProto = "gender_deploy.prototxt"
genderModel = "gender_net.caffemodel"

MODEL_MEAN_VALUES = (78.4263377603, 87.7689143744, 114.895847746)
genderList = ['Male', 'Female']

faceNet = cv2.dnn.readNet(faceModel, faceProto)
ageNet = cv2.dnn.readNet(ageModel, ageProto)
genderNet = cv2.dnn.readNet(genderModel, genderProto)

video = cv2.VideoCapture(args.image if args.image else 0)
padding = 20
while cv2.waitKey(1) < 0:
    hasFrame, frame = video.read()
    if not hasFrame:
        cv2.waitKey()
        break

    resultImg, faceBoxes = highlightFace(faceNet, frame)
    if not faceBoxes:
        print("No face detected")

    for faceBox in faceBoxes:
        face = frame[max(0, faceBox[1]-padding):
                     min(faceBox[3]+padding, frame.shape[0]-1), max(0, faceBox[0]-padding):min(faceBox[2]+padding, frame.shape[1]-1)]

        blob = cv2.dnn.blobFromImage(
            face, 1.0, (227, 227), MODEL_MEAN_VALUES, swapRB=False)
        genderNet.setInput(blob)
        genderPreds = genderNet.forward()
        gender = genderList[genderPreds[0].argmax()]
        print(f'Gender: {gender}')

        if mode == "custom":
            blobAge = cv2.dnn.blobFromImage(
                face, 1.0, (227, 227), swapRB=False)
            blobAge /= 255
            ageNet.setInput(blobAge)
        else:
            ageNet.setInput(blob)
        agePreds = ageNet.forward()
        argmax_index = agePreds[0].argmax()
        age = ageList[agePreds[0].argmax()]
        confidence = agePreds.max()
        print(f'Age: {age[1:-1]} years')
        print("Confidence: {:.4f}%".format(confidence * 100))

        cv2.putText(resultImg, f'{gender}, {age} {str(confidence * 100)[:2]}%', (
            faceBox[0], faceBox[1]-10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2, cv2.LINE_AA)
        cv2.imshow("Detecting age and gender", resultImg)

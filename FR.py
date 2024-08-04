import cv2
import os
import numpy as np

# Load the face detector
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Load training images
def load_training_images(training_dir):
    face_samples = []
    ids = []

    for person_name in os.listdir(training_dir):
        person_dir = os.path.join(training_dir, person_name)
        if os.path.isdir(person_dir):
            for image_name in os.listdir(person_dir):
                image_path = os.path.join(person_dir, image_name)
                gray_image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
                faces = face_cascade.detectMultiScale(gray_image, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
                for (x, y, w, h) in faces:
                    face_samples.append(gray_image[y:y + h, x:x + w])
                    ids.append(person_name)
    return face_samples, ids

# Train the recognizer
def train_recognizer(face_samples, ids):
    recognizer = cv2.face.LBPHFaceRecognizer_create()
    label_ids = {name: idx for idx, name in enumerate(set(ids))}
    labels = [label_ids[name] for name in ids]
    recognizer.train(face_samples, np.array(labels))
    return recognizer, label_ids

# Recognize faces in an image
def recognize_faces(frame, recognizer, label_ids, threshold=100):
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray_frame, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

    for (x, y, w, h) in faces:
        face = gray_frame[y:y + h, x:x + w]
        label, confidence = recognizer.predict(face)
        if confidence < threshold:
            name = [k for k, v in label_ids.items() if v == label][0]
        else:
            name = "Unrecognized"

        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
        cv2.putText(frame, name, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

    return frame

# Open the webcam and perform face recognition
def open_webcam(recognizer, label_ids):
    cap = cv2.VideoCapture(0)
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame = recognize_faces(frame, recognizer, label_ids)

        cv2.imshow("Webcam Face Recognition", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    training_dir = "training_images"
    face_samples, ids = load_training_images(training_dir)
    recognizer, label_ids = train_recognizer(face_samples, ids)
    open_webcam(recognizer, label_ids)

# Face-Recognition
Face detection and recognition using CascadeClassifier

# Face Recognition System with OpenCV

This project demonstrates how to build a face recognition system using OpenCV in Python. The system detects faces, trains a recognizer on a set of training images, and then recognizes faces in real-time from a webcam feed.

## Requirements

- Python 3.x
- OpenCV
- NumPy

Install the required libraries using pip:

```bash
pip install opencv-python-headless numpy
```

## Project Structure

```
.
├── face_recognition.py
└── training_images/
    ├── person1/
    │   ├── image1.jpg
    │   ├── image2.jpg
    │   └── ...
    ├── person2/
    │   ├── image1.jpg
    │   ├── image2.jpg
    │   └── ...
    └── ...
```

- `README.md`: This file, providing information about the project.
- `face_recognition.py`: The main Python script containing the face recognition logic.
- `training_images/`: Directory containing subdirectories for each person, with their respective training images.

## Usage

1. **Prepare Training Images:**
   - Create a directory `training_images`.
   - For each person, create a subdirectory named after them.
   - Add their face images (in `.jpg` format) to the respective subdirectories.

2. **Run the Script:**

   ```bash
   python face_recognition.py
   ```

## Code Explanation

### Importing Libraries

```python
import cv2
import os
import numpy as np
```

### Load the Face Detector

Load the pre-trained Haar Cascade face detector from OpenCV.

```python
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
```

### Load Training Images

Function to load and preprocess training images from the specified directory.

```python
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
```

### Train the Recognizer

Function to train the LBPH face recognizer on the loaded face samples and labels.

```python
def train_recognizer(face_samples, ids):
    recognizer = cv2.face.LBPHFaceRecognizer_create()
    label_ids = {name: idx for idx, name in enumerate(set(ids))}
    labels = [label_ids[name] for name in ids]
    recognizer.train(face_samples, np.array(labels))
    return recognizer, label_ids
```

### Recognize Faces in an Image

Function to recognize faces in a given frame using the trained recognizer.

```python
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
```

### Open the Webcam and Perform Face Recognition

Function to capture video from the webcam and perform face recognition in real-time.

```python
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
```

### Main Function

Load training images, train the recognizer, and start the webcam for face recognition.

```python
if __name__ == "__main__":
    training_dir = "training_images"
    face_samples, ids = load_training_images(training_dir)
    recognizer, label_ids = train_recognizer(face_samples, ids)
    open_webcam(recognizer, label_ids)
```

## Notes

- Make sure the training images are clear and contain only the face of the person.
- Adjust the `scaleFactor` and `minNeighbors` parameters in `detectMultiScale` if needed to improve face detection accuracy.
- Modify the `threshold` parameter in `recognize_faces` to control the recognition confidence level.

## Troubleshooting

- If the recognizer fails to recognize faces, ensure that the training images are correctly labeled and that the faces are properly detected during training.
- Verify that the webcam is working correctly and accessible by OpenCV.

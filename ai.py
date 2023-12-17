import cv2
import dlib

def detect_faces(image_path):
    # Load the image
    image = cv2.imread(image_path)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Load the pre-trained Haar cascade for face detection
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

    # Detect faces in the image
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5)

    # Draw rectangles around the faces
    for (x, y, w, h) in faces:
        cv2.rectangle(image, (x, y), (x+w, y+h), (255, 0, 0), 2)

    # Display the image with faces
    cv2.imshow('Detected Faces', image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def detect_landmarks(image_path):
    # Load the image
    image = cv2.imread(image_path)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Load the pre-trained shape predictor for facial landmarks
    predictor_path = 'shape_predictor_68_face_landmarks.dat'  # You can download this file from dlib's website
    predictor = dlib.shape_predictor(predictor_path)

    # Load the pre-trained face detector
    detector = dlib.get_frontal_face_detector()

    # Detect faces in the image
    faces = detector(gray)

    # Draw facial landmarks
    for face in faces:
        landmarks = predictor(gray, face)
        for n in range(68):
            x = landmarks.part(n).x
            y = landmarks.part(n).y
            cv2.circle(image, (x, y), 1, (0, 0, 255), -1)

    # Display the image with facial landmarks
    cv2.imshow('Facial Landmarks', image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

# Example usage:
image_path = 'path/to/your/image.jpg'
detect_faces(image_path)
detect_landmarks(image_path)

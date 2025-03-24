import cv2
import numpy as np
from keras.models import load_model

# Load the pre-trained age prediction model
model = load_model('age_prediction_model.keras')

# Load the Haar cascade for face detection
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

def predict_age(image_path):
    # Read the image
    img = cv2.imread(image_path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Detect faces
    faces = face_cascade.detectMultiScale(gray, 1.1, 4)

    for (x, y, w, h) in faces:
        # Extract the face region
        face = img[y:y+h, x:x+w]
        face = cv2.resize(face, (224, 224))  # Resize to the input size of the model
        face = face / 255.0  # Normalize the image
        face = np.expand_dims(face, axis=0)  # Add batch dimension

        # Predict age
        predicted_age = model.predict(face)
        print(f"Predicted Age: {predicted_age[0][0]}")

        # Draw rectangle around the face and put predicted age
        cv2.rectangle(img, (x, y), (x+w, y+h), (255, 0, 0), 2)
        cv2.putText(img, f'Age: {int(predicted_age[0][0])}', (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (36,255,12), 2)

    # Display the output
    cv2.imshow('Age Prediction', img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

# Example usage
if __name__ == "__main__":
    predict_age('photo.jpg')
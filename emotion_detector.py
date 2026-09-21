import cv2
import numpy as np
from tensorflow.keras.models import load_model

# =========================
# Load Face Detector
# =========================
face_cascade = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")

# =========================
# Load Emotion Model
# =========================
emotion_model = load_model("emotion_model.h5")

# =========================
# Emotion Labels
# =========================
emotion_labels = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']

# =========================
# Start Webcam
# =========================
cap = cv2.VideoCapture(0)

print("🎥 Camera started...")
print("🧠 Emotion AI running... Press 'q' to quit")

while True:
    ret, frame = cap.read()
    if not ret:
        print("❌ Camera not accessible")
        break

    # Convert to gray
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # =========================
    # Face Detection
    # =========================
    faces = face_cascade.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(60, 60)
    )

    # Debug
    # print("Faces detected:", len(faces))

    for (x, y, w, h) in faces:

        # Draw face box
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)

        # =========================
        # Face Preprocessing
        # =========================
        face = gray[y:y+h, x:x+w]
        face = cv2.resize(face, (48, 48))
        face = face / 255.0
        face = np.reshape(face, (1, 48, 48, 1))

        # =========================
        # Emotion Prediction
        # =========================
        preds = emotion_model.predict(face, verbose=0)
        emotion_index = np.argmax(preds)
        emotion = emotion_labels[emotion_index]
        confidence = float(np.max(preds))

        label = f"{emotion} ({confidence:.2f})"

        # =========================
        # Label UI
        # =========================
        # Background box
        cv2.rectangle(frame, (x, y-30), (x+w, y), (0, 0, 0), -1)

        # Text
        cv2.putText(frame, label, (x+5, y-8),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7, (0, 255, 255), 2)

    # =========================
    # Show Window
    # =========================
    cv2.imshow("🧠 Emotion Detection AI | Ayushman Chourasiya Project", frame)

    # Exit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        print("🛑 Exiting...")
        break

# =========================
# Cleanup
# =========================
cap.release()
cv2.destroyAllWindows()
print("✅ Camera closed | System shutdown")

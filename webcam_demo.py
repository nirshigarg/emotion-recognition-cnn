import cv2
import numpy as np
from tensorflow.keras.models import Model
from tensorflow.keras.layers import (
    Input,
    Conv2D,
    BatchNormalization,
    Activation,
    MaxPooling2D,
    Dropout,
    Flatten,
    Dense
)

# ==========================
# Build Model Architecture
# ==========================
def build_emotion_model(input_shape=(48, 48, 1), num_classes=7):
    inputs = Input(shape=input_shape)

    x = Conv2D(32, (3, 3), padding='same')(inputs)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = MaxPooling2D(2, 2)(x)

    x = Conv2D(64, (3, 3), padding='same')(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = MaxPooling2D(2, 2)(x)

    x = Conv2D(128, (3, 3), padding='same')(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = MaxPooling2D(2, 2)(x)
    x = Dropout(0.25)(x)

    x = Conv2D(256, (3, 3), padding='same', name='last_conv')(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = MaxPooling2D(2, 2)(x)
    x = Dropout(0.25)(x)

    x = Flatten()(x)

    x = Dense(256)(x)
    x = Activation('relu')(x)
    x = Dropout(0.4)(x)

    outputs = Dense(num_classes, activation='softmax')(x)

    return Model(inputs, outputs)


# ==========================
# Load Model Weights
# ==========================
model = build_emotion_model()
model.load_weights("emotion_weights.weights.h5")

print("✅ Model loaded successfully")


# ==========================
# Emotion Labels
# ==========================
class_names = [
    'angry',
    'disgust',
    'fear',
    'happy',
    'neutral',
    'sad',
    'surprise'
]


# ==========================
# Emotion Colors (BGR)
# ==========================
emotion_colors = {
    'angry':    (60, 60, 220),
    'disgust':  (60, 150, 90),
    'fear':     (180, 120, 60),
    'happy':    (60, 200, 230),
    'neutral':  (180, 180, 180),
    'sad':      (200, 130, 60),
    'surprise': (60, 200, 160)
}


# ==========================
# Face Detector
# ==========================
face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades +
    "haarcascade_frontalface_default.xml"
)


# ==========================
# Start Webcam
# ==========================
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("❌ Error: Could not open webcam")
    exit()

print("📷 Webcam started. Press 'q' to quit.")


# ==========================
# Main Loop
# ==========================
while True:

    ret, frame = cap.read()

    if not ret:
        break

    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    faces = face_cascade.detectMultiScale(
        gray_frame,
        scaleFactor=1.3,
        minNeighbors=5
    )

    for (x, y, w, h) in faces:

        # Face crop
        face_roi = gray_frame[y:y+h, x:x+w]

        # Resize
        face_resized = cv2.resize(face_roi, (48, 48))

        # Normalize
        face_normalized = face_resized.astype("float32") / 255.0

        # Shape -> (1,48,48,1)
        face_input = np.expand_dims(
            face_normalized,
            axis=(0, -1)
        )

        # Prediction
        prediction = model.predict(
            face_input,
            verbose=0
        )

        emotion_idx = np.argmax(prediction)
        confidence = prediction[0][emotion_idx]

        emotion_label = class_names[emotion_idx]

        # Confidence threshold
        if confidence < 0.50:
            emotion_label = "Detecting..."
            box_color = (120, 120, 120)
        else:
            box_color = emotion_colors.get(
                emotion_label,
                (0, 255, 0)
            )

        # Face Box
        cv2.rectangle(
            frame,
            (x, y),
            (x + w, y + h),
            box_color,
            2
        )

        # Label Text
        label_text = f"{emotion_label.capitalize()}  {confidence*100:.0f}%"

        (text_w, text_h), _ = cv2.getTextSize(
            label_text,
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            2
        )

        label_bg_top = max(
            y - text_h - 18,
            0
        )

        # Label Background
        cv2.rectangle(
            frame,
            (x, label_bg_top),
            (x + text_w + 16, y),
            box_color,
            -1
        )

        # Label Text
        cv2.putText(
            frame,
            label_text,
            (x + 8, y - 8),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (255, 255, 255),
            2,
            cv2.LINE_AA
        )

    # Slight polish effect
    overlay = frame.copy()

    cv2.addWeighted(
        overlay,
        0.95,
        np.zeros_like(frame),
        0.05,
        0,
        frame
    )

    cv2.imshow(
        "Emotion Recognition",
        frame
    )

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break


# ==========================
# Cleanup
# ==========================
cap.release()
cv2.destroyAllWindows()
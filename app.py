import streamlit as st
import cv2
import numpy as np
from PIL import Image
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Conv2D, BatchNormalization, Activation, MaxPooling2D, Dropout, Flatten, Dense

# ---- Page config ----
st.set_page_config(page_title="Emotion Recognition", page_icon="🙂", layout="centered")

# ---- Rebuild architecture and load weights (cached so it loads once) ----
@st.cache_resource
def load_emotion_model():
    inputs = Input(shape=(48, 48, 1))
    x = Conv2D(32, (3,3), padding='same')(inputs)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = MaxPooling2D(2,2)(x)

    x = Conv2D(64, (3,3), padding='same')(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = MaxPooling2D(2,2)(x)

    x = Conv2D(128, (3,3), padding='same')(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = MaxPooling2D(2,2)(x)
    x = Dropout(0.25)(x)

    x = Conv2D(256, (3,3), padding='same', name='last_conv')(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = MaxPooling2D(2,2)(x)
    x = Dropout(0.25)(x)

    x = Flatten()(x)
    x = Dense(256)(x)
    x = Activation('relu')(x)
    x = Dropout(0.4)(x)
    outputs = Dense(7, activation='softmax')(x)

    model = Model(inputs, outputs)
    model.load_weights('emotion_weights.weights.h5')
    return model

model = load_emotion_model()

class_names = ['angry', 'disgust', 'fear', 'happy', 'neutral', 'sad', 'surprise']

emotion_colors = {
    'angry': '#DC3C3C', 'disgust': '#5A966A', 'fear': '#7878B4',
    'happy': '#E6C83C', 'neutral': '#B4B4B4', 'sad': '#3C82C8',
    'surprise': '#3CA0A0'
}

face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# ---- Core prediction function ----
def predict_emotion(image_bgr):
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5)

    results = []
    for (x, y, w, h) in faces:
        face_roi = gray[y:y+h, x:x+w]
        face_resized = cv2.resize(face_roi, (48, 48))
        face_normalized = face_resized.astype('float32') / 255.0
        face_input = np.expand_dims(face_normalized, axis=(0, -1))

        prediction = model.predict(face_input, verbose=0)
        emotion_idx = np.argmax(prediction)
        emotion_label = class_names[emotion_idx]
        confidence = float(prediction[0][emotion_idx])

        results.append((x, y, w, h, emotion_label, confidence))

    return results

def draw_results(image_bgr, results):
    for (x, y, w, h, label, confidence) in results:
        color_hex = emotion_colors[label]
        color_bgr = tuple(int(color_hex[i:i+2], 16) for i in (5, 3, 1))
        cv2.rectangle(image_bgr, (x, y), (x+w, y+h), color_bgr, 2)
        text = f"{label.capitalize()} {confidence*100:.0f}%"
        cv2.putText(image_bgr, text, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX,
                    0.7, color_bgr, 2, cv2.LINE_AA)
    return image_bgr

# ---- UI ----
st.title("🙂 Facial Emotion Recognition")
st.write("A CNN trained on FER-2013 to recognize 7 emotions: angry, disgust, fear, happy, sad, surprise, neutral.")

mode = st.radio("Choose input method:", ["Upload a photo", "Use camera"])

image_bgr = None

if mode == "Upload a photo":
    uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])
    if uploaded_file is not None:
        image = Image.open(uploaded_file).convert("RGB")
        image_bgr = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)

else:
    camera_photo = st.camera_input("Take a photo")
    if camera_photo is not None:
        image = Image.open(camera_photo).convert("RGB")
        image_bgr = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)

if image_bgr is not None:
    results = predict_emotion(image_bgr)

    if len(results) == 0:
        st.warning("No face detected. Try a clearer, front-facing photo.")
    else:
        annotated = draw_results(image_bgr.copy(), results)
        annotated_rgb = cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB)
        st.image(annotated_rgb, caption="Prediction result", use_container_width=True)

        for (_, _, _, _, label, confidence) in results:
            st.write(f"**{label.capitalize()}** — {confidence*100:.1f}% confidence")

st.markdown("---")
st.caption("Model: Custom CNN (4 conv blocks, ~980K params) | Dataset: FER-2013 | Test accuracy: ~60.8%")
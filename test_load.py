from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Conv2D, BatchNormalization, Activation, MaxPooling2D, Dropout, Flatten, Dense

def build_emotion_model(input_shape=(48, 48, 1), num_classes=7):
    inputs = Input(shape=input_shape)

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
    outputs = Dense(num_classes, activation='softmax')(x)

    return Model(inputs, outputs)

model = build_emotion_model()
model.load_weights('emotion_weights.weights.h5')
print("Model rebuilt and weights loaded successfully")
model.summary()
import numpy as np

dummy_input = np.random.rand(1, 48, 48, 1).astype('float32')
prediction = model.predict(dummy_input, verbose=0)
print("Prediction probabilities:", prediction)
print("Sum of probabilities (should be ~1.0):", prediction.sum())
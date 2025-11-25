import os
import numpy as np
import tensorflow as tf
import librosa
import soundfile as sf

DATA_DIR = "data"
SAMPLE_RATE = 16000
DURATION = 1.0  # each clip ~1 second
SAMPLES = int(SAMPLE_RATE * DURATION)

N_MFCC = 13
FRAME_LENGTH = 0.025
FRAME_HOP = 0.010

BATCH_SIZE = 32
EPOCHS = 20


def load_audio(path):
    audio, sr = sf.read(path)
    if len(audio.shape) > 1:
        audio = audio[:, 0]  # convert stereo → mono

    audio = librosa.resample(audio, orig_sr=sr, target_sr=SAMPLE_RATE)

    if len(audio) > SAMPLES:
        audio = audio[:SAMPLES]
    else:
        audio = np.pad(audio, (0, SAMPLES - len(audio)))

    return audio.astype(np.float32)


def extract_mfcc(audio):
    mfcc = librosa.feature.mfcc(
        y=audio,
        sr=SAMPLE_RATE,
        n_mfcc=N_MFCC,
        n_fft=int(SAMPLE_RATE * FRAME_LENGTH),
        hop_length=int(SAMPLE_RATE * FRAME_HOP),
        htk=True
    )
    mfcc = mfcc.T  # (time, coeffs)
    return mfcc


classes = ["silence", "random", "hello"]
X = []
Y = []

for label_idx, label in enumerate(classes):
    folder = os.path.join(DATA_DIR, label)
    if not os.path.isdir(folder):
        print(f"Warning: {folder} missing")
        continue

    for file in os.listdir(folder):
        if file.endswith(".wav"):
            path = os.path.join(folder, file)
            audio = load_audio(path)
            mfcc = extract_mfcc(audio)
            X.append(mfcc)
            Y.append(label_idx)

X = np.array(X)
Y = np.array(Y)

# reshape for CNN (samples, time, features, channels)
X = X[..., np.newaxis]

print("Dataset loaded:", X.shape, Y.shape)

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
    X, Y, test_size=max(0.15, 3 / len(X)), random_state=42, stratify=Y
)


def build_model(input_shape, num_classes):
    model = tf.keras.Sequential([
        tf.keras.layers.Conv2D(16, (3, 3), activation='relu', padding='same',
                               input_shape=input_shape),
        tf.keras.layers.MaxPooling2D((2, 2)),
        tf.keras.layers.Conv2D(32, (3, 3), activation='relu', padding='same'),
        tf.keras.layers.MaxPooling2D((2, 2)),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(64, activation='relu'),
        tf.keras.layers.Dropout(0.3),
        tf.keras.layers.Dense(num_classes, activation='softmax')
    ])
    return model


model = build_model(X_train.shape[1:], len(classes))
model.compile(
    optimizer="adam",
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)

model.summary()

history = model.fit(
    X_train, y_train,
    batch_size=BATCH_SIZE,
    epochs=EPOCHS,
    validation_data=(X_test, y_test)
)

model.save("hello_model.h5")
print("Saved: hello_model.h5")


def representative_dataset():
    for i in range(100):
        idx = np.random.randint(0, X_train.shape[0])
        data = X_train[idx:idx + 1]
        yield [data.astype(np.float32)]


converter = tf.lite.TFLiteConverter.from_keras_model(model)
converter.optimizations = [tf.lite.Optimize.DEFAULT]
converter.representative_dataset = representative_dataset
converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
converter.inference_input_type = tf.int8
converter.inference_output_type = tf.int8

tflite_model = converter.convert()

with open("hello_model_int8.tflite", "wb") as f:
    f.write(tflite_model)

print("Saved: hello_model_int8.tflite")

import pyaudio
import wave
import cv2
import os
import pickle
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.cluster import KMeans
import time
import librosa
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.applications import VGG16
from tensorflow.keras.applications.vgg16 import preprocess_input

def extract_audio_features(audio_file):
    y, sr = librosa.load(audio_file, sr=None)
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=20)
    mfcc_mean = np.mean(mfcc, axis=1)
    mfcc_std = np.std(mfcc, axis=1)
    return np.concatenate((mfcc_mean, mfcc_std))

def extract_facial_features(face_image):
    base_model = VGG16(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
    model = Model(inputs=base_model.input, outputs=base_model.get_layer('block5_pool').output)
    img = cv2.resize(face_image, (224, 224))
    img = np.expand_dims(img, axis=0)
    img = preprocess_input(img)
    features = model.predict(img)
    return features.flatten()

def build_fingerprint_model(input_shape):
    model = Sequential()
    model.add(Conv2D(32, (3, 3), activation='relu', input_shape=input_shape))
    model.add(MaxPooling2D((2, 2)))
    model.add(Conv2D(64, (3, 3), activation='relu'))
    model.add(MaxPooling2D((2, 2)))
    model.add(Conv2D(128, (3, 3), activation='relu'))
    model.add(MaxPooling2D((2, 2)))
    model.add(Flatten())
    model.add(Dense(128, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(10, activation='softmax'))
    return model

def extract_fingerprint_features(fingerprint_image_path):
    img = cv2.imread(fingerprint_image_path)
    if img is None:
        raise FileNotFoundError(f"Failed to load image at path: {fingerprint_image_path}")
    img = cv2.resize(img, (128, 128))
    img = np.expand_dims(img, axis=0)
    img = img / 255.0
    fingerprint_model = build_fingerprint_model((128, 128, 3))
    fingerprint_model.compile(optimizer='adam', loss='categorical_crossentropy')
    return fingerprint_model.predict(img).flatten()

def capture_image():
    capture = cv2.VideoCapture(0)
    if not capture.isOpened():
        raise RuntimeError("Failed to open camera.")
    ret, frame = capture.read()
    capture.release()
    if not ret:
        raise RuntimeError("Failed to capture image from camera.")
    return frame

def add_user():
    new_name = input("Enter Name:")
    if not new_name:
        print("No name entered. Exiting.")
        return

    VOICE_PATH = f"./voice_database/{new_name}/"
    FINGERPRINT_PATH = f"./fingerprint_database/{new_name}/"
    FACE_PATH = f"./face_database/{new_name}/"
    os.makedirs(VOICE_PATH, exist_ok=True)
    os.makedirs(FINGERPRINT_PATH, exist_ok=True)
    os.makedirs(FACE_PATH, exist_ok=True)

    FORMAT = pyaudio.paInt16
    CHANNELS = 2
    RATE = 44100
    CHUNK = 1024
    RECORD_SECONDS = 3

    for i in range(3):
        audio = pyaudio.PyAudio()

        if i == 0:
            for j in range(3, -1, -1):
                time.sleep(1.0)
                print(f"Speak your name in {j} seconds", end='\r')
        else:
            time.sleep(2.0)
            print("Speak your name one more time" if i == 1 else "Speak your name one last time")
            time.sleep(0.8)

        stream = audio.open(format=FORMAT, channels=CHANNELS, rate=RATE, input=True, frames_per_buffer=CHUNK)
        print("Recording...")
        frames = []

        for _ in range(0, int(RATE / CHUNK * RECORD_SECONDS)):
            data = stream.read(CHUNK)
            frames.append(data)

        stream.stop_stream()
        stream.close()
        audio.terminate()

        wave_file = wave.open(f"{VOICE_PATH}{i + 1}.wav", 'wb')
        wave_file.setnchannels(CHANNELS)
        wave_file.setsampwidth(audio.get_sample_size(FORMAT))
        wave_file.setframerate(RATE)
        wave_file.writeframes(b''.join(frames))
        wave_file.close()
        print("Voice recording saved")

    print("Capture Facial image: ")
    try:
        face_image = capture_image()
        face_image_path = f"{FACE_PATH}face.jpg"
        cv2.imwrite(face_image_path, face_image)
        print("Facial image captured and saved")
    except RuntimeError as e:
        print(f"Error: {e}")
        return

    print("Capture Fingerprint image: ")
    try:
        fingerprint_image = capture_image()  # Assuming the same method for capturing fingerprint
        fingerprint_image_path = f"{FINGERPRINT_PATH}fingerprint.jpg"
        cv2.imwrite(fingerprint_image_path, fingerprint_image)
        print("Fingerprint image captured and saved")
    except RuntimeError as e:
        print(f"Error: {e}")
        return

    X_voice, X_fingerprint, X_face, Y = [], [], [], []
    for user in os.listdir("./voice_database/"):
        user_voice_path = f"./voice_database/{user}/"
        user_fingerprint_path = f"./fingerprint_database/{user}/fingerprint.jpg"
        user_face_path = f"./face_database/{user}/face.jpg"

        user_voice_features = []
        for voice_file in os.listdir(user_voice_path):
            audio_file = os.path.join(user_voice_path, voice_file)
            try:
                audio_features = extract_audio_features(audio_file)
                user_voice_features.append(audio_features)
            except Exception as e:
                print(f"Error extracting audio features for {audio_file}: {e}")

        if not user_voice_features:
            print(f"Skipping user {user} due to missing voice features.")
            continue

        try:
            fingerprint_image = cv2.imread(user_fingerprint_path)
            if fingerprint_image is None:
                raise FileNotFoundError(f"Fingerprint image not found for user {user}.")
            fingerprint_features = extract_fingerprint_features(user_fingerprint_path)
        except Exception as e:
            print(f"Error extracting fingerprint features for user {user}: {e}")
            fingerprint_features = None

        try:
            face_image = cv2.imread(user_face_path)
            if face_image is None:
                raise FileNotFoundError(f"Facial image not found for user {user}.")
            face_features = extract_facial_features(face_image)
        except Exception as e:
            print(f"Error extracting facial features for user {user}: {e}")
            face_features = None

        if fingerprint_features is None or face_features is None:
            print(f"Skipping user {user} due to missing fingerprint or face features.")
            continue

        X_voice.append(np.mean(user_voice_features, axis=0))
        X_fingerprint.append(fingerprint_features)
        X_face.append(face_features)
        Y.append(user)

    if not (X_voice and X_fingerprint and X_face):
        print("Not enough data to train the model.")
        return

    X_voice = np.array(X_voice)
    X_fingerprint = np.array(X_fingerprint)
    X_face = np.array(X_face)

    # Scale the data
    scaler = StandardScaler()
    X_voice_scaled = scaler.fit_transform(X_voice)
    X_fingerprint_scaled = scaler.fit_transform(X_fingerprint)
    X_face_scaled = scaler.fit_transform(X_face)

    try:
        X_combined = np.concatenate((X_voice_scaled, X_fingerprint_scaled, X_face_scaled), axis=1)
    except ValueError as e:
        print(f"Error combining feature arrays: {e}")
        return

    le = LabelEncoder()
    Y_trans = le.fit_transform(Y)

    clf = LogisticRegression(random_state=0, max_iter=1000).fit(X_combined, Y_trans)

    MODEL_DIR = "gmm_models/"
    os.makedirs(MODEL_DIR, exist_ok=True)
    with open(os.path.join(MODEL_DIR, "voice_auth.gmm"), 'wb') as file:
        pickle.dump((clf, le, scaler), file)

    print(new_name + ' added successfully')

if __name__ == '__main__':
    add_user()

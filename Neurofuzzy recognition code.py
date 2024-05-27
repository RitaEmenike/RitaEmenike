import numpy as np
import cv2
import os
import pickle
import pyaudio
import wave
import librosa
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.applications import VGG16
from tensorflow.keras.applications.vgg16 import preprocess_input


# Step 2: Data Preprocessing
def preprocess_voice(voice_file):
    # Placeholder code for preprocessing voice data
    print("Preprocessing voice...")
    return voice_file

def preprocess_fingerprint(fingerprint_data):
    # Placeholder code for preprocessing fingerprint data
    print("Preprocessing fingerprint...")
    return fingerprint_data

def preprocess_facial_image(facial_image):
    # Placeholder code for preprocessing facial image
    print("Preprocessing facial image...")
    return facial_image

# Step 6: Database Integration
def retrieve_user_details(user_id):
    # Placeholder code for retrieving user details from the database
    user_details = {}  # Retrieve user details from the database based on user_id
    return user_details

# Feature extraction functions
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
    model.add(Dense(64, input_shape=input_shape, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(32, activation='relu'))
    model.add(Dense(10, activation='softmax'))
    return model

def extract_fingerprint_features(fingerprint_image_path):
    img = cv2.imread(fingerprint_image_path)
    if img is None:
        return None
    img = cv2.resize(img, (128, 128))
    img = np.expand_dims(img, axis=0)
    img = img / 255.0
    fingerprint_model = build_fingerprint_model((128, 128, 3))
    fingerprint_model.compile(optimizer='adam', loss='categorical_crossentropy')
    return fingerprint_model.predict(img).flatten()

# Build neuro-fuzzy model
def build_neuro_fuzzy_model(input_dim):
    model = Sequential()
    model.add(Dense(128, input_dim=input_dim, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(64, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(32, activation='relu'))
    model.add(Dense(10, activation='softmax'))
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    return model

# Recording functions for user input
def record_voice():
    FORMAT = pyaudio.paInt16
    CHANNELS = 2
    RATE = 44100
    CHUNK = 1024
    RECORD_SECONDS = 3
    audio = pyaudio.PyAudio()
    stream = audio.open(format=FORMAT, channels=CHANNELS, rate=RATE, input=True, frames_per_buffer=CHUNK)
    print("Recording voice...")
    frames = []
    for _ in range(0, int(RATE / CHUNK * RECORD_SECONDS)):
        data = stream.read(CHUNK)
        frames.append(data)
    stream.stop_stream()
    stream.close()
    audio.terminate()
    voice_path = "new_user_voice.wav"
    wave_file = wave.open(voice_path, 'wb')
    wave_file.setnchannels(CHANNELS)
    wave_file.setsampwidth(audio.get_sample_size(FORMAT))
    wave_file.setframerate(RATE)
    wave_file.writeframes(b''.join(frames))
    wave_file.close()
    print(f"Voice recording saved to {voice_path}")
    return voice_path

def record_fingerprint():
    print("Please place your finger on the fingerprint scanner.")
    fingerprint_path = "new_user_fingerprint.jpg"
    sample_image = np.random.randint(0, 255, (128, 128, 3), dtype=np.uint8)
    cv2.imwrite(fingerprint_path, sample_image)
    print(f"Fingerprint saved to {fingerprint_path}")
    return fingerprint_path

def record_facial_image():
    print("Please look at the camera.")
    facial_image_path = "new_user_face.jpg"
    sample_image = np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)
    cv2.imwrite(facial_image_path, sample_image)
    print(f"Facial image saved to {facial_image_path}")
    return facial_image_path

def main():
    # Load and preprocess data
    X_voice, X_fingerprint, X_face, Y = [], [], [], []
    for user in os.listdir("./voice_database/"):
        user_voice_path = f"./voice_database/{user}/"
        user_fingerprint_path = f"./fingerprint_database/{user}/fingerprint.jpg"
        user_face_path = f"./face_database/{user}/face.jpg"

        user_voice_features = []
        for voice_file in os.listdir(user_voice_path):
            audio_file = os.path.join(user_voice_path, voice_file)
            audio_features = extract_audio_features(audio_file)
            if audio_features is not None:
                user_voice_features.append(audio_features)

        if not user_voice_features:
            print(f"Skipping user {user} due to missing voice features.")
            continue

        fingerprint_features = extract_fingerprint_features(user_fingerprint_path)
        face_features = extract_facial_features(cv2.imread(user_face_path))

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

    X_combined = np.concatenate((X_voice_scaled, X_fingerprint_scaled, X_face_scaled), axis=1)

    le = LabelEncoder()
    Y_trans = le.fit_transform(Y)

    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(X_combined, Y_trans, test_size=0.2, random_state=42)

    # Build and train the model
    model = build_neuro_fuzzy_model(X_train.shape[1])
    model.fit(X_train, y_train, epochs=50, batch_size=16, validation_data=(X_test, y_test))

    # Evaluate the model
    y_pred = np.argmax(model.predict(X_test), axis=1)
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Model accuracy: {accuracy * 100:.2f}%")

    # Save the model and the label encoder
    MODEL_DIR = "neuro_fuzzy_models/"
    os.makedirs(MODEL_DIR, exist_ok=True)
    model.save(os.path.join(MODEL_DIR, "user_recognition_model.h5"))
    with open(os.path.join(MODEL_DIR, "label_encoder.pkl"), 'wb') as file:
        pickle.dump(le, file)

    # Predict new user
    voice_path = record_voice()
    fingerprint_path = record_fingerprint()
    facial_image_path = record_facial_image()

    if not (os.path.exists(voice_path) and os.path.exists(fingerprint_path) and os.path.exists(facial_image_path)):
        print("Error: Missing recorded data.")
        return

    new_user_voice_features = extract_audio_features(voice_path)
    new_user_fingerprint_features = extract_fingerprint_features(fingerprint_path)
    new_user_facial_features = extract_facial_features(cv2.imread(facial_image_path))

    if new_user_voice_features is None or new_user_fingerprint_features is None or new_user_facial_features is None:
        print("Error: Could not extract features from the recorded data.")
        return

    new_user_voice_scaled = scaler.transform([new_user_voice_features])
    new_user_fingerprint_scaled = scaler.transform([new_user_fingerprint_features])
    new_user_facial_scaled = scaler.transform([new_user_facial_features])

    new_user_combined = np.concatenate((new_user_voice_scaled, new_user_fingerprint_scaled, new_user_facial_scaled), axis=1)

    new_user_prediction = model.predict(new_user_combined)
    new_user_predicted_label = le.inverse_transform(np.argmax(new_user_prediction, axis=1))

    print(f"Predicted user: {new_user_predicted_label[0]}")

if __name__ == '__main__':
    main()

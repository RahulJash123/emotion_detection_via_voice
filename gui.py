import tkinter as tk
from tkinter import filedialog
import numpy as np
import librosa
from keras.models import model_from_json

# Load model architecture and weights
model_architecture_path = 'model_a1.json'
model_weights_path = 'model_weights1.h5'

with open(model_architecture_path, 'r') as f:
    model_json = f.read()

loaded_model = model_from_json(model_json)
loaded_model.load_weights(model_weights_path)

# Function to extract features from audio file
def extract_features(file_path, max_pad_len=180):
    signal, sample_rate = librosa.load(file_path, sr=None)
    mfccs = librosa.feature.mfcc(y=signal, sr=sample_rate, n_mfcc=13)
    
    if mfccs.shape[1] < max_pad_len:
        mfccs_padded = np.pad(mfccs, ((0, 0), (0, max_pad_len - mfccs.shape[1])), mode='constant')
    else:
        mfccs_padded = mfccs[:, :max_pad_len]
    
    mfccs_processed = np.expand_dims(mfccs_padded, axis=-1)
    return mfccs_processed

# Function to predict emotion
def predict_emotion(file_path, model):
    features = extract_features(file_path)
    prediction = model.predict(features)
    emotion_label = np.argmax(prediction)
    return emotion_label

# Function to handle file upload and prediction
def upload_and_predict():
    file_path = filedialog.askopenfilename()
    if file_path:
        try:
            predicted_emotion = predict_emotion(file_path, loaded_model)
            emotion_mapping = {0: 'Sad', 1: 'Calm', 2: 'Angry', 3: 'Happy', 4: 'Neutral', 5: 'Fearful', 6: 'Disgust', 7: 'Surprised'}
            predicted_emotion_label = emotion_mapping[predicted_emotion]
            result_label.config(text=f"Predicted Emotion: {predicted_emotion_label}", pady=10,font=('arial',15,'bold'))
        except Exception as e:
            result_label.config(text=f"An error occurred: {str(e)}")

# Create main GUI window
root = tk.Tk()
root.geometry('800x600')
root.title("Emotion Detection")
root.configure(background='#CDCDCD')

# Create and place upload button
upload_button = tk.Button(root, text="Upload Voice", command=upload_and_predict)
upload_button.configure(background="#364156",foreground='white',font=('arial',20,'bold'))
upload_button.pack(pady=20)

result_label = tk.Label(root, text="")
result_label.pack(pady=10)

# Run the GUI application
root.mainloop()

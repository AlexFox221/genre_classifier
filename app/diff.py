import os
import math
import subprocess
import librosa
import json
from statistics import mode
import numpy as np
import tensorflow as tf
import tensorflow.keras as keras

from flask import Flask, render_template, url_for, request, redirect
from flask_dropzone import Dropzone

basedir = os.path.abspath(os.path.dirname(__file__))
diff = Flask(__name__)


diff.config.update(

    UPLOADED_PATH=os.path.join(basedir, 'uploads'),
    # Flask-Dropzone config:
    
    DROPZONE_ALLOWED_FILE_CUSTOM=True,
    DROPZONE_ALLOWED_FILE_TYPE='.mp3',
    DROPZONE_MAX_FILES=1,
    DROPZONE_MAX_FILE_SIZE=1000)


@diff.route('/')
def first():
    return render_template('index.html')

@diff.route('/upload', methods=['POST'])
def upload():    
    f = request.files.get('file')
    f.save(os.path.join(diff.config['UPLOADED_PATH'], "uploaded_song.mp3"))
    return render_template('newtoday.html')
        


def cut_song():
    from pydub import AudioSegment

    startMin = 0
    startSec = 0

    endMin = 0
    endSec = 30

    # Time to miliseconds
    startTime = startMin * 60 * 1000 + startSec * 1000
    endTime = endMin * 60 * 1000 + endSec * 1000

    # Opening file and extracting segment
    song = AudioSegment.from_mp3("uploads/uploaded_song.mp3")
    extract = song[startTime:endTime]

    # Saving
    extract.export("uploads/test_cut.wav", format="wav")


def save_mfcc(file_path, json_path, samples_per_track, n_mfcc=13, n_fft=2048, hop_length=512, n_segments=5):
    # dictionary to store data
    data = {
        "mapping": [],
        "labels": [],
        "mfcc": [],
    }

    num_samples_per_segment = int(samples_per_track / n_segments)
    expected_num_mfcc_vectors_per_segment = math.ceil(
        num_samples_per_segment / hop_length)  # 1.2 -> ~2

    # save the semantic label
    data["mapping"].append(file_path)

    # process files for a specific genre
    signal, sr = librosa.load(file_path, sr=22050)

    # process segments extracting mfcc and storing data
    for s in range(n_segments):
        start_sample = num_samples_per_segment * s  # s=0 -> 0
        # s=0 -> num_samples_per_segment
        finish_sample = start_sample + num_samples_per_segment

        mfcc = librosa.feature.mfcc(signal[start_sample:finish_sample],
                                    sr=sr,
                                    n_fft=n_fft,
                                    n_mfcc=n_mfcc,
                                    hop_length=hop_length)

        mfcc = mfcc.T

        # store mfcc for segment if it has expected length
        if len(mfcc) == expected_num_mfcc_vectors_per_segment:
            data["mfcc"].append(mfcc.tolist())
            data["labels"].append(s * 0)

    with open(json_path, "w") as fp:
        json.dump(data, fp, indent=4)


def load_data():
    with open("json/song_data.json", "r") as fp:
        data = json.load(fp)

    # convert lists into numpy arrays
    X = np.array(data["mfcc"])
    return X


def load_model():
    # load the model from disk
    new_model = tf.keras.models.load_model("cnn_model/cnn_model.h5")
    new_model.summary()
    return new_model


def predict(model, X):
    X = X[..., np.newaxis]
    X = X[np.newaxis, ...]

    prediction = model.predict(X)  # X -> (1, 130, 13, 1)

    # extract index with max value
    predicted_index = np.argmax(prediction, axis=1)
    return int(predicted_index)


def predict_all(model, X):
    predict_array = []
    for e in range(9):
        predict_array.append(predict(model, X[e]))
    final_indx = most_common(predict_array)
    print(predict_array)
    print(final_indx)
    return final_indx


def most_common(List):
    return(mode(List))


if __name__ == "__main__":
    diff.run(debug=True)
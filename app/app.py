import urllib.request
import os
import base64
import requests
from requests.adapters import HTTPAdapter
from requests.packages.urllib3.util.retry import Retry
from time import sleep
import math
import subprocess
import librosa
import json
from statistics import mode
import numpy as np
import tensorflow as tf
import tensorflow.keras as keras
import requests

from bs4 import BeautifulSoup
from collections import Counter
from flask import Flask, render_template, url_for, request, redirect
from flask_dropzone import Dropzone

basedir = os.path.abspath(os.path.dirname(__file__))

app = Flask(__name__)

app.config.update(

    UPLOADED_PATH=os.path.join(basedir, 'uploads'),
    # Flask-Dropzone config:

    DROPZONE_ALLOWED_FILE_CUSTOM=True,
    DROPZONE_ALLOWED_FILE_TYPE='.mp3',
    DROPZONE_MAX_FILES=1,
    DROPZONE_MAX_FILE_SIZE=1000,
)

dropzone = Dropzone(app)


@app.route('/')
def first():
    return render_template('index.html')


@app.route('/upload', methods=['POST'])
def upload():
    if os.path.exists("uploads/uploaded_song.mp3"):
        os.remove("uploads/uploaded_song.mp3")
    f = request.files.get('file')
    f.save(os.path.join(app.config['UPLOADED_PATH'], "uploaded_song.mp3"))
    return render_template('index.html')


@app.route('/getResult', methods=['GET'])
def getResult():
    cut_song()
    save_mfcc("uploads/test_cut.wav", "json/song_data.json",
              samples_per_track=22050*30, n_segments=10)
    X = load_data()
    model = load_model()
    result, gen1, p1, gen2, p2, gen3, p3 = predict_all(model, X)
    song_length("uploads/uploaded_song.mp3")

    names, authors, hrefs = scrap_google_page(result)

    rec1 = names[0]+" - " + authors[0]
    rec2 = names[1]+" - " + authors[1]
    rec3 = names[2]+" - " + authors[2]
    rec4 = names[3]+" - " + authors[3]
    rec5 = names[4]+" - " + authors[4]


    return render_template('result_form.html', result=result,
                           gen1=gen1,
                           gen2=gen2,
                           gen3=gen3,
                           per1=p1,
                           per2=p2,
                           per3=p3,
                           rec1=rec1,
                           rec2=rec2,
                           rec3=rec3,
                           rec4=rec4,
                           rec5=rec5,
                           href1="https://www.google.com"+hrefs[0],
                           href2="https://www.google.com"+hrefs[1],
                           href3="https://www.google.com"+hrefs[2],
                           href4="https://www.google.com"+hrefs[3],
                           href5="https://www.google.com"+hrefs[4])


def scrap_google_page(search):
    sleep(2)
    session = requests.Session()
    retry = Retry(connect=3, backoff_factor=0.5)
    adapter = HTTPAdapter(max_retries=retry)
    session.mount('http://', adapter)
    session.mount('https://', adapter)

    product = "Greatest "+search+" songs"
    if search == "Country":
        product = "Greatest Country love songs"
    if search == "Metal":
        product = "Greatest metal songs of all time"
    headers = {
            'authority': 'www.google.com',
            'user-agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/90.0.4430.212 Safari/537.36',
            'accept-language': 'en-US,en;q=0.9'}
    url = 'https://google.com/search?hl={}&q={}'.format('en', product)
    proxy={'http':'89.248.244.182:8080'}
    sleep(2)
    res = session.get(url, headers=headers).content
    soup = BeautifulSoup(res, 'html.parser')
    nameArray = []
    authorArray = []
    hrefsArray = []
    print(soup.title.text)

    for div in soup.findAll("a", {"class": "PZPZlf"}, 16):
        resultOne = div.findAll("div", {"class": "FozYP"})
        resultTwo = div.findAll("span")
        resultThree = div.get("href")
        
        if (len(resultOne) >= 1):
            textResult = resultOne[0].get_text()
            if (len(textResult) >= 1):
                nameArray.append(textResult)

        if (len(resultTwo) >= 1):
            textResult = resultTwo[0].get_text()
            if (len(textResult) >= 1):
                authorArray.append(textResult)

        if (len(resultThree) >= 1):
            hrefsArray.append(resultThree)

    print(nameArray)
    print(authorArray)
    #print(hrefsArray)

    return nameArray, authorArray, hrefsArray


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



def song_length(file_path):
    result = subprocess.run(
        ['ffprobe', '-v', 'error', '-show_entries', 'format=duration', '-of', 'default=noprint_wrappers=1:nokey=1',
         file_path], stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    print(round(float(result.stdout)))
    return round(float(result.stdout))



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
    genres = ["Blues", "Classical", "Country", "Disco",
              "Hip-hop", "Jazz", "Metal", "Pop", "Reggae", "Rock"]
    c = Counter(predict_array)
    persantage_res = [(i, c[i] / len(predict_array) * 100.0)
                      for i, count in c.most_common()]
    lenp = len(persantage_res)
    print(lenp)
    if lenp >= 1:
        pres1 = persantage_res[0]
        pres1 = [genres[pres1[0]], round(pres1[1], 2)]
        gen1 = pres1[0]
        p1 = pres1[1]
        print(pres1)
        if lenp >= 2:
            pres2 = persantage_res[1]
            pres2 = [genres[pres2[0]], round(pres2[1], 2)]
            gen2 = pres2[0]
            p2 = pres2[1]
            print(pres2)
            if lenp >= 3:
                pres3 = persantage_res[2]
                pres3 = [genres[pres3[0]], round(pres3[1], 2)]
                gen3 = pres3[0]
                p3 = pres3[1]
                print(pres3)
                return genres[final_indx], gen1, p1, gen2, p2, gen3, p3
            return genres[final_indx], gen1, p1, gen2, p2
        return genres[final_indx], gen1, p1


def most_common(List):
    return(mode(List))


if __name__ == "__main__":
     app.run(host='127.0.0.1', port=4000)
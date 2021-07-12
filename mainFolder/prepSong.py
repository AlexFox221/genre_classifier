import math
import subprocess
import librosa
import json

from os import path
from pydub import AudioSegment

JSON_PATH = "dataSong.json"


def cutsong():
    from pydub import AudioSegment

    files_path = ''
    file_name = ''

    startMin = 0
    startSec = 0

    endMin = 0
    endSec = 30

    # Time to miliseconds
    startTime = startMin * 60 * 1000 + startSec * 1000
    endTime = endMin * 60 * 1000 + endSec * 1000

    # Opening file and extracting segment
    song = AudioSegment.from_mp3("songs/blues/trip.mp3")
    extract = song[startTime:endTime]

    # Saving
    extract.export("songs/blues/test_cut.mp3", format="mp3")

    # files
    src = "songs/blues/test_cut.mp3"
    dst = "songs/blues/test.wav"

    # convert wav to mp3
    sound = AudioSegment.from_mp3(src)
    sound.export(dst, format="wav")


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
    expected_num_mfcc_vectors_per_segment = math.ceil(num_samples_per_segment / hop_length)  # 1.2 -> ~2

    # save the semantic label
    data["mapping"].append(file_path)

    # process files for a specific genre
    signal, sr = librosa.load(file_path, sr=22050)

    # process segments extracting mfcc and storing data
    for s in range(n_segments):
        start_sample = num_samples_per_segment * s  # s=0 -> 0
        finish_sample = start_sample + num_samples_per_segment  # s=0 -> num_samples_per_segment

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


if __name__ == "__main__":
    FILE_PATH = "songs/blues/test.wav"

    # measured in seconds
    cutsong()
    SAMPLES_PER_TRACK = 22050 * 30

    save_mfcc(FILE_PATH, JSON_PATH, SAMPLES_PER_TRACK, n_segments=10)

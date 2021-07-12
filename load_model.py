import json
from collections import Counter
from statistics import mode

import numpy as np
import tensorflow as tf
import tensorflow.keras as keras
from sklearn.model_selection import train_test_split

DATASET_PATH = "dataSong.json"


def load_data(dataset_path):
    with open(dataset_path, "r") as fp:
        data = json.load(fp)

    # convert lists into numpy arrays
    X = np.array(data["mfcc"])
    return X


def load_model():
    # load the model from disk
    new_model = tf.keras.models.load_model("cnn_model.h5")
    new_model.summary()
    return new_model


def predict(model, X):
    X = X[..., np.newaxis]
    X = X[np.newaxis, ...]

    prediction = model.predict(X)  # X -> (1, 130, 13, 1)

    # extract index with max value
    predicted_index = np.argmax(prediction, axis=1)
    print("Predicted index: {}".format(predicted_index))
    return int(predicted_index)


def most_common(List):
    return(mode(List))


if __name__ == "__main__":
    model = load_model()
    # make prediction
    X = load_data(DATASET_PATH)

    i1 = predict(model, X[0])
    i2 = predict(model, X[1])
    i3 = predict(model, X[2])
    i4 = predict(model, X[3])
    i5 = predict(model, X[4])
    i6 = predict(model, X[5])
    i7 = predict(model, X[6])
    i8 = predict(model, X[7])
    i9 = predict(model, X[8])
    i10 = predict(model, X[9])
    new_array = [i1, i2, i3, i4, i5, i6, i7, i8, i9, i10]
    print(most_common(new_array))

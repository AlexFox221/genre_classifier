# genre_classifier

Web page, that uses trained CNN in background to predict the uploaded song genre and recommend most listened/popular compositions of the given genre.

1. Preprocessing the data (songs - 200 instances of different genres) Soundwave -> Spectrum (using FFT) -> Spectrogram (using STFT) -> MFCC - preprocess.py
2. Saving training data as JSON (genre, MFCC,  labels) - preprocess.py
3. Create data split and feed the data to the CNN, optimize(Adam), train and save model - cnn_genre_classification.py

--------------------------------------------------------------------------------

1. Creating Web page, adding upload functionality - app.py and templates
2. Loading the cnn_model.h5, adding prediction function - app.py
3. Adding the recommendation function - app.py

Technologies - Python, TensorFlow (keras), SoapAPI , HTML&CSS, JS

Model: "sequential"
_________________________________________________________________
Layer (type)                 Output Shape              Param #   
_________________________________________________________________
conv2d (Conv2D)              (None, 128, 11, 32)       320
_________________________________________________________________
max_pooling2d (MaxPooling2D) (None, 64, 6, 32)         0
_________________________________________________________________
batch_normalization (BatchNo (None, 64, 6, 32)         128
_________________________________________________________________
conv2d_1 (Conv2D)            (None, 62, 4, 32)         9248
_________________________________________________________________
max_pooling2d_1 (MaxPooling2 (None, 31, 2, 32)         0
_________________________________________________________________
batch_normalization_1 (Batch (None, 31, 2, 32)         128
_________________________________________________________________
conv2d_2 (Conv2D)            (None, 30, 1, 32)         4128
_________________________________________________________________
max_pooling2d_2 (MaxPooling2 (None, 15, 1, 32)         0
_________________________________________________________________
batch_normalization_2 (Batch (None, 15, 1, 32)         128
_________________________________________________________________
flatten (Flatten)            (None, 480)               0
_________________________________________________________________
dense (Dense)                (None, 64)                30784
_________________________________________________________________
dropout (Dropout)            (None, 64)                0
_________________________________________________________________
dense_1 (Dense)              (None, 10)                650
_________________________________________________________________
Total params: 45,514
Trainable params: 45,322
Non-trainable params: 192

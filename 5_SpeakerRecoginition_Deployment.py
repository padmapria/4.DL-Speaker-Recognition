#------------------------------------------------------------------------------
# 
#  Project - Speaker Recognition
# Team 9
# Part 5 - Deployment using Flask
#------------------------------------------------------------------------------

from flask import Flask, flash, request, redirect, url_for, Response
from matplotlib import cm
import sklearn
from sklearn.model_selection import KFold
from sklearn.model_selection import train_test_split
import json
import sys
import numpy as np
import pandas as pd
import librosa
import pickle
import os
from shutil import copyfile
import matplotlib.pyplot as plt
import imageio

import cv2
import time
import multiprocessing

import tensorflow as tf
import tensorflow.keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.layers import Conv2D, ZeroPadding2D, Activation, Input, concatenate
from tensorflow.keras.models import Model

from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.layers import MaxPooling2D
from tensorflow.keras.layers import Concatenate
from tensorflow.keras.layers import Lambda, Flatten, Dense
from tensorflow.keras.initializers import glorot_uniform
from tensorflow.keras.utils import CustomObjectScope

from tensorflow.keras.layers import Layer
from tensorflow.keras.regularizers import l2
from tensorflow.keras import backend as K
from tensorflow.keras import initializers

import numpy.random as random

# Initialize our Flask app.
# NOTE: Flask is used to host our app on a web server, so that
# we can call its functions over HTTP/HTTPS.
#
app = Flask(__name__)

base_folder = "/Users/MacBookPro/AIandMLNYP/AIProject/SpeakerRecognition/TIMIT/"
data_folder = base_folder + "/data"
predict_audio_folder = base_folder + "/Audio/Prediction/"
pickle_path = '/Users/MacBookPro/AIandMLNYP/AIProject/SpeakerRecognition/data/'
output_folder = base_folder + "/npydata/"
model_path = base_folder + 'Model'


#------------------------------------------------------------------------------
# Gets the text content of the static file.
#------------------------------------------------------------------------------
def get_file(filename):  # pragma: no cover
    try:
        src = os.path.join(os.getcwd(), filename)
        # Figure out how flask returns static files
        # Tried:
        # - render_template
        # - send_file
        # This should not be so non-obvious
        return open(src).read()
    except IOError as exc:
        return str(exc)


#------------------------------------------------------------------------------
# This serves static files to the browser.
#------------------------------------------------------------------------------
@app.route('/', defaults={'path': ''})
@app.route('/<path:path>')
def get_resource(path):  # pragma: no cover
    mimetypes = {
        ".css": "text/css",
        ".html": "text/html",
        ".js": "application/javascript",
    }
    complete_path = os.path.join(os.getcwd(), path)
    ext = os.path.splitext(path)[1]
    mimetype = mimetypes.get(ext, "text/html")
    content = get_file(complete_path)
    return Response(content, mimetype=mimetype)


#------------------------------------------------------------------------------
# Process audio input
#------------------------------------------------------------------------------
# Each of our sample (16khz) lasts exactly from 3 - 5  seconds. We will truncate at 3 secs with 16000 * 3 samples.
#
mfcc_hop_length = 256
mfcc_max_frames = int(16000 * 3 / mfcc_hop_length) + 1

print ("MFCC Frames (for 3 sec audio):     %d" % (mfcc_max_frames))


num_classes = 10
max_samples = 16000 * 3  # 5 seconds
max_mfcc_features = 40

# Scale the values to be between 
def scale(arr):
    #arr = arr - arr.mean()
    safe_max = np.abs(arr).max()
    if safe_max == 0:
        safe_max = 1
    arr = arr / safe_max
    return arr


# Load a file and convert its audio signal into a series of MFCC
# This will return a 2D numpy array.
#
def convert_mfcc(file_name):
    signal, sample_rate = librosa.load(file_name) 
    signal = librosa.util.normalize(signal)
    signal_trimmed, index = librosa.effects.trim(signal, top_db=60)
    signal_trimmed = librosa.util.fix_length(signal_trimmed, max_samples)
    
    feature = (librosa.feature.mfcc(y=signal_trimmed, sr=sample_rate, n_mfcc=max_mfcc_features).T)

    if (feature.shape[0] > mfcc_max_frames):
        feature = feature[0:mfcc_max_frames, :]
    if (feature.shape[0] < mfcc_max_frames):
        feature = np.pad(feature, pad_width=((0, mfcc_max_frames - feature.shape[0]), (0,0)), mode='constant')
    
    # This removes the average component from the MFCC as it may not be meaningful.
    #
    feature[:,0] = 0
        
    feature = scale(feature)

    return feature


# generate pairs and targets
with open(os.path.join(pickle_path, "employeeaudio.pickle"), "rb") as f:
    (X, y) = pickle.load(f)


#----------------------------------------------
#load the model
#----------------------------------------------

def initialize_weights(shape, dtype=None):
    return np.random.normal(loc = 0.0, scale = 1e-2, size = shape)

def initialize_bias(shape, dtype=None):
    return np.random.normal(loc = 0.5, scale = 1e-2, size = shape)

def get_base_conv_encoder(input_shape):
     # Convolutional Neural Network
    model = Sequential()
    model.add(Conv2D(64, (3,3), activation='relu', input_shape=input_shape,
                   #kernel_initializer=initialize_weights, kernel_regularizer=l2(2e-4)))
                   kernel_initializer=initializers.RandomNormal(mean=0.0, stddev=0.01, seed=None), kernel_regularizer=l2(2e-4)))  
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Conv2D(128, (3,3), activation='relu',
                     kernel_initializer=initializers.RandomNormal(mean=0.0, stddev=0.01, seed=None),
                     bias_initializer=initialize_bias, kernel_regularizer=l2(2e-4)))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Flatten())
    model.add(Dense(516, activation='sigmoid',
                   kernel_regularizer=l2(1e-3),
                   kernel_initializer=initializers.RandomNormal(mean=0.0, stddev=0.01, seed=None),bias_initializer=initialize_bias))
    return model
    
def build_final_model(input_shape,  distance_metric='uniform_euclidean'):
    
    assert distance_metric in ('uniform_euclidean', 
                                'weighted_l1',
                                'cosine_distance')
    left_input = Input(input_shape)
    right_input = Input(input_shape)
    model = get_base_conv_encoder(input_shape)
    encoded_l = model(left_input)
    encoded_r = model(right_input)
    
    
    if distance_metric == 'weighted_l1':
        print("using Weighted_l1")
        L1_layer = Lambda(lambda tensors:K.abs(tensors[0] - tensors[1]))
        L1_distance = L1_layer([encoded_l, encoded_r])
        prediction = Dense(1,activation='sigmoid',bias_initializer=initialize_bias)(L1_distance)
      
    if distance_metric == 'uniform_euclidean':
        print("inside euclidian")
        L1_layer = Lambda(lambda tensors:K.sqrt(K.sum(K.square(K.abs(tensors[0] - tensors[1])),axis=-1, keepdims=True)))
        L1_distance = L1_layer([encoded_l, encoded_r])
        prediction = Dense(1,activation='sigmoid',bias_initializer=initialize_bias)(L1_distance)

   
    if distance_metric == 'cosine_distance':
        print("using cosine similarity")
        L1_layer = Lambda(cosine_similarity, output_shape=cos_dist_output_shape)
        L1_distance = L1_layer([encoded_l, encoded_r])
        prediction = Dense(1,activation='sigmoid',bias_initializer=initialize_bias)(L1_distance)
    
    
    # Connect the inputs with the outputs
    siamese_net = Model(inputs=[left_input,right_input],outputs=prediction)
    # return the model
    return siamese_net  

model = build_final_model((188, 40, 1),'weighted_l1')
model.load_weights(os.path.join(model_path, "seg_weights.best.hdf5"))
optimizer = Adam(lr = 0.00006)
model.compile(loss="binary_crossentropy",metrics=['accuracy'], optimizer=optimizer)


#------------------------------------------------------------------------------
# This is our predict URL 
#------------------------------------------------------------------------------
@app.route('/predict', methods=['POST'])
def predict():
    
    # Extracts the file from the upload and save it into
    # a temporary file.
    #
    f = request.files['file']
    filename = "%s.wav" % ((random.randrange(0, 1000000000)))
    f.save(filename)

    # Loads the file and convert features. 
    #

    mfcc = convert_mfcc(filename)
        
    input_l = []
    input_r = []
    s,w,h = X.shape
    input_l.append(mfcc.reshape(w,h,1))
    y_pred = np.zeros(shape=(s))
    
    ###loop through the audio mfccs of the baseline audio files.
     
    for  i in range(X.shape[0]):
        w,h = mfcc.shape
        pairs=[np.zeros((1 , w, h,1)) for i in range(2)]
        pairs[0][0,:,:,:] = input_l[0]
        pairs[1][0,:,:,:] = X[i].reshape(w,h,1)
        y_pred[i] = (model.predict(pairs).ravel())[0]
        
    
    # Use our Keras model to predict the output.
    #
    prediction = y[np.argmax(y_pred)]
    result = json.dumps(
        { 
            # Place the class index with the highest probability into
            # the "best_class" attribute.
            #
            # The use of item() converts best_class (which is a numpy.int64 
            # data type) to a native Python int.
            #
            #"best_class": best_class.item(), 
            "person_name" : prediction

            # Return the full prediction from Keras.
            # Convert a Numpy array to a native Python list.
            #
            #"full_prediction" : full_prediction.tolist()[0]
        })

    os.remove(filename)

    return Response(result, mimetype='application/json')                           


#------------------------------------------------------------------------------
# This starts our web server.
# Although we are running this on our local machine,
# this can technically be hosted on any VM server in the cloud!
#------------------------------------------------------------------------------
if __name__ == "__main__":
    app.debug = True
    app.run(host='127.0.0.1',port=5005)



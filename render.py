import os
import cv2
import json
import string
import numpy as np
import pandas as pd 
from keras import layers
from keras import callbacks
from keras.models import Model
import matplotlib.pyplot as plt
from keras.models import load_model

#Init main values
symbols = string.ascii_lowercase + "0123456789" # All symbols captcha can contain
num_symbols = len(symbols)
img_shape = (50, 200, 1)

def create_model():
    img = layers.Input(shape=img_shape) # Get image as an input and process it through some Convs
    conv1 = layers.Conv2D(16, (3, 3), padding='same', activation='relu')(img)
    bn1 = layers.BatchNormalization()(conv1)
    mp1 = layers.MaxPooling2D(padding='same')(bn1)  # 100x25
    conv2 = layers.Conv2D(32, (3, 3), padding='same', activation='relu')(mp1)
    bn2 = layers.BatchNormalization()(conv2)
    mp2 = layers.MaxPooling2D(padding='same')(bn2)  # 50x13
    conv3 = layers.Conv2D(32, (3, 3), padding='same', activation='relu')(mp2)
    bn3 = layers.BatchNormalization()(conv3)
    mp3 = layers.MaxPooling2D(padding='same')(bn3)  # 25x7
    
    # Get flattened vector and make 6 branches from it. Each branch will predict one letter
    flat = layers.Flatten()(mp3)
    outs = []
    for _ in range(6):
        dens1 = layers.Dense(64, activation='relu')(flat)
        drop = layers.Dropout(0.5)(dens1)
        res = layers.Dense(num_symbols, activation='sigmoid')(drop)

        outs.append(res)
    
    # Compile model and return it
    model = Model(img, outs)
    model.compile(loss='categorical_crossentropy', optimizer='adam',metrics=["accuracy"])
    return model

def preprocess_data():
    n_samples = len(os.listdir('./captcha_images_v1/'))
    X = np.zeros((n_samples, 50, 200, 1)) #1070*50*200
    y = np.zeros((6, n_samples, num_symbols)) #5*1070*36

    for i, pic in enumerate(os.listdir('./captcha_images_v1/')):
        # Read image as grayscale
        img = cv2.imread(os.path.join('./captcha_images_v1/', pic), cv2.IMREAD_GRAYSCALE)
        pic_target = pic[:-4]
        if img.shape != (50,200):
            img = cv2.resize(img, (200,50))
        if len(pic_target) != 6:
            print("Lengh Error:", img)
        else:
            # Scale and reshape image
            img = img / 255.0
            img = np.reshape(img, (50, 200, 1))
            # Define targets and code them using OneHotEncoding
            targs = np.zeros((6, num_symbols))
            for j, l in enumerate(pic_target):
                ind = symbols.find(l)
                targs[j, ind] = 1
            X[i] = img
            y[:, i] = targs
    
    # Return final data
    return X, y

X, y = preprocess_data()
X_train, y_train = X[:970], y[:, :970]
X_test, y_test = X[970:], y[:, 970:]

model=create_model();
model.summary();

#model = create_model()
hist = model.fit(X_train, [y_train[0], y_train[1], y_train[2], y_train[3], y_train[4], y_train[5]], batch_size=16, epochs=200,verbose=1, validation_split=0.2)

model.save('predict.h5')
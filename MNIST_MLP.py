# -*- coding: utf-8 -*-
"""
@author: P. Rajavel

"""
# MNIST dataset DNN


# fix dimension ordering issue
from keras import backend as K
K.set_image_dim_ordering('th')

import numpy as np
seedNo = 11
np.random.seed(seedNo)

# to know my computation time
import time
starttime = time.time()


# load data from keras
from keras.datasets import mnist # each image is 28x28 pixels
(X_train, y_train), (X_test, y_test) = mnist.load_data()


# Reshape 28*28 images to a 784 vector, for each image
Image_row = X_train.shape[1]
Image_col = X_train.shape[2]
Num_pixels = Image_row * Image_col
Num_trimage = X_train.shape[0]
Num_testimage = X_test.shape[0]
X_train = X_train.reshape(Num_trimage, Num_pixels).astype('float32')
X_test = X_test.reshape(Num_testimage, Num_pixels).astype('float32')


# Normalize each pixel to (0-1 range)
X_train = X_train / 255
X_test = X_test / 255


# Output is catagorical, so do one hot encoding
from keras.utils import np_utils
y_train = np_utils.to_categorical(y_train)
y_test = np_utils.to_categorical(y_test)
Num_class = y_test.shape[1]


from keras.models import Sequential
from keras.layers import Dense
def mykeras_model():
    # 1 Define model
    mykrmodel = Sequential()
    mykrmodel.add(Dense(Num_pixels, input_dim = Num_pixels, kernel_initializer = 'normal', activation = 'relu'))
    mykrmodel.add(Dense(Num_class, kernel_initializer = 'normal', activation = 'softmax'))
    
    # 2 Compile model
    mykrmodel.compile(loss = 'categorical_crossentropy', optimizer = 'adam', metrics = ['accuracy'])
    return mykrmodel


# 3. Fit the model
krmodel = mykeras_model()
krmodel.fit(X_train, y_train, validation_data = (X_test, y_test), epochs = 1, batch_size = 100, verbose = 2)

# 4. Evaluate using kfold cross validation
results_pred = krmodel.evaluate(X_test, y_test, verbose = 0)
print("Accuracy: %.2f%%" % (results_pred[1]*100))

stoptime = time.time()
print(stoptime-starttime)
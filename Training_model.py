#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np
import cv2

from numpy import load
from numpy import zeros
from numpy import ones
import os
import keras
from numpy.random import randint

import glob
from keras.utils import to_categorical

from keras.optimizers import Adam
from keras.initializers import RandomNormal
from numpy.random import randint
from numpy.random import randint
from keras.models import Model
from keras.models import Input
from keras.layers import Dense
from keras.utils import np_utils
from matplotlib import pyplot
from keras.layers import Conv2D
from keras.layers import Conv2DTranspose
from keras.layers import LeakyReLU
from keras.layers import Activation
from keras.layers import Concatenate
from keras.layers import Dropout
from numpy import load
from keras.layers import BatchNormalization
from keras.layers import LeakyReLU
from keras.utils.vis_utils import plot_model
import tensorflow as tf
import numpy as np
import sys
import matplotlib.pyplot as plt
from numpy import argmax
from keras.preprocessing.image import ImageDataGenerator
from sklearn.preprocessing import LabelEncoder
from keras.callbacks import ModelCheckpoint
import random
from scipy import ndimage
import cv2
from tensorflow.keras.applications import EfficientNetB5,DenseNet121
from sklearn.metrics import confusion_matrix


# In[ ]:


def model_dense(image_shape=(512,512,3)):
    in_src = Input(shape=image_shape)
    #d = BatchNormalization()(in_src)
    m = EfficientNetB5(include_top=False, weights='imagenet', input_shape=(512,512,3))(in_src)
    #Here I am making all the layer of the last layer to be non trainable
   # for layer in m.layers[:len(m.layers)-46]:
   #     layer.trainable = False
    #x = tf.keras.layers.GlobalMaxPool2D()(model)
    x = tf.keras.layers.GlobalAveragePooling2D()(m)
    #x = Flatten()(model)
    x = Dense(2048,activation='relu')(x)
    #x = Dropout(0.3)(x, training=True)
    x = Dense(1024,activation='relu')(x)
    #x = Dropout(0.3)(x, training=True)
    x = Dense(3, activation='softmax')(x)
    model = Model(in_src,x)
    opt = Adam(lr=0.0002, beta_1=0.5)
    model.compile(loss='categorical_crossentropy', optimizer=opt, metrics=['accuracy'])
    return model

model = model_dense()


# In[ ]:


# Here I am loading the testing data
path_normal = 'TEST_NORMAL/*'
path_cap = 'TEST_CAP/*'
path_covid = 'TEST_COVID/*'

normal_file = []
cap_file = []
covid_file = []


for i in glob.glob(path_normal):
    normal_file.append(i)
for i in glob.glob(path_cap):
    cap_file.append(i)
for i in glob.glob(path_covid):
    covid_file.append(i)
normal_file.sort()
cap_file.sort()
covid_file.sort()


# In[ ]:


Test_X = []
Test_Y = []

for i in normal_file:
    img = cv2.imread(i)
    Test_X.append(img)
    Test_Y.append(0)


for i in cap_file:
    img = cv2.imread(i)
    Test_X.append(img)
    Test_Y.append(1)
    
    
for i in covid_file:
    img = cv2.imread(i)
    Test_X.append(img)
    Test_Y.append(2)
    
    
    
Test_X = np.reshape(Test_X,(len(Test_X),512,512,3))
Test_Y_C = Test_Y
Test_Y = to_categorical(Test_Y)


print(Test_X.shape)
# In[ ]:


path_normal = 'NORMAL_TRAIN/*'
path_cap = 'CAP_TRAIN/*'
path_covid = 'COVID_TRAIN/*'

normal_file = []
cap_file = []
covid_file = []


for i in glob.glob(path_normal):
    normal_file.append(i)
for i in glob.glob(path_cap):
    cap_file.append(i)
for i in glob.glob(path_covid):
    covid_file.append(i)
    
normal_file.sort()
cap_file.sort()
covid_file.sort()

# def scipy_rotate(volume):
#     # define some rotation angles
#     angles = [-20, -10, -5, 5, 10, 20]
#     # pick angles at random
#     angle = random.choice(angles)
#     # rotate volume
#     volume = ndimage.rotate(volume, angle, reshape=False)
#     return volume


# In[ ]:
filepath = "saved-model-{epoch:02d}-{val_loss:.2f}.hdf5"
checkpoint = ModelCheckpoint(filepath, monitor='val_loss', verbose=1, save_best_only=False, mode='max')
callbacks_list = [checkpoint]





Temp = 5
for i in range(Temp):
    Train_X = []
    Train_Y = []
    for i in cap_file:
        img = cv2.imread(i)
        Train_X.append(img)
        Train_Y.append(1)

    ix = randint(0, len(covid_file), 2500)
    print(ix.shape)
    print(ix[0])

    for i in ix:
        path = covid_file[i]
        img = cv2.imread(path)
        Train_X.append(img)
        Train_Y.append(2)

    ix = randint(0, len(normal_file), 2500)    

    for i in ix:
        path = normal_file[i]
        img = cv2.imread(path)
        Train_X.append(img)
        Train_Y.append(0)
    Train_Y = to_categorical(Train_Y)
    Train_X = np.reshape(Train_X,(len(Train_X),512,512,3))

    model.fit(
        Train_X,
        Train_Y,
        validation_data=(Test_X,Test_Y),
        batch_size=10,
        epochs=5,
        callbacks=callbacks_list,
        shuffle=True,
    )
    del Train_X
    del Train_Y

k=0
predict = []
for i in Test_X:
    img = np.reshape(i,(1,512,512,3))
    pred = model(img)
    pred = argmax(pred)
    predict.append(pred)
    print(pred,"   actual = ",Test_Y[k])
   
    k+=1

matrix = confusion_matrix(Test_Y_C, predict, labels=[0, 1, 2])
print(matrix)



model.save('ICASSP_3way10.h5')
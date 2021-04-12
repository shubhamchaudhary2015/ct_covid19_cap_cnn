#!/usr/bin/env python
# coding: utf-8

# In[ ]:


#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
import pydicom as dicom
from numpy import savez_compressed
import os
import scipy.ndimage
import matplotlib.pyplot as plt
import matplotlib.pyplot as plt
import pylibjpeg
import cv2
import mpl_toolkits
import re
import pylibjpeg

from keras.optimizers import Adam
from numpy.random import randint
from keras.models import Model
from keras.models import Input
from keras.layers import Dense
from keras.utils import np_utils
from matplotlib import pyplot
from keras.layers import Activation
from keras.layers import Concatenate
from numpy import load
from keras.layers import BatchNormalization
from keras.layers import LeakyReLU
from keras.utils.vis_utils import plot_model
import tensorflow as tf
import numpy as np
from keras.utils import to_categorical
from tensorflow.keras.applications import EfficientNetB5
import sys
import matplotlib.pyplot as plt
from numpy import argmax
import pandas as pd


def weighted_majority_vote(pred,name):
    covid=0
    normal=0
    cap=0
    if(len(pred)>=80):  
        for i in pred[0:20]:
            if(i==2):
                covid+=0.7
            elif(i==1):
                cap+=0.7
            else:
                normal+=0.5
        for i in pred[60:]:
            if(i==2):
                covid+=0.7
            elif(i==1):
                cap+=0.7
            else:
                normal+=0.5
        for j in pred[20:60]:
            if(j==2):
                covid+=1
            elif(j==1):
                cap+=1
            else:
                normal+=1          
    else:
        for i in pred[:10]:
                if(i==2):
                    covid+=0.7
                elif(i==1):
                    cap+=0.7
                else:
                    normal+=0.5
        for i in pred[30:]:
            if(i==2):
                covid+=0.7
            elif(i==1):
                cap+=0.7
            else:
                normal+=0.5
        
        for j in pred[10:30]:
            if(j==2):
                covid+=1
            elif(j==1):
                cap+=1
            else:
                normal+=1
        
    if(covid>= normal and covid>=cap):
                label = 'COVID-19'
    elif(cap>covid and cap>normal):
        label = 'CAP'
    else:
        label = 'NORMAL'
    print(name,"  Prediction class = ",label,"  Covid = ",covid," cap=",cap,"  normal",normal)
    return label
            
# In[2]:


#Load the path of folder that contains test patient
INPUT_FOLDER_test1 = 'test-directory-folder/'  
patients_test1 = os.listdir(INPUT_FOLDER_test1)
patients_test1.sort()


# Load the scans in given folder path
def load_scan(path):
  slices = [dicom.read_file(path + '/' + s) for s in os.listdir(path)]
  slices.sort(key = lambda x: float(x.ImagePositionPatient[2]))
  try:
    slice_thickness = np.abs(slices[0].ImagePositionPatient[2] - slices[1].ImagePositionPatient[2])
  except:
    slice_thickness = np.abs(slices[0].SliceLocation - slices[1].SliceLocation)

  for s in slices:
    s.SliceThickness = slice_thickness
  return slices

def get_pixels_hu(slices):
  image = np.stack([s.pixel_array for s in slices])
  # should be possible as values should always be low enough(<32k)
  image = image.astype(np.int16)

  #Set outside-of-scan pixels to 0
  # The intercept is usually -1024, so air is approximately 0 
  image[image == -2000] = 0

  #Convert to Hounsfield units(HU)
  for slice_number in range(len(slices)):

    intercept = slices[slice_number].RescaleIntercept
    slope = slices[slice_number].RescaleSlope

    if slope != 1:
      image[slice_number] = slope * image[slice_number].astype(np.float64)
      image[slice_number] = image[slice_number].astype(np.int16)

    image[slice_number] += np.int16(intercept)

  return np.array(image, dtype=np.int16)




def three_channel(img):
    #img = cv2.resize(img,(256,256))
    img = np.stack((img,)*3, axis=-1)
    
    return img

def normalize(volume):
    """Normalize the volume"""
    min = -1150
    max = 150
    volume[volume < min] = min
    volume[volume > max] = max
    volume = (volume - min) / (max - min)
    volume = (volume*255).astype("uint8")
    return volume


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
    x = Dense(1024,activation='relu')(x)
    x = Dense(3, activation='softmax')(x)
    model = Model(in_src,x)
    model.load_weights("saved-model-05-0.35.hdf5")
    opt = Adam(lr=0.0002, beta_1=0.5)
    model.compile(loss='categorical_crossentropy', optimizer=opt, metrics=['accuracy'])
    return model
model = model_dense()


# In[ ]:

data = []
for i in range(len(patients_test1)):
    

    first_patient = load_scan(INPUT_FOLDER_test1 + patients_test1[i])
    first_patient_pixels = get_pixels_hu(first_patient)
    patient_slice = first_patient_pixels
    patient_name = str(patients_test1[i])
    #folder = path_test1+patient_name
    mid = int(len(patient_slice)/2)
    if(len(patient_slice) > 80):
        start = mid-40
        end = mid+40
    else:
        start = mid-20
        end = mid-20
    covid = 0
    cap = 0
    normal = 0
    prediction = []
    for j in range(start,end):
        img = patient_slice[j-1]
        img = normalize(img)
        img = three_channel(img)
        img = np.reshape(img,(1,512,512,3))
        pred = argmax(model(img))
        prediction.append(pred)
    label = weighted_majority_vote(prediction,patient_name)
    data.append([patient_name,label])
df = pd.DataFrame(data, columns = ['Patient name', 'Labels'])
df.to_csv('Test_result.csv')


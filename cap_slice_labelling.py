#!/usr/bin/env python
# coding: utf-8

# In[ ]:


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

from matplotlib import pyplot as plt
import glob


from keras.optimizers import Adam
from keras.initializers import RandomNormal
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
import cv2
from tensorflow.keras.applications import EfficientNetB5,DenseNet121
from sklearn.model_selection import train_test_split



#This mean was calculated while training the model infection classification for CAP
mean = [[[0.03516484, 0.03516484 ,0.03516484]]]
covid_image = []
path1 = "cap26to60/*"

for file in glob.glob(path1):
    covid_image.append(file)
covid_image.sort()
print(len(covid_image))

print(covid_image[0])
patient_name = covid_image[0].split('/')[1].split('_')[0]
patient_number = covid_image[0].split('/')[1].split('_')[1].split('.')[0]
print(patient_name,"  ",patient_number)


# In[ ]:


covid_19 = []
for i in covid_image:
    img = cv2.imread(i)
    covid_19.append(img)
print(len(covid_19))


# In[ ]:

if not os.path.exists('cap26to_label'):
    os.makedirs('cap26to_label')
    
if not os.path.exists('cap26to_normal_label'):
    os.makedirs('cap26to_normal_label')
model = keras.models.load_model('cap_label.h5')

k=0
for i in covid_19:
    #print(img)
    img = np.array(i)
    img = (img/255).astype('float32')
    img = img-mean
    img = np.reshape(img,(1,512,512,3))
    predict = model(img)
    class_one = predict.numpy() > 0.5
    if(class_one):
        patient_name = covid_image[k].split('/')[1].split('_')[0]
        patient_number = covid_image[k].split('/')[1].split('_')[1].split('.')[0]
        filename = 'cap21to_label/'+patient_name+'_'+str(patient_number)+'.jpg'
        cv2.imwrite(filename,i)
        print("cap",k+1)
    else:
        patient_name = covid_image[k].split('/')[1].split('_')[0]
        patient_number = covid_image[k].split('/')[1].split('_')[1].split('.')[0]
        filename = 'cap21to_normal_label/'+patient_name+'_'+str(patient_number)+'.jpg'
        cv2.imwrite(filename,i)
        print("Noncap",k+1)
    print(k)
    k+=1
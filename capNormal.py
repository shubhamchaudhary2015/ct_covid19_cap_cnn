#!/usr/bin/env python
# coding: utf-8

# In[ ]:


#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import cv2

from numpy import load
from numpy import zeros
from numpy import ones
from PIL import Image

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


# In[2]:


covid_image = []
normal = []
path1 = "cap/*"
path2 = "normal/*"

for file in glob.glob(path1):
    covid_image.append(file)
for file in glob.glob(path2):
    normal.append(file)
covid_image.sort()
normal.sort()


# In[6]:


covid19 = []
Y = []
for i in covid_image:
    img = cv2.imread(i)
    covid19.append(img)
    Y.append(1)


# In[7]:


# choose random instances
ix = randint(0, len(normal), len(covid19))
print(ix.shape)
print(ix[0])
# retrieve selected images
for i in ix:
    img = normal[i]
    img = cv2.imread(img)
    covid19.append(img)
    Y.append(0)


# In[ ]:
datagen = ImageDataGenerator(featurewise_center=True)
covid19 = np.array(covid19)
covid19 = (covid19/255).astype('float32')
datagen.fit(covid19)
mean = datagen.mean
covid19 = covid19-mean
covid19 = np.reshape(covid19,(len(covid19),512,512,3))
print(covid19.shape)
Y = np.array(Y)
print(mean)






X_train, X_test, y_train, y_test = train_test_split(covid19, Y, test_size=0.20, random_state=42)


# In[ ]:


def model_dense(image_shape=(512,512,3)):
    in_src = Input(shape=image_shape)
    #d = BatchNormalization()(in_src)
    m = DenseNet121(include_top=False, weights='imagenet', input_shape=(512,512,3))(in_src)
    #Here I am making all the layer of the last layer to be non trainable
   # for layer in m.layers[:len(m.layers)-46]:
   #     layer.trainable = False
    #x = tf.keras.layers.GlobalMaxPool2D()(model)
    x = tf.keras.layers.GlobalAveragePooling2D()(m)
    #x = Flatten()(model)
    #x = Dense(2048,activation='relu')(x)
    x = Dense(1024,activation='relu')(x)
    x = Dense(1, activation='sigmoid')(x)
    model = Model(in_src,x)
    opt = Adam(lr=0.0002, beta_1=0.5)
    model.compile(loss='binary_crossentropy', optimizer=opt, metrics=['accuracy'])
    return model


# In[ ]:


model = model_dense()
model.fit(
    X_train,
    y_train,
    validation_data=(X_test,y_test),
    batch_size=10,
    epochs=10,
    shuffle=True,
)
model.save('cap_label.h5')


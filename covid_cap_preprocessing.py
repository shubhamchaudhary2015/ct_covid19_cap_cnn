#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np
import pandas as pd
import pydicom as dicom
from numpy import savez_compressed
import os
import scipy.ndimage
import re
import cv2





#Some constants
INPUT_FOLDER_cap = 'Cap subjects/' #This is to pick CAP patient
INPUT_FOLDER_normal = 'Normal subjects/' #This is to pick Normal subjects
INPUT_FOLDER_covid = 'COVID-19 subjects/' #This is to pick covid19 subjects
patients_cap = os.listdir(INPUT_FOLDER_cap)
patients_normal = os.listdir(INPUT_FOLDER_normal)
patients_covid = os.listdir(INPUT_FOLDER_covid)
patients_cap.sort()
patients_normal.sort()
patients_covid.sort()


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
  image = image.astype(np.int16)


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



import pandas as pd
df = pd.read_csv('Stats.csv')



# Here i am creating a dictionary and storing its corresponding label
covid19_label = {} # It contains name of Normal patient and its corresponding label
covid19_category = {}
for i in range(len(df)):
  
  patient_name = df['Folder'].iloc[i]
  category = df['Category'].iloc[i]
  x = re.search("^P.*$", patient_name)
  label = df['Annotation'].iloc[i]
  covid19_label[patient_name] = label
  covid19_category[patient_name] = category


#This will read slice wise labels
slice_dict = {} #This will contain slice wise labels already given

#This will read slice wise classification
import pandas as pd
df2 = pd.read_csv('Slice_level_labels.csv')


for row in range(len(df2)):
    patient_name = df2.iloc[row][0]  #Here i am getting all the name of the patients
    patient_name = str(patient_name)   #Converting label into string
    #print(patient_name)
    #print(list(df2.iloc[row][:]))
    slice_dict[patient_name] = list(df2.iloc[row][1:])
    
    
#This funcion will convert greayscale image into three channel    
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


import os

if not os.path.exists('covid'):
    os.makedirs('covid')
if not os.path.exists('covid_normal'):
    os.makedirs('covid_normal')
    

for i in range(1,56,1):

    first_patient = load_scan(INPUT_FOLDER_covid + patients_covid[i])
    first_patient_pixels = get_pixels_hu(first_patient)
    patient_name = patients_covid[i]
    mid = int(len(first_patient_pixels)/2)
    start = mid-40
    end = mid+40
     #Here i am reading through all the slices
    print(patient_name,"  ",start,"  ",end,"  ",len(first_patient_pixels))
    for j in range(start,end,1):
        slice1 = first_patient_pixels[j]
        slice_label = slice_dict[patient_name]
        if(slice_label[j] == 1):
            slice1 = normalize(slice1)
            img = three_channel(slice1)
            filename = 'covid/'+patient_name+'_'+str(j+1)+'.jpg'
            cv2.imwrite(filename,img)
        else:
            slice1 = normalize(slice1)
            img = three_channel(slice1)
            filename = 'covid_normal/'+patient_name+'_'+str(j+1)+'.jpg'
            cv2.imwrite(filename,img)  #One means the slice shows that the patient have the covid infection    
    print(i)


# In[ ]:


if not os.path.exists('cap'):
    os.makedirs('cap')
if not os.path.exists('cap_normal'):
    os.makedirs('cap_normal')

for i in range(1,26,1):

    first_patient = load_scan(INPUT_FOLDER_cap + patients_cap[i])
    first_patient_pixels = get_pixels_hu(first_patient)
    patient_name = patients_cap[i]
    mid = int(len(first_patient_pixels)/2)
    start = mid-40
    end = mid+40
     #Here i am reading through all the slices
    for j in range(start,end,1):
        slice1 = first_patient_pixels[j]
        slice_label = slice_dict[patient_name]
        if(slice_label[j] == 1):
            slice1 = normalize(slice1)
            img = three_channel(slice1)
            filename = 'cap/'+patient_name+'_'+str(j+1)+'.jpg'
            cv2.imwrite(filename,img)
        else:
            slice1 = normalize(slice1)
            img = three_channel(slice1)
            filename = 'cap_normal/'+patient_name+'_'+str(j+1)+'.jpg'
            cv2.imwrite(filename,img)  #One means the slice shows that the patient have the covid infection
    print(i)
    
 
    
    
if not os.path.exists('normal'):
    os.makedirs('normal')

for i in range(1,len(patients_normal),1):

    first_patient = load_scan(INPUT_FOLDER_normal + patients_normal[i])
    first_patient_pixels = get_pixels_hu(first_patient)
    patient_name = patients_normal[i]
     #Here i am reading through all the slices
    for j in range(len(first_patient_pixels)):
        slice1 = first_patient_pixels[j]
        slice1 = normalize(slice1)
        img = three_channel(slice1)
        filename = 'normal/'+patient_name+'_'+str(j+1)+'.jpg'
        cv2.imwrite(filename,img)
   
    print(i)
    
   
    
if not os.path.exists('covid56to171'):
    os.makedirs('covid56to171')

for i in range(56,len(patients_covid),1):

    first_patient = load_scan(INPUT_FOLDER_covid + patients_covid[i])
    first_patient_pixels = get_pixels_hu(first_patient)
    patient_name = patients_covid[i]
    mid = int(len(first_patient_pixels)/2)
    start = mid-40
    end = mid+40
     #Here i am reading through all the slices
    for j in range(start,end,1):
        slice1 = first_patient_pixels[j]
        slice1 = normalize(slice1)
        img = three_channel(slice1)
        filename = 'covid56to171/'+patient_name+'_'+str(j+1)+'.jpg'
        cv2.imwrite(filename,img)
   
    print(i,"  covid")

    
if not os.path.exists('cap26to60'):
    os.makedirs('cap26to60')

for i in range(26,len(patients_cap),1):

    first_patient = load_scan(INPUT_FOLDER_cap + patients_cap[i])
    first_patient_pixels = get_pixels_hu(first_patient)
    patient_name = patients_cap[i]
    mid = int(len(first_patient_pixels)/2)
    start = mid-40
    end = mid+40
     #Here i am reading through all the slices
    for j in range(start,end,1):
        slice1 = first_patient_pixels[j]
        slice1 = normalize(slice1)
        img = three_channel(slice1)
        filename = 'cap26to60/'+patient_name+'_'+str(j+1)+'.jpg'
        cv2.imwrite(filename,img)
   
    print(i,"  cap")
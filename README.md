# DETECTING COVID-19 AND COMMUNITY ACQUIRED PNEUMONIA USING CHEST CT SCAN IMAGES WITH DEEP LEARNING






### Dependencies Needed:
* python
* Tensorflow
* Keras
* Numpy
* Sklearn
* Pandas
* Pylibjpeg
* Open cv
* Pydicom
* Os
* sys

Paper:https://arxiv.org/abs/2104.05121

### Dataset required
SPGC-COVID dataset


### To get result on Test set which is given:
Run Test_main.py, it contains the path for folder("test-directory-folder") which is suppose to contain ct scan of each patients(one subfolder folder for each patient), it will return the csv file which will contain prediction of all the patient.
To run Test_main.py we need pretrained model, which can be downloded from https://drive.google.com/file/d/1aBSl44i927SQpACQ7gKPL8xONAq7gOxr/view?usp=sharing



### Steps to be followed to run the entire experiment again.
#### Step1: Preprocessing Run:covid_cap_preprocessing.py(stage 1)
* Create a folder with jpeg images of ct slices of 55 covid and 25 cap patient
with infection only with name ‘covid’ and ‘cap’ folder respectively.
* Create a folder of 116 covid patients(name = ‘covid56to171’) and 35 cap
patients(name = ‘cap26to60’) whose slice level infection is not given.
* To run this we need COVID-19,CAP and Normal subjects in the same
directory in which the covid_cap_preprocessing.py will be placed.
#### Step2(stage 1):
* A. Run covidNormal.py
    * a. This will train the densenet121 as mentioned in report for classifying
       infection and non infection slice of covid19 patient 
    * b. It will take the input as folders which will be created in step 1 
* B. Run capNormal.py 
    * a. This will train the densnet121 as mentioned in report for classifying
       infection and non infection slice of CAP patient
#### Step3(stage 1):
* A. Run covid_slice_labelling.py 
    * a. This will use the model trained in step2(A),to select only the infection
       slices from CT scan of covid patient(‘covid56to171’) into the folder
       covid56to_label(This will be created when we run the script covid_slice_labelling.py)
* B. Run cap_slice_labelling.py
    * a. This will use the model trained in step2(B),to select only the infection slice from CT scan of CAP patient(‘cap26to60’) into the folder
       cap26to_label(This will be created when we run the script cap_slice_labelling.py)
#### Step4(stage 1):
* Combine all the infection slices for covid patient = ‘covid’(created in step1)+covid56to_label(created in step3 a)
* Combine all the infection slices for CAP patient = ‘cap’(created in step1)+cap26to_label(created in step3 b)
#### Step5(stage 2):
* Here we can split the CT image patient wise in test and train and save them in different
folder. Split the dataset into 90% train and 10% test for patient of each category.
* Run the Training_model.py
   *  This will save the weights for the model at different epoch, use the model with minimum validation loss.
   *  This is the final model which can be used to classify the patient.

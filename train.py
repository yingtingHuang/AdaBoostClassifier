
# coding: utf-8

# In[1]:


import os
import cv2
import numpy as np
import pandas as pd
import random  
import pickle
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
import sys
sys.path.append(r"D:\ML2017-lab-03-master")
import ensemble
import feature


# In[ ]:


def getDataX(path):
    """
    Inputs:
    - path  : the path of the image
    
    Onputs:
    - dataSetX :  a list indicating characteristic data
    """    
    
    newSize=[24,24]
    dataSetX=[]
    for filename in os.listdir(path):           # the parameter of  listdir is the path of the image
        #print ( path+ filename )                # print the path of every file
        img = cv2.imread(path+"\\"+ filename,cv2.IMREAD_GRAYSCALE)   # read the grayscale image   
        if img is None:         
            continue
        res1= cv2.resize(img,(newSize[0],newSize[1])) # resize the img to 24 * 24 
        #res1_1 = res1.reshape(1,24*24)/255   # 2D -> 1D ; norming
        NPD=feature.NPDFeature(res1)   # Extract features using the NPDFeature class in feature.py. 
        feat=NPD.extract()             # extract NPD features
        res2 = feat.tolist()           # matrix -> list
        dataSetX.append(res2)          # append new list to the exisiting list  

    return dataSetX
    
# Read the images ,extract NPD feature
dataX=[]
dataY=[]
# when x is a face image , y is equal to 1
face_x = getDataX(path=r"D:\ML2017-lab-03-master\datasets\original\face")
dataX.append(face_x)
dataY.append(np.ones(len(face_x))) 
# when x is a nonface image , y is equal to -1
nonface_x = getDataX(path=r"D:\ML2017-lab-03-master\datasets\original\nonface")
dataX.append(nonface_x)
dataY.append(-1 * np.ones(len(nonface_x))) 

# Save the data
with open('dataX.pickle', 'wb') as f:
    # Pickle the 'data' dictionary using the highest protocol available.
    pickle.dump(dataX, f, pickle.HIGHEST_PROTOCOL)
with open('dataY.pickle', 'wb') as f:
    # Pickle the 'data' dictionary using the highest protocol available.
    pickle.dump(dataY, f, pickle.HIGHEST_PROTOCOL)


# In[2]:


# Load the data
with open('dataX.pickle', 'rb') as f:
    # reads the characteristic data from cache
    dataX = pickle.load(f)
with open('dataY.pickle', 'rb') as f:
    # reads the label data from cache
    dataY = pickle.load(f)
  
# Data preprocessing
# list -> array
dataX = np.array(dataX) 
dataY = np.array(dataY)  
# axis=0 means the array of the corresponding columns is spliced horizontally
# axis=1 means the array of the corresponding rows is spliced vertically
dataX = np.concatenate((dataX[0],dataX[1]),axis=0) 
dataY = np.concatenate((dataY[0],dataY[1]),axis=0) 
dataY = dataY.reshape((len(dataY),1)) #make sure the shape of the label data is (n_samples,1).

# Devide dataset
x_train, x_validation, y_train, y_validation = train_test_split(dataX, dataY, test_size=0.2, random_state=42)


# In[3]:


# Training model
model = ensemble.AdaBoostClassifier( weak_classifier=DecisionTreeClassifier, n_weakers_limit=40)
model.fit(X=x_train,y=y_train)


# In[4]:


# Predict
y_train_pred=model.predict(X=x_train,threshold=0)
y_validation_pred=model.predict(X=x_validation,threshold=0)


# In[12]:


# Verify the accuracy on the validation set 
target_names = ['nonface', 'face']
report=classification_report(y_validation, y_validation_pred, target_names=target_names)
print(report)
# Writes the predicted result to report.txt .
f = open('report.txt', 'w')
f.write(report)
f.close()


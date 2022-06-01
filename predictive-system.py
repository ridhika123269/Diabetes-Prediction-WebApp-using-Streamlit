# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

import numpy as np
import pickle

#loading the saved model
loaded_model=pickle.load(open('C:/Users/Lenovo/Desktop/Deploying Machine Learning Model/trained_model.sav', 'rb'))

input_data=(7,196,90,0,0,39.8,0.451,41)

#changing the data into numpy array
input_data_as_nparray=np.asarray(input_data)

#reshaping the data since there is only one instance
input_data_reshaped=input_data_as_nparray.reshape(1,-1)

prediction=loaded_model.predict(input_data_reshaped)
print(prediction)

if(prediction==0):
  print("Non Diabetic")
else:
  print('Diabetic')
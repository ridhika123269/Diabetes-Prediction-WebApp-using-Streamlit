# -*- coding: utf-8 -*-
"""
Created on Wed Jun  1 11:41:30 2022

@author: Lenovo
"""

import numpy as np
import pickle
import streamlit as st

#loading the saved model
loaded_model=pickle.load(open('C:/Users/Lenovo/Desktop/Deploying Machine Learning Model/trained_model.sav', 'rb'))

#creating function for prediction
def diabetes_prediction(input_data):
   
    #changing the data into numpy array
    input_data_as_nparray=np.asarray(input_data)

    #reshaping the data since there is only one instance
    input_data_reshaped=input_data_as_nparray.reshape(1,-1)

    prediction=loaded_model.predict(input_data_reshaped)
    print(prediction)

    if(prediction==0):
      return 'Non Diabetic'
    else:
      return 'Diabetic'
  
    
def main():
    
    #giving a title
    st.title('Diabetes Prediction Web App')
    
    #getting input from the user
    Pregnancies=st.text_input('No. of Pregnancies:')
    Glucose=st.text_input('Glucose level:')
    BloodPressure=st.text_input('Blood Pressure value:')
    SkinThickness=st.text_input('Skin thickness value:')
    Insulin=st.text_input('Insulin level:')
    BMI=st.text_input('BMI value:')
    DiabetesPedigreeFunction=st.text_input('Diabetes pedigree function value:')
    Age=st.text_input('Age:')
    
    #code for prediction
    diagnosis=''
    
    #making a button for prediction
    if st.button('Predict'):
        diagnosis=diabetes_prediction([Pregnancies,Glucose,BloodPressure,SkinThickness,Insulin,BMI,DiabetesPedigreeFunction,Age])
        
    st.success(diagnosis)
    
if __name__=='__main__':
    main()
    
    
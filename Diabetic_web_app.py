# -*- coding: utf-8 -*-
"""
Created on Thu Apr 20 19:38:15 2023

@author: mkond
"""
import pickle

import numpy as np

from sklearn.preprocessing import StandardScaler

import streamlit as st

#model_load = None

def diabetic_prediction(user_input_data):
    
    #global model_load
    
    model_load = pickle.load(open('C:/Users/mkond/Diabetic_ML_Model/diabetic_model.sav', 'rb'))


    # user_input_data = (1, 89, 66, 23, 94, 28.1, 0.167, 21)

    input_data = np.asarray(user_input_data)

    input_data = input_data.reshape(1,-1)

    scaler = StandardScaler()

    scaler.fit(input_data)

    scal = scaler.transform(input_data)


    predict = model_load.predict(scal)

    #print(predict)

    if predict == 0:
        return 'Patient is not diabetic' 
    else:
        return 'Patient is diabetic'
    
    
def main():
    
    st.title('Diabetic web app prediction')
    
    # getting input data from  'Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness', 'Insulin',
     #  'BMI', 'DiabetesPedigreeFunction', 'Age', 
    
    Pregnancies = st.text_input('Number of pregnancies')
    Glucose = st.text_input('Glucose_levels')
    BloodPressure = st.text_input('Enter bloodpressure value')
    SkinThickness = st.text_input('Enter SKinthickness value')
    Insulin = st.text_input('Insulin')
    BMI = st.text_input('Enter BMI')
    DiabetesPedigreeFunction = st.text_input('DiabetesPedigreeFunction')
    Age = st.text_input('Enter the Age')
    
    # code for the prediction
    
    diagnosis = ''
    
    
    # create a button
    
    
    
    if st.button('Diabetes test result'):
        diagnosis = diabetic_prediction([Pregnancies, Glucose, BloodPressure, SkinThickness, Insulin, BMI, DiabetesPedigreeFunction, Age,])
                
    st.success(diagnosis) 
   
   
if __name__=='__main__':
    main()




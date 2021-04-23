
###################################################
# Making Prediction with the Neural Network

# Author: Michael Holdbrooke (mickybroox@gmail.com)

# Date: 20th April, 2021
###################################################


##########################
### Load Packages ###
##########################

import pandas as pd
import numpy as np

# Import tensorflow
import tensorflow as tf


## Load Model

loaded_model = tf.keras.models.load_model('saved_models/hepatitis_mortality.h5')

# Input features
inputs = [
    "Ascites",
    
    'Spiders', 
    
    'Malaise', 
    
    'Spleen Palpable', 
    
    'SGOT', 
    
    'Bilirubin', 
    
    'Protime', 
    
    'Age'
]

# Take Inputs
input_data = []

for i in range(len(inputs)):

    print('-------------------------------------')

    answer = float(input(inputs[i] + ": "))
    
    input_data.append(answer)


input_data = np.array([input_data])

# Reshape input data
input_data = np.reshape(
    input_data, 
    (
        input_data.shape[0], 1, 
        input_data.shape[1]
    )
)

# Get prediction
prediction = loaded_model.predict(input_data)


if (prediction < 0.5):
    print('#####################')
    print("DEATH SUSPECTED")
    print(prediction)
    print('#####################')

elif(prediction >= 0.5):
    print('#####################')
    print("YOU WILL SURVIVE")
    print(prediction)
    print('#####################')

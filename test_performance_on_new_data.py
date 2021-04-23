
###################################################
# Testing Classifier's performance on new data

# Author: Michael Holdbrooke (mickybroox@gmail.com)

# Date: 20th April, 2021
####################################################


###########################
### Load Packages ###
###########################

# Load the iconic trio 🔥
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

import seaborn as sns
plt.style.use('fivethirtyeight')

# Import model_selection
from sklearn import model_selection

# Import pickle
import pickle

# Set Seed
np.random.seed(81)

seed = 81


#####################
### Load Data ###
#####################

future_matrix = pd.read_csv("data/resampled_future_matrix.csv")

target_label = pd.read_csv("data/resampled_target_label.csv")

top_10_features = pd.read_csv("data/top_10_features.csv")

top_8_features = pd.read_csv("data/top_8_features.csv")

print("Data Loaded Successfully!")

#################################
## Load Serialized Objects ##
#################################

with open('saved_models/xgboost_classifier','rb') as model_file, open('saved_models/scaler_object', 'rb') as scaler_file:
    loaded_classifier = pickle.load(model_file)
    loaded_scaler = pickle.load(scaler_file)


###################################################
### Split Data into Train and Test sets ###
###################################################

X_train, X_test, y_train, y_test = model_selection.train_test_split(
    
    future_matrix[top_10_features['top_10_features'].values], 
    
    target_label, 
    
    test_size=0.3, 
    
    stratify=target_label, 
    
    random_state=seed,
)


##########################
### Get Data Sample ###
##########################

# Get Future matrix
data_sample = X_train[top_8_features['top_8_features'].values].iloc[50:100, :].copy()

### Get Targets

new_target = y_train.iloc[50:100].copy()


###########################
### Get Prediction ###
###########################

new_prediction = loaded_classifier.predict(data_sample)

### Get Predicted Probability

predicted_probability = loaded_classifier.predict_proba(data_sample)[:, 0]


###########################
### Final Data Output ###
###########################

data_sample["class"] = new_target

data_sample["predicted_outputs"] = new_prediction

data_sample["prediction_probability"] = predicted_probability


############################
## Incorrect Predictions ##
############################

incorrect_predictions = data_sample[data_sample['class'] != data_sample['predicted_outputs']]

incorrect_predictions.to_csv('data/incorrect_predictions_from_sampled_data.csv', index=False)


##############################
### Save Data to `csv` ###
##############################

data_sample.to_csv('data/data_with_predictions.csv', index=False)

print('Predictions by the classifier on new data has been saved successfully!')

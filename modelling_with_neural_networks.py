
###################################################
# Modelling with Neural Nets

# Author: Michael Holdbrooke (mickybroox@gmail.com)

# Date: 20th April, 2021
####################################################


##########################
### Load Packages ###
##########################

# Load the iconic trio 🔥
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

import seaborn as sns

plt.style.use('fivethirtyeight')

# Import tensorflow
import tensorflow as tf

# Import model_selection
from sklearn import model_selection

# Import pickle
import pickle

# Import time
import time

# Set Seed
np.random.seed(81)

tf.random.set_seed(81)

seed = 81


########################
### Load Data ###
########################

future_matrix = pd.read_csv("data/resampled_future_matrix.csv")

target_label = pd.read_csv("data/resampled_target_label.csv")

top_8_features = pd.read_csv("data/top_8_features.csv")

print("Data Loaded Successfully!")

#################################
## Load Serialized Objects ##
#################################

with open('saved_models/scaler_object', 'rb') as scaler_file:
    scaler = pickle.load(scaler_file)


###################################################
### Split Data into Train and Test sets ###
###################################################

X_train, X_test, y_train, y_test = model_selection.train_test_split(
    
    future_matrix[top_8_features['top_8_features'].values], 
    
    target_label, 
    
    test_size=0.3, 
    
    stratify=target_label, 
    
    random_state=seed,
)

###################
## Data Scaling ##
###################

# Make a copy of X_train
normalized_X_train = X_train.copy()

# Make a copy of X_test
normalized_X_test = X_test.copy()

# Initialize Columns to Scale
column_names_to_scale = ['bilirubin', 'sgot']

# Normalize X_train
normalized_X_train[column_names_to_scale] = scaler.fit_transform(normalized_X_train[column_names_to_scale])

# Normalize X_test
normalized_X_test[column_names_to_scale] = scaler.transform(normalized_X_test[column_names_to_scale])    


# Reshape Normalized_X_test
normalized_X_train = np.reshape(
    np.array(normalized_X_train),
    (
        np.array(normalized_X_train).shape[0], 1, 
        np.array(normalized_X_train).shape[1]
    )
)

# Reshape Normalized_X_test
normalized_X_test = np.reshape(
    np.array(normalized_X_test),
    (
        np.array(normalized_X_test).shape[0], 1, 
        np.array(normalized_X_test).shape[1]
    )
)


###########################################
## Create the Neural Network ##
###########################################

# Get number of features
number_of_features = len(normalized_X_test.columns)

lstm_model = tf.keras.Sequential()

lstm_model.add(tf.keras.layers.LSTM(25, input_shape=(1, number_of_features)))

lstm_model.add(tf.keras.layers.Dropout(0.1))

lstm_model.add(tf.keras.layers.Dense(1))

lstm_model.add(tf.keras.layers.Dense(1, activation='sigmoid'))

lstm_model.compile(loss="binary_crossentropy", optimizer="adam", metrics=['accuracy'])

lstm_model.summary()

print('A Neural Network has been created to be trained on the data!')


#####################################################
## Fit and Score the Neural Network ##
#####################################################

history = lstm_model.fit(
    normalized_X_train, 
    y_train, 
    epochs=300, 
    batch_size=10,
    validation_data=(normalized_X_test, y_test),
    verbose=2,
)

scores = lstm_model.evaluate(normalized_X_test, y_test)

print(lstm_model.metrics_names[1], scores[1] * 100)


##################################
## Plot Training Loss ##
##################################

training_loss = history.history['loss']
validation_loss = history.history['val_loss']
epochs = history.epoch

fig = plt.figure(figsize=(12, 8), dpi=300)

plt.plot(epochs, training_loss, label='Training loss', color='deepskyblue')
plt.plot(epochs, validation_loss, label='Validation loss', color='darkorange')

plt.title('Model loss', fontsize=28, pad=20)
plt.xlabel('Epochs', fontsize=20)
plt.ylabel('Loss', fontsize=20)

plt.grid(b=True, which="both", axis="both", color="grey", linewidth=0.4)

plt.legend(
    bbox_to_anchor=(1, 1), 
    loc="upper left", 
    fontsize='large', 
    frameon=True, 
    shadow=True, 
    fancybox=True
)

plt.tight_layout()

plt.savefig('figures/neural_network_loss.png', dpi=600, transparent=True)

print('Figure of the Neural Network training loss at each epoch has been saved successfully!')


########################################
## Plot Training Accuracy ##
########################################

training_accuracy = history.history['accuracy']
validation_loss = history.history['val_accuracy']
epochs = history.epoch


fig = plt.figure(figsize=(12, 8), dpi=300)

plt.plot(epochs, training_accuracy, label='Training acc', color='deepskyblue')
plt.plot(epochs, validation_loss, label='Validation acc', color='darkorange')

plt.title('Model Accuracy', fontsize=28, pad=20)
plt.xlabel('Epochs', fontsize=20)
plt.ylabel('Accuracy', fontsize=20)

plt.grid(b=True, which="both", axis="both", color="grey", linewidth=0.4)

plt.legend(
    bbox_to_anchor=(1, 1), 
    loc="upper left", 
    fontsize='large', 
    frameon=True, 
    shadow=True, 
    fancybox=True,
)

plt.tight_layout()

plt.savefig('figures/neural_network_accuracy.png', dpi=600, transparent=True)

print('Figure of the Neural Network training accuracy at each epoch has been saved successfully!')


###########################################
## Save Training History Data ##
###########################################

history_data = pd.DataFrame(history.history)

history_data['epochs'] = history.epoch

history_data.to_csv("csv_tables/neural_network_history.csv", index=False)

print('Neural Network history has been saved successfully!')


################################
## Save Neural Net ##
################################

saved_time = int(time.time())

path = f'./saved_models/{saved_time}'

# Save the model in tf format
lstm_model.save(path, save_format='tf')

# Save the model as a single HDF5 file
lstm_model.save('saved_models/hepatitis_mortality.h5')

print('Nueral Network has been saved Successfully!')

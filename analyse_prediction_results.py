
###################################################
# Analysing Prediction Results

# Author: Michael Holdbrooke (mickybroox@gmail.com)

# Date: 20th April, 2021

###################################################


##############################
### Load Packages ###
##############################

# Load the iconic trio 🔥
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

import seaborn as sns

plt.style.use('fivethirtyeight')


#####################
### Load Data ###
#####################

prediction_results = pd.read_csv('data/data_with_predictions.csv')

# Get Columns Names
col_names = prediction_results.columns[:8]


#########################
## Plot Results ##
#########################

for col_name in col_names:
    
    plt.figure(figsize=(10, 7), dpi=300)
    
    sns.scatterplot(
        x=col_name, 
        y='prediction_probability', 
        data=prediction_results, 
        hue='class',
        size='predicted_outputs',
        sizes=(90, 150),
        palette='Set2',
    )
    
    plt.title(f'{col_name.title()} Mortality Probability', fontsize=28, pad=20)
    
    plt.legend(
        bbox_to_anchor=(1, 1), 
        loc="upper left", 
        fontsize='medium', 
        frameon=True, 
        shadow=True, 
        fancybox=True
    )
    
    plt.tight_layout()
    
    plt.savefig(f'figures/{col_name.lower()}_mortality_probability.png', dpi=600, transparent=True)


###############
## Relplot ##
###############

hue_colors = {
    0: 'orangered', 
    1: 'deepskyblue',
}

markers = {
    0: 'X',
    1: 'o',
}


# Plot results
for col_name in col_names:

    sns.relplot(
        x=col_name, 
        y='prediction_probability', 
        data=prediction_results, 
        hue='class',
        size='class',
        sizes=(250, 250),
        style='class',
        col='predicted_outputs',
        markers=markers,
        palette=hue_colors,
        height=5,
        aspect=1.2,

    )
    
    plt.savefig(f'figures/relplot_of_{col_name.lower()}_mortality_probability.png', dpi=600, transparent=True)


print('Predictions from the classifier has been analyzed and the results have been saved successfully!')


#######################################################
# Predictive Analysis of Hepatitis Mortality

# Author: Michael Holdbrooke (mickybroox@gmail.com)

# Date: 20th April, 2021
#######################################################


###########################
### Load Packages ###
###########################

# Load the iconic trio 🔥
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Set plot style
plt.style.use('fivethirtyeight')

# Import seaborn for style and beauty
import seaborn as sns

# Set context 
sns.set_context('paper')

# Import model_selection
from sklearn import model_selection

# Import StandardScaler for data normalization
from sklearn.preprocessing import StandardScaler

# Import SMOTE to handle imbalance classes
from imblearn.over_sampling import SVMSMOTE

# Load models
from sklearn import tree
from sklearn import ensemble
import xgboost
from sklearn import linear_model

# custom
from custom import helper

# Load evaluation metrics
from sklearn import metrics

# Import pickle for model serialization
import pickle

# Import feature selection functions
from sklearn.feature_selection import SelectKBest, RFE, chi2

# Import lime for model interpretation
import lime
import lime.lime_tabular

# Set Seed
np.random.seed(81)

# Ignore warnings
import warnings
warnings.filterwarnings('ignore')


#####################
### Load Data ###
#####################

clean_data = pd.read_csv("data/cleaned_data.csv")

print("Data Loaded Successfully!")

### Create a Copy of the Loaded Data

data_with_targets = clean_data.copy()


###################################
## Data Preprocessing ##
###################################

### Split the Data into Feature Matrix and Target Label

target_variable = 'class'

# Unscaled Features
X = data_with_targets.drop([target_variable], axis=1)

# Target Variable
y = data_with_targets[target_variable]

print('Data has been splitted into fature matrix and target label')


#################################################################
### `SMOTE` Sampling to deal with imbalance classes ###
#################################################################

# Set Seed Value
seed = 81

smote = SVMSMOTE(random_state=seed)

resampled_X, resampled_y = smote.fit_resample(X, y)

# Save Resampled Data
resampled_X.to_csv('data/resampled_future_matrix.csv', index=False) 

resampled_y.to_csv('data/resampled_target_label.csv', index=False)

print("Data has been resampled using SMOTE and saved successfully!")


#############################
### Feature Selection ###
#############################

#### Use `SelectKBest()`

select_k_best = SelectKBest(score_func=chi2, k=10)

best_features = select_k_best.fit(resampled_X, resampled_y)

best_features_values = best_features.transform(resampled_X)

# Get Scores
feature_scores = pd.DataFrame(best_features.scores_, columns=['feature_scores'])

# Get Features
feature_column_names = pd.DataFrame(resampled_X.columns, columns=['feature_name'])

# Add Scores and Features into a single DataFrame
best_features_df = pd.concat([feature_scores, feature_column_names], axis=1)

# Save DataFrame
best_features_df.to_csv("csv_tables/feature_scores_using_selectkbest.csv", index=True)

print("Feature scores using SelectKBest saved successfully!")

#### `SelectKBest` Top 10 Features

best_ten_features_using_selectkbest = best_features_df.nlargest(10,'feature_scores')

#### Get Names of Top 10 Features

top_10_features = best_ten_features_using_selectkbest['feature_name'].values

# Save top 10 features from SelectKBest to csv
pd.DataFrame(top_10_features, columns=['top_10_features']).to_csv("data/top_10_features.csv", index=False)

###################################################
### Split Data into Train and Test sets ###
###################################################

X_train, X_test, y_train, y_test = model_selection.train_test_split(
    
    resampled_X[top_10_features], 
    
    resampled_y, 
    
    test_size=0.3, 
    
    stratify=resampled_y, 
    
    random_state=seed,
)

print('Data has been splitted into train and test sets')


########################################
### Scale Train and Test sets ###
########################################

# Instantiate Scaler Object
scaler = StandardScaler()

# Make a copy of X_train
normalized_X_train = X_train.copy()

# Make a copy of X_test
normalized_X_test = X_test.copy()

# Initialize Columns to Scale
column_names_to_scale = ['bilirubin', 'alk_phosphate', 'sgot']

# Normalize X_train
normalized_X_train[column_names_to_scale] = scaler.fit_transform(normalized_X_train[column_names_to_scale])

# Normalize X_test
normalized_X_test[column_names_to_scale] = scaler.transform(normalized_X_test[column_names_to_scale])

print("Selected Data features have been scaled successfully!")


#####################
## Modelling ##
#####################

# Instantiate baseline models
models = [
    ("DecisionTree", tree.DecisionTreeClassifier(random_state=seed)),
    ("RandomForest", ensemble.RandomForestClassifier(random_state=seed)),
    ("AdaBoost", ensemble.AdaBoostClassifier(random_state=seed)),
    ("ExtraTree", ensemble.ExtraTreesClassifier(random_state=seed)),
    ("GradientBoosting", ensemble.GradientBoostingClassifier(random_state=seed)),
    ("XGBOOST", xgboost.XGBClassifier(random_state=seed)),
]


###########################################################
### Cross-Validate Models on all Data Features ###
###########################################################

# Split data into 10 folds
cv_kfold = model_selection.KFold(n_splits=10, shuffle=True, random_state=seed)

scorer = "f1"

# Instantiate model_names as an empty list to keep the names of the models
model_names = []

# Instantiate cv_mean_scores as an empty list to keep the cross validation mean score of each model
cv_mean_scores = []

# Instantiate cv_std_scores as an empty list to keep the cross validation standard deviation score of each model
cv_std_scores = []

# Loop through the baseline models and cross validating each model
for model_name, model in models:
    model_scores = model_selection.cross_val_score(
        model, resampled_X, resampled_y, cv=cv_kfold, scoring=scorer, n_jobs=-1, verbose=1,
    )
    
    print(
        f"{model_name} Score: %0.2f (+/- %0.2f)"
        % (model_scores.mean(), model_scores.std() * 2)
    )

    # Append model names to model_name
    model_names.append(model_name)
    
    # Append cross validation mean score of each model to cv_mean_score
    cv_mean_scores.append(model_scores.mean())
    
    # Append cross validation standard deviation score of each model to cv_std_score
    cv_std_scores.append(model_scores.std())


# Parse model_names, cv_mean_scores and cv_std_scores and a pandas DataFrame object
cv_results = pd.DataFrame({"model_name": model_names, "mean_score": cv_mean_scores, "std_score": cv_std_scores})

# Sort the Dataframe in descending order
cv_results.sort_values("mean_score", ascending=False, inplace=True,)

# Save the DataFrame as a csv file
cv_results.to_csv("csv_tables/cross_validation_results.csv", index=True)

print('Models have been cross-validated and results have been saved successfully!')


#########################################################################
### Fit Models on `SelectKBest` Normalized Selected Features ###
#########################################################################

feature_importance_of_models, df_model_features_with_importance, model_summary = helper.baseline_performance(
    
    models=models,
    
    X_train=normalized_X_train, 
    
    y_train=y_train, 
    
    X_test=normalized_X_test, 
    
    y_test=y_test, 
    
    column_names=list(X_train.columns.values), 
    
    csv_path='csv_tables', 
    
    save_model_summary=True, 
    
    save_feature_importance=True, 
    
    save_feature_imp_of_each_model=True,
)

print('Models have been fitted on SelectKBest Normalized Selected Features')


#########################################
### Modelling on Top 8 Features ###
#########################################

# Select top 8 features
top_8_features = list(feature_importance_of_models['XGBOOST'].head(8))

# Save top 8 features to csv
pd.DataFrame(top_8_features, columns=['top_8_features']).to_csv("data/top_8_features.csv", index=False)

# Select X_train and X_test from the Top 8 models
X_train_new = normalized_X_train[top_8_features]

X_test_new = normalized_X_test[top_8_features]

print('Top 8 features of the best performing model have been selected')


#####################################
### Choose Model to Evaluate ###
#####################################

classifier = models[5][1]

classifier.fit(X_train_new, y_train)

print('Best performing model has been selected to evaluate.')


################################################################
### Get Train and Test Accuracy of the Chosen Model ###
################################################################

train_accuracy = classifier.score(X_train_new, y_train)

test_accuracy = classifier.score(X_test_new, y_test)

print(f"Train accuracy: {train_accuracy}")

print(f"Test accuracy: {test_accuracy}")


################################
### Evaluate Classifier ###
################################

# Get prediction
y_pred = classifier.predict(X_test_new)

# Get prediction probability
probabilities = classifier.predict_proba(X_test_new)

y_proba = probabilities[:, 1]

# Get accuracy score
test_accuracy_score = metrics.accuracy_score(y_test, y_pred)

# Get precision score
precision = metrics.precision_score(y_test, y_pred)

# Get recall score
recall = metrics.recall_score(y_test, y_pred)

# Get f1 score
f1_score = metrics.f1_score(y_test, y_pred)


####################################
## Write scores to a text file ##
#####################################

with open("metrics.txt", 'w') as output_text_file:
        
        output_text_file.write(f"Training Accuracy: {round(train_accuracy, 4)}\n")
        
        output_text_file.write(f"Test Accuracy: {round(test_accuracy_score, 4)}\n")
        
        output_text_file.write(f"Precision Score: {round(precision, 4)}\n")
        
        output_text_file.write(f"Recall Score: {round(recall, 4)}\n")
        
        output_text_file.write(f"F1 Score: {round(f1_score, 4)}\n")


#################################
### Plot Confusion Matrix ###
#################################

cm_output = metrics.confusion_matrix(y_test, y_pred)

# Convert to dataframe
cm_df = pd.DataFrame(cm_output)

fig, ax = plt.subplots(figsize=(7, 7), dpi=300)

sns.heatmap(
    cm_df, 
    annot=True, 
    square=True,
    annot_kws={"size": 18}, 
    cmap='Blues', 
    fmt='d', 
    linewidths=2, 
    linecolor="darkorange", 
    cbar=False, 
    xticklabels=[0, 1], 
    yticklabels=[0, 1]
)

plt.title("Confusion Matrix", fontsize=25, pad=20)

plt.xlabel("Predicted Label", fontsize=18, labelpad=3)
plt.xticks(fontsize=15)

plt.ylabel("Actual Label", fontsize=18)
plt.yticks(fontsize=15)

ax.text(2.25, -0.10,'Accuracy: '+str(round(test_accuracy_score, 3)), fontsize=14)

ax.text(2.25, 0.0,'Precision: '+str(round(precision, 4)), fontsize=14)

ax.text(2.25, 0.1,'Recall: '+str(round(recall, 4)), fontsize=14)

ax.text(2.25, 0.2,'F1 Score: '+str(round(f1_score, 4)), fontsize=14)

fig.tight_layout()

plt.savefig('confusion_matrix_plot.png', dpi=600, transparent=False)


######################
### Plot ROC_Curve ###
######################

false_positive_rate, true_positive_rate, thresholds = metrics.roc_curve(y_test, y_proba)

roc_score = metrics.roc_auc_score(y_test, y_proba)

fig = plt.figure(figsize=(12, 10), dpi=300)

plt.plot([0, 1], [0, 1], 'k--')

plt.plot(false_positive_rate, true_positive_rate, color='darkorange')

plt.fill_between(false_positive_rate, true_positive_rate, alpha=0.2, color='orange')

plt.title(f'ROC Curve - AUC : {round(roc_score, 3)}', fontsize=25, pad=20)

plt.xlabel('False Positive Rate', fontsize=20, labelpad=5)
plt.xticks(fontsize=20)

plt.ylabel('True Positive Rate', fontsize=20, labelpad=5)
plt.yticks(fontsize=20)

plt.grid(color='grey')

fig.tight_layout()

plt.savefig('ROC_Curve.png', dpi=600, transparent=False)


######################
### Plot Residuals ###
######################

fig = plt.figure(figsize=(7, 7), dpi=300)

y_pred_ = classifier.predict(X_test_new) + np.random.normal(0, 0.25, len(y_test))

y_jitter = y_test + np.random.normal(0, 0.25, len(y_test))

residuals_df = pd.DataFrame(list(zip(y_jitter, y_pred_)), columns=["Actual Label", "Predicted Label"])

ax = sns.scatterplot(
    
    x="Actual Label", 
    
    y="Predicted Label",
    
    data=residuals_df,
    
    facecolor='dodgerblue',
    
    linewidth=1.5,
)
    
ax.set_xlabel('Actual Label', fontsize=14) 

ax.set_ylabel('Predicted Label', fontsize=14)#ylabel

ax.set_title('Residuals', fontsize=20)

min = residuals_df["Predicted Label"].min()

max = residuals_df["Predicted Label"].max()

ax.plot([min, max], [min, max], color='black', linewidth=1)

plt.tight_layout()

plt.savefig('Residuals_plot.png', dpi=600, transparent=False)

print('Best performing model has been evaluated and the evaluation results have been saved successfully!')


#####################################
## Intepret with Lime ##
#####################################

target_names = ['Die', 'Live']

feature_names = X_test_new.columns

class_names = ["Die(0)", "Live(1)"]

number_of_features = len(feature_names)

X_test_sample = X_test_new.iloc[1]

sample_label = y_test.iloc[1]


##############################
## Create Explainer ##
##############################

explainer = lime.lime_tabular.LimeTabularExplainer(
    
    training_data=X_train_new.values, 
    
    feature_names=feature_names, 
    
    class_names=class_names, 
    
    discretize_continuous=True, 
    
    random_state=81,
)

explainer_instance = explainer.explain_instance(
    
    data_row=X_test_sample,
    
    predict_fn=classifier.predict_proba,
    
    num_features=number_of_features, 
    
    top_labels=0
)

print("An explainer has been initialized to interpret the model's predictions")


#################################################
## Save Explainer Instance as a DataFrame ##
#################################################

explainer_df = pd.DataFrame(explainer_instance.as_list())

# Save to csv
explainer_df.to_csv('csv_tables/explainer_results.csv', index=False)


####################################
## Plot Explainer Instance ##
####################################

plt.figure(figsize=(25, 10), dpi=300)

explainer_instance.as_pyplot_figure(label=1)

plt.title('Local Explanation for class Live(1)', fontsize=12, pad=20)

plt.yticks(fontsize=15)

plt.grid(b=True, alpha=0.6, color='grey')

plt.tight_layout()

plt.savefig('figures/lime_tabular_explainer.png', dpi=600, transparent=True)

print("An instance of the model's prediction has been interpreted by the explainer and the results have been saved successfully!")


###############################
## Save Classifier ##
###############################

helper.create_folder('./saved_models/')

with open('saved_models/xgboost_classifier', 'wb') as file:
    pickle.dump(classifier, file)


####################################
## Save Scaler Object ##
####################################

with open('saved_models/scaler_object', 'wb') as file:
    pickle.dump(scaler, file)

print('The classifier and scaler object have been serialized successfully!')


####################################################################
# EXPLORATORY DATA ANALYSIS & DATA CLEANING of Hepatitis Data

# Author: Michael Holdbrooke (mickybroox@gmail.com)

# Date: 20th April, 2021
####################################################################


#######################
## Import Libraries ##
#######################

# The iconic trio
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from matplotlib import rcParams, font_manager

# Set plots style
plt.style.use('fivethirtyeight')

# Import seaborn for style and beauty
import seaborn as sns

# Set context
sns.set_context('paper')

# custom
from custom import helper

# Ignore warnings
import warnings
warnings.filterwarnings('ignore')


##############################
## Load Dataset ##
##############################

raw_data = pd.read_csv("data/hepatitis.data")

print('Data Loaded Successfully!')


#############################################################
## Name the Column Heads of the Dataset Appropriately ##
#############################################################

column_heads = [
    "Class",
    "AGE",
    "SEX",
    "STEROID",
    "ANTIVIRALS",
    "FATIGUE",
    "MALAISE",
    "ANOREXIA",
    "LIVER BIG",
    "LIVER FIRM",
    "SPLEEN PALPABLE",
    "SPIDERS",
    "ASCITES",
    "VARICES",
    "BILIRUBIN",
    "ALK PHOSPHATE",
    "SGOT",
    "ALBUMIN",
    "PROTIME",
    "HISTOLOGY"
]

print(f'Total number of columns: {len(column_heads)}')

## Assign column head names to the dataset 
raw_data.columns = column_heads

## Convert column head names to snakecase ##
raw_data.columns = raw_data.columns.str.lower().str.replace(' ', '_')

## Create a Copy of the Dataset

df = raw_data.copy()

## Create Folders to keep figures and tables

helper.create_folder('./csv_tables/')
helper.create_folder('./figures/')


####################################
## Treat Missing Values ##
####################################

### Missing Attribute Values: (indicated by "`?`")

# Replace `?` with `NaNs`
df.replace('?', np.nan, inplace=True)

# Get missing values
missing_values = helper.missing_data(df)
missing_values.to_csv("csv_tables/missing_values.csv", index=True)

print('Missing Values Info Saved Successfully!')

### Check Total Number of Missing Values

total_number_of_misssing_values = missing_values.loc['Total', :].sum()

print(f'Total number of missng values: {total_number_of_misssing_values}')

### Get Column Heads with Missing Values

columns_with_missing_values = list(missing_values.columns[missing_values.loc['Total', :] > 0])

### Get the Median Value of Columns with Missing Values

median_values = df[columns_with_missing_values].median()

### Replace Missing Values with Median Values

df.fillna(value=median_values, inplace=True)

print('Missing Values Treated!')


###############################################
## Get Column Names and their Data Types ##
###############################################

dataset_columns = pd.DataFrame({'column_names':list(df.columns)})

data_types = []
for column in df.columns:
    dtype = str(df[column].dtypes)
    data_types.append(dtype)

dataset_columns['data_type'] = data_types
dataset_columns.to_csv("csv_tables/column_heads_of_dataset.csv", index=True)


###############################################################
## Treat Datatypes of the Column Heads ##
###############################################################

### Convert Columns with Integer Values to the `int` type

# Get all Columns of type `object`
object_columns_to_convert_to_ints = df.columns[df.dtypes == 'object']

# Columns to Omit
columns_to_omit = ['bilirubin', 'albumin']

#### Drop Columns to omit from the list

object_columns_to_convert_to_ints = object_columns_to_convert_to_ints.drop(columns_to_omit)

#### Convert Columns with `Integer` Values to the `int` type

df[object_columns_to_convert_to_ints] = df[object_columns_to_convert_to_ints].astype(int)

#### Convert Columns with `Float` Values to the `float` type

object_columns_to_convert_to_floats = ['bilirubin', 'albumin']

df[object_columns_to_convert_to_floats] = df[object_columns_to_convert_to_floats].astype(float)


###############################################
## Check Duplicated Values ##
###############################################

print(f'The number of Data Obersvation is: {len(df)}')

total_number_of_duplicated_values = df.duplicated().sum()

print(f'Total number of duplicated values: {total_number_of_duplicated_values}')


##########################################################
## Create Another copy of the Dataset ##
##########################################################

treated_df = df.copy()


#######################################################
## Transform Categorical Columns ##
#######################################################

# Convert the "class" column head to object type
treated_df['class'].replace(
    {
        1: 'Die', 
        2: 'Live',
    }, inplace=True
)

# Convert the "sex" column head to object type
treated_df['sex'].replace(
    {
        1: 'Male', 
        2: 'Female',
    }, inplace=True
)


# Columns with binary ("yes" and "no") values
yes_no_columns = [
    'steroid', 
    'antivirals', 
    'fatigue', 
    'malaise', 
    'anorexia', 
    'liver_big', 
    'liver_firm', 
    'spleen_palpable', 
    'spiders', 
    'ascites', 
    'varices', 
    'histology',
]


# Convert binary column heads to object type
for column in yes_no_columns:
    treated_df[column].replace(
        {
            1: 'No', 
            2: 'Yes',
        }, inplace=True
    )

    
###################################################################
## Get Statistical Summary of Full Dataset ##
###################################################################

data_statistical_summary = df.describe(include='all')
data_statistical_summary.to_csv("csv_tables/data_statistical_summary.csv", index=True)


####################################################################
## Statistical Summary of Categorical Features ##
####################################################################

statistical_summary_of_categorical_columns = treated_df.describe(include=[object])
statistical_summary_of_categorical_columns.to_csv("csv_tables/statistical_summary_of_categorical_columns.csv", index=True)


##################################################################
## Statistical Summary of Numerical Features ##
##################################################################

statistical_summary_of_numerical_columns = treated_df.describe(include=[np.number])
statistical_summary_of_numerical_columns.to_csv("csv_tables/statistical_summary_of_numerical_columns.csv", index=True)


####################################################################
## Summary of Individual Categorical Columns ##
####################################################################

categorical_columns = treated_df.select_dtypes(np.object).columns.values.tolist()

print('Saving column(s) summary')

for column in categorical_columns:

    summary_df = treated_df[column].value_counts().reset_index()

    summary_df.columns = [column, 'frequency']

    percentage = (treated_df[column].value_counts() / treated_df[column].count() * 100).values.tolist()

    summary_df['percentage'] = percentage

    total_df = pd.DataFrame(summary_df.sum(axis=0).to_dict(), index=[0])

    total_df.loc[0, column] = 'Total'

    final_summary_df = pd.concat([summary_df, total_df], axis=0, ignore_index=True)

    final_summary_df.to_csv(f"csv_tables/summary_table_of_{column}.csv", index=False)
    
    print('*' * 10)

    
###################################################
## Statistical Summary Per Gender ##
###################################################

satistical_summary_per_gender = treated_df.groupby('sex').describe(include='all')

satistical_summary_per_gender = satistical_summary_per_gender.T

satistical_summary_per_gender.to_csv("csv_tables/satistical_summary_per_gender.csv", index=True)


#################################################################################
## Statistical Summary of Numerical Features per Gender ##
#################################################################################

satistical_summary_of_numerical_columns_per_gender = treated_df.groupby('sex').describe(include=[np.number])

satistical_summary_of_numerical_columns_per_gender = satistical_summary_of_numerical_columns_per_gender.T

satistical_summary_of_numerical_columns_per_gender.to_csv("csv_tables/satistical_summary_of_numerical_columns_per_gender.csv", index=True)


###################################################################################
## Statistical Summary of Categorical Features per Gender ##
###################################################################################

satistical_summary_of_categorical_columns_per_gender = treated_df.groupby('sex').describe(include=[object])

satistical_summary_of_categorical_columns_per_gender = satistical_summary_of_categorical_columns_per_gender.T

satistical_summary_of_categorical_columns_per_gender.to_csv("csv_tables/satistical_summary_of_categorical_columns_per_gender.csv", index=True)

print('All Statistical Summary Info has been Saved Successfully!')


#################################################################################################
## Replace `1s` and `2s` in the Categorical Columns with `0s` and  `1s` ##
#################################################################################################

cols = [
    'class', 
    'sex', 
    'steroid', 
    'antivirals', 
    'fatigue', 
    'malaise', 
    'anorexia', 
    'liver_big', 
    'liver_firm', 
    'spleen_palpable', 
    'spiders', 
    'ascites', 
    'varices', 
    'histology',
]


for col in cols:
    df[col].replace(
        {
            1:0, 
            2:1,
        }, inplace=True
    )


####################################
## Check Outlier Info ##
####################################

columns_to_check_for_outliers = ['age', 'bilirubin', 'alk_phosphate', 'sgot', 'albumin', 'protime']

outliers = helper.outlier_info(df[columns_to_check_for_outliers])

outliers.to_csv("csv_tables/outlier_info.csv", index=True)

### Check Total Number of Outliers

total_number_of_outliers = outliers.loc['Number of Outliers', :].sum()

print(f'Total number of outliers is: {total_number_of_outliers}')


#########################
## Detect Outliers ##
#########################

for i, column in enumerate(df[columns_to_check_for_outliers]):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 8), dpi=300, clear=False)


    df[column].hist(bins=10, ax=ax1)

    ax = sns.boxplot(x=column, data=df, ax=ax2, color='deepskyblue')
    ax = sns.stripplot(x=column, data=df, color="maroon", jitter=0.2, size=4.5)


    ax1.set_title('Distribution of ' + column, fontsize=22)
    ax2.set_title('Boxplot of ' + column, fontsize=22)



    plt.setp(ax1.get_xticklabels(), fontsize=15)
    plt.setp(ax1.get_yticklabels(), fontsize=15)

    plt.setp(ax2.get_xticklabels(), fontsize=15)
    ax2.set_xlabel(ax2.get_xlabel(), fontsize=18)

    plt.grid(b=True, axis='both', color='white', linewidth=0.5)

    fig.tight_layout()

    plt.savefig(f"figures/Outlier{i}.png", dpi=600, transparent=True)

print('Outlier Info Has Been Saved Successfully!')


##################################################
## Correlation of Dataset Features ##
##################################################

# Get Correlation betwwen target variable and data features
correlation_with_target_variable = df.select_dtypes(np.number).corr()['class'].sort_values(ascending=False)
correlation_with_target_variable = pd.DataFrame(correlation_with_target_variable)

correlation_with_target_variable.to_csv('csv_tables/correlation_between_label_and_features.csv', index=True)


#################################################################
## Plot correlation between Target Variable and Data Features ##
#################################################################

# Set font_scale
sns.set(font_scale=1.3)

ax = correlation_with_target_variable.plot(
        kind='bar',
        figsize=(12, 10),
        rot=90, 
        color=['chocolate'], 
        edgecolor='white', 
        linewidth=5,
        legend=False,
)

plt.title('Correlation Between Target Variable and Data Features', fontsize=20, pad=20)

plt.ylabel('Correlation Value', fontsize=18)

for patch in ax.patches:
    width = patch.get_width()
    height = patch.get_height()
    x, y = patch.get_xy()
    ax.annotate(f'{round(height, 2)}', (x + width/2, y + height*1.02), ha='center')

plt.xticks(fontsize=15)
plt.yticks(fontsize=15)

plt.tight_layout()

plt.savefig(f'figures/correlation_between_label_and_features.png', dpi=600, transparent=True)


################################################################
## Boxplot distribution between features and target variable ##
################################################################

# Set font_scale
sns.set(font_scale=3)

correlation_features_with_continuous_values = ['albumin', 'protime', 'sgot', 'alk_phosphate', 'age', 'bilirubin']

for correlation_feature in correlation_features_with_continuous_values:
    
    plt.figure(figsize=(9, 8), dpi=300)
    
    sns.boxplot(
        x='class', 
        y=correlation_feature, 
        data=df,
        palette="Set2",
    )
    
    plt.title(f'Boxplot of {correlation_feature.title()}', fontsize=35, pad=25)

    plt.grid(b=False, alpha=0.5, color='silver')

    plt.tight_layout()

    plt.savefig(f'figures/boxplot_distribution_of_{correlation_feature}.png', dpi=600, transparent=True)


#########################################
## Plot correlation for full dataset ##
#########################################

correlation_data = df.select_dtypes(np.number).corr()

plt.figure(figsize=(30, 40))

sns.heatmap(
    correlation_data, 
    annot=True, 
    fmt='.2g',
    center=0,
    cmap='Reds', 
    square=True,
    annot_kws={"size": 25}, 
    cbar_kws={
        'label':'correlation bar', 
        'orientation': 'horizontal', 
        'shrink':0.8,
    },
    linewidths=3, 
    linecolor='white',
)

plt.savefig(f'figures/Data_Features_Correlation.png', dpi=300, transparent=True)


###############################################################
## Highly Positive and Highly Negative Correlated Columns ##
###############################################################

highly_correlated_col = df.select_dtypes(np.number).corr()

highly_correlated_col = highly_correlated_col[
    ((highly_correlated_col > 0.52) | (highly_correlated_col < -0.3))  & (highly_correlated_col != 1.0)
]

plt.figure(figsize=(30, 40))

sns.heatmap(
    highly_correlated_col, 
    annot=True, 
    cmap='Reds', 
    square=True,
    annot_kws={"size": 30}, 
    cbar_kws={
        'label':'correlation bar', 
        'orientation': 'horizontal', 
        'shrink':0.8,
    },
    linewidths=2, 
    linecolor='black',

)

plt.savefig(f'figures/highly_correlated_columns.png', dpi=300, transparent=True)


#######################################################
## Get Data Teable of Higly Correlated Columns ##
#######################################################

correlated_columns = df.select_dtypes(np.number).corr()

correlated_columns = correlated_columns[
    ((correlated_columns > 0.52) | (correlated_columns < -0.3)) & (correlated_columns != 1.0)
]

correlated_columns_df = correlated_columns.unstack().sort_values().drop_duplicates().dropna()

correlated_columns_df = pd.DataFrame(correlated_columns)

correlated_columns_df.columns = ['correlation_value']

correlated_columns_df.to_csv('csv_tables/highly_correlated_columns.csv', index=True)


print('Correlation Info of the Dataset has been Saved Sucessfully!')


########################################################
## Relationship between `bilirubin` and `class` ##
########################################################

hue_colors = {
    0: 'orangered', 
    1: 'turquoise',
}

markers = {
    0: 'X',
    1: 'o',
}

sns.relplot(
    x='class', 
    y='bilirubin', 
    data=df, 
    hue='class',
    size='class',
    sizes=(100, 100),
    style='class',
    markers=markers,
    palette=hue_colors,
    height=7,
    aspect=1.5,
    
);

plt.title('Relationship Between Bilirubin and Class', fontsize=30, pad=20)

plt.savefig(f'figures/relationship_between_bilirubin_and_class.png', dpi=600, transparent=True)

print('Relationship info Between bilirubin and class has been saved sucessfully!')


#######################################################
## Distribution Plot of Target Variable ##
#######################################################

target_variable = 'class'

plt.figure(figsize=(15, 10), dpi=300)

sns.distplot(df[target_variable], color='darkorange', bins=10, rug=True)

plt.title(f'Distribution of {target_variable}', fontsize=35, pad=20)

plt.grid(alpha=0.6, color='grey')

plt.tight_layout()

plt.savefig(f'figures/Distribution_Plot_of_{target_variable}.png', dpi=600, transparent=True)

print('Distribution plot of the target variable has been saved sucessfully!')


###############################################
## Count Plot of Target Variable ##
###############################################

def count_plot(df, column_name, color=['deepskyblue', 'dodgerblue'], edgecolor='white', dpi=300, transparent=True, path=''):

    fig = plt.figure(figsize=(10, 8), dpi=dpi)

    ax = df[column_name].value_counts().plot(
        kind='bar', 
        rot=0, 
        color=color, 
        edgecolor=edgecolor, 
        linewidth=5,
    )
    
    plt.title(column_name.title(), fontsize=30, pad=18)
    
    for patch in ax.patches:
        width = patch.get_width()
        height = patch.get_height()
        x, y = patch.get_xy()
        ax.annotate(f'{height}', (x + width/2, y + height*1.02), ha='center')
    
    top_value = df[column_name].value_counts().max() + 10
    
    plt.ylim(bottom=0, top=top_value)
    
    plt.xticks(fontsize=25)
    
    plt.yticks(fontsize=25)
    
    plt.grid(alpha=0.5, color='grey')
    
    plt.tight_layout()
    
    plt.savefig(f'{path}/{column_name}_count_plot.png', dpi=600, transparent=transparent)
    

count_plot(
    df=df, 
    column_name=target_variable, 
    color=['cyan', 'red'],
    edgecolor='white',
    dpi=300, 
    transparent=True,
    path='figures'
)

print('Count Plot of the target variable has been saved successfully!')


#####################################################
## Pie Chart Representation of the Target Variable ##
#####################################################

def draw_pie_chart(df, column_name, labels, colors=['darkturquoise', 'silver'], dpi=300, transparent=True, path=''):

    # Get number of values that belongs to the `die` class
    class_1 = len(df[df[column_name] == 0])

    # Get number of values that belongs to the `live` class
    class_2 = len(df[df[column_name] == 1])

    plt.figure(figsize=(10, 8), dpi=dpi)

    # Sizes of each class attribute
    sizes = [class_1, class_2]

    # explode 1st slice of the pie chart
    explode = (0.1, 0)  

    # Plot pie chart
    plt.pie(
        sizes, 
        explode=explode, 
        labels=labels, 
        colors=colors, 
        autopct='%1.1f%%', 
        shadow=False, 
        startangle=90,
        wedgeprops={'linewidth': 3, 'width': 1, 'edgecolor': 'white'},
        textprops={'fontsize': 24},
    )
    
    center_circle = plt.Circle(
        (0, 0), 
        0.50, 
        fc='white'
    )
    fig = plt.gcf()
    fig.gca().add_artist(center_circle)
    
    plt.title(column_name.upper(), fontsize=25)

    # Set axis to equal
    plt.axis('equal')
    
    plt.tight_layout()

    # Save pie chart
    plt.savefig(f'{path}/pie_chart_of_{column_name}.png', dpi=600, transparent=transparent)


draw_pie_chart(
    df=df,
    column_name='class', 
    labels=['Die', 'Live'], 
    colors=['red', 'cyan'], 
    dpi=300, 
    transparent=True,
    path='figures',
)

print('Pie chart of the target plot has been saved sucessfully!')


########################################
## Get Age Distribution ##
########################################

labels = [
    "Less than 10",
    "10-20",
    "20-30",
    "30-40",
    "40-50",
    "50-60",
    "60-70",
    "70 and more"
]

bins = [0, 10, 20, 30, 40, 50, 60, 70, 80]

frequency_df = df.groupby(pd.cut(df['age'], bins=bins, labels=labels)).size()

frequency_df = frequency_df.reset_index(name='count')

frequency_df.to_csv("csv_tables/frequency_distribution_of_age.csv", index=False)


#########################################
## Plot Age Distribution ##
#########################################

ax = frequency_df.plot(
    kind='bar', 
    figsize=(13, 12),
    rot=90, 
    color=['silver'], 
    edgecolor='white', 
    linewidth=5,
    legend=False
)


plt.title('Frequency Distribution of Age', fontsize=30, pad=50)

plt.xticks(ticks=np.arange(len(frequency_df)), labels=frequency_df['age'], fontsize=30)

for patch in ax.patches:
    width = patch.get_width()
    height = patch.get_height()
    x, y = patch.get_xy()
    ax.annotate(f'{round(height, 3)}', (x + width/2, y + height*1.02), ha='center')


plt.tight_layout()

plt.savefig(f'figures/frequency_distribution_of_age.png', dpi=600, transparent=True)

print('Age distribution info has been saved successfully!')


#####################################################
## Count Plot of `SEX` Column Head ##
#####################################################

count_plot(
    df=df, 
    column_name='sex', 
    color=['deepskyblue', 'magenta'],
    edgecolor='white',
    dpi=600, 
    transparent=True,
    path='figures',
)

print('Count plot of the sex column head has been saved successfully!')


#########################################################################
## Pie Chart Representation of "Sex" Column Head ##
#########################################################################

draw_pie_chart(
    df=df,
    column_name='sex', 
    labels=['Male', 'Female'], 
    colors=['deepskyblue', 'magenta'],
    dpi=600, 
    transparent=True,
    path='figures',
)

print('Pie chart of the sex column head has been saved sucessfully!')


###############################################
## Drop `SEX` Data Feature ##
###############################################

df.drop('sex', inplace=True, axis=1)

print('Sex Dropped!')


############################################################################
## Bar Plot of the Various `Bianry (Yes/No)` Attributes ##
############################################################################

binary_columns = [
    'steroid', 
    'antivirals', 
    'fatigue', 
    'malaise', 
    'anorexia', 
    'liver_big', 
    'liver_firm', 
    'spleen_palpable', 
    'spiders', 
    'ascites', 
    'varices', 
    'histology',
]

labels = ['No', 'Yes']

colors = [
    ['lime', 'orangered'],
    ['orange', 'lightgreen'],
    ['skyblue', 'yellowgreen'],
    ['orange', 'lightgreen'],
    ['orange', 'deepskyblue'],
    ['chocolate', 'orange'],
    ['orange', 'chocolate'],
    ['darkturquoise', 'silver'],
    ['limegreen', 'orange'],
    ['deepskyblue', 'orange'],
    ['darkorange', 'skyblue'],
    ['orangered', 'turquoise']
]

for i in range(len(binary_columns)):
    count_plot(
    df=df, 
    column_name=binary_columns[i], 
    color=colors[i],
    edgecolor='white',
    dpi=600, 
    transparent=True,
    path='figures'
)

    
####################################################################################################
## Pie Chart Representation of the Various `Bianry (Yes/No)` Attributes ##
####################################################################################################

for i in range(len(binary_columns)):
    draw_pie_chart(
        df=df, 
        column_name=binary_columns[i], 
        labels=labels, 
        colors=colors[i], 
        dpi=600, 
        transparent=True,
        path='figures',
    )

print('Bar charts and pie charts of the various binary columns have been saved successfully!')


#######################################################################################################
## Scatter Plots of Continuous Attributes According to the `Age` Attributes ##
#######################################################################################################

cmap = sns.cubehelix_palette(dark=.3, light=.8, as_cmap=True)

col_names = ['bilirubin', 'alk_phosphate', 'sgot', 'albumin', 'protime']

for col_name in col_names:
    
    plt.figure(figsize=(12, 8), dpi=300)
    
    sns.scatterplot(
        x='age', 
        y=col_name, 
        data=df, 
        hue='class',
        size='class',
        sizes=(120, 150),
        palette='Set1',
    )
    
    plt.title(f'{col_name.title()} Values According to AGE', fontsize=28, pad=20)
    
    plt.legend(bbox_to_anchor=(1, 1), loc="upper left", fontsize='large', frameon=True, shadow=True, fancybox=True)
    
    plt.tight_layout()
    
    plt.savefig(f'figures/scatter_plot_of_{col_name}.png', dpi=600, transparent=True)

print('Scatter plots of continuous features according to age have been saved successfully!')


##############################################################################################
## Relationship Between Continuous Attributes and Age per Class ##
##############################################################################################

col_names = ['bilirubin', 'alk_phosphate', 'sgot', 'albumin', 'protime']

hue_colors = {
    0: 'orange', 
    1: 'dodgerblue',
}

markers = {
    0: 'X',
    1: 'o',
}


for col_name in col_names:
    sns.relplot(
        x='age', 
        y=col_name, 
        data=df, 
        hue='class',
        size='class',
        sizes=(300, 300),
        style='class',
        col='class',
        markers=markers,
        palette=hue_colors,
        height=7,
        aspect=1.4,

    );
    
    plt.savefig(f'figures/relationship_between_{col_name}_and_age_per_class.png', dpi=600, transparent=True)


print('Relationship info between continuous attributes and age per class has been saved sucessfully!')


#############################################################################
## Feature Statistics per target variable - `Mean Value` ##
#############################################################################

features_mean_stats = df[['age', 'bilirubin', 'alk_phosphate', 'sgot', 'albumin', 'protime', target_variable]].groupby(target_variable).mean()

features_mean_stats.to_csv("csv_tables/features_mean_stats_per_target_variable.csv", index=True)


######################################################################
## Plot Feature Statistics per Target Variable ##
######################################################################

for col in list(features_mean_stats.columns.values):
    subset = features_mean_stats[col]
    
    fig = plt.figure(figsize=(12, 8), dpi=300)
    
    ax = subset.plot(
        kind='bar',
        color=['#EF5350', '#FFE0B2'], 
        edgecolor='#F5F5F5', 
        linewidth=7,
        rot=0,
    )
    
    plt.title(f'Mean Value of {col.title()} per Target Label', fontsize=35, pad=50)
    
    plt.xticks(ticks=np.arange(len(subset)), labels=['Die', 'Live'])
    
    plt.grid(alpha=0.2, color='grey')
    
    for patch in ax.patches:
        width = patch.get_width()
        height = patch.get_height()
        x, y = patch.get_xy()
        ax.annotate(f'{round(height, 2)}', (x + width/2, y + height*1.02), ha='center')
    
    plt.tight_layout()
    
    plt.savefig(f'figures/Mean_Value_of_{col}_per_Target_Label.png', dpi=600, transparent=True)


#######################################################################################
## Feature Statistics per target variable - `Standard Deviation` ##
#######################################################################################

features_std_stats = df[['age', 'bilirubin', 'alk_phosphate', 'sgot', 'albumin', 'protime', target_variable]].groupby(target_variable).std()

features_std_stats.to_csv("csv_tables/features_std_stats_per_target_variable.csv", index=True)

print('Feature Statistics info per target variable has been saved successfully!')


#####################################################################
## Create a Final Copy of the modified Dataset ##
#####################################################################

df_modified = df.copy()


#######################################################
## Save the Modified Dataset as csv ##
#######################################################

df_modified.to_csv("data/cleaned_data.csv", index=False)

print('Final Dataframe has been saved SUCCESSFULLY!')

print('\n')
print('\n')
print('\n')
print('DONE!')

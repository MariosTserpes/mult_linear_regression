#Importing libraries
import numpy as np 
import pandas as pd

import matplotlib.pyplot as plt
from matplotlib import style
import seaborn as sns

import pingouin as pg 
import statsmodels.api as sm
from statsmodels.stats import diagnostic as diag
from statsmodels.stats.outliers_influence import variance_inflation_factor

from sklearn.preprocessing import StandardScaler,MinMaxScaler
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.model_selection import train_test_split,cross_val_score

import warnings
warnings.filterwarnings("ignore")


#reading dataset
df = pd.read_csv('insurance.csv')
df_copy = df.copy() # a copy with our dataset

df.sample(10)

for value in df.columns:
    print(df[value].dtype)
    print(df[value].unique())
    
#View dimensions of dataset and checking for NaN values
print(df.shape)
print(df.info())
print(df.isnull().any())


#Categorical features
categorical = [var for var in df.columns if df[var].dtype == 'O']
categorical_df = df[categorical]
#Numerical features
numerical = [var for var in df.columns if df[var].dtype != 'O']
numerical_df = df[numerical]

print(f'Categorical variables : {categorical}')
print(f'Numerical variables : {numerical}')


# 1. About categorical variables

#3 plots in 1 row
fig, axes = plt.subplots(1, 3, figsize = (16, 5))
fig.suptitle('Number of observations for categorical variables', fontweight = 'bold')

#about gender
sns.countplot(categorical_df['sex'], ax = axes[0], palette = ['pink', 'blue'], alpha = 0.6)
axes[0].set_title('Number of males and females', fontweight = 'light')
axes[0].text(0.95, 650, f"{round(categorical_df['sex'].value_counts()[0]/len(df)* 100, 1)} %", fontweight = 'bold')
axes[0].text(-0.05, 640, f"{round(categorical_df['sex'].value_counts()[1]/len(df)* 100, 1)} %", fontweight = 'bold')


#about smokers
sns.countplot(categorical_df['smoker'], ax = axes[1], palette = ['pink', 'blue'], alpha = 0.6)
axes[1].set_title('Smokers and non-smokers', fontweight = 'light')
axes[1].text(0.95, 1020, f"{round(categorical_df['smoker'].value_counts()[0]/len(df) * 100, 1)} %", fontweight = 'bold')
axes[1].text(-0.08, 210, f"{round(categorical_df['smoker'].value_counts()[1]/len(df) * 100, 1)} %", fontweight = 'bold')


#about region 
sns.countplot(categorical_df['region'], ax = axes[2], palette = ['pink', 'blue'], alpha = 0.6)
axes[2].set_title('Residential area of beneficiaries', fontweight = 'light')
axes[2].text(0.75, 350, f"{round(categorical_df['region'].value_counts()[0]/len(df) * 100, 1)} %", fontweight = 'bold')
axes[2].text(-0.25, 310, f"{round(categorical_df['region'].value_counts()[1]/len(df) * 100, 1)} %", fontweight = 'bold')
axes[2].text(1.75, 310, f"{round(categorical_df['region'].value_counts()[2]/len(df) * 100, 1)} %", fontweight = 'bold')
axes[2].text(2.75, 310, f"{round(categorical_df['region'].value_counts()[3]/len(df) * 100, 1)} %", fontweight = 'bold');


# 2. Distributions, outliers and the relationship between bmi and charges.

fig, axes = plt.subplots(3, 3, figsize = (16, 24))
fig.suptitle('Understanding data from distributions, scatter plots and boxplots', fontweight = 'bold', fontsize = 20)

#About bmi
sns.distplot(df['bmi'], ax = axes[0, 0], color = 'blue')
axes[0, 0].set_title('Distribution of bmi', fontweight = 'bold')

#About bmi per gender
sns.scatterplot(y = df['charges'], x = df['bmi'], hue = df['sex'], palette = ['pink', 'blue'], ax = axes[0, 1])
axes[0, 1].set_title('Dispersion of charges VS bmi.', fontweight = 'bold')

#About bmi and smokers
sns.boxplot(x = df['bmi'], y = df['smoker'], ax = axes[0, 2], palette = ['pink', 'blue'])
axes[0, 2].set_title('Boxplot of bmi | smokers and non-smokers', fontweight = 'bold')

#About target variable('charges')
sns.kdeplot(df['charges'], color = 'red', shade = True,  ax = axes[1, 0]) 
axes[1, 0].set_title('Distribution of charges', fontweight = 'bold')

#Charges and gender
sns.kdeplot(df['charges'], hue = df['smoker'], shade = True, ax = axes[1, 1])
axes[1, 1].set_title('charges for smokers and non - smokers', fontweight = 'bold')

#charges and region
sns.boxplot(x = df['region'], y = df['charges'], ax = axes[1, 2])
axes[1, 2].set_title('Quadrants of charges per US suburb', fontweight = 'bold')

#About number of childrena
sns.countplot(df['children'], color = 'purple', ax = axes[2, 0], alpha = 0.5)
axes[2, 0].set_title('Number of childrens covered by insurance', fontweight = 'bold')

#About the ages 
sns.histplot(df['age'], bins = 7, color = 'red', edgecolor = 'black', alpha = 0.4, ax = axes[2, 1])
axes[2, 1].set_title('Histogram for the age', fontweight = 'bold')


#Charges vs bmi based on number of childrens
sns.kdeplot(df['charges'], hue = df['children'], ax = axes[2, 2], shade = True, palette = 'coolwarm')
axes[2, 2].set_title('Distribution of charges based on number of childrens', fontweight = 'bold')
plt.show()


#numerical features
num_descr_stats = round(numerical_df.describe().T, 2)
num_descr_stats['+3std'] = num_descr_stats['mean'] + (num_descr_stats['std'] * 3)
num_descr_stats['-3std'] = num_descr_stats['mean'] - (num_descr_stats['std'] * 3)
num_descr_stats


#categorical features
categ_num_stats = categorical_df.describe()
categ_num_stats
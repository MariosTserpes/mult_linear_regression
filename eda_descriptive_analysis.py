#Importing libraries
import numpy as np 
import pandas as pd

import matplotlib.pyplot as plt
from matplotlib import style
import seaborn as sns

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

#3. Visualizations, grouping ages and average number of charges based on Number of childs and region of residence
fig, axes = plt.subplots(1, 3, figsize = (15, 6))

#Visualizing Number Of Patients Per Age group
age_18_25 = df.age[(
            df.age >= 18) 
            & (df.age <= 25)]

age_26_35 = df.age[(
            df.age >= 26) 
            & (df.age <= 35)]

age_36_45 = df.age[(
            df.age >= 36) 
            & (df.age <= 45)]

age_46_55 = df.age[(
            df.age >= 46) 
            & (df.age <= 55)]

age_55_above = df.age[df.age >= 55]

x_age_groups = ['18-25', '26-35', '36-45', '46-55', '55+']
y_age = [len(age_18_25), len(age_26_35), len(age_36_45), len(age_46_55), len(age_55_above)]

sns.barplot(x = x_age_groups, y = y_age, data = df, 
            palette = ['blue', 'pink', 'red', 'grey', 'orange'], ax = axes[0])
axes[0].set_title('Number of patients per age group', fontweight = 'bold')
axes[0].set_xlabel('Counr')
axes[0].set_xlabel('Age Groups')


#Average Charges Based On Number Of Childrens
df_child = df.groupby(['children'])['charges'].mean().reset_index(drop = True)
sns.barplot(x = df_child.index, y = df_child.values, ax = axes[1])
axes[1].set_title('Average Charges Based on No of Childs', fontweight = 'bold')
axes[1].set_xlabel('Charges')
axes[1].set_xlabel('Children')

#Average Charges Per Region
df_region = df.groupby(['region'])['charges'].mean()
sns.barplot(x = df_region.index, y = df_region.values, ax = axes[2])
axes[2].set_title('Average Charges Per Region', fontweight = 'bold')
axes[2].set_xlabel('Charges')
axes[2].set_xlabel('Region');
#numerical features
num_descr_stats = round(numerical_df.describe().T, 2)
num_descr_stats['+3std'] = num_descr_stats['mean'] + (num_descr_stats['std'] * 3)
num_descr_stats['-3std'] = num_descr_stats['mean'] - (num_descr_stats['std'] * 3)
num_descr_stats


#categorical features
categ_num_stats = categorical_df.describe()
categ_num_stats


'''
In this step i will try to take some insights for 
benefeciaries  where the charges from the insurance 
are greater than 3 standard deviations from the average number of charges.  
'''

#3+ stand deviations from the mean of charges
stdev3 = round(df['charges'].mean() + (df['charges'].std() * 3), 2)
charges_df_outliers = df[df['charges'] > stdev3].reset_index(drop = True)# a data frame with charges > 3stdeviations
charges_df_outliers = charges_df_outliers.sort_values(by = 'age')
charges_df_outliers

'''
How many of beneficiaries are more than +3 standard deviations from the mean 
number of bmi and how many of these people are between 18.5 and 24.9 where according to literarure 
the ideal bmi is in this range.  
'''
#more than 3std bmi
more_than_3std_bmi = round(df['bmi'].mean() + (df['bmi'].std() * 3), 2)
#bmi_df_outliers
bmi_df_outliers = df[df['bmi'] > more_than_3std_bmi].reset_index(drop = True).sort_values(by = 'bmi')
bmi_df_outliers

'''
Some informations for the beneficiaries whose body mass index(BMI) is at ideal levels,
ie 18.5 to 24.9. 
'''

#setting the lower and upper limit of bmi
lower_limit = 18.5
upper_limit = 24.9

#condition in order the dataframe based on the lower and upper limit
ideal_body_mass = (df[
    (df['bmi'] >= lower_limit)
    & (df['bmi'] <= upper_limit)
])
print(f"The number of beneficiaries whose bmi is between the ideal bmi range is : {ideal_body_mass.shape[0]}, i.e {round(ideal_body_mass.shape[0]/len(df) * 100, 2)}% of the overall population.")

# 4. Visualizing some informations for the subgroup of beneficiaries with bmi between 18.5 and 24.9
fig, axes = plt.subplots(2, 3, figsize = (20, 17))
fig.suptitle('Exploring the structure of data for beneficiaries with bmi between 18.4 and 24.9', fontweight = 'bold', fontsize = 20)

#Let's see the how many of females and males are in this range
sns.countplot(ideal_body_mass['sex'], palette = ['blue', 'pink'], ax = axes[0, 0])
axes[0, 0].set_title('No of males and females with ideal body mass', fontweight = 'bold')
axes[0, 0].text(0.90, 117,ideal_body_mass['sex'].value_counts()[0], fontweight = 'bold', fontsize = 15)
axes[0, 0].text(-0.1, 107,ideal_body_mass['sex'].value_counts()[1], fontweight = 'bold', fontsize = 15)

#distributions of bmi based on the sex
sns.kdeplot(ideal_body_mass['bmi'], hue = ideal_body_mass['sex'], ax = axes[0, 1], shade = True, palette = ['lightblue', 'pink'])
axes[0, 1].set_title('Distribution of bmi based on gender', fontweight = 'bold')

#how many of these beneficiaries are smokers and non-smokers
#Let's see the how many of females and males are in this range
sns.countplot(ideal_body_mass['smoker'], palette = ['red', 'grey'], ax = axes[0, 2])
axes[0, 2].set_title('Smokers and non-smokers for the subgroup', fontweight = 'bold')
axes[0, 2].text(0.95, 51,ideal_body_mass['smoker'].value_counts()[1], fontweight = 'bold', fontsize = 15)
axes[0, 2].text(-0.1, 173,ideal_body_mass['smoker'].value_counts()[0], fontweight = 'bold', fontsize = 15)

#distributions of charges for this subgroup
sns.kdeplot(ideal_body_mass['charges'], ax = axes[1, 0], shade = True, color = 'red')
axes[1, 0].set_title('Distribution of charges for the subgroup', fontweight = 'bold')

#about the region of beneficiaries in this range
sns.countplot(ideal_body_mass['region'], ax = axes[1, 1], alpha = 0.6)
axes[1, 1].set_title('Number of beneficiaries in each suburb in the subgroup', fontweight = 'bold')

#beneficiaries' children
sns.countplot(ideal_body_mass['children'], color = 'red', ax = axes[1, 2], edgecolor = 'white')
axes[1, 2].set_title('Number of beneficiaries childrens in subgroup', fontweight = 'bold')
plt.show()

#5. Correlation analysis

#Exploring linear relationships
df_corr = df.corr()
sns.heatmap(
    df_corr, annot = True, cmap = 'Reds',
    xticklabels = df_corr.columns.values,
    yticklabels = df_corr.columns.values,
    )
plt.title('Health Insurance Heatmap', fontsize = 15)
plt.xticks(fontsize = 12)
plt.yticks(fontsize = 12);

#6. Scatter plots for each predictor against target variable[Charges]

fig, axes = plt.subplots(2, 3, figsize = (20, 17))
fig.suptitle('Scatter Plots and Linear Relationships', fontweight = 'bold', fontsize = 20)

#Charges Vs Bmi based on gender
sns.scatterplot(x = df['bmi'], y = df['charges'], hue = df['sex'], 
                ax = axes[0, 0], s = 100, color = 'red', edgecolor = 'white', alpha = 0.6)
axes[0, 0].set_title('Charges and Bmi per Gender', fontweight = 'bold')

#Charges Vs bmi based on no of children
sns.scatterplot(x = df['bmi'], y = df['charges'], hue = df['children'], 
                ax = axes[0, 1], s = 100, color = 'black', edgecolor = 'white', alpha = 0.6)
axes[0, 1].set_title('Charges and bmi per no of children', fontweight = 'bold')

#Charges Vs bmi based on no of children
sns.scatterplot(x = df['age'], y = df['charges'], hue = df['sex'], 
                ax = axes[0, 2], s = 100, color = 'black', edgecolor = 'white', alpha = 0.6)
axes[0, 2].set_title('Charges and age per gender', fontweight = 'bold')

#Charges and bmi per smoker status
sns.scatterplot(x = df['bmi'], y = df['charges'], hue = df['smoker'],
                ax = axes[1, 0], color = 'red', s = 100, edgecolor = 'white')
axes[1, 0].set_title('Charges and bmi per smoker', fontweight = 'bold')

#Charges Vs sex based on suburb
sns.boxplot(x = df['sex'], y = df['charges'], hue = df['region'],
                ax = axes[1, 1], color = 'blue')
axes[1, 1].set_title('Charges and sex based on suburb', fontweight = 'bold');


#Charges Vs sex based on suburb
sns.boxplot(x = df['children'], y = df['charges'], hue = df['sex'],
                ax = axes[1, 2], color = 'red')
axes[1, 2].set_title('Charges and children based on gender', fontweight = 'bold');
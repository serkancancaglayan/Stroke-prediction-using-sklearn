#dataset : https://www.kaggle.com/fedesoriano/stroke-prediction-dataset
import pandas as pd 
from sklearn.impute import SimpleImputer
import numpy as np


df = pd.read_csv('healthcare-dataset-stroke-data.csv')
"""print(df.columns)
(['id', 'gender', 'age', 'hypertension', 'heart_disease', 'ever_married',
       'work_type', 'Residence_type', 'avg_glucose_level', 'bmi',
       'smoking_status', 'stroke']"""

#dropping id's 
df = df.drop('id', axis = 1)


#turning NaN bmi values to mean of all
imp = SimpleImputer(missing_values = np.nan, strategy = 'mean')
imp.fit(df[['bmi']])
df['bmi'] = imp.transform(df[['bmi']])


#changing non numerical gender data to numerical, gender(male = 1, female = 0)
df['gender'] = df['gender'].replace(['Male', 'Female'], [1, 0])

#changing non numerical marriage data to numerical, ever_married(yes = 1, no = 0)
df['ever_married'] = df['ever_married'].replace(['Yes', 'No'], [1, 0])

#changing non numerical residence type data to numerical, Residence_type(urban = 1, rural = 0)
df['Residence_type'] = df['Residence_type'].replace(['Urban', 'Rural'], [1, 0])

#changing non numerical work type data to numerical
#print(df['work_type'].unique()) --> ['Private' 'Self-employed' 'Govt_job' 'children' 'Never_worked']
df['work_type'] = df['work_type'].replace(['Private','Self-employed', 'Govt_job', 'children', 'Never_worked'],
										  [0, 1, 2, 3, 4])

#changing non numerical data to numerical, smoking_status(formerly smoked = 0, never smoked = 1, smokes = 2, Unknown = 3)
#print(df['smoking_status'].unique()) --> ['formerly smoked' 'never smoked' 'smokes' 'Unknown']
df['smoking_status'] = df['smoking_status'].replace(['formerly smoked', 'never smoked', 'smokes', 'Unknown'],
												  [0, 1, 2, 3])


#writing changes on csv
df.to_csv('healthcare-dataset-stroke-data.csv', index = False)

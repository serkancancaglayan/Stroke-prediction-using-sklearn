import numpy as np
import pandas as pd
import pickle
from matplotlib import pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split


df = pd.read_csv('healthcare-dataset-stroke-data.csv')

#checking for NaN
for column in df.columns:
	if df[column].isnull().values.any():
		print(column + " column contains NaN values")

#gender column contains 1 NaN value, using pandas.dropna() function to delete that row
df = df.dropna()

#turning all the non numerical values to numerical, if there is any
for column in df.columns:
	df[column] = pd.to_numeric(df[column], errors = 'coerce')


X = df.drop('stroke', axis = 1).values
y = df['stroke'].values

#splitting data as test and train
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.1, random_state = 21)

#creating and traning the model
rfc = RandomForestClassifier()
rfc.fit(X_train, y_train)


print("Train score = ", rfc.score(X_train, y_train))
print("Test score = ", rfc.score(X_test, y_test))


#calculating feature importance using permutaion based method
feature_names = df.drop('stroke', axis = 1).columns
feature_importances_ = rfc.feature_importances_
sorted_idx = feature_importances_.argsort()
plt.barh(feature_names[sorted_idx], feature_importances_[sorted_idx])
plt.xlabel("Feaute importances on stroke")
plt.show()


#saving model to disk
filename = "final_model.sav"
pickle.dump(rfc, open(filename, 'wb'))

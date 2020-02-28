# =============================================================================
# Simple linear regression for Salary Prediction
# =============================================================================

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import pickle

## Load Dataset

dataset = pd.read_csv('C:/Users/Prateek/Desktop/content/ag dsb1/1Case Study/Flask App - Class Exercise/salary pridiction/Data/Salary_Data.csv')
#Separate Dependent and Independent Variables
x = dataset.iloc[:,0].values.reshape(-1,1)
y = dataset.iloc[:,-1]
#Since we have a very small dataset, we will train our model with all availabe data.
from sklearn.linear_model import LinearRegression

model = LinearRegression()
#Fitting model with trainig data
model.fit(x,y)

model.score(x,y)
# Saving model to disk use pickle.dump
pickle.dump(model,open('model.pkl','wb'))

# Verify by reloading model and predict results : Use pickle.load
model_load = pickle.load(open('model.pkl','rb'))

print(model_load.predict([[2.5]]))
# =============================================================================
# Simple linear regression for Salary Prediction
# =============================================================================

# Importing the libraries
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

data = pd.read_excel('Data/slr06.xls')

X= data.iloc[:,0].values.reshape(-1,1)

Y= data.iloc[:,1]

#pair plot
import seaborn as sns; sns.set(style="ticks", color_codes=True)
sns.pairplot(data)


# Total number of values
#N = len(X)

#Using scikit-learn method
# Import libraries and tools
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import mean_squared_error

# Cannot use Rank 1 matrix in scikit learn
#X = X.reshape((N, 1))

# Creating Model
model_reg = LinearRegression()

# Fitting training data
model_reg_params = {}

reg_grid = GridSearchCV(model_reg,model_reg_params,cv=5)

reg_grid.fit(X, Y)

reg_score = reg_grid.cv_results_

print(reg_score)

# Y Prediction
Y_pred = model_reg.predict(X)

# Calculating RMSE and  Score
mse = mean_squared_error(Y, Y_pred)
rmse = np.sqrt(mse)
r2_score = reg.score(X, Y)
#print(np.sqrt(mse))
#print(r2_score)

#score of model
model.score(X,Y)
# Saving model to disk use pickle.dump
pickle.dump(model_reg,open('modelInsurance.pkl','wb'))

# Verify by reloading model and predict results : Use pickle.load
model_load = pickle.load(open('modelInsurance.pkl','rb'))

print(model_load.predict([[0]]))








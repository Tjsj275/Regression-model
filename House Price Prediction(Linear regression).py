import pandas as pd
import numpy as np
import matplotlib.pyplot as pll
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score

customers=pd.read_csv("house.csv")

""""Print the first few values alongside dimensions"""
#print(customers.head())

""""Print the statistical data of the table"""
#print(customers.describe())

""""Print the non-null count and dtype of the data"""
#print(customers.info())

""""Plotting various types of plots possible"""
sns.pairplot(customers)

""""Scaling the data to some specific range"""
scaler=StandardScaler()
X=customers.drop(['Price','Address'],axis=1)
y=customers['Price']

cols=X.columns
X=scaler.fit_transform(X)

""""Training and Testing"""
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.3,random_state=101)

""""Linear Regression Plot"""
lr=LinearRegression()
lr.fit(X_train,y_train)

pred=lr.predict(X_test)

""""Accuracy of prediction model"""
#print(r2_score(y_test,pred))

""""Plotting the data"""
#sns.scatterplot(x=y_test,y=pred)

""""Histogram"""
#sns.histplot((y_test-pred),bins=50,kde=True)

""""Prediction Model"""
cdf=pd.DataFrame(lr.coef_,cols,['coefficients']).sort_values('coefficients',ascending=False)
print(cdf)




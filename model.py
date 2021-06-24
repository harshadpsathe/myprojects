# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import pickle

dataset = pd.read_csv('SWIFT_CAR_DATA.csv')

x = dataset.iloc[:, 1:3]
y = dataset.iloc[:, 0]

"""print(x)
print(y)"""
#Splitting Training and Test Set
#Since we have a very small dataset, we will train our model with all availabe data.

from sklearn.linear_model import LinearRegression
model = LinearRegression()

#Fitting model with trainig data
model.fit(x, y) 

# Saving model to disk
pickle.dump(model, open('model.pkl','wb'))

'''
# Loading model to compare the results
model = pickle.load(open('model.pkl','rb'))
print(model.predict([[2, 9, 6]]))
'''
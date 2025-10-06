# Artificial Neural Network

# Installing Theano
# pip install --upgrade --no-deps git+git://github.com/Theano/Theano.git

# Installing Tensorflow
# pip install tensorflow

# Installing Keras
# pip install --upgrade keras

# Part 1 - Data Preprocessing

# Importing the libraries
from sklearn.compose import ColumnTransformer

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
dataset = pd.read_csv('Churn_Modelling.csv')
X = dataset.iloc[:, 3:13].values
y = dataset.iloc[:, 13].values

# Encoding categorical data
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
labelencoder_X_1 = LabelEncoder()
X[:, 1] = labelencoder_X_1.fit_transform(X[:, 1])
labelencoder_X_2 = LabelEncoder()
X[:, 2] = labelencoder_X_2.fit_transform(X[:, 2])
#onehotencoder = OneHotEncoder(categorical_features = [1])## Replace this with
columnTransformer = ColumnTransformer([('encoder', OneHotEncoder(), [1])], remainder = 'passthrough')
#X = onehotencoder.fit_transform(X).toarray()##Replace this with
X = np.array(columnTransformer.fit_transform(X), dtype = np.float64)
X = X[:, 1:]

# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)

# Feature Scaling
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

# Part 2 - Now let's make the ANN!

# Importing the Keras libraries and packages
import keras
from keras.models import Sequential
from keras.layers import Dense

# Initialising the ANN
classifier = Sequential()

# Adding the input layer and the first hidden layer
classifier.add(Dense(units = 6, kernel_initializer = 'uniform', activation = 'relu', input_dim = 11))

# Adding the second hidden layer
classifier.add(Dense(units = 6, kernel_initializer = 'uniform', activation = 'relu'))

# Adding the output layer
classifier.add(Dense(units = 1, kernel_initializer = 'uniform', activation = 'sigmoid'))

# Compiling the ANN
classifier.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])

# Fitting the ANN to the Training set
classifier.fit(X_train, y_train, batch_size = 10, epochs = 100)

# Part 3 - Making predictions and evaluating the model

# Predicting the Test set results
y_pred = classifier.predict(X_test)
y_pred = (y_pred > 0.5)

# 🔟 Predicting a single new observation
"""
Predict if the customer with the following information will leave the bank:
Geography: los Angeles
Credit Score: 800
Gender: female
Age: 40
Tenure: 2
Balance: 120000
Number of Products: 1
Has Credit Card: No
Is Active Member: Yes
Estimated Salary: 70000
"""

# Note:
# After one-hot encoding and dropping one dummy variable, 
# the input order should be:
# [Geography_Germany, Geography_Spain, CreditScore, Gender, Age, Tenure, 
#  Balance, NumOfProducts, HasCrCard, IsActiveMember, EstimatedSalary]

# France → [0, 0]
# Male → 1

new_data = np.array([[0, 0, 600, 1, 40, 3, 60000, 2, 1, 1, 50000]])

# Scale features
new_data_scaled = sc.transform(new_data)

# Predict
new_pred = classifier.predict(new_data_scaled)
print("\nPrediction probability:", float(new_pred))
print("Will the customer leave?:", "Yes" if new_pred > 0.5 else "No")

# Making the Confusion Matrix for model evaluation
from sklearn.metrics import confusion_matrix, accuracy_score
cm = confusion_matrix(y_test, y_pred)
acc = accuracy_score(y_test, y_pred)
print("\nConfusion Matrix:\n", cm)
print("Test Accuracy:", acc)

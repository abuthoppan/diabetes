'''
Dataset : https://www.kaggle.com/uciml/pima-indians-diabetes-database
'''

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
#import the keras library and pakage 
import keras
#intializ our NN
from keras.models import Sequential
#pakage to create layar
from keras.layers import Dense

dataset = pd.read_csv('diabetes.csv')

X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, -1].values

# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)

# Feature Scaling
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

#inatilize the ANN
classifier =Sequential()

# create input layer
classifier.add(Dense(units=6,kernel_initializer='uniform' , activation='tanh' , input_dim=8))

# create hidden layer
classifier.add(Dense(units=6,kernel_initializer='uniform' , activation='tanh'))
# create hidden layer

# create output layer
classifier.add(Dense(units=1 , kernel_initializer='uniform' , activation='sigmoid'))

#compiling the ANN
classifier.compile(optimizer='adam' , loss='binary_crossentropy' , metrics=['accuracy'])

# Fitting the ANN to the Training set
classifier.fit(X_train, y_train, batch_size = 10, nb_epoch = 100)

classifier.save("diabetes_classifier.h5s")

# Predicting the Test set results
y_pred = classifier.predict(X_test)
y_pred = (y_pred > 0.5)


# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)

print(cm)
print("Accuracy: \n",(((93+29)/154) *100))

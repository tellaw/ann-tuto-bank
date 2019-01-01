# ANN : Artificial Neural Network DEMO

# STEP 1 : READING INPUT DATA
# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
dataset = pd.read_csv('bank-additional-full.csv', sep=';')
X = dataset.iloc[:, 0:16].values
y = dataset.iloc[:, 16].values

# Encoding categorical data
from sklearn.preprocessing import LabelEncoder, OneHotEncoder

# bank client data:
#0 - age (numeric)
#1 - job : type of job (categorical: 'admin.','blue-collar','entrepreneur','housemaid','management','retired','self-employed','services','student','technician','unemployed','unknown')
#2 - marital : marital status (categorical: 'divorced','married','single','unknown'; note: 'divorced' means divorced or widowed)
#3 - education (categorical: 'basic.4y','basic.6y','basic.9y','high.school','illiterate','professional.course','university.degree','unknown')
#4 - default: has credit in default? (categorical: 'no','yes','unknown')
#5 - housing: has housing loan? (categorical: 'no','yes','unknown')
#6 - loan: has personal loan? (categorical: 'no','yes','unknown')
# related with the last contact of the current campaign:
#8 NON RELEVANT - contact: contact communication type (categorical: 'cellular','telephone') 
#9 NON RELEVANT - month: last contact month of year (categorical: 'jan', 'feb', 'mar', ..., 'nov', 'dec')
#10 NON RELEVANT - day_of_week: last contact day of the week (categorical: 'mon','tue','wed','thu','fri')
#7 - duration: last contact duration, in seconds (numeric). Important note: this attribute highly affects the output target (e.g., if duration=0 then y='no'). Yet, the duration is not known before a call is performed. Also, after the end of the call y is obviously known. Thus, this input should only be included for benchmark purposes and should be discarded if the intention is to have a realistic predictive model.
# other attributes:
#8 - campaign: number of contacts performed during this campaign and for this client (numeric, includes last contact)
#9 - pdays: number of days that passed by after the client was last contacted from a previous campaign (numeric; 999 means client was not previously contacted)
#10 - previous: number of contacts performed before this campaign and for this client (numeric)
# social and economic context attributes
#11 - emp.var.rate: employment variation rate - quarterly indicator (numeric)
#12 - cons.price.idx: consumer price index - monthly indicator (numeric) 
#13 - cons.conf.idx: consumer confidence index - monthly indicator (numeric) 
#14 - euribor3m: euribor 3 month rate - daily indicator (numeric)
#15 - nr.employed: number of employees - quarterly indicator (numeric)
#
#16 - y - has the client subscribed a term deposit? (binary: 'yes','no')

# Not required, this is an issue with my laptop CPU and Keras Lib
import os
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"


# STEP 2 : ENCODING LABEL TO NUMERIC VALUES, FITTING for INPUT
labelencoder_X_1 = LabelEncoder()
X[:, 1] = labelencoder_X_1.fit_transform(X[:, 1])
labelencoder_X_2 = LabelEncoder()
X[:, 2] = labelencoder_X_2.fit_transform(X[:, 2])
labelencoder_X_3 = LabelEncoder()
X[:, 3] = labelencoder_X_3.fit_transform(X[:, 3])
labelencoder_X_4 = LabelEncoder()
X[:, 4] = labelencoder_X_4.fit_transform(X[:, 4])
labelencoder_X_5 = LabelEncoder()
X[:, 5] = labelencoder_X_5.fit_transform(X[:, 5])
labelencoder_X_6 = LabelEncoder()
X[:, 6] = labelencoder_X_6.fit_transform(X[:, 6])

onehotencoder = OneHotEncoder(categorical_features = [1,2,3,4,5,6])
X = onehotencoder.fit_transform(X).toarray()


# STEP 3 : ENCODING OUTPUT TO NUMERICAL VALUES
#labelencoder_y_1 = LabelEncoder()
#y[:] = labelencoder_y_1.fit_transform(y[:])

# STEP 4 : SPLITTING DATA INTO TWO DATASET TRAIN & TEST
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)

# STEP 5 : SCALING DATAS
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

# STEP 6 : BUILDING THE ANN
# Importing the Keras libraries and packages
import keras
from keras.models import Sequential
from keras.layers import Dense

X_test.shape

# Initialising
myann = Sequential()

# Adding input layer and first hidden layer
myann.add(Dense(units = 10, kernel_initializer = 'uniform', activation = 'relu', input_dim = 43))

# Adding a second hidden layer
myann.add(Dense(units = 10, kernel_initializer = 'uniform', activation = 'relu'))

# Adding the output layer
myann.add(Dense(units = 1, kernel_initializer = 'uniform', activation = 'sigmoid'))

# Compiling the ANN
myann.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])

# Fitting the ANN to the Training set
myann.fit(X_train, y_train, batch_size = 10, epochs = 100)

# STEP 7 : EVALUATING

# Predicting the Test set results
y_pred = myann.predict(X_test)
y_pred = (y_pred > 0.5)
print (y_test.tolist())
# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)

# That's all folk, ready to play :-)
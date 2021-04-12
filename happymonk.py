#importing the bank note dataset
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
df=pd.read_csv("BankNote_Authentication.csv")
df.head()
#Seperating the input variables from target variable
X=df.drop(["class"], axis = 1)
X

# Target variable
Y=df["class"]
Y
# Splitting the data into train and test
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size = 0.25, random_state = 123)
# Scaling the data
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)
#importing the necessary libraries
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
# creating the neural network
classifier = Sequential()
classifier.add(Dense(units = 1, activation = 'sigmoid', kernel_initializer = 'uniform', input_dim = 4))
classifier.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])
history=classifier.fit(X_train, y_train,validation_split = 0.25,  batch_size=10,epochs = 20)
#predicting 
y_pred = classifier.predict(X_test) > 0.5
# Building confusion matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)
cm
# Getting F1_score
from sklearn.metrics import f1_score
print(f1_score(y_test, y_pred, average='macro'))
print(f1_score(y_test, y_pred, average='micro'))
print(f1_score(y_test, y_pred, average='weighted'))

print(history.history.keys())

# Visualizing the loss updates with the number of epochs 
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title(' Loss vs Epochs')
plt.ylabel(' Loss')
plt.xlabel('Number of Epochs')
plt.legend(['train', 'validation'], loc='upper left')
plt.show()

#Visualizing the accuracy updates with the number of epochs
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('Accuracy vs Epochs')
plt.ylabel(' Accuracy')
plt.xlabel('Number of Epochs')
plt.legend(['train', 'validation'], loc='upper left')
plt.show()

# Building the neural network with cross validation and knowing the accuracies   
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import StratifiedKFold
from keras.models import Sequential
from keras.layers import Dense
def build_classifier():
    classifier = Sequential()
    classifier.add(Dense(units = 1, kernel_initializer = 'uniform', activation = 'sigmoid', input_dim = 4))
    classifier.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])
    return classifier
classifier = KerasClassifier(build_fn = build_classifier, batch_size = 10, epochs = 20)
kfold = StratifiedKFold(n_splits=10, shuffle=True, random_state=7)
accuracies = cross_val_score(estimator = classifier, X = X_train, y =y_train, cv = kfold)
print(accuracies)
mean = accuracies.mean()
print(mean)
variance = accuracies.std()
print(variance)
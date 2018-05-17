
# Author: Jacob Bailey


import sklearn as skl
from sklearn import svm
import pandas as pd
import scipy as sy
import os
import time
import matplotlib.pyplot as plt


# change directory so that we can read in the data
os.chdir('Python-Banknotes')


# read in banknote authentication set
banknotes = pd.read_csv('data/data_banknote_authentication.txt', names=['variance', 'skewness', 'curtosis', 'entropy', 'class'], header=0)

# convert to array
X = banknotes[['variance', 'skewness', 'curtosis', 'entropy']].as_matrix()
y = banknotes[['class']].as_matrix()



# Reference for building histogram (https://docs.scipy.org/doc/numpy-1.14.0/reference/generated/numpy.histogram.html)
# We will make four histograms

# Variance
plt.hist(X[:,0], bins='auto')
plt.title('Histogram of Variance')
plt.savefig('../plots/variance-histogram.png')
plt.close()

# Skewness
plt.hist(X[:,1], bins='auto')
plt.title('Histogram of Skewness')
plt.savefig('../plots/skewness-histogram.png')
plt.close()

# Curtosis
plt.hist(X[:,2], bins='auto')
plt.title('Histogram of Curtosis')
plt.savefig('../plots/curtosis-histogram.png')
plt.close()

# Entropy
plt.hist(X[:,3], bins='auto')
plt.title('Histogram of Entropy')
plt.savefig('../plots/entropy-histogram.png')
plt.close()

# now let us start the fun part....building models
# we reference (http://scikit-learn.org/stable/tutorial/basic/tutorial.html#learning-and-predicting)
clf = svm.SVC(gamma=0.001, C=100.)

# create training and test datasets
X_train, X_test, y_train, y_test = skl.model_selection.train_test_split(X, y, test_size=0.2, random_state=0)



## LOGISTIC REGRESSION
# https://towardsdatascience.com/logistic-regression-using-python-sklearn-numpy-mnist-handwriting-recognition-matplotlib-a6b31e2b166a
logisticRegr = skl.linear_model.LogisticRegression()
# now let us fit our model
logisticRegr.fit(X_train, y_train[:,0])
# predictions
predictions = logisticRegr.predict(X_test)
# get the score of the model
score = logisticRegr.score(X_test, y_test)


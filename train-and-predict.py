
# Author: Jacob Bailey


from sklearn.model_selection import train_test_split
from sklearn import tree
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn import svm
from sklearn.linear_model import LogisticRegression
import pandas as pd
import scipy as sy
import os
import matplotlib.pyplot as plt


# change directory so that we can read in the data
os.chdir('python/Python-Banknotes')


# read in banknote authentication set
banknotes = pd.read_csv('data/data_banknote_authentication.txt', names=['variance', 'skewness', 'curtosis', 'entropy', 'class'], header=0)

# convert to array
X = banknotes[['variance', 'skewness', 'curtosis', 'entropy']].as_matrix()
y = banknotes[['class']].as_matrix()[:,0]




# We will make four histograms

# Variance
plt.hist(X[:,0], bins='auto')
plt.title('Histogram of Variance')
plt.savefig('plots/variance-histogram.png')
plt.close()

# Skewness
plt.hist(X[:,1], bins='auto')
plt.title('Histogram of Skewness')
plt.savefig('plots/skewness-histogram.png')
plt.close()

# Curtosis
plt.hist(X[:,2], bins='auto')
plt.title('Histogram of Curtosis')
plt.savefig('plots/curtosis-histogram.png')
plt.close()

# Entropy
plt.hist(X[:,3], bins='auto')
plt.title('Histogram of Entropy')
plt.savefig('plots/entropy-histogram.png')
plt.close()



# create training and test datasets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)



## LOGISTIC REGRESSION
logisticRegr = LogisticRegression()
# now let us fit our model
logisticRegr.fit(X_train, y_train)
# predictions
predictions = logisticRegr.predict(X_test)
# get the score of the model
score = logisticRegr.score(X_test, y_test)
# achieves score of 0.989090



## LINEAR DISCRIMINANT ANALYSIS
linearDA = LinearDiscriminantAnalysis()
# fit linear discriminant model
linearDA.fit(X_train, y_train)
# make predictions
lda_predictions = linearDA.predict(X_test)
# get the score of the model
score_lda = linearDA.score(X_test, y_test)
# achieves score of 0.974545



## SUPPORT VECTOR MACHINE
supportVecMach = svm.LinearSVC()
# fit support vector machine
supportVecMach.fit(X_train, y_train)
# make predictions
svm_predictions = supportVecMach.predict(X_test)
# get the score of the model
score_svm = supportVecMach.score(X_test, y_test)
# achieves score of 0.989009



## DECISION TREE
decisionTree = tree.DecisionTreeClassifier()
# fit decision tree
decisionTree.fit(X_train, y_train)
# make predictions
tree_predictions = decisionTree.predict(X_test)
# get the score of the model
score_tree = decisionTree.score(X_test, y_test)
# achieves score of 0.996363



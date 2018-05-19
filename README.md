# Python-Banknotes
Here we will be using the UCI Machine Learning Repository Banknotes dataset (https://archive.ics.uci.edu/ml/datasets/banknote+authentication) to showcase some machine learning examples in python. Many of the ideas here are found in the book *Introduction to Statistical Learning*.

The goal here is to determine which banknotes are authentic. Therefore, we will be solving a classification problem. We will try several different algorithms and compare the results.

## Examining our variables

It's hard to stray away from my statistics background, so I always like to first visually examine some of the variables that I'll be using. We make one for each one of our predictor variables.






We have quite a few packages/libaries to import here. Note that we are separating out scikit learn, so that it is clear exactly what we are using here.

```python
from sklearn.model_selection import train_test_split
from sklearn import tree
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.ensemble import GradientBoostingClassifier
from sklearn import svm
from sklearn.linear_model import LogisticRegression
import pandas as pd
import scipy as sy
import os
import matplotlib.pyplot as plt
```

Next, we'll want to set up our working directory.

```python
os.chdir('python/Python-Banknotes')
```

We are then going to read in data that we have loaded from the UCI Machine Learning Repository (https://archive.ics.uci.edu/ml/datasets/banknote+authentication). Personally, I've had a lot of experience using R, so I chose to use Pandas to read in the data. We then need to convert that pandas data frame into a matrix and an array, for  **X** and **y** respectively.


```python
# read in banknote authentication set
banknotes = pd.read_csv('data/data_banknote_authentication.txt', names=['variance', 'skewness', 'curtosis', 'entropy', 'class'], header=0)

# convert to array
X = banknotes[['variance', 'skewness', 'curtosis', 'entropy']].as_matrix()
y = banknotes[['class']].as_matrix()[:,0]
```

Now that we have read in some data, we will take a look at the distribution of our variables. With a statistics background, I typically perfer to use histograms to examine the data. Now, it's not always terribly important that we do this since we are using our variables for prediction rather than inference, but *meiner meinung* you have to build a relationship with your data and get to know it.


### Variance
![](plots/variance-histogram.png)

```python
# Variance
plt.hist(X[:,0], bins='auto')
plt.title('Histogram of Variance')
plt.savefig('plots/variance-histogram.png')
plt.close()
```

### Skewness
![](plots/skewness-histogram.png)

```python
# Skewness
plt.hist(X[:,1], bins='auto')
plt.title('Histogram of Skewness')
plt.savefig('plots/skewness-histogram.png')
plt.close()
```

### Curtosis
![](plots/curtosis-histogram.png)

```python
# Curtosis
plt.hist(X[:,2], bins='auto')
plt.title('Histogram of Curtosis')
plt.savefig('plots/curtosis-histogram.png')
plt.close()
```

### Entropy
![](plots/entropy-histogram.png)

```python
# Entropy
plt.hist(X[:,3], bins='auto')
plt.title('Histogram of Entropy')
plt.savefig('plots/entropy-histogram.png')
plt.close()
```

## Splitting up our data and building our models

We are only using an 80/20 split here for the sake of simplicity. In this case, we're likely to overestimate the bias of the model. Cross-validation might be used to better assess what exactly our true error rate, but we'll save that for another day.

```python
# create training and test datasets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)
```

### Logistic Regression

Our problem is a classification model, so logistic regression is our first stop. Even though this is a pretty simple model, we get an error rate of 98.9%.

```python
## LOGISTIC REGRESSION
logisticRegr = LogisticRegression()
# now let us fit our model
logisticRegr.fit(X_train, y_train)
# predictions
predictions = logisticRegr.predict(X_test)
# get the score of the model
score = logisticRegr.score(X_test, y_test)
# achieves score of 0.989090
```

### Linear Discriminant Analysis

The next model we try is an LDA (i.e., linear discrimant analysis) model. Linear discriminant analysis is fundamentally different from logistic regression in that it tries to model our classification problem using the distribution of the predictors given the response and flips the results instead of trying to directly model the response given the predictors. This model performs slightly less well than the logistic regression model with an error rate of 97.5%.

```python
## LINEAR DISCRIMINANT ANALYSIS
linearDA = LinearDiscriminantAnalysis()
# fit linear discriminant model
linearDA.fit(X_train, y_train)
# make predictions
lda_predictions = linearDA.predict(X_test)
# get the score of the model
score_lda = linearDA.score(X_test, y_test)
# achieves score of 0.974545
```


### Support Vector Machine

Thirdly, we model our classification problem using a support vector machine method. When our classes are well separated, this is method works well. We won't worry investigate the separation of our classes here, but it is something of which we might want to make note. Clearly, the algorithm was appropriate with a score of 98.9%.

```python
## SUPPORT VECTOR MACHINE
supportVecMach = svm.LinearSVC()
# fit support vector machine
supportVecMach.fit(X_train, y_train)
# make predictions
svm_predictions = supportVecMach.predict(X_test)
# get the score of the model
score_svm = supportVecMach.score(X_test, y_test)
# achieves score of 0.989009
```

### Decision Tree

Saving the best for last, we model our classification problem using the classification decision tree algorithm. Classification trees are easy to interpret and offer a mix of predictive and inferential power. We achieve a score of 99.6% with this model.

```python
## DECISION TREE
decisionTree = tree.DecisionTreeClassifier()
# fit decision tree
decisionTree.fit(X_train, y_train)
# make predictions
tree_predictions = decisionTree.predict(X_test)
# get the score of the model
score_tree = decisionTree.score(X_test, y_test)
# achieves score of 0.996363

```
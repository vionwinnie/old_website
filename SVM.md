---
title: Support Vector Machine 
shorttitle: SVM
layout: project
nav_include: 4
noline: 1
---
## Contents
{:.no_toc}
*  
{: toc}

## Introduction

I am making a series of blog posts to introduce some of the common models in the machine learning space. My first post is about support vector machine (SVM). Support vector machine is a **supervised learning** technique and for **classification** problems. 

## Intuition behind it and the math

If given a set of points, we want to find a line that best separates the two groups. But there are many lines that can do so. SVM has a specific definition for the line that we look for:
for each possible line, it can extend outwards with a margin of buffer until the margin reaches the nearest points (for both clusters). The line that maxmimize the distance of the margin is the optimum solution for SVM.

![Alt Text](/assets/SVM/plots_0.png)

For more complicated separation like this, 

![Alt Text](/assets/SVM/plots_2.png)

we can also take the error of misclassification data points into consideration. 

Total Error Equation:
$$ minimize  \sum_{j=1}^{n} \max \left\{0,1-(a_0+ \sum_{i=1}^{n} (a_i)^2) y_j\right\} + min \cdot \lambda \left\( sum_{i=1}^{n} \|a_i^2\| \right\)$$

As we can see in the total error equation, the error is divided into two parts: 1) minimizing penalty for misclassifying individual points and 2) maxmizing margin distance. It is upon us to decide how to strike a balance between the two. This is called the **trade-off** between margin and misclassification error. Mathematically, the tradeoff is represented by $\lambda$ in the equation: The lower the lambda is, the more important it is to get the individual points correct. Geometrically, the $\lambda$ would change the width of the margin.

The margins are narrowed when you increase the penalty coefficient as shown in the plot below:
![Alt Text](/assets/SVM/plots_3.png)

## How to give weighted cost of misclassification

Suppose in the example of diagnosing cancer, a False-negative (Telling the patient he/she has no cancer while he/she actually has) has a much higher penalty than a False-positive ((Telling the patient he/she has cancer while he/she actually does not have), how can we express that in the model? 

We can reformulate the penalty for misclassification ( i.e. the first component) in the total error equation. 
$$ minimize  \sum_{j=1}^{n}  \textcolor{red}{m_j} \max \left\{0,1-(a_0+ \sum_{i=1}^{n} (a_i)^2) y_j\right\} + min \cdot \lambda \left\( sum_{i=1}^{n} \|a_i^2\| \right\)$$

$$ \left\{ 
\begin{array}{c}
m_j > 5 \\ 
m_j < 0.5 \\ 
\end{array}
\right. 
$$

$ m_j $ is the penalty cost. For False-negative, we set it to 1000 (for example) and 1 for false-positive to discount for the difference in penalty. Such setup would refine the boundary for better distinction for false-negatives.

![Alt Text](/assets/SVM/plots_4.png)

## Kernel Trick
What if the data doesn't look linearly separable?
![Alt Text](/assets/SVM/plots_5.png)

We could recompute (x,y) into $ (x,y, x^2 + y^2) $ to map the 2-dimensional data into 3-dimensional space and from there it would be easier to draw the separation boundary. After we project the boundary back to 2-D space, the boundary might not be linear anymore. There are multiple kernels that does this projection into higher dimension: tanh, rbf, and so on. 

## Scaling and standardization 
SVM is very sensitive to range of numbers. If one variable ranges from [0,1] and the other variable ranges [-1000000,10000000], the separation boundary can be sensitive to the variable with larger range than the smaller range. To remove this effect, we can apply scaling or standardization to remove such effect.  

## A simple Python implementation 
I have taken this dataset from https://www.kaggle.com/mlg-ulb/creditcardfraud/home
On the documentation it says that features are preprocessed using PCA except Time and Amount 

```
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

## Reading in the data 
data_path = r"C:\Users\USER\Documents\Fall2018\ISYE6501\blogposts\SVM\creditcard.csv"
df = pd.read_csv(data_path,nrows=10000)

X = df[['V1','V2']]
y = df[['Class']]
del df

## Train-test split 
from sklearn.model_selection import train_test_split
train_X, val_X, train_y, val_y = train_test_split(X, y,
                                                  random_state = 0,
                                                  train_size=0.6,
                                                  test_size=0.4)

unique, counts = np.unique(val_y, return_counts=True)
print('Distribution of labels:')
print(dict(zip(unique,counts)))

## Since this is an imbalanced dataset, I have used a class weight to put more emphasis to classify getting the 1's right as more important

class_weight = {0: 1,
                1: 100}

plt.scatter(train_X['V1'], train_X['V2'], c=train_y.values.ravel())
plt.title('Non-linear distribution')

## Implementing SVM using different kernel (linear, rbf, polynomial)
from sklearn import svm
C = 1.0  # SVM regularization parameter
models = (svm.SVC(kernel='linear', C=C,class_weight=class_weight),
          svm.SVC(kernel='rbf', gamma=0.6, C=C,class_weight=class_weight),
          svm.SVC(kernel='poly', degree=4, C=C,class_weight=class_weight))
models = (clf.fit(train_X, train_y) for clf in models)

# Title for the plots
titles = ('SVC with linear kernel',
          'SVC with RBF kernel',
          'SVC with polynomial (degree 4) kernel')

from sklearn.metrics import confusion_matrix

## Visualize the result 
for model,title in zip(models,titles):    
    y_pred = model.predict(val_X)
    print(title)
    print(confusion_matrix(val_y, y_pred))
```

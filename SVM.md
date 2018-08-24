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

$ m_j $ is the penalty cost. For False-negative, we set it to 5 (for example) aand less than 0.5 for false-positive to discount for the difference in penalty. 

## Python Implementation 


## Kernel Trick

## Scaling and standardization 


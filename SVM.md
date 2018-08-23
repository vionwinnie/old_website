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

we can also take the error of misclassification data points into consideration. The next thing we need to decide is whether maximizing the margin is more important or reducing the error of misclassification is more important. This is called the trade-off between margin and misclassification error.
Mathematically, the tradeoff is represented by $\lambda$ in the equation: The lower the lambda is, the more important it is to get the individual points correct. Geometrically, the lambda would change the width of the margin.

## Kernel Trick

## Python Implementation 

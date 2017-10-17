---
title: Quantitative Stock Picking in Hang Seng Index
shorttitle: Star Stock Picker
layout: default
nav_include: 4
noline: 1
---

## Introduction

Traditional quant models combine types of signals using a fixed percentage for each component in order to classify stocks. I am curious to see if deploying machine learning techniques can help improve betting averages of the stocks. My hypothesis is that these machine learning techniques can detect closer relationship of different metrics, assign better weights for different metrics, and thus draw a better separation plane between stocks likelky to go up or down. 

## Topic: Comparing Performance Between Random Forest Classifier Model (RF) and Support Vector Machine (SVM)

**Benchmark:** Hang Seng Index (HSI) 
**Timeframe:** 2000-2017 Monthly Data 
** Machine Learning Techniques:** Random Forest Classifier, Support Vector Machine

## Data Collection And Feature Selection

I have extracted HSI constituents and the dates of changes in index composition. I saved each security as an individual csv. There are about 60 securities.

## Feature Selection and Preprocessing:

I have extracted 25 base features, including price, cash from operations, earnings, forward P/E, dividend and so on. 

---

| Category | Factor | Description |  
|--------- | ------ | ----------- | 
|Valuation | P/E    | Price to Last Quarter Earnings|



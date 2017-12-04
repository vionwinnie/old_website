---
title: True Rating of San Francisco Restaurants on Yelp
shorttitle: Yelp Review
layout: default
nav_include: 3
noline: 1
---

## Background:

### Theories and Techniques Covered:

Bayesian Statistics, Monte Carlo Markov Chain, Gibbs Sampling

### Dataset Description & Problem Statement:

I have received this dataset from AM207. The data transformed quantified Yelp reviews for a number of restaurants in San Francisco. The goal of the project is to build a model to help a machine classify any given restaurant (or qualities of a restaurant) as "good" or "bad" given Yelp reviews. The intricacy of the problem is multi-fold: firstly, the distribution of samples in each restaurant is quite unevent, simply taking the average of stars of all reviews would skew the result for restaurants with very few reviews. Secondly, different users have bias or unique user behavior. One user might give 5 stars for most restaurants while the other one is an average 3 star and a 5-star is only reserved for the truly outstanding one? We need a sophistical Bayesian statistical model to normalize these bahaviors using certain features (so-called pooling) before we can further analyze the problem.  

This is how the raw data looks like:

| review_id |	topic |	rid |	count |	max |	mean |	min |	stars |	uavg |	var |
| IFrK2Blir_oq2qlDQmkqfw |	0 |	_0ZajBG5CSBSyxeeZV276g |	1 |	0.754263539 |	0.754263539 |	0.754263539 |	5 |	3.943396226 |	0 |
| IFrK2Blir_oq2qlDQmkqfw |	0 |	_0ZajBG5CSBSyxeeZV276g |	2 |	0.555812921 |	0.532285551 |	0.508758181 |	3 |	2.28 |	0.001107074 |
| dw84hAi3u0JE9CJn3vA |	1 |	_0ZajBG5CSBSyxeeZV276g |	5 |	0.871084844 |	0.469637675 |	0.067404631 |	3 |	2.28 |	0.121701096 |
| dw84hAi3u0JE9CJn3vA |	1 |	_0ZajBG5CSBSyxeeZV276g |	3 |	0.896720364 |	0.815665403 |	0.751170103 |	5 |	4.457142857 |	0.005501886 |
| gqnwsPW0DCurRxdRMCaGdA |	1 |	_0ZajBG5CSBSyxeeZV276g |	3 |	0.749899647 |	0.703165398 |	0.609696901 |	5 |	4.045454545 |	0.00655227 |
| izTGBN8dp5Q2dogVzRSQ |	1 |	_0ZajBG5CSBSyxeeZV276g |	4 |	0.733182386 |	0.637034849 |	0.529671453 |	5 |	5 |	0.007453149 |
| kc0ZIUV5wOsOBlm4o8eaw |	0 |	_2FUpthYr7h9VLNbe9Gxyw |	6 |	0.816901117 |	0.737071614 |	0.648147685 |	5 |	4 |	0.004913717 |
| BnAVLSivW3TDWk91SuTTag |	0 |	_2FUpthYr7h9VLNbe9Gxyw |	7 |	0.632203671 |	0.535985358 |	0.306211693 |	2 |	3.310344828 |	0.013015496 |
| FJmJITBLZDtc4Hf_NkShsw |	1 |	_2FUpthYr7h9VLNbe9Gxyw |	1 |	0.816823591 |	0.816823591 |	0.816823591 |	2 |	3.310344828 |	0 |

We have the following variables:
1. "review_id" - the unique identifier for each Yelp review
2. "topic" - the subject addressed by the review (0 stands for food and 1 stands for service)
3. "rid" - the unique identifier for each restaurant
4. "count" - the number of sentences in a particular review on a particular topic
5. "mean" - the probability of a sentence in a particular review on a particular topic being positive, averaged over total number of sentences in the review related to that topic.
6. "var" - the variance of the probability of a sentence in a particular review on a particular topic being positive, taken over all sentences in the review related to that topic.
7. "uavg" - the average star rating given by a particular reviewer (taken across all their reviews)
8. "stars" - the number of stars given in a particular review
9. "max" - the max probability of a sentence in a particular review on a particular topic being positive
10. "min" - the min probability of a sentence in a particular review on a particular topic being positive

## Preprocessing and Results:

https://github.com/AM207/2017HWs/blob/master/Long_HW3/Long_HW3_Solutions.ipynb

## Reflection and Future Application:

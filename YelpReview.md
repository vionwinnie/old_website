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

I have received this dataset from AM207. The data is collected from online Yelp reviews for a number of restaurants in San Francisco. The goal of the project is to build a model to estimate the mean rating for any given restaurant in the sample set.. The intricacy of the problem is multi-fold: firstly, the distribution of samples in each restaurant is quite uneven, simply taking the average of stars of all reviews would skew the result for restaurants with very few reviews. Secondly, different users have bias or unique user behavior. One user might give 5 stars for most restaurants while the other one is an average 3 star and a 5-star is only reserved for the truly outstanding one? We need a sophistical Bayesian statistical model to normalize these bahaviors using certain features (so-called pooling) before we can further analyze the problem.  

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

## Let's get started!

'''
Loading the setups
import itertools
import pymc
import scipy.stats as stats
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
%matplotlib inline
from scipy.special import erf
import seaborn as sns
sns.set_style("whitegrid")
sns.set_context("poster")
'''

Read the raw data and break down the data into restuarants and further into food and service segments:
'''
df_restaurant = pd.read_csv(r"C:\Users\Dell\Documents\Python Scripts\Data\dftouse4.csv")

## Compute standard error 
df_restaurant['var_sample']=df_restaurant['var']/df_restaurant['count']
df_restaurant['sigma']=np.sqrt(df_restaurant['var_sample'])

df_food = df_restaurant[:][df_restaurant.topic ==0]
df_service = df_restaurant[:][df_restaurant.topic ==1]

##View the restautnts with most number of review
restaurant_breakdown = df_restaurant.groupby('rid')['review_id'].count().sort_values(ascending=False)

#generate unique names 
names=df_restaurant['rid'].unique().tolist()

#create a data frame dictionary to store my spit dataframes
Food =  {elem : pd.DataFrame for elem in names}
Service = {elem : pd.DataFrame for elem in names}

for key in Food.keys():
    Food[key] = df_food[:][df_food.rid == key]

for key in Service.keys():
    Service[key] = df_service[:][df_service.rid == key]

'''

## Data Exploration for a number of restaurants with varying sample sizes:
'''
large = restaurant_breakdown.index[0:5]
mid = restaurant_breakdown.index[5000:5002]
small = restaurant_breakdown.index[-4:-1]
selected_rid = large|mid|small

### Selected Restaurant food model
Service_Selected = {elem : Service[elem] for elem in selected_rid}
Food_Selected = {elem : Food[elem] for elem in selected_rid}

### Create a visualization function
import itertools

def shrinkage_plot(means, thetas, theta_vars, counts):
    data = zip(means, thetas, theta_vars / counts, theta_vars, counts)
    palette = itertools.cycle(sns.color_palette())
    with sns.axes_style('white'):
        for m,t, me2, te2, c in data:
            color=next(palette)
            noise=0.04*np.random.randn()
            noise2=0.04*np.random.randn()
            plt.plot([m,t],[noise,1+noise2],'o-', color=color, lw=1)
            if me2==0:
                me2=4
            plt.errorbar([m,t],[noise,1+noise2], xerr=[np.sqrt(me2), np.sqrt(te2)], color=color,  lw=1)
        plt.yticks([])
        plt.xlim([0,1])
        plt.ylim([0, 1.05])
        sns.despine(offset=-2, trim=True, left=True)
    return plt.gca()
'''
## Need to add a more detailed explanation of the model setup with diagrams

### Set up the hierarchical model for food ratings:
A Hierarchical model with non-centered parametrization. Theta of a reviewer is drawn from a Normal hyper-prior distribution with parameters μμ and ττ. Once we get a θjθj then can draw the means from it given the data σjσj and one such draw corresponds to our data.

'''
dict_trace_food = {}
dict_pp_food = {}

for elem in selected_rid:
    observed = np.array(Food_Selected[elem]['mean'])
    sigma = np.array(Food_Selected[elem]['sigma'])
    with pm.Model() as rest_1:
        mu = pm.Normal('mu', mu=0.5, sd=0.15)   # A normal prior 
        tau = pm.HalfCauchy('tau', beta=0.00001) 
        nu = pm.Normal('nu', mu=0.5, sd=0.00001,shape = len(observed))
        theta = pm.Deterministic('theta', mu + tau * nu)
       obs = pm.Normal('obs', mu=theta, sd=sigma, observed=observed)
    with rest_1:
        trace2 = pm.sample(10000, init=None, njobs=2, tune=1000)
    
    ppc = pm.sample_ppc(trace2, samples=1000, model=rest_1)
    
    dict_trace_food[elem] = trace2
    dict_pp_food[elem] = ppc['obs']


dict_trace_service = {}
dict_pp_service = {}

for elem in selected_rid:
    observed = np.array(Service_Selected[elem]['mean'])
    sigma = np.array(Service_Selected[elem]['sigma'])    
    bool = np.where( sigma <= 0 )
    
    observed = np.delete(observed, bool[0]) 
    sigma = np.delete(sigma,bool[0])

    with pm.Model() as rest_1:
        
        ## Prior 
        mu = pm.Normal('mu', mu=0.5, sd=0.15)
        tau = pm.HalfCauchy('tau', beta=0.00001)
        nu = pm.Normal('nu', mu=0.5, sd=0.00001,shape = len(observed))        
        theta = pm.Deterministic('theta', mu + tau * nu)
        
        ##Likelihood
        obs = pm.Normal('obs', mu=theta, sd=sigma, observed=observed)
    with rest_1:
        trace2 = pm.sample(10000, init=None, njobs=2, tune=1000)

    ppc = pm.sample_ppc(trace2, samples=1000, model=rest_1)
    
    dict_trace_service[elem] = trace2
    dict_pp_service[elem] = ppc['obs'] 

 

## Reflection and Future Application:

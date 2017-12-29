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

```python

#Loading the setups
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
```

Read the raw data and break down the data into restuarants and further into food and service segments:
```python
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
```
## Model setup
We know that $ Posterior \sim Likelihood \cdot Prior $
But what is posterior, likelihood and prior in this case?

We know that there are $Q=1000$ restaurants, and in the $qth$ restaurant, there are $n_q$ reviews. 
What we know about each review is mean probability of a sentence being positive. 
We don't know $ p( being positive | sentence)$, but the mean of all the sentences in that review. 

For each $qth$ restaurant, the sample mean of each group $q$ is  $\bar{y_q} = \frac{1}{n_q} \sum_{i=1}^{n_q} y_{iq}$
With sampling variance: $\sigma_q^2 = \sigma^2/n_q$
Likelihood of each $\theta_q$, $\bar{y}_q$ is $$\bar{y_q} \vert \theta_q \sim N(\theta_q,\sigma_q^2).$$

If we choose the prior to be: 
$$ \theta_j \sim N(\mu, \tau^2) $$

Multiplying likelihood and prior together will give us the posterior distribution, after simplication then we will see that The _posterior mean_ is a weighted average of the prior mean and the observed average. 

Our non-centered hierarchical model will look like the following:
We estimate $\theta_{j}$ from a Normal hyper-prior distribution with parameters of $\nu $ and $\tau $. With the $\mu$ and $\tau$ drawn, we can then compute $\theta_{j}$. After that, we can then estimate the mean, $\bar{y_{j}}$ with the $\sigma_{j}$ given in our data.
The deterministic setup between $\theta$, $\mu$, and $\tau$ reduces the correlation of actively sampled variables, mitigating curvature and steepness.

Hyperpriors:

$$
\mu \sim \mathcal{N}(0, 5)\\
\tau \sim \text{Half-Cauchy}(0, 5)\\
$$

Prior:

$$
\nu_{j} \sim \mathcal{N}(0, 1)\\
\theta_{j} = \mu + \tau\nu_j \\
$$

Likelihood:

$$
\bar{y_{j}} \sim \mathcal{N}(\theta_{j}, \sigma_{j})
$$


```python
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
```

Let's view the traceplot and the autocorrelation plot of the food and service plots:

```python

restaurants = list(food.keys())

for i,rest in enumerate(restaurants):
    print('rid=',rest)
    pm.traceplot(food[rest])
    pm.autocorrplot(food[rest])

for i,rest in enumerate(restaurants):
    print('rid=',rest)
    pm.traceplot(service[rest])
    pm.autocorrplot(service[rest])
 
```


For resturant rid= JhupPnWfNlMJivnWB5druA:
Food Rating Model:
![Alt Text](/assets/rest1.png)
Service Rating Model:
![Alt Text](/assets/service1.png)


For resturant rid= muqFM9Hoamh_fGL4MPeZqg:
Food Rating Model:
![Alt Text](/assets/rest2.png)
Service Rating Model:
![Alt Text](/assets/service2.png)
 

Our resampling looks smooth on the traceplots for both restaurants. 

## Visualizing pooling effects of the hierarchical model:

```python

for i,p in enumerate(selected_rid):
    mean_est = np.array(df_food[df_food.rid==p]['mean']).tolist()
    counts = np.array(df_food[df_food.rid==p]['count']).tolist()
    
    sigma = np.array(Food_Selected[p]['sigma'])
    bool = np.where( sigma <= 0 )
    
    mean_est = np.delete(mean_est, bool[0]) 
    counts = np.delete(counts,bool[0])
    df = food[p]['theta']
    
    theta_2 = df[1::2, :]
    theta_est = np.mean(theta_2,axis=0)
    theta_var = np.var(theta_2,axis=0)
    
    plt.subplot(10,1,i+1)
    shrinkage_plot(mean_est,theta_est,theta_var,counts)
    plt.title(('rest_',i,len(theta_est),'review of Food'))
```
![Alt Text](/assets/pooling.png)
 
 For shrinkage plots for service ratings:
```python

 for i,p in enumerate(restaurants):
    mean_est = np.array(df_service[df_service.rid==p]['mean']).tolist()
    counts = np.array(df_service[df_service.rid==p]['count']).tolist()
    
    sigma = np.array(Service_Selected[p]['sigma'])
    bool = np.where( sigma <= 0 )
    
    mean_est = np.delete(mean_est, bool[0]) 
    counts = np.delete(counts,bool[0])
    df = service[p]['theta']
    
    theta_2 = df[1::2, :]
    theta_est = np.mean(theta_2,axis=0)
    theta_var = np.var(theta_2,axis=0)
    
    plt.subplot(10,1,i+1)
    shrinkage_plot(mean_est,theta_est,theta_var,counts)
    plt.title(('rest_',i,len(theta_est),'review of Service'))
```
![Alt Text](/assets/pooling1.png)

We can see that for each restaurant, the hierarchical model we set up pass information from one review to another. The raw mean for each review can be dispersed widely. But after the modeling, the resampled estimation of the same parameter is drawn towards the middle due to the pooling effects.

## Prototype Ranking Order:
With the resampling for both food and service ratings, we can now estimate with more comfort for each aspect of a restaurants. What happens next is that we extract the score for both food and service and combine into one rating. 

```python
food_score = {}
service_score = {}
restaurants = list(food.keys())

for i,p in enumerate(restaurants):
    df_food = food[p]['theta']
    df = service[p]['theta']
    
    theta_2 = df[1::2, :]
    theta_1 = df_food[1::2, :]

    theta_est_2 = np.mean(np.mean(theta_2,axis=0)) #service
    theta_est_1 = np.mean(np.mean(theta_1,axis=0))  #food
    
    food_score[p] = theta_est_1
    service_score[p] = theta_est_2
    
food_list = [food_score]
service_list = [service_score]

df1 = pd.DataFrame(food_list).transpose()
df1.columns = ['Food_Score']
df1['rid'] = df1.index
df1 = df1[column_order]
df2 = pd.DataFrame(service_list).transpose()
df2.columns = ['Service_Score']
df2['rid'] = df2.index
df_final=df1.merge(df2,left_on='rid',right_on='rid',how='outer')
overall_score = [np.nanmean([df['food_score'][i],df['service_score'][i]]) for i in range(0,10)]
df['overall_score'] = overall_score
df['overall_rank'] = df['overall_score'].rank(ascending=False)

```
This is how the final ranking looks like:

|rid|food score|service score|overall_score|overall ranking|
|ABrSt3fsirLrUYNVrD3fbQ|0.758|0.682|0.720|1|
||0.708|NaN|0.708|2|
||0.628|0.758|0.693|3|
||0.669|0.699|0.684|4|
||0.645|NaN|0.645|5|
||0.591|0.668|0.629|6|
||0.618|0.624|0.621|7|
||0.675|0.553|0.614|8|
||0.536|0.549|0.543|9|
||0.525|0.288|0.406|10|


---
title: Quantitative Stock Picking in Hang Seng Index
shorttitle: Star Stock Picker
layout: project
nav_include: 4
noline: 1
---
## Contents
{:.no_toc}
*  
{: toc}

# Introduction

Traditional quant models combine signals classify stocks by a top down approach: it assigns a fixed percentage for each signal and combine them altogether for a composite score. I am curious to see if deploying machine learning techniques can help improve betting averages of the stockpicking. My hypothesis is that these machine learning techniques can detect closer relationship of different metrics, assign better weights for different metrics, and thus draw a better separation plane between rising and declining stocks. 

**Topic:** Comparing Performance Between Random Forest Classifier Model (RF) and Support Vector Machine (SVM)

**Benchmark:** Hang Seng Index (HSI) 

**Timeframe:** 2000-2017 Monthly Data 

**Machine Learning Techniques:** Random Forest Classifier, Support Vector Machine

# Project Flow 

## Data Collection & Feature Selection

I have extracted HSI constituents and the dates of changes in index composition. I saved each security as an individual csv. There are about 60 securities.

For each security, I have extracted 25 base features, including price, cash from operations, earnings, forward P/E, daily volume, short interest, dividend and so on. I then derived 194 factors I have engineered from the base features, grouped into different categories. These categories capture different parts of the stock, from fundamental, momentum, market expectation. Some of them are averages, differences, and Z score of one metrics. Click [Here]{: Feature Table} to see the feature table.

## Data Preprocessing 

1. Before standardizing the data, I first load it into my iPython interpreter through the following function *DataProcessing()*. The function will return two data items: 1) a dictionary of stock-level data with classifier label ( 1,0 based on forward 1-month excess return).  2) a dictionary of constituents at each time period

```python
import pandas as pd
import numpy as np
import glob, time, os

## Loading Dateparse for reading dataframes
dateparse = lambda x: pd.datetime.strptime(x, '%Y-%m-%d')


def DataProcessing(hsi_path, feature_path, stock_path, index_path):
    ### Reading Index Data
    dateparse = lambda x: pd.datetime.strptime(x, '%Y-%m-%d')
    HSI = pd.read_csv(hsi_path,header=2,parse_dates=['date'],date_parser=dateparse)
    new_name = ['date']
    for s in HSI.columns.tolist():
        if s != 'date':
            new_name.append('HSI_'+s) 
        
    HSI.columns = new_name
    
    ## Loading the column name lists to select the desired features
    column_selection= pd.read_excel(feature_path,sheetname=0,header=0,index_col=False)['Column'].tolist()
    column_selection = [s.encode("utf-8") for s in column_selection]   ### Encode as string 
    
    ## Reading Stock Level Data
    paths = glob.glob(stock_path)
    stock = {}
    for path in paths:
        stock_name = path.split("\\")[-1].split(".")[0]
        intermediary = pd.read_csv(path,header=2,parse_dates=['date'],date_parser=dateparse,usecols=column_selection)
        
        ## Merge HSI with stock level 
        intermediary = pd.merge(intermediary,HSI,how='left',on='date')
        
        ## Adding the classifier tab
        intermediary['classifier'] = intermediary['TotalReturnIndex_Chg_1M']> intermediary['HSI_Chg_1M']
        intermediary['classifier'] = intermediary['classifier'].shift(-1)
        intermediary['Forward_Chg_1M'] = intermediary['TotalReturnIndex_Chg_1M'].shift(-1)
        intermediary['stock_tag'] = stock_name.replace('-HK','')  ##Added Stock Tags 
        stock[stock_name] = intermediary    
    
    ## Loading Index Constituents Addition And Deletion File 
    dateparse2 = lambda x: pd.datetime.strptime(x, '%Y%m%d')
    HSI_changes= pd.read_excel(index_path,sheetname=0,header=0,index_col=False)
    
    ## Create Dictionary of Constituents According to Each Period In Time  
    ticker_date_dict = {}
    index = []
    
    for i in range(0,len(HSI_changes)):  
        ticker = HSI_changes.iloc[i,:]
        name = ticker['Ticker']
        dates = pd.Series(pd.date_range(start=dateparse2(str(ticker['Start_Date'])), end=dateparse2(str(ticker['End_Date'])), freq='M')).tolist()  
        for date in dates:
            index.append(date)
        ticker_date_dict[name]=dates
    
    unique_dates= list(set(index))
    inverted = {}
    for cool in unique_dates:
        inverted.setdefault(cool, [])  ## Default an empty index
    for k,v in ticker_date_dict.items():
        for m in v:
            inverted[m].append(k)

    return inverted, stock

IndexData_path = os.path.join(os.path.dirname(__file__), '..', 'Output/ExchangeFactor/HSI.csv') 
FeatureSelection_path =  os.path.join(os.path.dirname(__file__), '..', 'Input/StockLevel/FeatureSelection.xlsx')
StockLevelSearch_path =  os.path.join(os.path.dirname(__file__), '..', 'Output/StockLevel/*.csv')
Index_Add_Drop_path = os.path.join(os.path.dirname(__file__), '..', 'Input/StockLevel/Constituent_Addition_Removal.xlsx')
    
timeframe_dict, stock_dictionary= DataProcessing(IndexData_path,FeatureSelection_path,StockLevelSearch_path,Index_Add_Drop_path)
```
2. With timeframe_dict and stock_dictionary at hand, we can now proceed to reshape the data to form cross-section for each time period and then split the data into training and testing set. I'm using an expanding window approach, therefore this is my home-made test_train_split, different from the sklearn standardized function. 

```python
def TestTrainSplit_TimeSeries(timeframe_dict, stock_dictionary,periodicity):
    
    " timeframe_dict: The list of time periods in the stocks"
    " stock_dictionary: The dictionary in which I loaded the stock level data"
    " periodicity: The number of periods I want to include in each training period" 
    
    
    # Sorting the list of time periods 
    times = timeframe_dict.keys()
    times.sort()
    
    ## forming a cross section 
    month_set = {}     
    
    for period in times:
        securities = timeframe_dict[period]
        
        for i,security in enumerate(securities):
            test_df = stock_dictionary[security]
            line = test_df[test_df['date']==period]
            
            if i ==0:
                month = line
            else:
                month = pd.concat([month,line])
        
        month = month.drop(['date'],1) ## Remove Date Column
        npp = change_value(np.array(month).transpose()) ## Convert to Numpy Array for easing the operation
        
        month_set[period]=npp
    
    ## Splitting Features and Labels in training and testing sets
    train_set_num = xrange(periodicity,len(timeframe_dict))
    train_set = []   
    train_label_set = []
    
    test_set = []
    test_label_set = []
    test_stock_set = []
    
    for i in train_set_num:
        
        if i < len(timeframe_dict)-1 :
            
            allArrays = np.concatenate([month_set[times[x]] for x in range(i)],axis=1)
        
            labels = allArrays[-2]  ## Separate the labels column
            allArrays = allArrays[:-2] ## Drop the labels column
            
            train_set.append(allArrays)
            train_label_set.append(labels)
            
            testArrays = month_set[times[i+1]]
            testlabels = testArrays[-2]
            testtags = testArrays[-1]
            testArrays = testArrays[:-2]
        
            test_set.append(testArrays)
            test_label_set.append(testlabels)
            test_stock_set.append(testtags)
    
    return train_set, train_label_set, test_set, test_label_set,test_stock_set
 
 TrainFeatureSet, TrainLabelSet, TestFeatureSet, TestLabelSet, TestStockTagsSet = TestTrainSplit_TimeSeries(timeframe_dict, stock_dictionary,para.period)
``` 
This is how a raw cross section would look like:
| Index | Beta_Raw | EPS_LTM | EPS | DPS | TotalReturnIndex_Chg_1M | TotalReturnIndex_Chg_3M | TotalReturnIndex_Chg_6M | TotalReturnIndex_Chg_1Y | PB_LTM | PE_LTM | EV/EBITDA | EV/EBIT | EV/Sales | P/Sales | ROE |
|--------- | ------ | ----------- | --------- | ------ | ----------- | --------- | ------ | ----------- | --------- | ------ | ----------- |--------- | ------ | ----------- | 
|--------- | ------ | ----------- | --------- | ------ | ----------- | --------- | ------ | ----------- | --------- | ------ | ----------- |--------- | ------ | ----------- | 

|0|1.3639|3.39|3.35|1.73|0.00879863|0.013264331|0.216134088|0.21701794|1.034873396|16.8879056|19.52420504|26.11366935|5.633693988|0.000760949|0.063172043|
|1|0.7843|0.732264163|0.721631299|0.23346895|-0.027606711|0.05909564|0.371368673|0.369727789|1.177625671|11.01515055|8.633889691|12.94489533|0.80991864|0.00026371|0.105576436|
|2|1.1395|nan|0.01737304|0.006154287|-0.007546115|0.078348011|0.320074349|0.3200743490|nan|nan|nan|nan|nan|0.001281707|0.242847707|
|3|1.3818|1.761432|1.71|0.34|0.060018148|0.16913692|0.306561568|0.322479784|2.549068821|13.54011963|4.778318986|6.729102843|3.564941666|0.000185515|0.296332214|
|4|0.9269|nan|0.67|0.53|0.029415152|0.187269201|0.360964765|0.523349619|nan|nan|nan|nan|nan|0.002778431|0.120525452|
|5|1.485|3.831|3.83|1.6|0.0081556|0.014077946|0.324343506|0.258543165|0.485366245|9.284521013|221.9177996|260.3835515|63.89780405|0.014547648|0.05190279|

# Result 


## Feature Table 

| Category | Factor | Description |  
|--------- | ------ | ----------- | 
|Beta|Beta_Raw|Raw Beta|
|EarningsMomentum|EPS_Chg_T-4|Change in EPS from 4 months ago|
|EarningsMomentum|EPS_Chg_T-4_Zscore_3Y|3 Year Z-Score for Change in EPS from 4 months ago|
|EarningsMomentum|EPS_LTM|Earnings Per Share for the last twelve months|
|EarningsMomentum|EPS_NTM|Earnings Per Share Estimate for the next twelve months, Fiscal Year n, n=1,2,3|
|EarningsMomentum|EPS_NTM_Chg_EPS_LTM|EPS difference for twleve months|
|EarningsMomentum|EPSEst_1FY_Revision_1M|Change in n months of 1st fiscal year EPS Estimates, n = 1,3,6|
|EarningsMomentum|EPSEst_2FY_Chg_EPSEst_1FY|Difference between EPS in the first and second fiscal year|
|EarningsMomentum|EPSEst_2FY_Revision_1M|Change in n months of 2nd fiscal year EPS Estimates, n = 1,3,6|
|EarningsMomentum|EPSEst_3FY_Chg_EPSEst_2FY|Difference between EPS in the second and third fiscal year|
|EarningsMomentum|EPSEst_3FY_Revision_1M|Change in n months of 3rd fiscal year EPS Estimates, n = 1,3,6|
|MarketFactors|TotalReturnIndex_Chg_1Y-1M|Difference in total return for the last twelve month and the last one month,  also taking z score of n years, n=1,2,3,5,7|
|MarketFactors|TotalReturnIndex_Chg_nM|Change in total return for the last n monhts, n=1,3,6,12|
|MarketFactors|TotalReturnIndex_Chg_nMAvg|Average in monthly change in total return for the last n months, n=1,3,6,12|
|MarketFactors|TotalReturnIndex_Zscore_nY|Z Score of Total Return for n years, n=1,2,3,5,7|
|MarketFactors|TurnoverRatio|Daily Volume / Shares Outstanding, also taking average of n-months, n = 3,6,12|
|PriceMomentum|MoneyFlowIndex|Money Flow Index,  also taking z score of n years, n=1,2,3,5,7|
|PriceMomentum|ShortSellTurnoverPct|Short Selling Interest/ Total Volume, also taking z score of n years, n=1,2,3,5,7|
|Quality|Capex/Asset|Capex / Asset|
|Quality|Capex/Asset_Diff_T-4|The difference between 4-month period of Capex/Asset|
|Quality|CCR|(CFO + Capex)/ Profit, also taking the difference between 4-month period|
|Quality|Debt/Equity|Debt/Equity|
|Quality|Debt/Equity_Diff_T-4|The difference between 4-month period of Debt/ Equity|
|Quality|DividendYield|Dividend/ Price,  also taking z score of n years, n=1,2,3,5,7|
|Quality|DPS|Dividend Per Share for the last twleve months|
|Quality|FCFYield|FCF / Market Cap,  also taking z score of n years, n=1,2,3,5,7|
|Quality|Leverage|Total Assets / Total Equity, also taking the 4-month difference|
|Quality|NetDebt/Equity|Net Debt/Equity|
|Quality|NetDebt/Equity_Diff_T-4|The difference between 4-month period of Net Debt / Equity|
|Quality|NetDebt/MarketCap|Net Debt / Market Cap,  also taking z score of n years, n=1,2,3,5,7|
|Quality|ROE|Return On Equity,  also taking z score of n years, n=1,2,3,5,7, and difference of 4-month period|
|Quality|ROE_Diff_T-4|The difference between 4-month period of ROE|
|Quality|ROEEst_nFY|Return On Equity Estimate for fiscal year n, n=1,2,  also taking z score of n years, n=1,2,3,5,7|
|Quality|ROEVolatility_nY|n-year mean of ROE / n-year standard deviation of ROE , n=3,5|
|Risk|EPSEst_1FY_STDEV|Standard Deviation of ROE Estimate for fiscal year 1|
|Risk|PriceVolatility_nY|n-year mean of Price / n-year standard deviation of Price , n=3,5|
|Value|Cash/MarketCap|Cash / MarketCap,  also taking z score of n years, n=1,2,3,5,7|
|Value|EV/EBIT|Entreprise Value to EBIT,  also taking z score of n years, n=1,2,3,5,7|
|Value|EV/EBITDA|Entreprise Value to EBITDA,  also taking z score of n years, n=1,2,3,5,7|
|Value|EV/Sales|Entreprise Value to Sales,  also taking z score of n years, n=1,2,3,5,7|
|Value|P/CFO|Price / CFO,  also taking z score of n years, n=1,2,3,5,7|
|Value|P/FCF|Price / FCF,  also taking z score of n years, n=1,2,3,5,7|
|Value|P/Sales|Price to Sales,  also taking z score of n years, n=1,2,3,5,7|
|Value|PB_LTM|Price to Book Ratio for the last twelve months, also taking z score of n years, n=1,2,3,5,7|
|Value|PB_NTM|Price To Book Estimate for the next twelve months,  also taking z score of n years, n=1,2,3,5,7|
|Value|PE_LTM|Price to Earnings Ratio for the last twelve months, also taking z score of n years, n=1,2,3,5,7|
|Value|PE_NTM|Price to Earnings Ratio Estimate for the next twelves months,  also taking z score of n years, n=1,2,3,5,7|



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

Before standardizing the data, I will have to load it into my iPython interpreter through the following command: 

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



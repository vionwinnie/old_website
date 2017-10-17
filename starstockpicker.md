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

Traditional quant models combine types of signals using a fixed percentage for each component in order to classify stocks. I am curious to see if deploying machine learning techniques can help improve betting averages of the stocks. My hypothesis is that these machine learning techniques can detect closer relationship of different metrics, assign better weights for different metrics, and thus draw a better separation plane between stocks likelky to go up or down. 

**Topic:** Comparing Performance Between Random Forest Classifier Model (RF) and Support Vector Machine (SVM)

**Benchmark:** Hang Seng Index (HSI) 

**Timeframe:** 2000-2017 Monthly Data 

**Machine Learning Techniques:** Random Forest Classifier, Support Vector Machine


# Data Collection & Feature Selection

I have extracted HSI constituents and the dates of changes in index composition. I saved each security as an individual csv. There are about 60 securities.

For each security, I have extracted 25 base features, including price, cash from operations, earnings, forward P/E, daily volume, short interest, dividend and so on. I then derived 194 factors I have engineered from the base features, grouped into different categories. These categories capture different parts of the stock, from fundamental, momentum, market expectation. Some of them are averages, differences, and Z score of one metrics. Click [Here] to see the feature table.


# Feature Table 

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



# time_series_realEstate-
Forecasting real estate prices in Spain and Portugal 

In the right market, buying or renting out real-estate can be a lucrative option. In many European countries, real estate is once again regaining its pre-2008 value. For this analysis, we will dive deeper into Spain and Portugal’s real estate market over 2016 through early August, 2017. 

This real estate dataset tracks metrics such as the price, square footage, and house/rental description. The average price for this real estate market is around 3,639,388 euros. It should come as little surprise that these rentals are geared towards wealthy European individuals, who can afford these high-valued properties. 

Using a stepwise model, we will choose an ARIMA (auto-regressive integrated moving-average) model to forecast Spain and Portugal’s future real estate prices. For this forecast, we will use an ARIMA (1,1,1) using one lag (auto-regression term) and one lagged forecast error (moving average). 

The time series training real estate data consists of 297 observations starting from January 18th, 2016 through March 31st, 2017. The testing data runs from April 1st, 2017 through August 10th, 2017. Our ARIMA model had a root-mean-squared-error of 3871425 between the actual listed average real estate prices and the predicted average real estate prices. 

Read Here: https://beyondtheaverage.wordpress.com/2018/06/27/forecasting-spain-and-portugals-high-end-housing-market/

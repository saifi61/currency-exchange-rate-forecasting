### INTRODUCTION

In finance, an exchange rate is the rate at which one currency will be exchanged for another currency. It reflects the value of one currency relative to another currency. It is an important aspect in the global economy as it affects international trade and investment. The currency exchange rate can be determined by various factors such as government policies, interest rates, inflation, political stability and more.

The International Monetary Fund (IMF) is an international organization that aims to promote international monetary cooperation, facilitate international trade, and reduce poverty around the world. The IMF works to ensure the stability of the international monetary system—the system of exchange rates and international payments that enables countries to transact with each other. It provides policy advice, financial support, and technical assistance to its member countries, and it acts as a forum for international economic cooperation and decision-making. The IMF's main objectives are to promote international monetary cooperation, facilitate the balanced growth of international trade, promote exchange rate stability, and assist in the establishment of a multilateral system of payments.

### PROBLEM STATEMENT

Forecasting the exchange rate of multiple currencies using historical exchange rate data and external economic indicators. To get the findings on how to control inflation and have a study on the development state by means of economic statistics of all countries by analysing whether the government has introduced better monetary policies based on IMF (International Monetary Fund) rules and regulations.

### OBJECTIVE

To predict and forecast the closing exchange rate of any specific currencies. Here we perform a Time Series Analysis and build an ARIMA, Prophet and Regression model. Results will be compared by evaluating the metrics then predicting and forecasting future values of Euro (EUR) or United States Dollar (USD) currency with respect to the other currencies.

In addition,
The objectives that we can provide with the currency exchange rate dataset include:

> Analyzing the trend and patterns in the currency exchange rate over a certain period of time.
>
> Evaluating the performance of different currencies and assessing their stability and risk.

### DATASET DESCRIPTION

This is a historical dataset of currency foreign exchange rates. This dataset includes 113 currencies in total with exchange values in varied permutations. It can be used for various task like time series analysis and forecasting. There are approximately 1.4million rows and 7 columns in the dataset.

The data has two main columns in particular:

- slug
- currency
  > Slug column is of the form "ABC/PQR" where ABC and PQR are standard active currency codes as of ISO 4217.
  >
  > Currency contains a currency code which is the unit of the value in the tables. For examples, if slug is JPY/INR and currency is INR it means that
  <center>1 JPY = xINR</center>
  where x is in INR

The other columns are the regular OHLC (Open - High - Low - Close) columns along with the date

### PROCESS FLOW

![](image/2023-02-20-09-18-36.png)

### Auto Arima

This is a type of ARIMA model that automatically chooses the most optimal values for the parameters like :

- p: the order of auto-regeression,
- d: the order of integration, and
- q: the order of moving average.

Auto arima works by performing differencing tests(Canova-Hansen method of finding optimal order of seasonal differencing is used.).
It moves through all combinations of p, q, d in a given range and selects the fit with the least value for a chosen metric.
Metrics commonly used are Kwiatkowski–Phillips–Schmidt–Shin, Augmented Dickey-Fuller, or Phillips–Perron.

- We have chosen to use ADF metric for our time series analysis.

We have used auto arima to save time while building and tuning our model since it would be tedious to select optimal p, q, d values for all 300+ currency pairs in our dataset.

The end result is a better fitted model and more accurate prdeictions for our forecasting project.

Summary plots:

- The residual errors appear to have a uniform variance and fluctuate around a mean of zero.
- The density plot on the top right suggests a normal distribution with a mean of zero.
- The red line is aligned with all of the dots. Any significant deviations would indicate a skewed distribution.
- The residual errors are not autocorrelated. Any autocorrelation would imply that the residual errors have a pattern that isn’t explained by the model.

- Dependent Variable – the close values which is what we’re trying to predict.
- Model – ARIMA.
- Date – The date we ran the model
- Time – Wed, 22 Feb 2023
- Sample – 12-21-2003 - 11-17-2019
- No. Observations – 831

- The independent variables are the intercept.
- The error term is sigma2 or epsilon in our equation above.
- Our lag variables is ar.L1

#### Term Significance

We want to make sure each term in our model is statistically significant.
The null for this section is that each coefficient is NOT statistically significant.
Therefore, we want each term to have a p-value of less than 0.05, so we can reject the null hypothesis with statistically significant values.

In our model result, only the intercept is not statistically significant as the p-value is above the 0.05 threshold.

#### Fit Analysis

- Log-Likelihood
  The log-likelihood function identifies a distribution that fits best with the sampled data. While it’s useful, AIC and BIC punish the model for complexity, which helps make our ARIMA model parsimonious.

- Akaike’s Information Criterion
  Akaike’s Information Criterion (AIC) helps determine the strength of the linear regression model. The AIC penalizes a model for adding parameters since adding more parameters will always increase the maximum likelihood value.

- Bayesian Information Criterion
  Bayesian Information Criterion (BIC), like the AIC, also punishes a model for complexity, but it also incorporates the number of rows in the data.

- Hannan-Quinn Information Criterion
  Hannan-Quinn Information Criterion (HQIC), like AIC and BIC, is another criterion for model selection; however, it’s not used as often in practice.

- The aim is to choose the model with the highest log likelihood and lowest AIC. However because we used auto arima to select the best model, we didn't compare the log likelihoods of other parameters for p,d,q.

#### Review Assumptions

Next, we want to make sure our model meets the assumption that the residuals are independent, known as white noise.

If the residuals are not independent, we can extract the non-randomness to make a better model.

- Ljung-Box
  The Ljung Box test, pronounced “Young” and sometimes called the modified Box-Pierce test, tests that the errors are white noise.

The Ljung-Box (L1) (Q) is the LBQ test statistic at lag 1 is 0.01, and the p-value is 0.86.
Since the probability is above 0.05, we can’t reject the null that the errors are white noise.

- Heteroscedasticity
  Heteroscedasticity tests that the error residuals are homoscedastic or have the same variance. The summary performs White’s test. Our summary statistics show a test statistic of 0.63 and a p-value of 0.00, which means we reject the null hypothesis and our residuals show variance.

- Jarque-Bera
  Jarque-Bera tests for the normality of errors. It tests the null that the data is normally distributed against an alternative of another distribution. We see a test statistic of 934.92 with a probability of 0, which means we reject the null hypothesis, and the data is not normally distributed. Also, as part of the Jarque-Bera test, we see the distribution has a slight negative skew and a large kurtosis.

### ARIMA Model

An ARIMA model is created by using pmdarima module.

The order parameter expects a tuple of three integers representing the number of autoregressive (p), differencing (d), and moving average (q) terms to include in the model.
These parameters are extracted from the auto-arima model in the previous step by using the order() function.
The model information, coefficient estimates, model diagnostics, Information criteria and residuals are printed using the fitted summary function.

The forecast() method is used to generate a forecast for the next n time periods in the time series using the fitted ARIMA model. n is the number of test records.

To evaluate the fitting by ARIMA, we have used the below metrics:

1. Mean Squared Error (MSE): MSE measures the average squared difference between the predicted and actual values of a time series. A lower MSE value indicates better performance. The formula for MSE is: MSE = (1/n) \* Σ(actual_i - predicted_i)^2 where n is the number of observations in the time series, actual_i is the actual value of the i-th observation, and predicted_i is the predicted value of the i-th observation.

2. Root Mean Squared Error (RMSE): RMSE is the square root of the MSE, and it measures the average distance between the predicted and actual values of a time series. Like MSE, a lower RMSE value indicates better performance. The formula for RMSE is: RMSE = sqrt((1/n) \* Σ(actual_i - predicted_i)^2)

3. Mean Absolute Error (MAE): MAE measures the average absolute difference between the predicted and actual values of a time series. It is less sensitive to outliers than MSE and RMSE. The formula for MAE is: MAE = (1/n) \* Σ|actual_i - predicted_i|

4. Mean Absolute Percentage Error (MAPE): MAPE measures the average percentage difference between the predicted and actual values of a time series. It is often used to evaluate the accuracy of forecasting models. The formula for MAPE is: MAPE = (1/n) \* Σ|(actual_i - predicted_i)/actual_i|

- These metrics are commonly used to evaluate the performance of ARIMA models.

The metrics from the model based on USD/INR

- MSE: 0.0011339080488869154
- MAE: 0.029262777070431054
- RMSE: 0.033673551177250606
- MAPE: 0.006786148855821739

### Prophet Model

The second model we have tried is the open source prophet model provided by Facebook.
The seasonality mode parameter controls how seasonality is modeled in the time series data. We have chosen additive
by assuming seasonal effects are additive to the trend component.
The parameter daily_seasonality is a boolean that is set to True to include a daily
seasonality component, which means that patterns that repeat on a daily basis will be captured.
Similarly weekly_seasonality is also set to true to capture any weekly seasonaloitys that might be present.

The training data is used to fit the prophet model and it contains a "ds" column which is the
datetime column that represents the time periods in the time series and a "y" column representing the close prices.

The make_future_dataframe() is used to create a new dataframe for future time periods. this is a build in method of the prophet model.
periods parameter specifies the number of time periods for which you want to make predictions. In this case, periods=len(test_data) means
that the number of future time periods is equal to the length of the test data. freq='W-SUN' means that the time series has a weekly
frequency and the week ends on Sunday.

After creating the Prophet model instance and fitting it to the training data, we used it to make predictions for the future time
periods having the length of the test data
using the model.predict() function. The output of the predict() function is a pandas DataFrame that contains the predicted values,
along with the uncertainty intervals for each prediction. The yhat column of the predictions is then used to measure the performance metrics of the model,.

We have use the below metrics to evaluate the prophet model fitting:
1.Mean Squared Error (MSE)
2.Root Mean Squared Error (RMSE)
3.Mean Absolute Error (MAE)
4.Mean Absolute Percentage Error (MAPE)

### Summary

In this project we forecasted the closing price of currency exchange.

There are 7 unique currency to the left of the currency pair

> ['GBP' 'EUR' 'AUD' 'JPY' 'CHF' 'USD' 'INR']

while there are 113 currencies to the right of the pair

> ['EGP' 'SCR' 'NOK' 'SZL' 'MYR' 'PKR' 'NZD' 'CUP' 'CAD' 'CHF' 'ZMW' 'FJD'
> 'XOF' 'IDR' 'JPY' 'UAH' 'TWD' 'MMK' 'BRL' 'CNY' 'HUF' 'MXN' 'BMD' 'KWD'
> 'BBD' 'TTD' 'SEK' 'BWP' 'HNL' 'CZK' 'ISK' 'LKR' 'USD' 'RON' 'EUR' 'TND'
> 'ZAR' 'SGD' 'BYN' 'PEN' 'XPF' 'THB' 'GBP' 'KRW' 'AUD' 'MOP' 'PGK' 'PHP'
> 'BSD' 'CRC' 'JOD' 'MZN' 'TRY' 'BGN' 'DJF' 'COP' 'INR' 'BAM' 'PLN' 'SAR'
> 'ARS' 'IRR' 'NIO' 'MWK' 'LBP' 'IQD' 'KES' 'UGX' 'KZT' 'NGN' 'GTQ' 'PYG'
> 'ETB' 'SVC' 'VND' 'BND' 'MAD' 'KYD' 'ILS' 'MKD' 'HKD' 'RUB' 'AED' 'GMD'
> 'BZD' 'SLL' 'MVR' 'DKK' 'MUR' 'JMD' 'OMR' 'DZD' 'BOB' 'GHS' 'RSD' 'TZS'
> 'UZS' 'BDT' 'CLP' 'MGA' 'QAR' 'ANG' 'HTG' 'BHD' 'ALL' 'BIF' 'SOS' 'MDL'
> 'PAB' 'XAF' 'SDG' 'DOP' 'UYU']

Due to the different currency pair, we designed a function that will take in currency A and display the possible currency B for the selected currency. Moving on, we downsample the data to weekly which resulted to eliminating the missing entries of the daily record.

The weekly data was plotted to check for independent features but none was found, this also means high correlation between variables and the target variable (Closing Value). KDE plot was used to check the distribution of the target variable over time and it does not appear to be normally distributed.

Stationarity test were carried out on the time series by plotting its rolling mean and standard deviation, also using Dickey Fuller Test. Non-stationary series was converted to series using log.

The data was splitted into 90/10 because of the few observations we have as a result of downsampling the data to weekly.

Two models were used in this project to forecast the closing price, ARIMA and PROPHET MODEL.

#### ARIMA MODEL

An ARIMA model is created by using pmdarima module.
The order parameter expects a tuple of three integers representing the number of autoregressive (p), differencing (d), and moving average (q) terms to include in the model.
These parameters are extracted from the auto-arima model in the previous step by using the order() function.

#### PROPHET MODEL

The second model we are considering is the open source prophet model provided by Facebook.
The seasonality mode parameter controls how seasonality is modeled in the time series data. We have chosen additive by assuming seasonal effects are additive to the trend component.
The parameter daily_seasonality is a boolean that is set to True to include a daily seasonality component, which means that patterns that repeat on a daily basis will be captured.
Similarly weekly_seasonality is also set to true to capture any weekly seasonaloitys that might be present.

### Some Currency Evaluation

![](image/summary.jpg)

### Live Exchange Comparison

Using the deployed [Project Dashboard](https://swathi0710-forexprediction-timeseries-dqrtk2.streamlit.app/) below are the live excahnge price and model prediction for the written date

![](image/live_xchang.jpg)

### Limitations

1. Our Dataset does not contain data for the past 2 years which may have helped with more accurate predictions.
2. The project has not taken into consideration the effects of change in one currency pair on another.
3. The project has not acquired data about external factors that may have an effect on exchange rates like socio-economic conditions, international relationships and natural disasters.

### Future Scope for Improvement

1. Data should be collected for the years 2021-2023 and more currency pairs can be added.
2. Additional seasonal models should be explored and compared with the performance of FBprophet.
3. A user friendly Mobile application can be built using the streamlit API.
4. The application can use the currency rate predictions to estimate stock and trading options to help investors make better decisions.
# Option-price-prediction---volatality-based-and-greek-parameters
This research work focused on developing machine learning models that were able to capture the relation between the greek parameters and the option prices.

Option analysis has mathematically derived 
values that are affected by underlying asset value and four 
parameters theta (time value), gamma which is second order 
derivative, delta which is the primary derivative or the 
probability of the event happening, Vega refers to the 
volatility, these are the four parameters that are affecting the 
options prices, used for option price prediction. The main aim 
of this research is to understand the effect of these various 
Greek parameters on predicting option prices. In this 
research we have trained 4 machine learning models which
are, linear regression, random forest, SVR, K-Nearest 
Neighbors. Training data set used is NIFTY for the year 2022 
CE options (European options), a total dataset of 3661 rows, 
was split into 2931 and 731 for training and testing 
respectively. One more test was conducted other than original 
training data that is considering KNIFTY data of 17-April 
2023 with expiry 27-April 2023 to analyze the results of abovementioned models. A stochastic volatality model called Heston 
model was also experimented to compare it with the ML 
models. After analysis, Random Forest was found to have the 
minimum Mean Absolute Error. This research is helpful for 
the algorithm traders and machine learning enthusiasts, who 
try understand such a complex financial system

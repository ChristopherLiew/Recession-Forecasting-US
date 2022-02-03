# Recession Forecasting US
A time series classification project focused on forecasting US economic recessions using regression based and tree based methods. 

![rec](https://user-images.githubusercontent.com/52255272/115595915-13ac6e00-a30a-11eb-84b0-784200875240.png)

Inspiration for this project comes from the seminal paper by Kauppi and Saikkonen (2008) where dynamic probit models were employed to predict historical recessions. The motivation for this projects stems from our interest in validating the efficacy of modern classical machine learning methods and sequence models in predicting U.S. recessions. To this endeavour, we tested:
1) Dynamic probit (Benchmark)
2) Random Forest
3) XGBoost
4) Simple Recurrent Neural Network model
5) Granger Ramanathan Forecast Combination

Deep learning is generally not very useful in macroeconomic predictive modelling due to the low frequency and volume of economic data used (i.e. Only 50 years of monthly or quarterly data) which yields low signal to noise ratios, making data hungry deep neural nets generally poor performers vis a vis statistical learning methods.

```Forecasters``` contains sklearn model classes with useful methods for generating grids given various forecasting horizons and performing hyperparameter tuning on said models.

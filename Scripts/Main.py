'''
A model to forecast next 12 months of Emergency and Normal service calls.
Data: The dataset provided is related to service calls, specifically when they were received,
the priority, and the town where the service is needed.
The priority is contained within the request_type column and is either an emergency (E) or normal (N).
This data is spread across two tables:
              1. RECEIVED: Date and time when the call was received
              2. REQ_INFO: Request type and town for the call
Data provided as a SQLite DB.

Approach to Problem Solving:
Injest data -> EDA -> Model fit -> Frecast
Refer to README.txt for additional details
For best view, read in PyCharm - sections can be concatenated
'''

#region Import
import pandas as pd
import numpy as np
import seaborn as sns
from EDA import ExploratoryDataAanalysis
from DataIngestion import DataInjestion
from Model import Model
pd.set_option("display.max_columns", 120)
pd.set_option("display.max_rows", 120)
#endregion Import

###region DefineInput
filepath = r'../Data/takehomeDB.db'
skipEDA = False
skipModel = False
LogTranform = False
#endregion DefineInput

###region Data Injection and Preprocessing
#Short Description
'''
Read in SQLLite db and conver to pandas; merge the data frames. Check for missing values.
Ensure no loss of information by checking the shapes of the df and sql tables.
Merge date and time and convert time to normal datetime format. 
Encode E/N for summing the number of requests
Extract/create features for ML/forecast/EDA. 
'''
DI = DataInjestion(filepath=filepath,tablenames=['RECEIVED','REQ_INFO'])
data = DI.get_data()
#endregion Data Injection and Preprocessing

### region Exploratory Data Analysis
'''
Class to sample/aggregate data at different frequency.
Class holds different dataframe and can be accessed via dictionary key.
Idea is not to have large df sporadically distributed and a single
container object helps with a clean implementation. Furthermore, repetitive 
plotting calls can be encapsulated by simply passing data frame names as opposed to 
passing large dataframes around. Same goes for other classes
-This step is done to understand, transform and plot the data
-Perform basic QA/QC of the data and check for outliers: Request_E & Request_N outliers 
-Extract trends, seasonility and error components; perform hypothesis testing for stationarity, trends and seasonality
-Check for correlation between variables to decide between univariate vs multivariate forecasting methods
-Fit simple models (Holt Winters) to extract early patterns 
'''
EDA = ExploratoryDataAanalysis(data=data, agg={'REQUEST_TYPE_E': 'sum','REQUEST_TYPE_N':'sum'})
data_daily = EDA.get_data(table_name='Daily')
data_weekly = EDA.get_data(table_name='Weekly')
if not skipEDA:
    dataframe_names = ['Daily', 'Weekly']
    plotvars = ['REQUEST_TYPE_E', 'REQUEST_TYPE_N','REQUEST_TYPE_TOTAL']
    EDA.plot_from_dataframe(dataframe_names=dataframe_names, plotvarsy=plotvars, kind='line', log_transform=False, suffix='REQ', alpha=0.9)
    EDA.plot_from_dataframe(dataframe_names=dataframe_names, plotvarsy=plotvars, kind='line', log_transform=True, suffix='REQ', alpha=0.9)
    EDA.plot_by_category(dataframe_names=['Daily_Town'], x='RECEIVED_DATE', y='REQUEST_TYPE_TOTAL', hue='YEAR', kind='line', alpha=0.9)
    EDA.plot_by_category(dataframe_names=['Daily_Town'], x='RECEIVED_DATE', y='REQUEST_TYPE_TOTAL', hue='TOWN_NAME', kind='line', alpha=0.9)
    EDA.plot_by_category(dataframe_names=['Weekly'], x='WEEK', y='REQUEST_TYPE_TOTAL', hue='YEAR', kind='line', alpha=0.9)
    EDA.plot_by_category(dataframe_names=['Weekly'], x='WEEK', y='REQUEST_TYPE_E', hue='YEAR', kind='line', alpha=0.9)
    EDA.plot_by_category(dataframe_names=['Weekly'], x='WEEK', y='REQUEST_TYPE_N', hue='YEAR', kind='line', alpha=0.9)
    EDA.plot_by_category(dataframe_names=['Daily_Town'], x='TOWN_NAME', y='REQUEST_TYPE_TOTAL', hue='YEAR', kind='bar', alpha=0.9)
    EDA.plot_by_category(dataframe_names=['Daily'], x='REQUEST_TYPE_TOTAL', kind='dist', col='YEAR', alpha=0.9)
    EDA.plot_by_category(dataframe_names=['Daily'], x='REQUEST_TYPE_E', kind='dist', col='YEAR', alpha=0.9)
    EDA.plot_by_category(dataframe_names=['Daily'], x='REQUEST_TYPE_N', kind='dist', col='YEAR', alpha=0.9)
    EDA.plot_by_category(dataframe_names=['Daily_Town'], x='REQUEST_TYPE_TOTAL', y='REQUEST_TYPE_E', kind='line',hue='YEAR', alpha=0.9)
    print('Pearson Correlation (between E vs N) for Daily Data', EDA.pearsonr(df_name='Daily'))
    print('Pearson Correlation (between E vs N) for Daily Data', EDA.pearsonr(df_name='Weekly'))
    EDA.plot_by_category(dataframe_names=dataframe_names, x='REQUEST_TYPE_E',
                         y='REQUEST_TYPE_N', kind='regplot', alpha=0.9)
    EDA.plot_sns_facet_grid(df_name='Daily', facet_row='YEAR',
                            x='REQUEST_TYPE_N', y='REQUEST_TYPE_E',
                            hue=None, sns_plot_obj=sns.regplot,
                            figname='face_plot_regplot_E_vs_N_by_Year')

    ###
    '''
    Check for correlation between Normal and Emergency calls; 
    expectaion is to have an imbalance class for
    sufficiently large number of calls and for very small calls we may see 1:1 relation.
    Later stationarity and seaonsal variations shed more light on statistical properties
    '''
    EDA.plot_from_dataframe(dataframe_names=dataframe_names,plotvarsy='REQUEST_TYPE_E',
                            plotvarsx='REQUEST_TYPE_N',kind='scatter',alpha=0.9, subplots=False)
    EDA.plot_from_dataframe(dataframe_names=['Town'], plotvarsy=plotvars, kind='bar', alpha=0.9, suffix='vardist')
    pearsons_corr = EDA.pearsonr(df_name = 'Town', col_name_1 = 'REQUEST_TYPE_N', col_name_2 = 'REQUEST_TYPE_E')
    print(f'Corelation coefficient between Normal and Emergency calls observed in the daily frequency: ', pearsons_corr)

    ###Plot rolling averages
    lags = [180, 26]
    df_names = ['Daily', 'Weekly']
    prop = ['REQUEST_TYPE_TOTAL']
    for lag, df_name in zip(lags,df_names):
        for name in prop:
            EDA.plot_moving_average(df_name=df_name, col_name=name, windows=[7,30])
            EDA.plot_exponential_smoothing(df_name=df_name, col_name=name, alphas=[0.7, 0.9])
            EDA.plot_auto_correlations(df_name=df_name, col_name=name, lags=lag)
            if df_name == 'Subdaily': continue
            EDA.plot_ETS(df_name=df_name, col_name=name, method='additive')

    for df_name in ['Daily', 'Weekly']:
        EDA.plot_DoubleExpontentialSmoothing(df_name=df_name, cols=plotvars, trend='add', seasonal_periods=2,
                                             include_seasonal=True)

    ### Statistical testing for stationarity
    '''
    For practical purposes we can assume the series to be stationary 
    if it has constant statistical properties over time, ie. the following:
    constant mean, constant variance, an autocovariance that does not depend on time.
    Augmented Dickey fuller test conducts a univariate hypothesis testing on the time series 
    to determine if the series is stationary. Here the HO is that the series is non-stationary 
    (unit root i.e.value of a =1)). If p < 0.05, then the series is stationary. 
    '''
    for df_name, window in zip(['Daily', 'Weekly'],[7, 4]):
        EDA.stationarity_test(df_name, cols=['REQUEST_TYPE_E', 'REQUEST_TYPE_N'], window=window)


#### Clip & Transform Data
data_daily[data_daily['REQUEST_TYPE_E'] > 80].REQUEST_TYPE_E = 80
data_weekly[data_weekly['REQUEST_TYPE_E'] > 400].REQUEST_TYPE_E = 400
if LogTranform:
    for col in ['REQUEST_TYPE_E','REQUEST_TYPE_TOTAL', 'REQUEST_TYPE_N']:
        data_daily[col] = np.log(data_daily[col])
        data_weekly[col] = np.log(data_weekly[col])
###endregion Exploratory Data Analysis

#region Model
'''
The origian data has 0.5M entries. Unfortunately, the data as such could not be used 
for the model building method. This is because if we drill down to request by town we would end up with
handful of data points that are not sufficient for prediction. It is possible to treat this as a classification problem 
if we have lot more data per town. Due to this, the data is being agregated by date and town information is discarded. 
So the prediction is being performed by date and treated as a multivariate time series problem. 
Not all models do a good job in this case but the following models have been explored. Amongst the different approaches,
Prophet (facebook's time series forecasting at scale) is the most straightforward and reasonable. It can also handle 
multiple variables (one could add regressions as constraints) so that dependency between predictors/output is captured.
For additional reading, refer: https://facebook.github.io/prophet/docs/seasonality,_holiday_effects,_and_regressors.html#additional-regressors 
Multiple models for training, testing and forecasting available. 
Models have been tested with varying degrees of predictability.
Tuned models have hyperparameters stored
Untuned models are under 
'''
if not skipModel:
    params_prophet = {'changepoint_prior_scale': 0.1, 'seasonality_prior_scale': 30.0,
                            'n_changepoints': 100, 'daily_seasonality': False,
                            'weekly_seasonality': True, 'yearly_seasonality': False} # tuned params
    params_RNN = {'num_lstm_units':100, 'ts_generator_period':7, 'epochs':30, 'batch_size':1,
                  'num_hidden_lstm_units':[100], 'num_hidden_lstm_layers':1, 'lr':0.01,
                  'num_hidden_dense_units':[60], 'num_hidden_dense_layers':1} # not tuned -> need to work on
    params_ML = {'period':5, 'max_depth': 20, 'n_estimators': 20} # tuned params
    # name: Prophet, RNN, ML, SARIMA, VAR (doens't work well for non-stationary),
    name = 'ML'
    tune_params = False
    cols = ['REQUEST_TYPE_TOTAL','REQUEST_TYPE_E']
    data_list = [data_daily]
    periods = [365]
    freqs = ['D']
    params_list = [params_ML]
    for data, period, freq, params in zip(data_list, periods, freqs, params_list):
        model = Model(name = name, cols = cols, data=data,
                      split_time_str ='20150101', end_time_str = '20151201',
                      future_period = period, freq=freq, params=params,
                      resample=False, resample_freq='1H')
        if tune_params: model.hyper_parameter_tuning()
        model.fit()
        # model.cross_validation()
        model.predict()
        model.plot()
        model.evaluate()
#endregion Model

# Tuned params:
# params_weekly_prophet = {'changepoint_prior_scale': 30.0, 'seasonality_prior_scale': 30.0,
#                          'n_changepoints': 25, 'daily_seasonality': False,
#                          'weekly_seasonality': True, 'yearly_seasonality': False} # tuned params

# Untuned Params:
# parameters_SARIMA = {'p':2, 'd': 1, 'q': 3, 'seasonal_order':((1, 0, 1, 12))} # not tuned -> need to work on
# parameters_VAR = {'maxlag':15} # not tuned -> need to work on
# params_RNN = {'num_lstm_units':100, 'ts_generator_period':7, 'epochs':30, 'batch_size':1,
#                   'num_hidden_lstm_units':[100], 'num_hidden_lstm_layers':1, 'lr':0.01,
#                   'num_hidden_dense_units':[60], 'num_hidden_dense_layers':1} # not tuned -> need to work on
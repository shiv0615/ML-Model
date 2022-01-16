from collections import OrderedDict
import itertools
import matplotlib.pyplot as plt
import matplotlib as mpl
import pandas as pd
import numpy as np
from FBProphet import FBProphet
from ARIMAModel import SARIMAModel, VARModel
from MultivariateTimeSeriesRNN import MultivariateTimeSeriesRNN
from ML import ML
from sklearn.metrics import r2_score, median_absolute_error, mean_absolute_error
from sklearn.metrics import mean_squared_log_error
from sklearn.model_selection import train_test_split
from statsmodels.tools.eval_measures import rmse
from sklearn.model_selection import ParameterGrid
from statsmodels.tsa.statespace.sarimax import SARIMAX
from pmdarima import auto_arima
from sklearn.metrics import mean_squared_error
from prophet import Prophet
from prophet.diagnostics import cross_validation, performance_metrics
from prophet.plot import plot_cross_validation_metric
import warnings
warnings.filterwarnings('ignore')
mpl.rcParams['figure.figsize'] = 17,8
mpl.rcParams.update({'font.size': 16})

size=15
params = {'legend.fontsize': 'large',
          'figure.figsize': (20,8),
          'axes.labelsize': size,
          'axes.titlesize': size,
          'xtick.labelsize': size*0.75,
          'ytick.labelsize': size*0.75,
          'axes.titlepad': 25}
mpl.rcParams.update(params)

'''
Module to build model framework
Peforms test train split
Attaches appropriate model
Calls fit, predict and forecast
Althought tempting to define a superclass, 
this is a deliberate to avoid class inheritence
'''

class Model(object):

    def __init__(self, name, data, split_time_str, end_time_str, params,
                 cols, future_period=730, freq='1D', resample=True, resample_freq='1H'):
        print('=======================================')
        print('Building Model')
        print('Model Name: ', name)
        print('')
        self.data = OrderedDict()

        self.name = name
        self.model_class = None
        self.cols = cols
        self.future_period = future_period
        self.freq = freq
        self.params = params
        self.resample_freq = resample_freq
        self.data['Data'] = data
        self.data['Train'] = None
        self.data['Test'] = None
        self.split_time_str = split_time_str
        self.end_time_str = '20151201'
        self.split_time = pd.to_datetime(self.split_time_str,format='%Y%m%d')
        self.end_time = pd.to_datetime(self.end_time_str,format='%Y%m%d')

        if resample:
            self.resample_data()
        self.split_test_train_data()
        self.attach_model_class()

    def resample_data(self):
        '''
        Downsample data to lower frequency to artificially engineer data while conserving volume
        '''
        self.data['Data'].resample(self.resample_freq).ffill()
        print('Resampled Data Shape: \n', self.data['Data'].shape)

    def split_test_train_data(self):
        '''
        Method to split data to test and train
        '''
        mask_test = (self.data['Data'].index >= self.split_time)
        self.data['Train'] = self.data['Data'][~mask_test]
        self.data['Test'] = self.data['Data'][mask_test]
        print(f'Time to Split Test and Train Data: ', self.split_time)
        for df_name in ['Data', 'Test', 'Train']:
            print(f'Data Shapes for Data: {df_name} is {self.data[df_name].shape}')
        print('')

    def attach_model_class(self):
        '''
        Select appropriate model based on user input
        '''
        if self.name.lower() == 'prophet':
            self.model_class = FBProphet(self.data, self.cols,
                                         self.split_time, self.end_time,
                                         future_period=self.future_period,
                                         freq=self.freq, params=self.params)
        elif self.name.lower() == 'sarima':
            self.model_class = SARIMAModel(self.data, self.cols, params=self.params,
                                           split_time=self.split_time, end_time=self.end_time,
                                           freq=self.freq, future_period=self.future_period)
        elif self.name.lower() == 'var':
            self.model_class = VARModel(self.data, self.cols, params=self.params,
                                        split_time=self.split_time, end_time=self.end_time,
                                        freq=self.freq, future_period=self.future_period)
        elif self.name.lower() == 'rnn':
            self.model_class = MultivariateTimeSeriesRNN(data=self.data, cols=self.cols,
                                                         split_time=self.split_time,
                                                         end_time=self.end_time,
                                                         freq=self.freq, params=self.params,
                                                         future_period=self.future_period)
        elif self.name.lower() == 'ml':
            self.model_class = ML(data=self.data, cols=self.cols, split_time=self.split_time,
                                   end_time=self.end_time,freq=self.freq, params=self.params,
                                    future_period=self.future_period)

    def fit(self):
        '''
        Fit the model to training data
        '''
        self.model_class.fit()

    def predict(self):
        '''
        Predict the model to test data
        '''
        self.model_class.predict()

    def plot(self):
        '''
        Plot results
        '''
        self.model_class.plot()

    def evaluate(self):
        '''
        Evluate the model accuracy
        '''
        self.model_class.evaluate()

    def cross_validation(self):
        '''
        Cross validation to measure training accuracy
        '''
        self.model_class.cross_validation()

    def hyper_parameter_tuning(self):
        '''
        Tune parameters
        '''
        self.model_class.hyper_parameter_tuning()



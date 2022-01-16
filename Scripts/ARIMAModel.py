from collections import OrderedDict
import matplotlib.pyplot as plt
import matplotlib as mpl
import pandas as pd
import numpy as np
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.statespace.sarimax import SARIMAX
from pmdarima import auto_arima
from statsmodels.tsa.vector_ar.var_model import VAR
from statsmodels.tools.eval_measures import rmse
import warnings
warnings.filterwarnings('ignore')
mpl.rcParams['figure.figsize'] = 17,8
mpl.rcParams.update({'font.size': 16})

size=16
params = {'legend.fontsize': size,
          'figure.figsize': (20,8),
          'axes.labelsize': size,
          'axes.titlesize': size,
          'xtick.labelsize': 11,
          'ytick.labelsize': 11,
          'axes.titlepad': 25}
mpl.rcParams.update(params)

class SARIMAModel(object):

    def __init__(self, data, cols, params, split_time=None,
                 end_time=None, freq='D', future_period=None):
        self.data = data
        self.params = params
        self.split_time = split_time
        self.end_time = end_time
        self.freq = freq
        self.future_period = future_period
        self.p = self.params['p']
        self.d = self.params['d']
        self.q = self.params['q']
        self.seasonal_order = self.params['seasonal_order']
        self.cols = cols
        self.model = OrderedDict()
        self.model_fitted = OrderedDict()
        self.tuned_model = OrderedDict()
        self.full_df = OrderedDict()
        self.test_df = OrderedDict()
        self.train_df = OrderedDict()
        self.forecast = OrderedDict()
        self.rmse = OrderedDict()
        self.mean_error = OrderedDict()
        print('Params for ARIMA Model \n', self.params)

        self.prepare_data()
        self.initialize_model()

    def prepare_data(self):
        for col in self.cols:
            self.full_df[col] = self.data['Data'][col]
            self.train_df[col] = self.data['Train'][col]
            self.test_df[col] = self.data['Test'][col]

            # pd.plotting.register_matplotlib_converters()
            f, ax = plt.subplots(figsize=(14, 5))
            self.train_df[col].plot(kind='line', y=col, color='blue', label='Train', ax=ax)
            self.test_df[col].plot(kind='line', y=col, color='red', label='Test', ax=ax)
            plt.title('Test and Train Data Plotted for QA/QC for Column: ' + col)
            plt.tight_layout()
            plt.show()

    def initialize_model(self):
        for col in self.cols:
            self.model[col] = SARIMAX(self.train_df[col],
                                      order=(self.p, self.d, self.q),
                                      seasonal_order=self.seasonal_order)

    def cross_validation(self):
        model_cv = OrderedDict()
        for col in self.cols:
            model_cv[col] = auto_arima(y=self.train_df[col], seasonal=True, m=12)
            print(model_cv[col].summary())

    def fit(self):
        for col in self.cols:
            self.model_fitted[col] = self.model[col].fit()
            print('Summary for Fitted Model for Col: ', col)
            print(self.model_fitted[col].summary())

    def predict(self):
        for col in self.cols:
            start = len(self.train_df[col])
            end = len(self.train_df[col]) + len(self.test_df[col]) + 365
            print(start, end)
            self.forecast[col] = self.model_fitted[col].predict(start=start, end=end, dynamic=False, typ='levels')

    def plot(self):
        for col in self.cols:
            model_fit = self.model_fitted[col]

            # line plot of residuals
            residuals = pd.DataFrame(model_fit.resid)
            residuals.plot()
            plt.title('Line Plot of the Residuals for Column' + col)
            plt.tight_layout()
            plt.show()
            # density plot of residuals
            residuals.plot(kind='kde')
            plt.title('Density Plot of the Residuals for Column' + col)
            plt.tight_layout()
            plt.show()

            # summary stats of residuals
            print(residuals.describe())

            self.custom_plot()

    def custom_plot(self):
        for col in self.cols:
            ax = self.test_df[col].plot(y = col, legend=True, label='Test Data for Column: '+ col)
            self.forecast[col].plot(legend=True, label='Predictions for Column: '+ col, ax=ax)
            plt.tight_layout()
            plt.title('Comparison Between Test and Prediction')
            plt.show()

    def evaluate(self):
        for col in self.cols:
            print(self.forecast[col])
            end_time = self.test_df[col][len(self.test_df[col]) - 1]
            mask = (self.forecast[col].index >= self.split_time) & \
                   (self.forecast[col].index <= end_time)  # self.end_time)
            yhat = self.forecast[col][mask][col]
            test = self.test_df[col][col]

            self.rmse[col] = rmse(yhat, test)
            self.mean_error = np.abs(np.mean(yhat) - np.mean(test))
            print('Computing Model Metric for Prop: ', col)
            print(f'RMSE: {self.rmse[col]}')
            print(f'Mean Error: ', self.mean_error)
            print(f'Mean of Test Data: ', np.mean(test))
            print(f'Mean of Predicted Data: ', np.mean(yhat))

class VARModel(object):

    def __init__(self, data, cols, params, split_time=None,
                 end_time=None, freq='D', future_period=None):
        self.data = data
        self.params = params
        self.cols = cols
        self.maxlags = params['maxlag']
        self.split_time = split_time
        self.end_tme = end_time
        self.freq = freq
        self.future_period = future_period
        self.model = None
        self.model_fitted = None
        self.tuned_model = None
        self.full_df = None
        self.test_df = None
        self.train_df = None
        self.forecast = None
        self.rmse = None
        self.mean_error = None
        print('Params for Vector AR Model \n', self.params)

        self.prepare_data()
        self.initialize_model()

    def prepare_data(self):
        self.full_df = self.data['Data'][self.cols]
        self.train_df = self.data['Train'][self.cols]
        self.test_df = self.data['Test'][self.cols]

        # pd.plotting.register_matplotlib_converters()
        f, ax = plt.subplots(figsize=(14, 5))
        train_lbl = [col + '_TRAIN' for col in self.cols]
        test_lbl = [col + '_TEST' for col in self.cols]
        self.train_df.plot(kind='line', y=self.cols, color='blue', label=train_lbl, ax=ax)
        self.test_df.plot(kind='line', y=self.cols, color='red', label=test_lbl, ax=ax)
        plt.title('Test and Train Data')
        plt.tight_layout()
        plt.show()

    def cross_validation(self):
        pass

    def initialize_model(self):
        self.model = VAR(self.train_df)

    def fit(self):
        self.model_fitted = self.model.fit(maxlags=self.maxlags, ic='aic')
        print(self.model_fitted.summary())

    def plot(self):
        self.model_fitted.plot()
        plt.tight_layout()
        plt.show()

        self.model_fitted.plot_acorr()
        plt.tight_layout()
        plt.show()

    def predict(self):
        lag_order = self.model_fitted.k_ar
        pred = self.model_fitted.forecast(self.model_fitted.y, steps=len(self.test_df))
        self.model_fitted.plot_forecast(2*len(self.test_df))
        self.forecast = pd.DataFrame(index=range(0, len(pred)), columns=self.cols)

    def evaluate(self):
        print(self.forecast.head(5))
        for col in self.cols:

            end_time = self.test_df[col][len(self.test_df[col]) - 1]
            mask = (self.forecast[col].index >= self.split_time) & \
                   (self.forecast[col].index <= end_time)  # self.end_time)
            yhat = self.forecast[col][mask][col]
            test = self.test_df[col][col]

            self.rmse[col] = rmse(yhat, test)
            self.mean_error = np.abs(np.mean(yhat) - np.mean(test))
            print('Computing Model Metric for Prop: ', col)
            print(f'RMSE: {self.rmse[col]}')
            print(f'Mean Error: ', self.mean_error)
            print(f'Mean of Test Data: ', np.mean(test))
            print(f'Mean of Predicted Data: ', np.mean(yhat))

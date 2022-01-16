from collections import OrderedDict
import itertools

import keras.optimizers
import matplotlib.pyplot as plt
import matplotlib as mpl
import pandas as pd
import numpy as np
from sklearn.metrics import r2_score, median_absolute_error, mean_absolute_error
from sklearn.metrics import mean_squared_log_error
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.preprocessing.sequence import TimeseriesGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense,LSTM, Dropout
from tensorflow.keras.callbacks import EarlyStopping

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
__show_plot__ = True


'''
Class to perform RNN forecasting
This is a great method if the data is high frequency and particularly when there
is a loarge volume of data to crunch. 
Although the original data has 0.5M data points, it is sparse and spread across multiple 
towns. So data compression has been performed that has resulted in loss of data and subsequent poor performance of RNN
'''

class MultivariateTimeSeriesRNN(object):
    def __init__(self, data, cols, split_time, end_time, params, freq = '1D', future_period = None):
        self.data = data
        self.cols = cols
        self.split_time = split_time
        self.end_time = end_time
        self.params = params
        self.num_lstm_units = self.params['num_lstm_units']
        self.num_hidden_lstm_units = self.params['num_hidden_lstm_units']
        self.num_hidden_lstm_layers = self.params['num_hidden_lstm_layers']
        self.num_hidden_DENSE_units = self.params['num_hidden_dense_units']
        self.num_hidden_DENSE_layers = self.params['num_hidden_dense_layers']
        self.period = self.params['ts_generator_period']
        self.epochs = self.params['epochs']
        self.batch_size = self.params['batch_size']
        self.future_period = future_period
        self.freq = freq
        self.history = None
        self.forecast = None
        self.train_df = None
        self.test_df = None
        self.scaled_full_df = None
        self.scaled_train = None
        self.scaled_test = None
        self.generator_train = None
        self.generator_test = None
        self.generator_full = None
        self.model = None
        self.scaler = None
        print('Params for RNN Model \n', self.params)

        self.prepare_data()
        self.build_model()

    def prepare_data(self):
        '''
        Utility to function to save plots in the direcoty

        Parameters
        ----------
        plt: pyplot plot handle for the plot/figure frame
        figname: string, name of the figure to be saved to disk

        Returns
        -------
        None
        '''
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
        if __show_plot__: plt.show()

        self.normalize_data()
        self.generate_time_series_read_input()

    def normalize_data(self):
        '''
        Normalize data if using gradient descent for convergence
        Parameters
        ----------
        None

        Returns
        -------
        None
        '''
        self.scaler = MinMaxScaler()
        self.scaler.fit(self.train_df)
        self.scaled_train = self.scaler.transform(self.train_df)
        self.scaled_test = self.scaler.transform(self.test_df)
        self.scaled_full_df = self.scaler.transform(self.full_df)

    def hyper_parameter_tuning(self):
        '''
       Perform Hyper Parameter tuning

       Parameters
       ----------
       None

       Returns
       -------
       None
       '''
        raise NotImplementedError

    def generate_time_series_read_input(self):
        '''
        Method to convert normal time series to RNN ready input
        '''
        # define generator
        length = self.period
        batch_size = self.batch_size
        self.generator_train = TimeseriesGenerator(self.scaled_train, self.scaled_train, length=length, batch_size=batch_size)
        self.generator_test = TimeseriesGenerator(self.scaled_test, self.scaled_test, length=length, batch_size=batch_size)
        self.generator_full = TimeseriesGenerator(self.scaled_full_df, self.scaled_full_df, length=length, batch_size=batch_size)
        print(f'Type of Train Generator: {type(self.generator_train)}')

    def build_model(self):
        '''
        Build a RNN DNN model.
        '''
        keras.backend.clear_session()
        self.model = Sequential()
        self.model.add(LSTM(self.num_lstm_units, activation='relu', input_shape=(self.period, self.scaled_train.shape[1]),
                            return_sequences=True))
        self.model.add(Dropout(0.2))
        if self.num_hidden_lstm_layers > 0:
            for i in range(self.num_hidden_lstm_layers):
                self.model.add(LSTM(self.num_hidden_lstm_units[i], activation='relu',return_sequences=True))
                self.model.add(Dropout(0.2))
        if self.num_hidden_DENSE_layers > 0:
            for i in range(self.num_hidden_DENSE_layers):
                self.model.add(Dense(self.num_hidden_DENSE_units[i], activation='relu'))
        self.model.add(Dense(self.scaled_train.shape[1],activation='relu'))
        self.model.compile(optimizer=keras.optimizers.Adam(lr=self.params['lr']), loss='mse')
        self.model.summary()

    def fit(self):
        '''
        Fit the model
        '''
        early_stop_cb = EarlyStopping(monitor='val_loss', patience=5)
        self.history = self.model.fit_generator(self.generator_train, epochs=self.epochs,
                            validation_data=self.generator_test,
                            callbacks=[early_stop_cb])
        self.model.save('model.h5')

    def plot(self):
        hist = pd.DataFrame(self.history.history)
        hist.plot()
        plt.title('Loss')
        plt.tight_layout()
        if __show_plot__: plt.show()

        self.custom_plot()

    def predict(self):
        test_pred = []
        first_eval_batch = self.scaled_train[-self.period:]
        current_batch = first_eval_batch.reshape((self.batch_size, self.period, self.scaled_train.shape[1]))
        # print(f'first_eval_batch \n {first_eval_batch}')
        # print(f'current_batch\n{current_batch}')
        for i in range(2*(len(self.scaled_test)+1)):
            pred = np.squeeze(self.model.predict(current_batch))[0]
            test_pred.append(pred)
            # print(f'current_batch\n{current_batch}')
            current_batch = np.append(current_batch[:, 1:, :], [[pred]], axis=1)

        self.forecast = pd.DataFrame(data=self.scaler.inverse_transform(test_pred),columns=self.test_df.columns)
        self.forecast['TIME'] = pd.date_range(str(self.split_time), periods=2*(len(self.scaled_test)+1), freq=self.freq)
        self.forecast.set_index(keys='TIME',inplace=True)

    def custom_plot(self):
        train_pred = []
        for x, _ in self.generator_train:
            pred = np.squeeze(self.model.predict(x))[0]
            train_pred.append(pred)

        train_pred = pd.DataFrame(data=self.scaler.inverse_transform(train_pred), columns=self.train_df.columns)
        train_pred['TIME'] = self.train_df.iloc[self.period:].index
        train_pred.set_index(keys='TIME',inplace=True)
        for col in self.cols:
            f, ax = plt.subplots(figsize=(14, 5))
            # print(self.forecast[col], self.test_df[col])
            train_pred.plot(kind='line', y=col, color='blue', label=f'Prediction for Col: {col}', ax=ax)
            self.train_df.plot(kind='line', y=col, color='red', label=f'Train for Col: {col}', ax=ax)
            plt.title('Train vs Fitted Data')
            plt.tight_layout()
            if __show_plot__: plt.show()


        for col in self.cols:
            f, ax = plt.subplots(figsize=(14, 5))
            # print(self.forecast[col], self.test_df[col])
            self.forecast.plot(kind='line', y=col, color='blue', label=f'Prediction for Col: {col}', ax=ax)
            self.test_df.plot(kind='line', y=col, color='red', label=f'Test for Col: {col}', ax=ax)
            plt.title('Prediction vs Test Data (including Forecast)')
            plt.tight_layout()
            if __show_plot__: plt.show()

    def evaluate(self):
        raise NotImplementedError

    def cross_validation(self):
        raise NotImplementedError

    def create_windowed_data(self):
        raise NotImplementedError
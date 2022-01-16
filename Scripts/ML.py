from collections import OrderedDict
import itertools, os
import keras.optimizers
import matplotlib.pyplot as plt
import matplotlib as mpl
import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import GridSearchCV

'''
Approach to treat this as a ML model where in additional features are created by
differencing the data frame. This is a powerful approach for fitting and testing the data
Forecasting is really good for the near term. Long term forecasrting is challenged due to the 
lack of data/loss of data from data compression.
'''

class ML(object):
    def __init__(self, data, cols, split_time,
                 end_time, params, freq='D', future_period=None):
        self.data = data
        self.cols = cols
        self.split_time = split_time
        self.end_time = end_time
        self.params = params
        self.freq = freq
        self.future_period = future_period
        self.path = r'../Figure/ML'
        self.save_fig = True
        self.show_plots = True
        self.input_cols = None
        self.output_cols = None
        self.predicted_val = None
        self.train_df = None
        self.test_df = None
        self.full_df_supv = None
        self.train_df_supv = None
        self.test_df_supv = None
        self.scaled_train = None
        self.scaled_test = None
        self.model = None
        self.scaler = None

        self.create_dir()
        self.prepare_data()

    def create_dir(self):
        '''
        Create Directory to save figures

        Parameters
        ----------
        None

        Returns
        -------
        None
        '''
        if not os.path.exists(self.path):
            if not os.path.exists(os.path.dirname(self.path)):
                os.mkdir(os.path.dirname(self.path))
            os.mkdir(self.path)

    def show_and_save_plots(self, plt, figname):
        '''
        Utility function to show and save plots

        Parameters
        ----------
        None

        Returns
        -------
        None
        '''
        extn = '.png'
        plt.tight_layout()
        print('Saving Figure to: ', os.path.join((self.path), figname + extn))
        if self.save_fig: plt.savefig(os.path.join((self.path), figname + extn), dpi=150)
        if self.show_plots: plt.show()

    # convert series to supervised learning
    def series_to_supervised(self, df, t_bwd=1, t_fwd=1, dropnan=True):
        """
        This section of the code had been extracted from below:
        https://machinelearningmastery.com/convert-time-series-supervised-learning-problem-python/
        convert - time - series - supervised - learning - problem - python /
        Frame a time series as a supervised learning dataset.
        Arguments:
            df: data frame
            t_bwd: Number of lag observations as input (X).
            t_fwd: Number of observations as output (y).
            dropnan: Boolean whether or not to drop rows with NaN values.
        Returns:
            Pandas DataFrame of series framed for supervised learning.
        """
        n_vars = 1 if type(df) is list else df.shape[1]
        cols, names = list(), list()
        # input sequence (t-n, ... t-1)
        for i in range(t_bwd, 0, -1):
            cols.append(df.shift(i))
            names += [('var%d(t-%d)' % (j+1, i)) for j in range(n_vars)]
        # forecast sequence (t, t+1, ... t+n)
        for i in range(0, t_fwd):
            cols.append(df.shift(-i))
            if i == 0:
                names += [('var%d(t)' % (j + 1)) for j in range(n_vars)]
            else:
                names += [('var%d(t+%d)' % (j + 1, i)) for j in range(n_vars)]
        # put it all together
        agg = pd.concat(cols, axis=1)
        agg.columns = names
        # drop rows with NaN values
        if dropnan:
            agg.dropna(inplace=True)
        return agg

    def prepare_data(self):
        '''
        split data to test and train; drop unncessary columns;
        change column names to Prophet required name
        Plot data for QA/QC
        Parameters
        ----------
        None

        Returns
        -------
        None
        '''
        self.full_df = self.data['Data'][self.cols]
        self.train_df = self.data['Train'][self.cols]
        self.test_df = self.data['Test'][self.cols]

        self.full_df_supv = self.series_to_supervised(df=self.full_df, t_bwd=self.params['period'], t_fwd=1)
        self.train_df_supv = self.series_to_supervised(df=self.train_df, t_bwd=self.params['period'], t_fwd=1)
        self.test_df_supv = self.series_to_supervised(df=self.test_df, t_bwd=self.params['period'], t_fwd=1)

        self.output_cols = ['var%d(t)' % (i + 1) for i in range(len(self.cols))]
        self.input_cols = [col for col in self.train_df_supv.columns if col not in self.output_cols]
        print(f'Shape of the Supv Training Data: {self.train_df_supv.shape}')
        print(f'Shape of the Supv Test Data: {self.test_df_supv.shape}')
        print(f'Shape of the Supv Full Data: {self.full_df_supv.shape}')

        # pd.plotting.register_matplotlib_converters()
        f, ax = plt.subplots(figsize=(14, 5))
        train_lbl = [col + '_TRAIN' for col in self.cols]
        test_lbl = [col + '_TEST' for col in self.cols]
        self.train_df.plot(kind='line', y=self.cols, color='blue', label=train_lbl, ax=ax)
        self.test_df.plot(kind='line', y=self.cols, color='red', label=test_lbl, ax=ax)
        plt.title('Test and Train Data')
        self.show_and_save_plots(plt,figname='Train_test_QAQC')

        # self.normalize_data()
        self.build_model()

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
        self.scaler.fit(self.train_df_supv)
        self.scaled_train_supv = self.scaler.transform(self.train_df_supv)
        self.scaled_test_supv = self.scaler.transform(self.train_df_supv)

    def build_model(self):
        '''
        Instantiate model object with select parameters
        Parameters
        ----------
        None

        Returns
        -------
        None
        '''
        self.model = RandomForestRegressor(max_depth=self.params['max_depth'],
                                           n_estimators=self.params['n_estimators'])

    def fit(self):
        '''
        Create a fitted model object
        Parameters
        ----------
        None

        Returns
        -------
        None
        '''
        X = self.train_df_supv[self.input_cols].reset_index(drop=True)
        Y = self.train_df_supv[self.output_cols].reset_index(drop=True)
        self.model.fit(X,Y)

    def predict(self):
        '''
          Predict the model for the test period
          Parameters
          ----------
          None

          Returns
          -------
          None
        '''

        X = self.test_df_supv[self.input_cols].reset_index(drop=True)
        Y = self.test_df_supv[self.output_cols].reset_index(drop=True)
        self.predicted_val = self.model.predict(X)
        print('Random Forest Regression Score: ', self.model.score(X,Y))
        print('Random Forest Feature Importance: ', sorted(zip(self.model.feature_importances_,X.columns), reverse=True))
        self.forecast()

    def plot(self):
        '''
          Plot feature importance

          Parameters
          ----------
          None

          Returns
          -------
          None
        '''
        std = np.std([tree.feature_importances_ for tree in self.model.estimators_], axis=0)
        forest_importances = pd.Series(self.model.feature_importances_)
        fig, ax = plt.subplots()
        forest_importances.plot.bar(yerr=std, ax=ax)
        ax.set_title("Feature importances using MDI")
        ax.set_ylabel("Mean decrease in impurity")
        fig.tight_layout()
        self.show_and_save_plots(plt,figname='FeatureImportances')

        for i, col in enumerate(self.cols):
            self.test_df_supv['Predicted_var%d(t)'%(i+1)] = self.predicted_val[:,i]
            self.test_df_supv[[self.output_cols[i],'Predicted_var%d(t)'%(i+1)]].plot()
            plt.tight_layout()
            plt.title('Plotting Comparisong with Test Data and Predicted Data')
            self.show_and_save_plots(plt,figname='TestVsPredictedDataComparisons')

    def forecast(self):
        '''
          Forecast results for 12 months
          Parameters
          ----------
          None

          Returns
          -------
          None
        '''

        future_pred = []
        df = self.full_df[-(self.params['period']+1):].reset_index(drop=True)
        for i in range(self.future_period):
            forecast_supv_df = self.series_to_supervised(df[-(self.params['period']+1):], t_bwd=self.params['period'], t_fwd=1)
            pred = self.model.predict(forecast_supv_df[self.input_cols])
            future_pred.append(pred)
            add_df = {}
            for i, col in enumerate(self.cols):
                add_df[col] = pred[0,i]
            df = df.append(add_df, ignore_index=True)
            # print('After Fit')
            # print(f'forecast_supv_df.shape: ', forecast_supv_df.shape)
            # print(f'forecast_supv_df', forecast_supv_df)
            # print(f'df', df)
            # print(f'pred', pred)
            # import os
            # if i > 1: os.sys.exit()

        self.forecast = df[-self.future_period:]
        self.forecast['TIME'] = pd.date_range(str(self.end_time), periods=self.future_period, freq=self.freq)
        self.forecast.set_index('TIME',inplace=True)
        for col in self.cols:
            self.forecast[col].plot(label='Prediction',color='r')
            self.test_df[col].plot(label='Test',color='b')
            plt.tight_layout()
            plt.title('Forecast for ' + col)
            self.show_and_save_plots(plt,figname='ForecaseForCol_'+col)

    def cross_validation(self):
        parameters = {
            'max_depth':[5,10,50,100],
            'n_estimators':[2, 10, 20, None]
        }
        cv = GridSearchCV(self.model,param_grid=parameters,cv=5)
        X = self.train_df_supv[self.input_cols].reset_index(drop=True)
        Y = self.train_df_supv[self.output_cols].reset_index(drop=True)
        cv.fit(X,Y)
        for mean, std, param in zip(cv.cv_results_['mean_test_score'],
                                    cv.cv_results_['std_test_score'],
                                    cv.cv_results_['params']):
            print(f'Mean Test Score:{mean}, Std Test Score: {std}, Params:{param}')
        print(f'Best Results: \n', cv.best_params_)
        print(f'Best Score: \n', cv.best_score_)

    def hyper_parameter_tuning(self):
        self.cross_validation()

    def evaluate(self):
        X = self.train_df_supv[self.input_cols].reset_index(drop=True)
        Y = self.train_df_supv[self.output_cols].reset_index(drop=True)
        Ypred = self.model.predict(X)
        print('RMSE for Training Data: \n', np.sqrt(mean_squared_error(Y, Ypred)))
        print('R2Score for Training Data: \n', r2_score(Y, Ypred))
        print('Mean Error for Training Data \n', np.abs(np.mean(Ypred) - np.mean(Y)))

        Y = self.test_df_supv[self.output_cols].reset_index(drop=True)
        print('RMSE for Test Data: \n', np.sqrt(mean_squared_error(Y,self.predicted_val)))
        print('R2Score for Test Data: \n', r2_score(Y, self.predicted_val))
        print('Mean Error for Training Data \n', np.abs(np.mean(self.predicted_val) - np.mean(Y)))


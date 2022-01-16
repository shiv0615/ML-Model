from collections import OrderedDict
import itertools, os
import matplotlib.pyplot as plt
import matplotlib as mpl
import pandas as pd
import numpy as np
from sklearn.metrics import r2_score
from statsmodels.tools.eval_measures import rmse
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

class FBProphet(object):

    def __init__(self, data, cols, split_time, end_time, params, future_period=730, freq='D'):
        self.data = data
        self.cols = cols
        self.future_df = None
        self.future_period = future_period
        self.freq = freq
        self.split_time = split_time
        self.params = params
        self.path = r'../Figure/FBProphet'
        self.save_fig = True
        self.show_plots = True
        self.initial = str(2*365) + ' days'
        self.period = str(2*365) + ' days'
        self.horizon = str(180) + ' days'
        print('Params for Prophet Model \n', self.params)

        self.end_time = end_time
        self.model = OrderedDict()
        self.tuned_model = OrderedDict()
        self.tuned_model_fitted = OrderedDict()
        self.full_df = OrderedDict()
        self.test_df = OrderedDict()
        self.train_df = OrderedDict()
        self.forecast = OrderedDict()
        self.rmse = OrderedDict()
        self.mean_error = OrderedDict()

        self.create_dir()
        self.build_model()
        self.prepare_data()

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
        for i, col in enumerate(self.cols):
            print('Parameter for Prophet Model:')
            print(self.params)
            self.model[col] = Prophet(**self.params)
            self.model[col].add_seasonality(name='daily',period=3,fourier_order=15)
            self.model[col].add_seasonality(name='weekly', period=7, fourier_order=35)
            self.model[col].add_seasonality(name='yearly', period=365.25, fourier_order=35)
            for j in range(i):
                self.model[col].add_regressor('add' + str(j+1))


    def create_dir(self):
        '''
        Directory to save image
        :return:
        '''
        if not os.path.exists(self.path):
            if not os.path.exists(os.path.dirname(self.path)):
                os.mkdir(os.path.dirname(self.path))
            os.mkdir(self.path)

    def show_and_save_plots(self,plt, figname):
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
        extn = '.png'
        plt.tight_layout()
        print('Saving Figure to: ',os.path.join((self.path),figname+extn))
        if self.save_fig: plt.savefig(os.path.join((self.path),figname+extn),dpi=150)
        if self.show_plots: plt.show()

    def get_model(self, col):
        '''
        Get model handle corresponding to the variable being forecasted

        Parameters
        ----------
        col: string, to fetch corresponding model

        Returns
        -------
        model handle
        '''
        return self.model[col]

    def make_future_df(self):
        '''
        Create period to forecast

        Parameters
        ----------
        None

        Returns
        -------
        None
        '''
        self.future_df = self.model[self.cols[0]].make_future_dataframe(periods=self.future_period,freq=self.freq)

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
        for i, col in enumerate(self.cols):
            if i == 0:
                self.full_df[col] = self.data['Data'][col].reset_index()
                self.train_df[col] = self.data['Train'][col].reset_index()
                self.test_df[col] = self.data['Test'][col].reset_index()
                self.train_df[col].columns = ['ds', 'y']
                self.test_df[col].columns = ['ds', 'y']
                self.full_df[col].columns = ['ds', 'y']
            else:
                col_list = self.cols[:i+1]
                col_list.reverse()
                colnames = ['ds', 'y']
                for j in range(i): colnames.append('add' + str(j+1))
                self.full_df[col] = self.data['Data'][col_list].reset_index()
                self.train_df[col] = self.data['Train'][col_list].reset_index()
                self.test_df[col] = self.data['Test'][col_list].reset_index()
                self.train_df[col].columns = colnames
                self.test_df[col].columns = colnames
                self.full_df[col].columns = colnames

            self.full_df[col]['floor'] = 0
            self.train_df[col]['floor'] = 0
            self.test_df[col]['floor'] = 0
            if self.freq == 'D':
                self.full_df[col]['cap'] = 1000
                self.train_df[col]['cap'] = 1000
                self.test_df[col]['cap'] = 1000
            elif self.freq == 'W':
                self.full_df[col]['cap'] = 5000
                self.train_df[col]['cap'] = 5000
                self.test_df[col]['cap'] = 5000

            # pd.plotting.register_matplotlib_converters()
            f, ax = plt.subplots(figsize=(14, 5))
            self.train_df[col].plot(kind='line', x='ds', y='y', color='blue', label='Train', ax=ax)
            self.test_df[col].plot(kind='line', x='ds', y='y', color='red', label='Test', ax=ax)
            plt.title('Test and Train Data Plotted for QA/QC for Column: ' + col)
            self.show_and_save_plots(plt,figname='Train_test_QAQC_for_col_'+col)

    def fit(self):
        '''
        Fit the model to the training data
        Parameters
        ----------
        None

        Returns
        -------
        None
        '''
        for i, col in enumerate(self.cols):
            self.model[col].fit(self.train_df[col])
        self.make_future_df()

    def predict(self):
        for i, col in enumerate(self.cols):
            if i > 0:
                col_list = self.cols[:i + 1]
                col_list.reverse()
                n = len(col_list)-1
                for j in range(i):
                    self.future_df['add'+str(j+1)] = self.forecast[col_list[n-j]]['yhat']
            self.forecast[col] = self.model[col].predict(self.future_df)
            mask = (self.forecast[col]['yhat'] < 0)
            self.forecast[col].loc[mask,'yhat'] = 0

    def plot(self):
        '''
        Plot results through Prophet plotting functions

        Parameters
        ----------
        None

        Returns
        -------
        None
        '''

        for col in self.cols:
            self.model[col].plot(self.forecast[col],ylabel=col)
            plt.title('Model Forecast Plot: ' + col)
            self.show_and_save_plots(plt,figname='ModelForecastPlotWithTest_for_col_'+col)

            self.model[col].plot_components(self.forecast[col])
            plt.title('Model Forecast Plot: ' + col)
            self.show_and_save_plots(plt,figname='ModelForecastComponentPlots_for_col_'+col)

        self.custom_plot()

    def custom_plot(self):
        '''
        Generate custom plots

        Parameters
        ----------
        None

        Returns
        -------
        None
        '''

        for col in self.cols:
            ax = self.forecast[col].plot(x='ds',y='yhat',label='predictions', legend=True, c='b')
            self.test_df[col].plot(x='ds',y='y',label='test_data', legend=True, ax=ax, c='r', xlim=['2015-01-01','2017-01-01'])
            self.show_and_save_plots(plt,figname='CustomModelForecastPlot_for_col_'+col)

    def evaluate(self):
        '''
        Evaluate model metrics: rmse and r2_score

        Parameters
        ----------
        None

        Returns
        -------
        None
        '''
        for col in self.cols:
            end_time = self.test_df[col]['ds'][len(self.test_df[col]['ds'])-1]
            mask = (self.forecast[col].ds >= self.split_time) & \
                   (self.forecast[col].ds <= end_time) #self.end_time)
            yhat = self.forecast[col][mask]['yhat']
            test = self.test_df[col]['y']

            self.rmse[col] = rmse(yhat,test)
            self.mean_error = np.abs(np.mean(yhat) - np.mean(test))
            print('Computing Model Metric for Prop: ', col)
            print(f'RMSE: {self.rmse[col]}')
            print('R2Score for Training Data: \n', r2_score(yhat, test))
            print(f'Mean Error: ', self.mean_error)
            print(f'Mean of Test Data: ', np.mean(test))
            print(f'Mean of Predicted Data: ', np.mean(yhat))

    def cross_validation(self):
        '''
        Perform cross_validation
        https://facebook.github.io/prophet/docs/diagnostics.html

        Parameters
        ----------
        None

        Returns
        -------
        None
        '''
        df_cv = OrderedDict()
        for col in self.cols:
            df_cv[col] = cross_validation(self.model[col],horizon=self.horizon,
                                          period=self.period,initial=self.initial)
            mask = (df_cv[col]['yhat'] < 0)
            df_cv[col][mask]['yhat'] = 0
            performance_metrics(df_cv[col])
            plot_cross_validation_metric(df_cv[col],metric='rmse')

    def hyper_parameter_tuning(self):
        '''
        Perform Hyper Parameter tuning -
        This section of the code has been extracted from Prophet manual
        https://facebook.github.io/prophet/docs/diagnostics.html

        Parameters
        ----------
        None

        Returns
        -------
        None
        '''
        param_grid = {
                       'changepoint_prior_scale': [0.1, 0.5, 1.0, 10.0, 15.0, 30.0],
                       'seasonality_prior_scale': [0.1, 0.5, 1.0, 10.0, 15.0, 30.0],
                       'n_changepoints': [25, 100],
                       'daily_seasonality': [True,False],
                       'weekly_seasonality': [True, False],
                       'yearly_seasonality': [True, False]
        }

        # Generate all combinations of parameters
        all_params = [dict(zip(param_grid.keys(), v)) for v in itertools.product(*param_grid.values())]
        for i, col in enumerate(self.cols):
            print('Tuning parameter for col: ', col)
            rmses = []
            for params in all_params:
                self.tuned_model[col] = Prophet(**params)
                for j in range(i):
                    self.tuned_model[col].add_regressor('add' + str(j + 1))
                self.tuned_model_fitted[col] = self.tuned_model[col].fit(self.train_df[col])
                df_cv = cross_validation(self.tuned_model_fitted[col], horizon=self.horizon, period=self.period,
                                         initial=self.initial)
                df_p = performance_metrics(df_cv, rolling_window=1)
                rmses.append(df_p['rmse'].values[0])

            # Find the best parameters
            tuning_results = pd.DataFrame(all_params)

            tuning_results['rmse_'+col] = rmses
            print(tuning_results)
            tuning_results.to_excel(r'../Data/hyper_param_tuning_results_' + col + '.xlsx')

            best_params = all_params[np.argmin(rmses)]
            print(best_params)
#region Import
import random, os
import pandas as pd
import numpy as np
from collections import OrderedDict
import matplotlib.pyplot as plt
import matplotlib as mpl
import seaborn as sns
from sklearn.metrics import mean_absolute_error
from scipy.stats import pearsonr
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.stattools import kpss
import warnings
sns.set()
warnings.filterwarnings('ignore')
pd.set_option("display.max_columns", 120)
pd.set_option("display.max_rows", 120)
mpl.rcParams['figure.figsize'] = 17,8
mpl.rcParams["savefig.dpi"] = 300
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
#endregion Import

class ExploratoryDataAanalysis(object):

    def __init__(self, data, agg = {'REQUEST_TYPE_E': 'sum','REQUEST_TYPE_N':'sum'},
                 show_plots=True, save_fig=True):
        self.data = data
        self.agg = agg
        self.CI = 1.96
        self.dataframe_names = ['Subdaily', 'Daily', 'Weekly']
        self.plotvars = ['REQUEST_TYPE_E', 'REQUEST_TYPE_N', 'REQUEST_TYPE_TOTAL']
        self.path = r'../Figure/EDA'
        self.show_plots = show_plots
        self.save_fig = save_fig
        self.dataframes = OrderedDict()
        print('=======================================')
        print('Performing ExploratoryDataAanalysis')
        print('')
        self.create_dir()
        self.create_df()
        plt.close('all')

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
        plt.close('all')

    def create_df(self):
        '''
        Creates data frames at different frequencies
        Perform feature engineering for newly created data frame

        Parameters
        ----------
        plt: pyplot plot handle for the plot/figure frame
        figname: string, name of the figure to be saved to disk

        Returns
        -------
        None
        '''
        print(f'Shape of the input dataset: ', self.data.shape)

        self.data.reset_index(inplace=True)
        self.dataframes['Subdaily'] = self.create_groupby_df(keys=['RECEIVED_DATE_TIME'], agg=self.agg)
        self.data.set_index('RECEIVED_DATE_TIME', inplace=True)

        self.dataframes['Daily'] = self.create_groupby_df(keys=['RECEIVED_DATE'], agg=self.agg)
        self.dataframes['Daily'].index.freq = 'D'

        self.dataframes['Daily_Town'] = self.create_groupby_df(keys=['RECEIVED_DATE','TOWN_NAME'], agg=self.agg)
        self.dataframes['Daily_Town'].index.freq = 'D'
        self.dataframes['Daily_Town'].reset_index(level=1, inplace=True)

        self.dataframes['Midweek'] = self.create_groupby_df(keys='RECEIVED_DATE', agg=self.agg, freq='3D')
        self.dataframes['Midweek'].index.freq = '3D'

        self.dataframes['Weekly'] = self.create_groupby_df(keys='RECEIVED_DATE', agg=self.agg, freq='W')
        self.dataframes['Weekly'].index.freq = 'W'

        self.dataframes['Monthly'] = self.create_groupby_df(keys='RECEIVED_DATE', agg=self.agg, freq='M')
        self.dataframes['Monthly'].index.freq = 'M'

        self.dataframes['Town'] = self.create_groupby_df(keys=['TOWN_NAME'], agg=self.agg)
        self.dataframes['Town'].sort_values(by='REQUEST_TYPE_TOTAL', inplace=True, ascending=False)

        for key in self.dataframes:
            if key != 'Town':
                col = 'RECEIVED_DATE'
                if key == 'Subdaily': col = 'RECEIVED_DATE_TIME'
                self.dataframes[key].reset_index(inplace=True)
                self.dataframes[key]['YEAR'] = self.dataframes[key][col].apply(lambda date: date.year)
                self.dataframes[key]['MONTH'] = self.dataframes[key][col].apply(lambda date: date.month)
                self.dataframes[key]['WEEK'] = self.dataframes[key][col].apply(lambda date: date.week)
                self.dataframes[key]['DAYS'] = self.dataframes[key][col].apply(lambda date: date.day)
                self.dataframes[key].set_index(col, inplace=True)
            print(f'Shape of the {key} is {self.dataframes[key].shape}')
            # print(f'Columns in the dataframe: {key} are {self.dataframes[key].columns}')

    def get_table_names(self):
        '''
        Utility to function to save plots in the direcoty

        Parameters
        ----------
        plt: pyplot plot handle for the plot/figure frame
        figname: string, name of the figure to be saved to disk

        Returns
        -------
        self.dataframes.keys(): list of str containing tablenames (keys for dictionary)
        '''
        return self.dataframes.keys()

    def get_data(self, table_name):
        '''
        Fettch table by name

        Parameters
        ----------
        table_name: str, name of the dataframe or table

        Returns
        -------
        self.dataframes[table_name]: pd.DataFrame, table
        '''
        return self.dataframes[table_name]

    def covariance(self, data1, data2):
        '''
        Compute covariance between two datasets

        Parameters
        ----------
        data1, data2: np array

        Returns
        -------
        np.cov(data1, data2): The covariance matrix of the variables.
        '''
        return np.cov(data1, data2)

    def pearsonr(self, df_name = 'Daily', col_name_1 = 'REQUEST_TYPE_N', col_name_2 = 'REQUEST_TYPE_E'):
        '''
            Pearson correlation coefficient and p-value for testing non-correlation.

            The Pearson correlation coefficient [1]_ measures the linear relationship
            between two datasets.  The calculation of the p-value relies on the
            assumption that each dataset is normally distributed.  (See Kowalski [3]_
            for a discussion of the effects of non-normality of the input on the
            distribution of the correlation coefficient.)  Like other correlation
            coefficients, this one varies between -1 and +1 with 0 implying no
            correlation. Correlations of -1 or +1 imply an exact linear relationship.
            Positive correlations imply that as x increases, so does y. Negative
            correlations imply that as x increases, y decreases.

            The p-value roughly indicates the probability of an uncorrelated system
            producing datasets that have a Pearson correlation at least as extreme
            as the one computed from these datasets.

            Parameters
            ----------
           df_name: str, name of the data frame
           col_name_1: str, column name in the data frame
           col_name_2 = str, column name in the data frame

            Returns
            -------
            corr : float, Pearson's correlation coefficient.
        '''
        corr, _ = pearsonr(self.dataframes[df_name][col_name_1], self.dataframes[df_name][col_name_2])
        return corr

    def create_groupby_df(self, keys, agg=None, freq=None):
        '''
        Create new data frames by performing pandas group by operations

        Parameters
        ----------
        keys: str, column name to pivot
        agg: dictionary, strategy for aggregation/group by operation
        freq: str, to create groupe object

        Returns
        -------
        df: pd.DataFrame, returns grouped data frame
        '''
        if agg is None: agg = self.agg
        if freq is None:
            df = self.data.groupby(keys).agg(agg)
        else:
            df = self.data.groupby(pd.Grouper(key=keys, freq=freq)).agg(agg)
        df['REQUEST_TYPE_TOTAL'] = df['REQUEST_TYPE_E'] + df['REQUEST_TYPE_N']
        df['REQUEST_ET_RATIO'] = df['REQUEST_TYPE_E'] / df['REQUEST_TYPE_TOTAL']
        return df

    def plot_from_dataframe(self, dataframe_names=None, plotvarsy=None, plotvarsx=None,
                            kind='line', subplots=True, alpha=0.4, log_transform=False, suffix=None):
        '''
          Utility function to use pandas plotting capbility

          Parameters
          ----------
          dataframe_names: list, names of the dataframes
          plotvarsy: list, list of column names to plot
          plotvarsx: str, name of the column to use as x axis
          kind: str, Options include {'line', 'scatter', 'bar'} - strategy to plot
          subplots: bool, to turn on/off subplot option
          alpha: float, transparency for the plotting objects
          log_transform: bool, perform log transform
          suffix: str, to add to figname for saving

          Returns
          -------
          None
        '''
        if dataframe_names is None: dataframe_names = self.dataframe_names
        if plotvarsy is None: plotvarsy = self.plotvars
        for key in dataframe_names:
            vars = None
            if isinstance(plotvarsy,list):
                vars = plotvarsy.copy()
            else:
                vars = [plotvarsy]
            if isinstance(plotvarsx,list):
                vars += plotvarsx
            elif plotvarsx is not None:
                vars += [plotvarsx]

            df = self.dataframes[key][vars]
            if log_transform:
                for var in vars: df[var] = np.log(df[var])

            if kind == 'line':
                df.plot(y=plotvarsy, kind=kind, title='Line Plot for '+ key, subplots=subplots)
            elif kind == 'scatter':
                df.plot(x=plotvarsx, y=plotvarsy, alpha=alpha, kind='scatter',
                        title='Scatter Plot for '+ key, subplots=subplots)
            elif kind == 'bar':
                df.plot(y=plotvarsy,kind='bar',title='Plot for Data: '+key,
                        alpha=alpha, subplots=subplots)
            if log_transform: plt.title(f'LOG Transformation Applied for: {vars}')
            else: plt.title(f'Plots for {vars}')

            figname = kind + '_' + 'DF_' + key + '_'
            if suffix is not None:
                figname += suffix
            if log_transform:
                figname += '_Log'

            self.show_and_save_plots(plt,figname)

    def plot_by_category(self, dataframe_names=None, x=None, y=None, hue=None, kind='line',
                         col=None, alpha=0.4, figname=None):
        '''
          Utility function to use seaborn plotting capbilities

          Parameters
          ----------
          dataframe_names: list, names of the dataframes
          y: str, column name for y axis
          x: str, column name for x axis
          hue: str, column name for hue/color by
          alpha: float, transparency for the plotting objects
          kind: str, Options include {'line', 'scatter', 'bar', 'dist', 'violin', 'countplot','catplot'} - strategy to plot
          figname: str, figname for saving

          Returns
          -------
          None
        '''
        if dataframe_names is None: dataframe_names = self.dataframe_names
        for name in dataframe_names:
            if kind == 'line':
                sns.lineplot(x=x, y=y, hue = hue, data=self.dataframes[name].reset_index(), alpha=alpha)
            elif kind == 'scatter':
                sns.scatterplot(x=x, y=y, hue = hue, data=self.dataframes[name].reset_index(), alpha=alpha)
            elif kind == 'bar':
                sns.barplot(x=x, y=y, hue = hue, data=self.dataframes[name].reset_index(), alpha=alpha)
            elif kind == 'dist':
                sns.displot(data=self.dataframes[name].reset_index(), x=x, y=y, hue=hue, alpha=alpha)
            elif kind == 'violin':
                sns.violinplot(data=self.dataframes[name].reset_index(), x=x, y=y, hue=hue, col=col, alpha=alpha)
            elif kind == 'countplot':
                sns.countplot(data=self.dataframes[name].reset_index(), x=x, y=y, hue=hue, col=col, alpha=alpha)
            elif kind == 'catplot':
                sns.catplot(data=self.dataframes[name].reset_index(), x=x, y=y, hue=hue, col=col, alpha=alpha)
            elif kind == 'regplot':
                sns.regplot(data=self.dataframes[name].reset_index(), x=x, y=y, ci=95)
            plt.title(f"Plot for Data: {name}")
            plt.legend()
            if figname is None:
                figname = kind + '_' + 'sns' + '_' + 'data_' + name + '_' + str(y)
            self.show_and_save_plots(plt, figname)

    def plot_sns_facet_grid(self, x, y, hue, sns_plot_obj,
                            df_name = None, df = None, figname=None,
                            facet_row = None, facet_col = None):
        '''
        Plots Facet grid using the sns plot onbect
        :param df_name: str, dataframe name
        :param facet_col: str, name of the col to grid the plot
        :param facet_row: str, name of the row to grid the plot
        :param x: str, col name
        :param y: str, col name
        :param hue: str, col name
        :param sns_plot_obj: obj, seabornd plot object such as sns.scatterplot
        :param df: dataframe
        :param figname: str, for saving figure
        :return: None
        '''
        if df is None: df = self.dataframes[df_name]
        g = sns.FacetGrid(data=df, col=facet_col, row=facet_row, hue=hue, margin_titles=True)
        g.map_dataframe(sns_plot_obj, x=x, y=y)
        plt.title(f"Facet Plot for Data: {df_name}", loc='center')
        self.show_and_save_plots(plt, figname=figname)

    def plot_moving_average(self, df_name, col_name, windows, plot_intervals=False, scale=None):
        '''
        Create moving average plot
        :param df_name: str, name fo the data frame
        :param col_name: str, column name
        :param windows: float, rolling/moving average window
        :param plot_intervals: bool, to plot confidence interval
        :param scale: float, Confidence interval (1.96 for Gaussian)
        :return:
        None
        '''
        if scale is None: scale = self.CI
        plt.title('Moving average\n window size = {}'.format(windows))
        df = self.dataframes[df_name][col_name]
        for window in windows:
            r = random.random()
            b = random.random()
            g = random.random()
            color = (r, g, b)
            rolling_mean = df.rolling(window=window).mean()
            plt.plot(rolling_mean, 'g', label='Rolling mean trend: ' + str(window),c=color)

            # Plot confidence intervals for smoothed values
            if plot_intervals:
                mae = mean_absolute_error(df[window:], rolling_mean[window:])
                deviation = np.std(df[window:] - rolling_mean[window:])
                lower_bound = rolling_mean - (mae + scale * deviation)
                upper_bound = rolling_mean + (mae + scale * deviation)
                plt.plot(upper_bound, 'r--', label='Upper bound / Lower bound: ' + str(window))
                plt.plot(lower_bound, 'r--')

        plt.plot(df[windows[0]:], label='Actual values', alpha=0.2)
        plt.legend(loc='best')
        plt.title('Moving Average Plot for Data: ' + df_name + ' for Column: ' + col_name)
        plt.grid(True)

        figname = 'MovingAvgPlot_data_' + df_name + '_forCol_' + col_name
        self.show_and_save_plots(plt, figname)

    def exponential_smoothing(self, df, alpha):
        '''
        Perform exponential smoothing
        :param df: pd.DataFrame
        :param alpha: float, weight
        :return: result, list
        '''
        result = [df[0]]
        for n in range(1, len(df)):
            result.append(alpha * df[n] + (1 - alpha) * result[n - 1])
        return result
        # return df.ewm(alpha=alpha, adjust=False).mean() #, alpha=alpha)

    def plot_exponential_smoothing(self, df_name, col_name, alphas):
        '''
        Plot Exponential smoothing
        :param df_name: str, data frame name
        :param col_name: str, col name
        :param alphas: float, weight
        :return:
        '''
        df = self.dataframes[df_name][col_name]
        for alpha in alphas:
            r = random.random()
            b = random.random()
            g = random.random()
            color = (r, g, b)
            plt.plot(self.exponential_smoothing(df, alpha), label="Alpha {}".format(alpha),c=color)
        plt.plot(df.values, "c", label="Actual",alpha=0.2)
        plt.legend(loc="best")
        plt.axis('tight')
        plt.title("Exponential Smoothing Plot for Data: " + df_name + 'for Column: '+ col_name)
        plt.grid(True)
        figname = 'ExpSmoothingPlot_data_' + df_name + '_forCol_' + col_name
        self.show_and_save_plots(plt, figname)

    def plot_ETS(self, df_name, col_name, method='multiplicative'):
        '''
        Plot Error, Trend and seasonal components
        :param df_name: str, data frame name
        :param col_name: str, column name
        :param method: ("add", "mul", "additive", "multiplicative") type
        :return: None
        '''
        result = seasonal_decompose(self.dataframes[df_name][col_name],model=method)
        result.plot()
        plt.title('ETS Plot for Data: '+ df_name +' for Column: ' + col_name)
        plt.tight_layout()
        figname = 'ETSPlot_data_' + df_name + '_forCol_' + col_name
        self.show_and_save_plots(plt, figname)

    def plot_DoubleExpontentialSmoothing(self, df_name, cols, trend='add',seasonal_periods=12,include_seasonal=False):
        '''
        Plot and perform Double Exponential Smoothing/Holt Winters Method
        :param df_name: str,name of he data frame
        :param cols: str, column name
        :param trend: ("add", "mul", "additive", "multiplicative") trend components
        :param seasonal_periods: int, seasonal period
        :param include_seasonal: bool, to include seasonal component
        :return: None
        '''
        df = self.dataframes[df_name]
        for col in cols:
            r = random.random()
            b = random.random()
            g = random.random()
            color = (r, g, b)
            if include_seasonal:
                plt.plot(ExponentialSmoothing(df[col],trend=trend,seasonal=trend,
                                              seasonal_periods=seasonal_periods).fit().fittedvalues.shift(-1), label="TES: "+ col, c=color)
            else:
                plt.plot(ExponentialSmoothing(df[col], trend=trend).fit().fittedvalues.shift(-1), label="DES: " + col,
                         c=color)
            plt.plot(df[col], "c", label="Actual", alpha=0.2)
            plt.legend(loc="best")
            plt.axis('tight')
            plt.title("Holt Winters for Data: " + df_name + ' for Column: ' + col)
            plt.grid(True)
            figname = 'DoubleExpSmoothingPlot_data_' + df_name + '_forCol_' + col
            self.show_and_save_plots(plt, figname)

    def plot_auto_correlations(self, df_name, col_name, lags):
        '''
        Plot auto correlation and partial auto correlation
        :param df_name: str, name of the data frame
        :param col_name: str, name of the column
        :param lags: int, lag period to compute correlation
        :return: None
        '''
        title = 'ACF Plot for : '+ df_name + 'for Column ' + col_name
        plot_acf(self.dataframes[df_name][col_name], lags=lags, title=title)
        figname = 'ACFPlot_data_' + df_name + '_forCol_' + col_name
        self.show_and_save_plots(plt, figname)
        title = 'PACF Plot for : ' + df_name + 'for Column ' + col_name
        plot_pacf(self.dataframes[df_name][col_name], lags=lags, title=title)
        figname = 'PACFPlot_data_' + df_name + '_forCol_' + col_name
        self.show_and_save_plots(plt, figname)

    def stationarity_test(self, df_name, cols, window=12):
        '''
        Perform (Augmented Dickey-Fuller & Kwiatkowski-Phillips-Schmidt-Shin) stationarity test along with
        rolling mean and standard deviation statistics
        :param df_name:str, name of the data frame
        :param cols: str, column name for whicht eh test needs to be conducted
        :param window: float, rolling window to compute statistics
        :return:
        '''
        for col in cols:
            # # Determing rolling statistics
            # rolling_mean = self.dataframes[df_name][col].rolling(window=window).mean()
            # rolling_std = self.dataframes[df_name][col].rolling(window=window).std()
            #
            # # Plot rolling statistics:
            # plt.plot(self.dataframes[df_name][col], color='blue', label=col)
            # plt.plot(rolling_mean, color='red', label= col + ' ' + str(window) + ' Rolling Mean')
            # plt.plot(rolling_std, color='black', label= col + ' ' + str(window) + ' Rolling std')
            # plt.legend(loc='best')
            # plt.title('Rolling Stats for Data: ', df_name)
            # figname = 'RollinStats_data_' + df_name
            # self.show_and_save_plots(plt, figname)

            # Perform Augmented Dickey-Fuller test:
            print('----------------------------------------------------')
            print(f'Performing Augmented Dickey-Fuller Test for Stationairy for Data: {df_name} and Column: {col}')
            df_test = adfuller(self.dataframes[df_name][col], autolag='AIC')
            df_output = pd.Series(df_test[0:4],index=['Test Statistic', 'p-value', '#Lags Used', 'Number of Observations Used'])
            for key, value in df_test[4].items():
                df_output['Critical Value (%s)' % key] = value
            print(df_output)

            # Perform KPSS test:
            # Note: Null of KPSS is opposite of ADF; i.e. null hypothesis is trend stationary
            # vs alternate hypothesis non-stattionary.
            print(f'Performing Kwiatkowski-Phillips-Schmidt-Shin Test for Stationairy for Data: {df_name} and Column: {col}')
            kpss_test = kpss(self.dataframes[df_name][col], regression='c')
            kpss_output = pd.Series(kpss_test[0:3], index=['Test Statistic', 'p-value', 'Lags Used'])
            for key, value in kpss_test[3].items():
                kpss_output['Critical Value (%s)' % key] = value
            print(kpss_output)
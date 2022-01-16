#region Import
import sqlite3
import pandas as pd
from collections import OrderedDict
import warnings
warnings.filterwarnings('ignore')
pd.set_option("display.max_columns", 120)
pd.set_option("display.max_rows", 120)
#endregion Import
'''
Module to ingest data. The module reads  SQLite db
Converts it to pandas data frame; ensures proper data time formatting
Calculates columns
Performs quick QA/QC of the data -> data types, missing values, etc.
Performs quick feature engieering; converts str to integers (one hot encoding)
Concatenates data create one data frame ready to perform EDA and subsequently for ML/forecasting
'''

class DataInjestion(object):

    def __init__(self, filepath, tablenames):
        self.filepath = filepath
        self.tablenames = tablenames
        self.data = None
        print('=======================================')
        print('Performing DataInjestion')
        print('')
        self.create_dataframe()

    def create_dataframe(self):
        '''
        Create pandas data frame with add on features
        :return:
        '''
        df = self.load_sqllite_db()
        self.data = df['RECEIVED'].merge(df['REQ_INFO'], left_on='KEY', right_on='KEY').drop('KEY', axis=1)
        self.extract_features_from_datetime()
        self.data.set_index('RECEIVED_DATE_TIME', inplace=True)
        self.OneHotEncoding(['REQUEST_TYPE'])
        for col in ['REQUEST_TYPE_E','REQUEST_TYPE_N']:
            self.data[col].astype('int64').dtypes
        self.data['REQUEST_TYPE_TOTAL'] = self.data['REQUEST_TYPE_E'] + self.data['REQUEST_TYPE_N']
        self.data['REQUEST_ET_RATIO'] = self.data['REQUEST_TYPE_E'] / self.data['REQUEST_TYPE_TOTAL']
        print(f'# null data in the dataframe: ', self.data.isnull().sum())
        print(f' Shape of the merged dataset with engineered features: \n', self.data.shape)
        print(f' Columns in the merged dataset with engineered features: \n', self.data.columns)
        print(f' Data Info \n', self.data.info())
        assert self.data.shape[0] == df['RECEIVED'].shape[0], ' Shape of the merged data frame is different from the input'

    def extract_features_from_datetime(self):
        '''
        Feature engineering from date time object
        '''
        self.data['RECEIVED_DATE'] = pd.to_datetime(self.data['RECEIVED_DATE'])
        self.data['RECEIVED_TIME'] = pd.to_timedelta(self.data['RECEIVED_TIME'].astype('str').apply(lambda x: self.convert24(x)))
        self.data['YEAR'] = self.data['RECEIVED_DATE'].apply(lambda date: date.year)
        self.data['MONTH'] = self.data['RECEIVED_DATE'].apply(lambda date: date.month)
        self.data['WEEK'] = self.data['RECEIVED_DATE'].apply(lambda date: date.week)
        self.data['DAYS'] = self.data['RECEIVED_DATE'].apply(lambda date: date.day)
        self.data['RECEIVED_DATE_TIME'] = self.data['RECEIVED_DATE'] + self.data['RECEIVED_TIME']

    def load_sqllite_db(self):
        '''
        Connect to SQLLite DB and load the db and convert to pandas dataframe
        '''
        df = OrderedDict()
        with sqlite3.connect(self.filepath) as con:
            for tablename in self.tablenames:
                print(f'Reading tablename: {tablename} from sqllite_db: {self.filepath}')
                req_sql = "SELECT * FROM " + tablename
                df[tablename] = pd.read_sql_query(req_sql, con)
                print('----'+tablename+'---\n', df[tablename].head(5))
                print('Table Shape: ', df[tablename].shape)
        return df

    def convert24(self, s):
        '''
        Convert time to a consistent format; hours minutes and second frames
        :param s:
        :return:
        '''
        if s[-2:] == "AM":
            if s[:2] == '12':
                a = str('00' + s[2:4]) + ':00'
            else:
                a = s[:-3] + ':00'
        else:
            if s[:2] == '12':
                a = s[:-3] + ':00'
            else:
                a = str(int(s[:2]) + 12) + s[2:4] + ':00'
        return a

    def OneHotEncoding(self, non_numerical_columns):
        '''
        Convert string to
        :param non_numerical_columns:str, column names
        :return:
        '''
        self.data['REQUEST_TYPE_E'] = 0
        self.data['REQUEST_TYPE_N'] = 0
        self.data.loc[(self.data['REQUEST_TYPE'] == 'E'), 'REQUEST_TYPE_E'] = 1
        self.data.loc[(self.data['REQUEST_TYPE'] == 'N'), 'REQUEST_TYPE_N'] = 1
        # for i in non_numerical_columns:
        #     one_hot = pd.get_dummies(self.data[i], prefix=i)
        #     print('lenght of one_hot', len(one_hot))
        #     print(one_hot.head(5))
        #     self.data = self.data.merge(one_hot,left_on='RECEIVED_DATE_TIME', right_on='RECEIVED_DATE_TIME')
        #     self.data.drop(i, axis = 1, inplace=True)

    def get_data(self):
        '''
        get dataframe
        :return: pd.DataFrame
        '''
        return self.data
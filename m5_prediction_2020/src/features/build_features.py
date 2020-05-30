import pandas as pd 
import numpy as np
import gc

import logging
logger = logging.getLogger()
logger.setLevel(logging.DEBUG)

import collections

from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler

from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder

import joblib
import tqdm


class build_features:
    
    def __init__(self,file_path):
        self.file_path = file_path
        
        self.PRICE_DTYPES = {"store_id": "category", "item_id": "category", 
                "wm_yr_wk": "int16","sell_price":"float32" }
        
        self.CAL_DTYPES={"event_name_1": "category", "event_name_2": "category", "event_type_1": "category", 
         "event_type_2": "category", "weekday": "category", 'wm_yr_wk': 'int16', "wday": "int16",
        "month": "int16", "year": "int16", "snap_CA": "float32", 'snap_TX': 'float32', 'snap_WI': 'float32' }


        self.train_catcols = ['id', 'item_id', 'dept_id','store_id', 'cat_id', 'state_id']
        self.train_numcols = [f"d_{day}" for day in range(1,1969)]
        
        self.TRAIN_DTYPES = {col: "category" for col in self.train_catcols if col != "id"}
        self.TRAIN_DTYPES.update({numcol:"float32" for numcol in self.train_numcols})


    def read_data(self,n_samples = None):
        #Read in data
        filename = 'calendar'
        self.df_cal = pd.read_csv(self.file_path + '/' + filename + '.csv', dtype = self.PRICE_DTYPES)
        logging.info("Finished reading in {rows} rows and {cols} columns from {name}".format(
            rows = self.df_cal.shape[0],cols = self.df_cal.shape[1],name = filename))
        
        
        filename = 'sell_prices'
        self.df_sell = pd.read_csv(self.file_path + '/' + filename + '.csv', dtype = self.PRICE_DTYPES)
        logging.info("Finished reading in {rows} rows and {cols} columns from {name}".format(
            rows = self.df_sell.shape[0],cols = self.df_sell.shape[1],name = filename))
        
        
        filename = 'sales_train_validation'
        self.df_sales = pd.read_csv(self.file_path + '/' + filename + '.csv', dtype = self.PRICE_DTYPES, nrows = n_samples)
        logging.info("Finished reading in {rows} rows and {cols} columns from {name}".format(
            rows = self.df_sales.shape[0],cols = self.df_sales.shape[1],name = filename))
        

    def process_data_pre_merge(self):
        
        self.df_sales = self.df_sales.assign(id=self.df_sales.id.str.replace("_validation", ""))
        self.df_sales = self.df_sales.reindex(columns=self.df_sales.columns.tolist() + ["d_" + str(1913 + i + 1) for i in range(2 * 28)])
        self.df_sales = self.df_sales.melt(id_vars=["id", "item_id", "dept_id", "cat_id", "store_id", "state_id"],
                var_name='d', value_name='demand')
        self.df_sales = self.df_sales.assign(d=self.df_sales.d.str[2:].astype("int16"))

        #For calendar data
        self.df_cal["date"] = pd.to_datetime(self.df_cal["date"])
        self.df_cal = self.df_cal.assign(d=self.df_cal.d.str[2:].astype("int16"))

        #For sales prices
        self.df_sell['sell_price'] = self.df_sell['sell_price'].fillna(0)
    
        logging.info("Processed pre-merge...")
    
    def build_master(self):
        df_master = self.df_sales.merge(self.df_cal, how="left", on="d")
        df_master = df_master.merge(self.df_sell, how="left", on=["wm_yr_wk", "store_id", "item_id"])
        

        self.df_master = df_master

        logging.info("Processed merge - Master dataframe has {rows} rows and {cols} columns".format(
            rows = self.df_master.shape[0],cols = self.df_master.shape[1]))

        gc.collect()

    
    def process_data_post_merge(self):

        #Drop year,wday,wm_yr_wk
        self.df_master.drop(['wm_yr_wk','year','weekday','date'], axis = 1, inplace = True)

        #Encode categorical columns
        self.master_transformer = transformer()
        catcols = ['item_id','dept_id','store_id', 'cat_id', 'state_id',
        'wday','month']
        #Include the following later
        #'event_name_1','event_type_1','event_name_2','event_type_2']
        
        #Normalize numerical columns
        numcols = ['sell_price','demand']
        
        for col in numcols:
            self.df_master[col] = self.master_transformer.normalize(self.df_master,col,mode = 'fit')

        for col in catcols:
            ohe_df = self.master_transformer.one_hot_encode(self.df_master,col,mode = 'fit') 
            self.df_master = pd.concat([self.df_master,ohe_df],axis = 1)
            del self.df_master[col]

            '''
            self.df_master[col] = self.df_master[col].astype('category')
            self.df_master[col] = self.df_master[col].cat.codes.astype("int16")
            self.df_master[col] -= self.df_master[col].min()
            '''

        logging.info("Processed post-merge...")

        gc.collect()

        return self.df_master
    
    def process_data(self,n_samples = None):
        self.read_data(n_samples)
        self.process_data_pre_merge()
        self.build_master()
        return self.process_data_post_merge()

    def export_pickle_file(self,file_path):
        #Features to drop 
        drop_list = ['event_name_1','event_type_1','event_name_2','event_type_2']
        self.df_master.set_index('d')
        for id in (self.df_master.id.unique()):
            self.df_master.drop(drop_list, axis = 1)
            joblib.dump(self.df_master[self.df_master.id == id].values, file_path + '/processed/{}.npy'.format(id))

        logging.info("Finished exporting files...")


class transformer:

    def __init__(self):
        self.normalizer_dict = collections.defaultdict()
        self.standardizer_dict = collections.defaultdict()
        self.ohe_dict = collections.defaultdict()

    def normalize(self,df,col_name,mode):
        
        values = df[col_name].values.reshape(-1,1)
        if mode == 'fit':
            assert(col_name not in self.normalizer_dict.keys()), 'Column already transformed.'
            norm_scaler = MinMaxScaler(feature_range=(0, 1))
            norm_scaler = norm_scaler.fit(values)
            normalized = norm_scaler.transform(values)
            #Save state
            self.normalizer_dict[col_name] = norm_scaler
            #Convert to pandas DataFrame
            normalized_df = pd.DataFrame(normalized,columns = [col_name])
            
            return normalized_df
        
        if mode == 'inv':
            assert(col_name in self.normalizer_dict.keys()), 'Column not transformed.'
            norm_scaler = self.normalizer_dict[col_name]
            inversed = norm_scaler.inverse_transform(values)
            #Convert to pandas DataFrame
            inversed_df = pd.DataFrame(inversed,columns = [col_name])
            
            return inversed_df

    def standardize(self,df,col_name, mode):
        
        
        values = df[col_name].values

        if mode == 'fit':
            assert(col_name not in self.standardizer_dict.keys()), 'Column already transformed.'
            std_scaler = StandardScaler()
            standardized = std_scaler.fit_transform(values)
            #Save state
            self.standardizer_dict[col_name] = std_scaler
            #Convert to pandas DataFrame
            standardized_df = pd.DataFrame(standardized,columns = [col_name])

            return standardized_df
            
        if mode == 'inv_transform':
            assert(col_name in self.standardizer_dict.keys()), 'Column not transformed.'
            std_scaler = self.standardizer_dict[col_name]
            inversed = std_scaler.inverse_transform(values)
            #Convert to pandas DataFrame
            inversed_df = pd.DataFrame(inversed,columns = [col_name])

            return inversed_df

    def one_hot_encode(self,df,col_name,mode):
    
    
        values = df[col_name].values

        if mode == 'fit':
            assert(col_name not in self.ohe_dict.keys()), 'Column already transformed.'
            # integer encode
            label_encoder = LabelEncoder()
            integer_encoded = label_encoder.fit_transform(values)
            #Save state
            self.ohe_dict[col_name] = label_encoder
            # binary encode
            onehot_encoder = OneHotEncoder(sparse=False)
            integer_encoded = integer_encoded.reshape(len(integer_encoded), 1)
            onehot_encoded = onehot_encoder.fit_transform(integer_encoded)
            #Convert to pandas DataFrame
            onehot_encoded_df = pd.DataFrame(onehot_encoded,columns = [col_name + '_' + str(i) for i in range(onehot_encoded.shape[1])])

            return onehot_encoded_df

        if mode == 'inv_transform':
            assert(col_name in self.ohe_dict.keys()), 'Column not transformed.'
            label_encoder = self.ohe_dict[col_name]
            inversed = label_encoder.inverse_transform([np.argmax(onehot_encoded[0, :])])
            #Convert to pandas DataFrame
            inversed_df = pd.DataFrame(inversed,columns = [col_name + '_' + str(i) for i in range(inversed.shape[0])])

            return inversed_df
        
    
        

if __name__ == '__main__':
    #Main function to be implemented
    path = r'/Users/abhisheknamballa/Desktop/Kaggle/m5_prediction/kaggle-m5-accuracy-2020/m5_prediction_2020/data/'
    b_fea = build_features(path + 'raw/')
    
    
    df_master = b_fea.process_data(10)
    print(df_master.info())
    b_fea.export_pickle_file(path)

    gc.collect()
    
    #df_master.to_csv(path + 'external/master_dataframe')


        

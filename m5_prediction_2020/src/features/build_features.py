import pandas as pd 
import numpy as np
import gc

import logging
logger = logging.getLogger()
logger.setLevel(logging.DEBUG)


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
        self.df_master.drop(['wm_yr_wk','year','weekday'], axis = 1, inplace = True)

        #Encode categorical columns
        catcols = ['dept_id','store_id', 'cat_id', 'state_id','event_name_1',
            'event_type_1','event_name_2','event_type_2','snap_CA','snap_TX','snap_WI']
        
        for col in catcols:
            self.df_master[col] = self.df_master[col].astype('category')
            self.df_master[col] = self.df_master[col].cat.codes.astype("int16")
            self.df_master[col] -= self.df_master[col].min()


        logging.info("Processed post-merge...")

        return self.df_master

        gc.collect()
    
    #def add_features(self):
    
    def process_data(self,n_samples = None):
        self.read_data(n_samples)
        self.process_data_pre_merge()
        self.build_master()
        return self.process_data_post_merge()




if __name__ == '__main__':
    #Main function to be implemented
    path = r'/Users/abhisheknamballa/Desktop/Kaggle/m5_prediction/kaggle-m5-accuracy-2020/m5_prediction_2020/data/'
    b_fea = build_features(path + 'raw/')
    
    
    df_master = b_fea.process_data(10)
    
    gc.collect()
    
    df_master.to_csv(path + 'external/master_dataframe')


        

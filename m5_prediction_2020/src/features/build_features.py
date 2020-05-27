import pandas as pd 
import numpy as np
import gc
import utils.utils as utils

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


    #def normalize_data(self,df):


    def pre_process_data(self,filename):
        
        if filename == 'sell_prices':
            df = pd.read_csv(self.file_path + '/' + filename + '.csv', dtype = self.PRICE_DTYPES)
            for col, col_dtype in self.PRICE_DTYPES.items():
                if col_dtype == "category":
                    df[col] = df[col].cat.codes.astype("int16")
                    df[col] -= df[col].min()
                    
        if filename == 'sales_train_validation':
            
            df = pd.read_csv(self.file_path + '/' + filename + '.csv', dtype = self.TRAIN_DTYPES)

            for col in self.train_catcols:
                if col != "id":
                    df[col] = df[col].cat.codes.astype("int16")
                    df[col] -= df[col].min()
            
            df = df.assign(id=df.id.str.replace("_validation", ""))
            df = df.reindex(columns=df.columns.tolist() + ["d_" + str(1913 + i + 1) for i in range(2 * 28)])
            df = df.melt(id_vars=["id", "item_id", "dept_id", "cat_id", "store_id", "state_id"],
                 var_name='d', value_name='demand')
            df = df.assign(d=df.d.str[2:].astype("int16"))

        
        if filename == 'calendar':
            df = pd.read_csv(self.file_path + '/' + filename + '.csv', dtype = self.CAL_DTYPES)
            df["date"] = pd.to_datetime(df["date"])
            for col, col_dtype in self.CAL_DTYPES.items():
                if col_dtype == "category":
                    df[col] = df[col].cat.codes.astype("int16")
                    df[col] -= df[col].min()
            df = df.assign(d=df.d.str[2:].astype("int16"))
            
        return df

    def build_master(self,df_sales, df_cal, df_prices,n_sample = None):
        if not n_sample:
            df_master = df_sales.merge(df_cal, how="left", on="d")
        else:
            #Change functionality so that this picks n_samples of a particular id
            #More intuitive to work with in that way
            df_master = df_sales.sample(n_sample).merge(df_cal, how="left", on="d")

        df_master = df_master.merge(df_prices, how="left", on=["wm_yr_wk", "store_id", "item_id"])
        df_master.drop(['wm_yr_wk'], axis = 1, inplace = True)

        gc.collect()

        return df_master


if __name__ == '__main__':
    #Main function to be implemented
    path = r'/Users/abhisheknamballa/Desktop/Kaggle/m5_prediction/kaggle-m5-accuracy-2020/m5_prediction_2020/data/'
    b_fea = build_features(path + 'raw/')
    
    df_cal = b_fea.pre_process_data('calendar')
    print('Processed calendar data')
    df_prices = b_fea.pre_process_data('sell_prices')
    print('Processed sales price data')
    df_sales = b_fea.pre_process_data('sales_train_validation')
    print('Processed sales qnty data')

    df_master = utils.reduce_mem_usage(b_fea.build_master(df_sales,df_cal,df_prices, n_sample=10))
    print('Processed merge')

    del df_cal
    del df_prices
    del df_sales

    gc.collect()
    
    #df_master.to_feather(path + 'external/master_dataframe')


        

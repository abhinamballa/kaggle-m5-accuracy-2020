from features.build_features import build_features,transformer
from models.predict_model import DataLoader,run_model
import utils
from sklearn.preprocessing import LabelEncoder


import pandas

def main():
    raise NotImplemented




if __name__ == '__main__':
    
    path = r'/Users/abhisheknamballa/Desktop/Kaggle/m5_prediction/kaggle-m5-accuracy-2020/m5_prediction_2020/data/'
    b_fea = build_features(path + 'raw/')
    df_master = b_fea.process_data(10)
    #b_fea.export_pickle_file(path)
    
    

    #Inspired by https://www.kaggle.com/gopidurgaprasad/m5-forecasting-eda-lstm-pytorch-modeling
    # create dataframe for loading npy files and  train valid split

    data_info = df_master[["id", "d"]]

    # total number of days -> 1913
    # for training we are taking data between 1800 < train <- 1913-28-28 = 1857

    train_df = data_info[(1800 < data_info.d) &( data_info.d < 1857)]

    label = LabelEncoder()
    label.fit(train_df.id)


    # valid data is given last day -> 1885 we need to predict next 28days

    valid_df = data_info[data_info.d == 1885]

    ## for exaple one item

    datac = DataLoader(train_df, path + 'processed',label)
    n = datac.__getitem__(100)
    print(n["features"].shape, n["label"].shape)


    run_model(DataLoader,train_df,valid_df)
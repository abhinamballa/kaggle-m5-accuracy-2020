import torch
import torch.nn as nn
import tqdm
import joblib
import numpy as np
import sklearn

class DataLoader:
    def __init__(self, df, file_path,label_transfomer,train_window = 28, predicting_window=28):
        self.df = df.values #df contains the mapping of 
        self.train_window = train_window
        self.predicting_window = predicting_window
        self.file_path = file_path
        self.label = label_transfomer

    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, item):
        df_item = self.df[item]
        df_id = df_item[0]
        day_int = df_item[1]
        
        item_npy = joblib.load(f"{self.file_path}/{df_id}.npy")
        #print('item_npy is', item_npy)
        
        #Columns to be used as features - (normalized) demand, 
            # one hot encoded item_id, dept_id, cat_id, stode_id,
            #date features - (one hot encoded - wday, month, eventnames, 1 and 2, already encoded snap for CA, TX, and WI)
            #Selling price - (normalized) convert NaNs to 0s.

        item_npy_demand = np.array(item_npy[:,3]) #Assuming demand is that column
        #print('item_npy_demand is', item_npy_demand)

        features = np.array(item_npy[day_int-self.train_window:day_int,3:]).astype('float32')
        #print('features is', features)

        predicted_demand = np.array(item_npy_demand[day_int:day_int+self.predicting_window]).astype('float32')
        #print('predicted_demand is', predicted_demand)

        item_label = self.label.transform([df_id])
        item_onehot = [0] * 1000   #Hard coded - needs to change
        item_onehot[item_label[0]] = 1

        list_features = []
        for f in features:
            one_f = []
            one_f.extend(item_onehot)
            one_f.extend(f)
            list_features.append(one_f)

        return {
            "features" : torch.Tensor(list_features),
            "label" : torch.Tensor(predicted_demand)
        }


# ---> Code needs to be tested this point on

#LSTM + NN model (based on https://www.kaggle.com/gopidurgaprasad/m5-forecasting-eda-lstm-pytorch-modeling)

class LSTM(nn.Module):
    #Input size needs to change
    def __init__(self, input_size=1047, hidden_layer_size=100, output_size=28):
        super().__init__()
        self.hidden_layer_size = hidden_layer_size
        self.lstm = nn.LSTM(input_size, hidden_layer_size, batch_first=True)
        self.linear = nn.Linear(hidden_layer_size, output_size)

        self.hidden_cell = (torch.zeros(1,1,self.hidden_layer_size),
                            torch.zeros(1,1,self.hidden_layer_size))
        
    def forward(self, input_seq):

        lstm_out, self.hidden_cell = self.lstm(input_seq)

        lstm_out = lstm_out[:, -1]

        predictions = self.linear(lstm_out)

        return predictions


# loss function
def criterion1(pred1, targets):
    l1 = nn.MSELoss()(pred1, targets)
    return l1


def train_model(model,train_loader, epoch, optimizer, scheduler=None, history=None):
    model.train()
    total_loss = 0
    
    t = tqdm(train_loader)
    
    for i, d in enumerate(t):
        
        item = d["features"].cuda().float()
        y_batch = d["label"].cuda().float()

        optimizer.zero_grad()

        out = model(item)
        loss = criterion1(out, y_batch)

        total_loss += loss
        
        t.set_description(f'Epoch {epoch+1} : , LR: %6f, Loss: %.4f'%(optimizer.state_dict()['param_groups'][0]['lr'],total_loss/(i+1)))
        '''
        if history is not None:
            history.loc[epoch + i / len(X), 'train_loss'] = loss.data.cpu().numpy()
            history.loc[epoch + i / len(X), 'lr'] = optimizer.state_dict()['param_groups'][0]['lr']
        '''
        loss.backward()
        optimizer.step()
        

def evaluate_model(model, val_loader, epoch, scheduler=None, history=None):
    model.eval()
    loss = 0
    pred_list = []
    real_list = []
    RMSE_list = []
    with torch.no_grad():
        for i,d in enumerate(tqdm(val_loader)):
            item = d["features"].cuda().float()
            y_batch = d["label"].cuda().float()

            o1 = model(item)
            l1 = criterion1(o1, y_batch)
            loss += l1
            
            o1 = o1.cpu().numpy()
            y_batch = y_batch.cpu().numpy()
            
            for pred, real in zip(o1, y_batch):
                rmse = np.sqrt(sklearn.metrics.mean_squared_error(real, pred))
                RMSE_list.append(rmse)
                pred_list.append(pred)
                real_list.append(real)

    loss /= len(val_loader)
    
    if scheduler is not None:
        scheduler.step(loss)

    print(f'\n Dev loss: %.4f RMSE : %.4f'%(loss, np.mean(RMSE_list)))

#Run function

def run_model(DataLoader,train_df,valid_df):

    DEVICE = "cuda"
    TRAIN_BATCH_SIZE = 512
    TEST_BATCH_SIZE = 128
    EPOCHS = 1
    start_e = 1


    model = LSTM()
    model.to(DEVICE)

    train_dataset = DataLoader(train_df)

    train_loader = torch.utils.data.DataLoader(
        dataset=train_dataset,
        batch_size= TRAIN_BATCH_SIZE,
        shuffle=True,
        num_workers=4,
        drop_last=True
    )


    valid_dataset = DataLoader(valid_df)

    valid_loader = torch.utils.data.DataLoader(
        dataset=valid_dataset,
        batch_size= TEST_BATCH_SIZE,
        shuffle=False,
        num_workers=4,
        drop_last=True
    )


    optimizer = torch.optim.AdamW(model.parameters(), lr=0.0001)

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=5, mode='min', factor=0.7, verbose=True, min_lr=1e-5)

    for epoch in range(start_e, EPOCHS+1):
        train_model(model, train_loader, epoch, optimizer, scheduler=scheduler, history=None)
        evaluate_model(model, valid_loader, epoch, scheduler=scheduler, history=None)




if __name__ == '__main__':
    print("In main")
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import torch 
import random
import optuna
from optuna.trial import TrialState

from sklearn.preprocessing import MinMaxScaler
from sklearn import preprocessing

def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True

setup_seed(1)
time_window = 100  
time_step = 10     
data_origin = pd.read_csv(r"")
data_origin = data_origin.iloc[:,0]
data_origin.dropna(inplace=True)
data_y = data_origin
data_y = data_y.to_frame()

scaler = MinMaxScaler(feature_range=(0, 1))
data_y_scaled = scaler.fit_transform(data_y)

n = len(data_y_scaled)
test_x = np.zeros((n - time_window - time_step + 1, time_window, 1))
test_y = np.zeros((n - time_window - time_step + 1, time_step, 1))

for i in range(n - time_window - time_step + 1):
    test_x[i] = data_y_scaled[i:i + time_window, :]
    test_y[i] = data_y_scaled[i + time_window:i + time_window + time_step, :]

print(test_x.shape, test_y.shape)

from torch.utils.data import DataLoader, TensorDataset, Subset

x = torch.tensor(test_x, dtype=torch.float32)
y = torch.tensor(test_y, dtype=torch.float32)
dataset = TensorDataset(x, y)

allnumber = x.size(0)
print('allnumber', allnumber)

train_index = np.arange(int(allnumber * 0.8))
trainnum = len(train_index)
print('trainnum', trainnum)

test_index = np.arange(int(allnumber * 0.8), allnumber)
testnum = len(test_index)
print('testnum', testnum)

train_all = Subset(dataset, train_index)
test_all = Subset(dataset, test_index)

train_batch = DataLoader(train_all, batch_size=512, drop_last=False, shuffle= False)
test_batch = DataLoader(test_all, batch_size=512, drop_last=False, shuffle= False)

for bat_x, bat_y in train_batch:
    print(bat_x.shape)
    print(bat_y.shape)
    break
    
print('----------------')

for bat_x, bat_y in test_batch:
    print(bat_x.shape)
    print(bat_y.shape)
    break

y_true = data_y[time_window:]

y_true_train = y_true[0:trainnum + time_step - 1]
y_true_test = y_true[trainnum:]

print('y_true', y_true.shape)
print('y_true_train', y_true_train.shape)
print('y_true_test', y_true_test.shape)

def MakeWeeklyData(ori_step, batch_size, data):
    new_time_steps = ori_step // 5

    new_tensor_data = torch.zeros(batch_size, new_time_steps, 1)

    for i in range(batch_size):
        for j in range(new_time_steps):
            start_idx = j * 5
            end_idx = start_idx + 5
            for k in range(1):
                new_tensor_data[i, j, k] = torch.sum(data[i, start_idx:end_idx, k]) / 5
    
    return new_tensor_data

def MakeMonthlyData(ori_step, batch_size, weekly_data):
    new_time_steps = ori_step // 4

    new_tensor_data = torch.zeros(batch_size, new_time_steps, 1)

    for i in range(batch_size):
        for j in range(new_time_steps):
            start_idx = j * 5
            end_idx = start_idx + 5
            for k in range(1):
                new_tensor_data[i, j, k] = torch.sum(weekly_data[i, start_idx:end_idx, k]) / 4
    
    return new_tensor_data

import torch.nn as nn

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class InputAttention(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(InputAttention, self).__init__()
        self.W1 = nn.Linear(input_size, hidden_size)
        self.W2 = nn.Linear(hidden_size, hidden_size)
        self.v = nn.Linear(hidden_size, 1)

    def forward(self, x, h):
        h = h.repeat(x.size(1), 1, 1).transpose(0, 1)
        x = self.W1(x)
        h = self.W2(h)
        e = torch.tanh(x + h)
        attention_weights = torch.softmax(self.v(e), dim=1)
        return attention_weights
    
class DARNN_Encoder(nn.Module):
    def __init__(self, input_size, hidden_size, dropout4darnn):
        super(DARNN_Encoder, self).__init__()
        self.input_attention = InputAttention(input_size, hidden_size)
        self.lstm = nn.LSTMCell(input_size, hidden_size)
        self.dropout = nn.Dropout(p = dropout4darnn)

    def forward(self, x):
        h = torch.zeros(x.size(0), self.lstm.hidden_size).to(device)
        c = torch.zeros(x.size(0), self.lstm.hidden_size).to(device)

        enc_outputs = []

        for t in range(x.size(1)):
            input_weights = self.input_attention(x[:, -time_window+t:, :], h)  # 时间窗口为96
            weighted_input = torch.mul(input_weights, x[:, -time_window+t:, :])
            h, c = self.lstm(weighted_input.mean(dim=1), (h, c))
            h = self.dropout(h)
            enc_outputs.append(h.unsqueeze(1))

        enc_outputs = torch.cat(enc_outputs, dim=1)
        return enc_outputs[:, -16:, :]

class TemporalAttention(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(TemporalAttention, self).__init__()
        self.U = nn.Linear(input_size, hidden_size)
        self.W = nn.Linear(hidden_size, hidden_size)
        self.v = nn.Linear(hidden_size, 1)

    def forward(self, x, h):
        h = h.repeat(x.size(1), 1, 1).transpose(0, 1)
        x = self.U(x)
        h = self.W(h)
        e = torch.tanh(x + h)
        attention_weights = torch.softmax(self.v(e), dim=1)
        return attention_weights

class DARNN_Decoder(nn.Module):
    def __init__(self, hidden_size, output_size, dropout4darnn):
        super(DARNN_Decoder, self).__init__()
        self.temporal_attention = TemporalAttention(hidden_size, hidden_size)
        self.lstm = nn.LSTMCell(hidden_size, hidden_size)
        self.fc = nn.Linear(hidden_size, output_size)
        self.dropout = nn.Dropout(p = dropout4darnn)

    def forward(self, enc_outputs):
        h = torch.zeros(enc_outputs.size(0), self.lstm.hidden_size).to(enc_outputs.device)
        c = torch.zeros(enc_outputs.size(0), self.lstm.hidden_size).to(enc_outputs.device)

        dec_outputs = []
        for t in range(enc_outputs.size(1)):
            temporal_weights = self.temporal_attention(enc_outputs, h)
            context_vector = torch.sum(temporal_weights * enc_outputs, dim=1)
            h, c = self.lstm(context_vector, (h, c))
            out = self.dropout(self.fc(h))
            dec_outputs.append(out.unsqueeze(1))

        dec_outputs = torch.cat(dec_outputs, dim=1)
        return dec_outputs

class TrendSeasonalDecomposition(nn.Module):
    def __init__(self, in_features):
        super().__init__()
        self.trend = nn.Linear(in_features, in_features)
        self.seasonal = nn.Linear(in_features, in_features)

    def forward(self, x):
        trend_part = self.trend(x)
        seasonal_part = self.seasonal(x)
        return trend_part, seasonal_part

class EnhancedMultiHeadAttention(nn.Module):
    def __init__(self, embed_size, heads):
        super().__init__()
        self.heads = heads
        self.embed_size = embed_size

        self.values = nn.Linear(embed_size, embed_size * heads)
        self.keys = nn.Linear(embed_size, embed_size * heads)
        self.queries = nn.Linear(embed_size, embed_size * heads)
        self.fc_out = nn.Linear(heads * embed_size, embed_size)

    def forward(self, values, keys, queries, mask):
        B = queries.shape[0]
        values = self.values(values).view(B, -1, self.heads, self.embed_size).transpose(1,2)
        keys = self.keys(keys).view(B, -1, self.heads, self.embed_size).transpose(1,2)
        queries = self.queries(queries).view(B, -1, self.heads, self.embed_size).transpose(1,2)

        energy = torch.einsum("nqhd,nkhd->nhqk", [queries, keys])
        
        if mask is not None:
            energy = energy.masked_fill(mask == 0, float('-inf'))

        attention = torch.softmax(energy / (self.embed_size ** (1 / 2)), dim=3)
        out = torch.einsum("nhql,nlhd->nqhd", [attention, values]).reshape(B, -1, self.heads * self.embed_size)
        
        return self.fc_out(out)
    
class AutoformerEncoder(nn.Module):
    def __init__(self, feature_size, heads, num_layers):
        super().__init__()
        self.layer_stack = nn.ModuleList([
            nn.ModuleList([
                TrendSeasonalDecomposition(feature_size),
                EnhancedMultiHeadAttention(feature_size, heads)
            ]) for _ in range(num_layers)
        ])

    def forward(self, x, mask=None):
        for layer in self.layer_stack:
            trend_layer, attention_layer = layer
            trend, seasonal = trend_layer(x)
            x = attention_layer(seasonal, seasonal, seasonal, mask) + trend
        return x
    
class DataEncoder4DailyData(nn.Module):
    def __init__(self, input_size, hidden_size, dropout4darnn):
        super().__init__()
        self.DailyDataEncoder = DARNN_Encoder(input_size, hidden_size, dropout4darnn)

    def forward(self, x):
        enc_x = self.DailyDataEncoder(x)
        return enc_x

    
class DataEncoder4MonthlyData(nn.Module):
    def __init__(self, input_size, head_num, num_layers) :
        super().__init__()
        self.MonthlyDataEncoder = AutoformerEncoder(input_size, head_num, num_layers)
    
    def forward(self, x):
        enc_x = self.MonthlyDataEncoder(x)
        return enc_x

class IntergratedModel(nn.Module):
    def __init__(self, dropout, dropout4darnn, dropout4weekly, hiddensize4darnn, layer_num4weekly, layer_num4monthly):
        super().__init__()

        self.DataDailyProcessor = DataEncoder4DailyData(1, hiddensize4darnn, dropout4darnn)
        self.DataMonthlyProcessor = DataEncoder4MonthlyData(1, 1, layer_num4monthly)

        self.linear4daily1 = nn.Linear(16, 100)
        self.linear4daily2 = nn.Linear(hiddensize4darnn, 1)
        self.linear4out2 = nn.Linear(105, 10)
        self.linear4out1 = nn.Linear(hiddensize4darnn, 1)

        self.decoder = DARNN_Decoder(1, hiddensize4darnn, dropout4darnn)
        self.dropout = nn.Dropout(p = dropout)

    def forward(self, x, batchsize):
        datadaily = x.cuda()
        dataweekly = MakeWeeklyData(100, batchsize, datadaily).cuda()
        datamonthly = MakeMonthlyData(20, batchsize, dataweekly).cuda()

        enc_daily = self.DataDailyProcessor(datadaily)
        enc_daily = self.linear4daily2(self.linear4daily1(enc_daily.transpose(1, 2)).transpose(1, 2))
        enc_monthly = self.DataMonthlyProcessor(datamonthly)

        enc_in = torch.concat([enc_daily, enc_monthly], dim = 1)
        dec_out = self.decoder(enc_in)

        out = self.linear4out2(self.linear4out1(dec_out).transpose(1, 2)).transpose(1, 2)

        out = self.dropout(out)

        return out


def ReturnNet(trial, dropout, dropout4darnn, dropout4weekly, hiddensize4darnn, layer_num4weekly, layer_num4monthly):
    model = IntergratedModel(dropout, dropout4darnn, dropout4weekly, hiddensize4darnn, layer_num4weekly, layer_num4monthly).cuda()
    return model

def print_best_params(study, trial):
    print(f"Best trial so far: Trial #{study.best_trial.number} with params: {study.best_trial.params}")

def objective(trial):
    params = {
        "epoch":500,
        "lr": trial.suggest_float("lr", 0.0001, 0.01, log = True),
        "dropout": trial.suggest_float("dropout", 0, 1),
        "dropout4darnn": trial.suggest_float("dropout4darnn", 0, 1),
        "dropout4weekly": trial.suggest_float("dropout4weekly", 0, 1),
        "hiddensize4darnn":trial.suggest_int("hiddensize4darnn", 16, 128, step = 16),
        "layer_num4weekly":trial.suggest_int("layer_num4weekly", 1, 8, step = 1),
        "layer_num4monthly":trial.suggest_int("layer_num4monthly", 1, 8, step = 1),
    }

    model = ReturnNet(
        trial = trial,
        dropout = params['dropout'],
        dropout4darnn = params['dropout4darnn'],
        dropout4weekly = params['dropout4weekly'],
        hiddensize4darnn = params['hiddensize4darnn'],
        layer_num4weekly = params['layer_num4weekly'],
        layer_num4monthly = params['layer_num4monthly'],
    ).to(device)

    test_R2= train(trial, params, model)
    return test_R2

for x, y in train_batch:
    testmodel = IntergratedModel(0, 0, 0, 2, 2, 2).cuda()
    out = testmodel(x, x.shape[0])
    print(out.shape)
    break

def calculate_loss(y_true, y_pred):
    mse = mean_squared_error(y_true, y_pred)
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    
    def mape(y_true, y_pred):
        y_true = np.array(y_true)
        y_pred = np.array(y_pred)
        mask = y_true != 0
        if np.any(mask):
            return np.mean(np.abs((y_pred[mask] - y_true[mask]) / y_true[mask]))
        else:
            return 0

    mape_value = mape(y_true, y_pred)
    
    return {'MSE': mse, 'MAE': mae, 'R2': r2, 'MAPE': mape_value}

from torch.optim import *
def train(trial, params, model):
    gpu_available = torch.cuda.is_available()

    criterion = nn.MSELoss().cuda()
    epochs = params["epoch"] 

    optimizer = torch.optim.Adam(model.parameters(), lr=params["lr"])
    scheduler = lr_scheduler.StepLR(optimizer, step_size=20, gamma = 0.9)

    all_test_pred_df = pd.DataFrame(columns=['Epoch'] + [f'Time Step {i+1}' for i in range(10)])
    all_test_true_df = pd.DataFrame(columns=['Epoch'] + [f'Time Step {i+1}' for i in range(10)])

    all_train_pred_df = pd.DataFrame(columns=['Epoch'] + [f'Time Step {i+1}' for i in range(10)])
    all_train_true_df = pd.DataFrame(columns=['Epoch'] + [f'Time Step {i+1}' for i in range(10)])

    for epoch in range(epochs):

        print("epoch %d begins"%(epoch + 1))
        model.train()
        train_true = []
        train_pred = []
        sum_train_loss = 0
        
        for data, btrue in train_batch:
            if gpu_available:
                data, btrue = data.cuda(), btrue.cuda()
            
            y_pred = model(data, data.shape[0])
            
            train_true.append(btrue)
            train_pred.append(y_pred)
            
            loss = criterion(y_pred, btrue)
        
            if gpu_available:
                loss = loss.cpu()
            sum_train_loss += loss

            if optimizer is not None:
                optimizer.zero_grad()
                        
            loss.backward()
            optimizer.step()
            
        scheduler.step()

        print(r"epoch: ",epoch+1)
        
        train_true_sum = torch.cat(train_true, dim=0)
        train_true_sum = train_true_sum.view(-1, 10).cpu().detach().numpy() 

        train_pred_sum = torch.cat(train_pred, dim=0)
        train_pred_sum = train_pred_sum.view(-1, 10).cpu().detach().numpy()
        
        avg_train_loss = sum_train_loss / len(train_batch)
        print(r'avg_train_loss: {:.5f}'.format(avg_train_loss))
        
        #所有的训练误差
        all_train_loss = calculate_loss(train_true_sum, train_pred_sum)
        print(r"Train all_loss: ",all_train_loss) 
        
        if (epoch + 1) % 2000 == 0:
            
            epoch_data = pd.DataFrame([[epoch + 1] + list(train_pred_sum[i]) for i in range(len(train_pred_sum))],
                                    columns=['Epoch'] + [f'Time Step {i+1}' for i in range(10)])
            
            all_train_pred_df = pd.concat([all_train_pred_df, epoch_data], ignore_index=True)
            
            epoch_data = pd.DataFrame([[epoch + 1] + list(train_true_sum[i]) for i in range(len(train_true_sum))],
                                    columns=['Epoch'] + [f'Time Step {i+1}' for i in range(10)])
            all_train_true_df = pd.concat([all_train_true_df, epoch_data], ignore_index=True)

            train_pred_gather = []
            
            for i in range(time_step):
                train_true_sum_i = train_true_sum[0:,i]
                train_pred_sum_i = train_pred_sum[0:,i]
                train_pred_gather.append(train_pred_sum_i)  
                
            plt.figure(figsize=(60,20))
            x0 = range(0,5000)
            plt.plot(x0,y_true_train[:5000].squeeze(), label = 'true',color= 'red',linestyle= '--',linewidth=1.5)
            
            for i in range(10):
                x_i = range(i,5000 + i)
                plt.plot(x_i,train_pred_gather[i][:5000].squeeze(), label =str(i+1)+" step",linewidth=1)
                
            plt.legend()
            plt.title("Train")
            plt.show() 

        test_true=[]
        test_pred=[]
        model.eval()  
        sum_test_loss = 0
        with torch.no_grad():
            for data_test, btrue_test in test_batch:  
                if gpu_available:
                    data_test, btrue_test = data_test.cuda(), btrue_test.cuda()
                
                y_pred_test = model(data_test, data_test.shape[0])
                
                test_true.append(btrue_test)
                test_pred.append(y_pred_test)
        
        loss_test = criterion(y_pred_test, btrue_test)

        if gpu_available:
            loss_test = loss_test.cpu()
        sum_test_loss += loss_test
                
        test_pred_sum = torch.cat(test_pred, dim=0)
        test_pred_sum = test_pred_sum.view(-1, 10).cpu().detach().numpy() 
        
        test_true_sum = torch.cat(test_true, dim=0)
        test_true_sum = test_true_sum.view(-1, 10).cpu().detach().numpy()
        
        avg_test_loss = sum_test_loss / len(test_batch)
        print(r'avg_test_loss: {:.5f}'.format(avg_test_loss))

        all_test_loss = calculate_loss(test_true_sum, test_pred_sum)
        print(r"Test all_loss: ",all_test_loss)

        print("*****************************************")
            
        if (epoch + 1) % 2000 == 0:
            
            epoch_data = pd.DataFrame([[epoch + 1] + list(test_pred_sum[i]) for i in range(len(test_pred_sum))],
                                    columns=['Epoch'] + [f'Time Step {i+1}' for i in range(10)])
            
            all_test_pred_df = pd.concat([all_test_pred_df, epoch_data], ignore_index=True)
            
            epoch_data = pd.DataFrame([[epoch + 1] + list(test_true_sum[i]) for i in range(len(test_true_sum))],
                                    columns=['Epoch'] + [f'Time Step {i+1}' for i in range(10)])
            all_test_true_df = pd.concat([all_test_true_df, epoch_data], ignore_index=True)

            test_pred_gather = []
            
            for i in range(time_step):
                test_true_sum_i = test_true_sum[0:,i]
                test_pred_sum_i = test_pred_sum[0:,i]
                test_pred_gather.append(test_pred_sum_i)  
                
            plt.figure(figsize=(16,6))
            x0 = range(0,len(y_true_test))
            plt.plot(x0,y_true_test, label = 'true',color= 'red',linestyle= '--',linewidth=1.5)
            
            for i in range(10):
                x_i = range(i,len(test_pred_gather[i])+i)
                plt.plot(x_i,test_pred_gather[i], label =str(i+1)+" step",linewidth=1)
                
            plt.legend()
            plt.title("TEST")
            plt.show() 

        # if (epoch + 1) % 10 == 0:
        #     torch.save(model.state_dict(), r"")

        #     last_epoch_pred_data = all_test_pred_df[all_test_pred_df['Epoch'] == epoch + 1]
        #     last_epoch_pred_data.to_csv(r'', index=False)

        #     last_epoch_true_data = all_test_true_df[all_test_true_df['Epoch'] == epoch + 1]
        #     last_epoch_true_data.to_csv(r'', index=False)

    return all_test_loss['R2']

study = optuna.create_study(study_name="paper2024", direction="maximize",
                            sampler=optuna.samplers.TPESampler(seed = 42))
study.optimize(objective, n_trials=100, timeout=864000, callbacks=[print_best_params])

pruned_trials = study.get_trials(deepcopy=False, states=[TrialState.PRUNED])
complete_trials = study.get_trials(deepcopy=False, states=[TrialState.COMPLETE])
print("study stats")
print("number of finished trials: ", len(study.trials))
print("number of pruned trials: ", len(pruned_trials))
print("number of complete trials: ", len(complete_trials))
best_trial = study.best_trial

print("best trial: ", best_trial)
print("value: ", best_trial.value)
print("best trial params: ")
for k, v in best_trial.params.items():
    print("     {}: {}".format(k, v))
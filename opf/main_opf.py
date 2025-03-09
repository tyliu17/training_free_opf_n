import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.model_selection import train_test_split
import time
import math
from datetime import datetime
from torchsummary import summary
import os
import subprocess
import torch.optim as optim

from dnn_model import DNN
from loss_func import LossFunc
# from lstm_model import InferCell
from dataset import Dataset
from outputplot import output_plot
from evaluate_model import evaluate_model
from compute_error_metrics import compute_error_metrics
from flow_violation import compute_line_flow_violation

 # NAS GA
from InferCell import InferCell
from NAS_Search import generate_random_genotype



# Hyperparameters
max_epochs = 3
eval_epoch = 5
tolerance = 5  # 早停耐心值
min_delta = 1e-3


batch_size = 32

# 初始學習率
lr = 0.001
factor = 0.9
patience = 5

# 檢查是否有可用的 GPU，否則使用 CPU
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
timestamp = datetime.now().strftime('%m%d%H%M')
output_path_root = r'C:/Users/USER/Desktop/training_free_opf_n/result/'
output_path = os.path.join(output_path_root, timestamp)
output_path = output_path + '/'
root = r'C:/Users/USER\Desktop/training_free_opf_n/118_data/'
model_path = r'C:/Users/USER/Desktop/training_free_opf_n/model/'


#Injection Shift Factor Matrix 表示每個匯流排的功率注入變化如何影響每條輸電線的潮流。
# 186 行 (n_line = 186)：這代表 電網中有 186 條輸電線 (transmission lines)。
# 118 列 (n_bus = 118)：這代表 系統中有 118 個匯流排 (buses)，符合 IEEE 118-bus 測試系統。
filename = root + '118dc_quad_ISF.txt'
S_isf=pd.read_table(filename,sep=',',header=None).to_numpy() # ISF matrix

# 186
filename=root+'118ac_fmax.txt'
f_max=pd.read_table(filename,sep=',',header=None).to_numpy() # flow limit
n_line = np.size(S_isf,0)

# 將 Numpy 轉換為 PyTorch 張量 (Tensor) 並移至 GPU
S = torch.from_numpy(S_isf).to(device) # ISF
f_max = torch.from_numpy(f_max).to(device) # flow limit

# 讀取資料
x=np.load(root+'ac118_p10_x_v.npy')
y=np.load(root+'ac118_p10_y_v.npy') #電壓
W=np.load(root+'ac118_p10_w.npy')


# scaling on voltage 對電壓進行縮放
#V =(V−0.9)×100 
# 將電壓值放大 100 倍，使數值範圍更適合機器學習（避免梯度消失或過小）。
vy_deviation = 0.9  
vy_scale = 100
y[:,1,:] = (y[:,1,:] - vy_deviation) * vy_scale
# print('voltage range(scaled):',np.min(y[:,1,:]),np.max(y[:,1,:]))

# scaling on price 對價格 (price) 進行縮放
pi_deviation = 0 # 如果想要讓價格在某個範圍內變化，可以修改 pi_deviation 的值。
y[:,0,:] = y[:,0,:] + pi_deviation
# filter out extreme points in price 過濾價格中的極端值
y_sort_arg = np.argsort(np.amax(np.abs(y[:,0,:]),axis=0)) # max extreme
y_sort_arg1 = np.argsort(np.amin(y[:,0,:],axis=0),axis=0) # min extreme
# 移除極端值
del_idx0 = []
del_num = 0
for i in range(del_num):
  del_idx0.append(y_sort_arg[-i])
  del_idx0.append(y_sort_arg1[i])
# 去除重複的索引，並排序
del_idx = [] # keep only non-repeated
[del_idx.append(x) for x in del_idx0 if x not in del_idx]
del_idx = np.sort(del_idx)
# delete extreme points
print('price range old:',np.min(y[:,0,:]),np.max(y[:,0,:]))
print('voltage range old:',np.min(y[:,1,:]),np.max(y[:,1,:]))
# x = np.delete(x, del_idx, axis=2)
# y = np.delete(y, del_idx, axis=2)
print('price range new:',np.min(y[:,0,:]),np.max(y[:,0,:]))
print('voltage range new:',np.min(y[:,1,:]),np.max(y[:,1,:]))
print(x.shape,y.shape)

n_sample=y.shape[-1]
n_bus=y.shape[0]
x_total=x.transpose((1,0,2)).reshape(-1,x.shape[-1])
y_total=y.transpose((1,0,2)).reshape(-1,y.shape[-1])
x_train,x_test,y_train,y_test=train_test_split(x_total.T,y_total.T,test_size=0.2)

#train_set、val_set
params={'batch_size': batch_size,
        'shuffle': True,
        'num_workers': 0}
train=Dataset(x_train,y_train)
train_set=torch.utils.data.DataLoader(train,**params)
val=Dataset(x_test,y_test)
val_set=torch.utils.data.DataLoader(val,**params)

# 資料分析
# 從資料集中建立 DataLoader
train_loader = torch.utils.data.DataLoader(train, batch_size=batch_size, shuffle=True)
val_loader = torch.utils.data.DataLoader(val, batch_size=batch_size, shuffle=True)

# 取第一個 batch 進行分析
train_batch_x, _ = next(iter(train_loader))
val_batch_x, _ = next(iter(val_loader))

# 計算均值和標準差
print(train_batch_x.shape)
train_mean, train_std = train_batch_x.mean(dim=0), train_batch_x.std(dim=0)
val_mean, val_std = val_batch_x.mean(dim=0), val_batch_x.std(dim=0)


# mean_value_train = train_mean.mean().item()
# print('Train Mean:', mean_value_train)
# std_value_train = train_std.mean().item()
# print('Train Std:', std_value_train)
# mean_value_val = val_mean.mean().item()
# print('Validation Mean:', mean_value_val)
# std_value_val = val_std.mean().item()
# print('Validation Std:', std_value_val)

input_size = n_bus*6
seq_len = 6
hidden_size = 10
output_size = n_bus*2
# 測試模型
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# DNN
# net=DNN([n_bus*6,n_bus*10,n_bus*10,n_bus*10,n_bus*2]).to(device) #source code
# net = DNN([n_bus*6, n_bus*5, n_bus*5, n_bus*10, n_bus*10, n_bus*10, n_bus*10, n_bus*2]).to(device)#my code



# NAS
# Generate a random architecture
genotype = generate_random_genotype()
print("Generated Genotype:", genotype)

print("=====================================")
print("input_size", input_size)
print("seq_len", seq_len)
print("hidden_size", hidden_size)
print("output_size", output_size)


# Initialize model
net = InferCell(input_size, seq_len, hidden_size, output_size, genotype).to(device)
print("=====================================")
print("Model Summary:")
print(net)




# criterion = nn.MSELoss()

#lstm
# 設定參數
# input_size = n_bus*6  # 每個時間步的輸入維度
# seq_len = 6  # 設定時間步數 (相當於 n_bus*6)

# hidden_size = 10  # LSTM 的隱藏層維度
# num_layers_list = [1, 3, 2, 2]  # 每層 LSTM 的層數
# output_size = n_bus * 2  # 最終輸出維度 (跟 DNN 相同)

# # 初始化 LSTM-based 模型
# net = InferCell(input_size, seq_len, hidden_size, num_layers_list, output_size).to(device)

optimizer = optim.AdamW(net.parameters(), lr)

scheduler = optim.lr_scheduler.ReduceLROnPlateau(
    optimizer, 
    mode='min',       # 監測 loss
    factor=factor,       # 學習率變成 50% (下降更多)
    patience=patience,       # 3 個 epoch 沒有改善就降低學習率
    verbose=True,     
    min_lr=1e-6       
)

## params needed for S calculation
# line parameters
# ieee118_lineloc.txt：線路的連接資訊（節點之間的連接關係）。
# ieee118_lineparams.txt：輸電線路的物理參數（電阻 r、電抗 x、電納 b）。
# ieee118_Bmat.txt：B 矩陣（電網的導納矩陣）。
filename1 = root+'ieee118_lineloc.txt'
filename2 = root+'ieee118_lineparams.txt'
filename3 = root+'ieee118_Bmat.txt'
# incidence info 線從哪裡連到哪裡
line_loc = pd.read_table(filename1,sep=',',header=None).to_numpy()

# r, x, shunt, S_max
line_params = pd.read_table(filename2,sep=',',header=None).to_numpy()
B_mat=pd.read_table(filename3,sep=',',header=None).to_numpy()
B_r = np.delete(B_mat,68,axis=0)
B_r = np.delete(B_r,68,axis=1)
Br_inv = np.linalg.inv(B_r)

R_line = line_params[:, 0].copy()  # 電阻 R
X_line = line_params[:, 1].copy()  # 電抗 X
B_shunt = line_params[:, 2].copy()  # 電納 B_shunt

Z_line = R_line + 1j * X_line 
Y_line = 1 / Z_line

G_line = np.real(Y_line)
B_line = np.imag(Y_line)

# transformer indicator
a = (R_line > 0).astype(int)

# params to tensor and GPU
G_line_tensor = torch.from_numpy(G_line).to(device) # conductance
B_line_tensor = torch.from_numpy(B_line).to(device) # susceptance
B_shunt_tensor = torch.from_numpy(B_shunt/2).to(device) # conductance
Br_inv_tensor = torch.from_numpy(Br_inv).to(device) # reduced Bbus matrix
a_tensor = torch.from_numpy(a).double().to(device) # line/transformer

my_loss = LossFunc(f_max, G_line_tensor, B_line_tensor, B_shunt_tensor, Br_inv_tensor, a_tensor, line_loc, device, n_bus)

# 訓練模型
train_loss = []
val_loss = []
lr_list = []

t0 = time.time()
previous = float('inf')

feas = False  # 加入可行性標記
# local_batch	輸入特徵 x	(512, 118 * 6)
# local_label	真實標籤 y (512, 118 * 2)
for epoch in range(max_epochs):
    total_loss = 0.0
    for local_batch, local_label in train_set:
        optimizer.zero_grad()
        local_batch, local_label = local_batch.to(device), local_label.to(device)
        print("local_batch : ",local_batch.shape)
        print("local_label : ",local_label.shape)
        

        logits = net(local_batch)
        loss = my_loss.calc(logits, local_label, local_batch, feas)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    avg_loss = total_loss / len(train_set.dataset)
    train_loss.append(avg_loss)
    lr_list.append(scheduler.optimizer.param_groups[0]['lr'])
    print(f"Epoch {epoch} | Training loss: {avg_loss:.4f} | Learning Rate: {scheduler.optimizer.param_groups[0]['lr']:.6f}")
    
    if epoch % eval_epoch == 0:
        net.eval()
        total_loss = 0.0
        for local_batch, local_label in val_set:
            local_batch, local_label = local_batch.to(device), local_label.to(device)
            logits = net(local_batch)
            loss = my_loss.calc(logits, local_label, local_batch, feas)
            total_loss += loss.item()
        avg_loss = total_loss / len(val_set.dataset)
        val_loss.append([epoch, avg_loss])
        print(f"Epoch {epoch} | Validation loss: {avg_loss:.4f} | Learning Rate: {scheduler.optimizer.param_groups[0]['lr']:.6f}")
        
        # 更新 learning rate
        scheduler.step(avg_loss)
        
        # 早停機制
        # if epoch:
        #     if previous - avg_loss < min_delta:
        #         tolerance -= 1
        #     if tolerance == 0:
        #         print("Early stopping triggered")
        #         break
        previous = avg_loss
        net.train()

        final_epoch = epoch

t1 = time.time()
print(f"Training time: {t1 - t0:.4f}s")
#印出 lr validation loss training loss 圖




output_plot(train_loss, val_loss, lr_list, output_path)
# 輸出模型
model_path = model_path+'%s.pt'%(timestamp)

if feas==False: model_path.replace('feas','')
torch.save(net.state_dict(),model_path)


# 進行模型評估
y_pred2, y_test2 = evaluate_model(net, y, x_test, y_test, n_bus, vy_scale, vy_deviation, device)
y_pred1 = y_pred2.copy()

# 將預測值轉換回原始尺度
y_pred1[:,1,:] = y_pred1[:,1,:] / vy_scale + vy_deviation
y_test2[:,1,:] = y_test2[:,1,:] / vy_scale + vy_deviation

# 計算誤差
n_test = np.size(y_test2,2)
err_L2_mean, err_Linf_mean, err_L2_mean_v, err_Linf_mean_v, err_L2_std, err_L2_std_v = compute_error_metrics(y_test2, y_pred1, n_test)

# 將 x_total 和 y_total 轉換成適合的形狀，主要目的是根據 n_bus (節點數) 來重新排列資料
x_new = np.zeros([x.shape[0],x.shape[1],n_sample])
for i in range(x.shape[1]):
  x_new[:,i,:] = x_total[n_bus*i:n_bus*(i+1),:]

y_new = np.zeros([y.shape[0],y.shape[1],n_sample])
for i in range(y.shape[1]):
  y_new[:,i,:] = y_total[n_bus*i:n_bus*(i+1),:]

# 這部分將測試資料 x_total 轉換成張量，並讓模型 net 進行預測
n_sample=x_total.shape[-1]
x_feed = torch.from_numpy(x_total.T).float()
y_pred1=net(x_feed.to(device)).cpu().detach().numpy().T

# y_pred2 重新整理成 (n_bus, 2, n_sample) 的形狀
# [:,0,:] 是價格預測 (pi)
# [:,1,:] 是電壓預測 (V)
y_pred_temp = y_pred1.copy()
y_pred2=np.zeros([y.shape[0],y.shape[1],n_sample])
y_pred2[:,0,:]=y_pred_temp[:n_bus,:]
y_pred2[:,1,:]=y_pred_temp[n_bus:,:]
print(y_pred2.shape,y_pred1.shape)
y_pred1 = y_pred2.copy()


gen_limit0 = x_new[:,4,:].copy() # lin cost
gen_idx = []
gen_idx = np.arange(n_bus)
# for i in range(n_bus):
#   if gen_limit0[i,0] > 0:
#     gen_idx.append(i)
print(type(gen_idx),len(gen_idx),gen_idx)






# 計算市場利潤誤差
# 這裡的 gen_cost0 是發電機的線性成本
# lmp_data 是市場價格
# quadratic_a 是二次成本係數
# profit_pred 是預測的市場利潤
# profit_true 是真實的市場利潤
# profit_err 是市場利潤誤差
# profit_err_l2 是市場利潤誤差的 L2 范數
gen_cost0 = x_new[:,4,:].copy()
lmp_data = y_new[:,0,:].copy()
quadratic_a = x_new[:,5,:].copy()
profit_pred = y_pred1[:,0,:] - gen_cost0
print(np.min(np.abs(profit_pred)))
profit_true = lmp_data - gen_cost0
print(np.min(np.abs(profit_true)))
profit_pred=(y_pred1[:,0,:]-gen_cost0)/(quadratic_a+1e-10)/2
profit_true=(lmp_data-gen_cost0)/(quadratic_a+1e-10)/2
print(np.min(np.abs(profit_pred)))
print(np.min(np.abs(profit_true)))


print(profit_pred.shape,profit_true.shape)
profit_err = profit_true - profit_pred
profit_err_l2 = np.zeros([n_sample,1])

for i in range(n_sample):
  profit_err_l2[i] = np.linalg.norm(profit_err[:,i])/np.linalg.norm(profit_true[:,i])
print(np.mean(profit_err_l2))

p_pred_sort = np.reshape(profit_pred,n_bus*n_sample)
p_true_sort = np.reshape(profit_true,n_bus*n_sample)
print(p_pred_sort.shape)
print(np.min(p_pred_sort),np.min(p_true_sort))

#  檢查發電機出力
# x = [load, gen_cost, gen_lim]
binary_thres_true = 1e-5
binary_thres = x_new[:,0,:].copy() # upper
binary_thres_lo = x_new[:,1,:].copy() # lower
gen_pred_binary_full = np.zeros((n_bus,n_sample))
gen_true_binary_full = np.zeros((n_bus,n_sample))

for i in range(n_sample):
  for j in range(len(gen_idx)):
    # predicted generator limit
    if profit_pred[gen_idx[j],i] > binary_thres[gen_idx[j],i]:
      gen_pred_binary_full[gen_idx[j],i] = binary_thres[gen_idx[j],i]
    elif profit_pred[gen_idx[j],i] < binary_thres_lo[gen_idx[j],i]:
      gen_pred_binary_full[gen_idx[j],i] = binary_thres_lo[gen_idx[j],i]
    else:
      gen_pred_binary_full[gen_idx[j],i] = profit_pred[gen_idx[j],i]
    # true generator limit
    if profit_true[gen_idx[j],i] > binary_thres[gen_idx[j],i]:
      gen_true_binary_full[gen_idx[j],i] = binary_thres[gen_idx[j],i]
    elif profit_true[gen_idx[j],i] < binary_thres_lo[gen_idx[j],i]:
      gen_true_binary_full[gen_idx[j],i] = binary_thres_lo[gen_idx[j],i]
    else:
      gen_true_binary_full[gen_idx[j],i] = profit_true[gen_idx[j],i]

gen_inj=gen_pred_binary_full
gen_inj_true=gen_true_binary_full
# nodal injection
load0 = -x_new[:,1,:].copy() # load file
p_inj = gen_inj #- load0
p_inj_true = gen_inj_true #- load0
print(np.sum(p_inj),np.sum(gen_inj_true))
print(np.sum(p_inj),np.sum(load0),np.sum(gen_inj))



print(p_inj_true.shape,p_inj.shape)
p_inj_true_sort = np.reshape(p_inj_true,n_bus*n_sample)
p_inj_sort = np.reshape(p_inj,n_bus*n_sample)
p_err = np.zeros(n_sample)
for i in range(n_sample):
  p_err[i] = np.linalg.norm(p_inj_true[:,i]-p_inj[:,i]) / np.linalg.norm(p_inj_true[:,i])
print('mean p_inj l2 err:',np.mean(p_err))



filename=root+'118ac_fmax.txt'
f_max1=pd.read_table(filename,sep=',',header=None).to_numpy() # flow limit

# 計算線路潮流
n_line = np.size(S_isf,0)
flow_est = np.zeros((n_line,n_sample))
flow_est0 = np.zeros((n_line,n_sample))

f_binary = np.zeros((n_line,n_sample))
f_binary0 = np.zeros((n_line,n_sample))

# for i in range(n_sample):
flow_est = np.dot(S_isf,p_inj)
flow_est0 = np.dot(S_isf,p_inj_true)
# f_max
# f_max_numpy = f_max.cpu().detach().numpy()
f_max_numpy = f_max1.copy()
f_binary = (np.abs(flow_est)-f_max_numpy > 0)
f_binary0 = (np.abs(flow_est0)-f_max_numpy > 0)

print(f_max_numpy.shape,flow_est.shape,flow_est0.shape)
f_tot_sample = n_line * n_sample
print(np.sum(f_binary),np.sum(f_binary0))
print(np.sum(f_binary)/f_tot_sample,np.sum(f_binary0)/f_tot_sample)
print(n_line,n_sample,flow_est.shape)


# soft threshold
f_err_est = np.abs(flow_est)-f_max_numpy
f_err_true = np.abs(flow_est0)-f_max_numpy

f_err_est = np.maximum(np.abs(flow_est)-f_max_numpy,0) # identify violations
f_err_true = np.maximum(np.abs(flow_est0)-f_max_numpy,0)

print(np.max(f_err_est),np.max(f_err_true))
print(np.max(f_err_est/f_max_numpy),np.max(f_err_true/f_max_numpy))



f_binary_soft = (np.abs(flow_est)-f_max_numpy > 0.1*(f_max_numpy))
f_binary0_soft = (np.abs(flow_est0)-f_max_numpy > 0.1*(f_max_numpy))
print(np.sum(f_binary_soft),np.sum(f_binary0_soft))
print(np.sum(f_binary_soft)/f_tot_sample,np.sum(f_binary0_soft)/f_tot_sample)



f_pred_sort = np.reshape(f_err_est/f_max_numpy,n_line*n_sample)
f_true_sort = np.reshape(f_err_true/f_max_numpy,n_line*n_sample)




f_line = np.sum(f_binary,0)
f_samp = np.sum(f_binary,1)
print('max sample pred:',np.max(f_line))
print('max line pred:',np.max(f_samp))

f_line0  = np.sum(f_binary0,0)
f_samp0 = np.sum(f_binary0,1)
print('max sample true:',np.max(f_line0))
print('max line true:',np.max(f_samp0))



gen_cost_pred = np.zeros((n_bus,n_sample))
gen_cost_true = np.zeros((n_bus,n_sample))
objective_err = np.zeros(n_sample)

gen_cost_pred = np.multiply(np.multiply(p_inj,p_inj),quadratic_a) + np.multiply(p_inj,gen_cost0)
gen_cost_true = np.multiply(np.multiply(p_inj_true,p_inj_true),quadratic_a) + np.multiply(p_inj_true,gen_cost0)

objective_err = np.sum(np.abs(gen_cost_true-gen_cost_pred),axis=0) / np.sum(gen_cost_true,axis=0)
print(np.mean(objective_err))


print(p_inj_true.shape,p_inj.shape)

p_inj_true_sort = np.reshape(p_inj_true,n_bus*n_sample)
p_inj_sort = np.reshape(p_inj,n_bus*n_sample)

p_err = np.zeros(n_sample)
for i in range(n_sample):
  p_err[i] = np.linalg.norm(p_inj_true[:,i]-p_inj[:,i]) / np.linalg.norm(p_inj_true[:,i])

print('mean p_inj l2 err:',np.mean(p_err))




# Bbus and B_r inverse
filename1 = root+'ieee118_Bbus.txt'
Bbus=pd.read_table(filename1,sep=',',header=None).to_numpy()
B_r = np.delete(Bbus,68,axis=0)
B_r = np.delete(B_r,68,axis=1)
Br_inv = np.linalg.inv(B_r)

# Y = G + jB
filename1 = root+'ieee118_Gmat.txt'
G_mat=pd.read_table(filename1,sep=',',header=None).to_numpy()
filename1 = root+'ieee118_Bmat.txt'
B_mat=pd.read_table(filename1,sep=',',header=None).to_numpy()
print(G_mat.shape,B_mat.shape)

# line parameters
filename1 = root+'ieee118_lineloc.txt'
line_loc = pd.read_table(filename1,sep=',',header=None).to_numpy()

# load line params
filename1 = root+'ieee118_lineparams.txt'
line_params = pd.read_table(filename1,sep=',',header=None).to_numpy()
R_line = line_params[:,0].copy()
X_line = line_params[:,1].copy()
B_shunt = line_params[:,2].copy()
Z_line = R_line + 1j * X_line 
Y_line = 1 / Z_line
G_line = np.real(Y_line)
B_line = np.imag(Y_line)
# P_inj w/out reference bus in p.u.
p_inj_r = np.delete(p_inj,68,axis=0) / 100
p_inj_true_r = np.delete(p_inj_true,68,axis=0) / 100
p_inj_pu = p_inj / 100
p_inj_true_pu = p_inj_true / 100
print(Br_inv.shape,p_inj.shape,p_inj_true.shape)#p_inj_true

theta0 = np.matmul(Br_inv,p_inj_r)
theta_true0 = np.matmul(Br_inv,p_inj_true_r)
theta = np.insert(theta0,68,0,axis = 0)
theta_true = np.insert(theta_true0,68,0,axis = 0)
print(theta.shape,theta_true.shape)



print(np.max(theta),np.min(theta))
math.sin(math.pi/6)
print(G_line[0],B_line[0])



# Calculate real and reactive flow
f_p = np.zeros((n_line,n_sample))
f_q = np.zeros((n_line,n_sample))
fji_p = np.zeros((n_line,n_sample))
fji_q = np.zeros((n_line,n_sample))
print(f_q.shape)

v_pred = y_pred1[:,1,:].copy()
v_pred = v_pred / vy_scale + vy_deviation
print(np.max(v_pred),np.min(v_pred),v_pred.shape)

theta1 = theta[line_loc[:,0]-1,:]
theta2 = theta[line_loc[:,1]-1,:]
V1 = v_pred[line_loc[:,0]-1,:]
V2 = v_pred[line_loc[:,1]-1,:] 
f_p=(a*G_line*(V1*V1).T)-a*((V1*V2).T)*(G_line*np.cos(theta1-theta2).T+B_line*np.sin(theta1-theta2).T)
f_p=f_p.T
f_q=-a*(V1.T)*(a*V1.T)*(B_line+B_shunt/2)+a*((V1*V2).T)*(B_line*np.cos(theta1-theta2).T-G_line*np.sin(theta1-theta2).T)
f_q=f_q.T

theta1 = theta[line_loc[:,1]-1,:]
theta2 = theta[line_loc[:,0]-1,:]
V1 = v_pred[line_loc[:,1]-1,:]
V2 = v_pred[line_loc[:,0]-1,:]
fji_p=(a*G_line*(V1*V1).T)-a*((V1*V2).T)*(G_line*np.cos(theta1-theta2).T+B_line*np.sin(theta1-theta2).T)
fji_p=fji_p.T
fji_q=-a*(V1.T)*(a*V1.T)*(B_line+B_shunt/2)+a*((V1*V2).T)*(B_line*np.cos(theta1-theta2).T-G_line*np.sin(theta1-theta2).T)
fji_q=fji_q.T



s_pred = np.sqrt(f_p*f_p+f_q*f_q)*100
sji_pred = np.sqrt(fji_p*fji_p+fji_q*fji_q)*100
print(np.max(f_q),np.min(f_q))
flow_est.shape
print(np.mean(s_pred[0,:]),np.mean(f_max_numpy[0]))



sij_binary = (np.abs(s_pred)-f_max_numpy[:n_line] > 0)
sji_binary = (np.abs(sji_pred)-f_max_numpy[:n_line] > 0)
s_binary = np.maximum(sij_binary,sji_binary)
print(np.sum(s_binary))#,np.sum(f_binary0))
print('hard violation rate:',np.sum(s_binary)/n_sample/n_line)#,np.sum(f_binary0)/f_tot_sample)
s_binary_soft = (np.abs(s_pred)-f_max_numpy[:n_line] > 0.1*(f_max_numpy[:n_line]))
print(np.sum(s_binary_soft))#,np.sum(f_binary0_soft))
print(np.sum(s_binary_soft)/n_sample/n_line)#,np.sum(f_binary0_soft)/f_tot_sample)


# 違規分析
# violation level
sij_violation = np.abs(s_pred)-f_max_numpy[:n_line] #/ f_max_numpy
sij_violation_level = np.maximum(sij_violation,0)
sji_violation = np.abs(sji_pred)-f_max_numpy[:n_line] #/ f_max_numpy
sji_violation_level = np.maximum(sji_violation,0)
s_violation_level = np.maximum(sij_violation_level,sji_violation_level)
s_violation_level = np.divide(s_violation_level,f_max_numpy[:n_line])
s_vio_lvl = np.reshape(s_violation_level,n_line*n_sample)

print('S violation level:')
print('hard:',np.sum(s_binary)/f_tot_sample)
print('mean:',np.mean(s_vio_lvl))
print('median:',np.median(s_vio_lvl))
print('max:',np.max(s_vio_lvl))
print('std:',np.std(s_vio_lvl))
print('p99:',np.percentile(s_vio_lvl,99))

f_violation = np.abs(flow_est)-f_max_numpy #/ f_max_numpy
f_violation_level = np.maximum(f_violation,0)
f_violation_level = np.divide(f_violation_level,f_max_numpy)
f_vio_lvl = np.reshape(f_violation_level,n_line*n_sample)

print('f violation level:')
print('hard:',np.sum(f_binary)/f_tot_sample,np.sum(f_binary0)/f_tot_sample)
print('mean:',np.mean(f_vio_lvl))
print('median:',np.median(f_vio_lvl))
print('max:',np.max(f_vio_lvl))
print('std:',np.std(f_vio_lvl))
print('p99:',np.percentile(f_vio_lvl,99))


# # err_L2_mean = np.mean(err_L2)
# # err_Linf_mean = np.mean(err_Linf)
# print('Price L2 mean:', err_L2_mean,'L_inf mean:', err_Linf_mean )
# print('std:',np.std(err_L2))
# # err_L2_mean_v = np.mean(err_L2_v)
# # err_Linf_mean_v = np.mean(err_Linf_v)
# print('Voltage L2 mean:', err_L2_mean_v,'L_inf mean:', err_Linf_mean_v )
# print('std:',np.std(err_L2_v))

params = (sum(temp.numel() for temp in net.parameters() if temp.requires_grad))

 
# 建立輸出內容字串
output_content = f"""
date: {timestamp}
genotype: {genotype}
final_epoch: {final_epoch}
training time: {t1 - t0:.4e}s

factor: {factor}
patience: {patience}

start learning rate: {lr:.4e}
batch size: {batch_size}
params number: {params:.4e}

Price L2 mean: {err_L2_mean:.4e} 
Price std: {err_L2_std:.4e}

Voltage L2 mean: {err_L2_mean_v:.4e} 
Voltage std: {err_L2_std_v:.4e}
"""


# 設定儲存路徑和檔名
output_file = f'{output_path}/118ac_output_{timestamp}.txt'

# 確保目錄存在
os.makedirs(output_path, exist_ok=True)

# 將內容寫入 txt 檔案
with open(output_file, 'w') as file:
    file.write(output_content)

print(f"輸出內容已儲存到 {output_file}")


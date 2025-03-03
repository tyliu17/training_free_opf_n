import torch
import numpy as np

def evaluate_model(net, y, x_test, y_test, n_bus, vy_scale, vy_deviation, device):
    n_test = np.size(x_test,0)

    x_test_feed = torch.from_numpy(x_test).float()  # 轉換為 PyTorch 張量 (float)
    x_test_feed = x_test_feed.to(device)

    y_pred = net(x_test_feed) # 讓模型預測輸出

    #118*2 = 236
    # 將ypred轉numpy
    y_pred1 = y_pred.cpu().detach() # 移到 CPU，並取消梯度追蹤
    # 移除維度 (避免 shape 錯誤)，轉成 NumPy 陣列
    # ((batch_size, 1, n_bus*2) → (batch_size, n_bus*2))。
    y_pred1 = torch.squeeze(y_pred1,1).numpy() 

    # 重新調整 y_pred 的形狀
    # 轉置，使 y_pred 變成 (特徵數, 測試樣本數) → (測試樣本數, 特徵數), 便於切割
    y_pred_temp = y_pred1.copy().transpose()


    # y_pred2：
    # y.shape[0]：代表(n_bus)
    # y.shape[1]：代表價格 (pi) 和電壓 (V)
    # n_test：測試樣本數
    # y_pred2[:, 0, :] = y_pred_temp[:n_bus, :]：將 前 n_bus 個數值 作為價格 (pi) 預測值
    # y_pred2[:, 1, :] = y_pred_temp[n_bus:, :]：將 後 n_bus 個數值 作為電壓 (V) 預測值
    y_pred2=np.zeros([y.shape[0],y.shape[1],n_test])
    y_pred2[:,0,:]=y_pred_temp[:n_bus,:] # 預測的價格
    y_pred2[:,1,:]=y_pred_temp[n_bus:,:] # 預測的電壓

    y_test_temp = y_test.copy().transpose()
    # y_pred2=np.reshape(y_pred2,(y_pre.shape[0],y_pre.shape[1],n_test))
    y_test2=np.zeros([y.shape[0],y.shape[1],n_test])
    y_test2[:,0,:]=y_test_temp[:n_bus,:]
    y_test2[:,1,:]=y_test_temp[n_bus:,:]
    # y_pred1 = y_pred2.copy()


    # # recover the original p.u. scale
    # # vy_deviation) * vy_scale
    # y_pred1[:,1,:] = y_pred1[:,1,:] / vy_scale + vy_deviation
    # y_test2[:,1,:] = y_test2[:,1,:] / vy_scale + vy_deviation
    return y_pred2, y_test2

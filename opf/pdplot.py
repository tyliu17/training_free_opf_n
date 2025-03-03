import matplotlib.pyplot as plt
import numpy as np
from datetime import datetime

def plot_loss(train_loss, val_loss, lr, batch_size):

    val_len = len(val_loss)
    print(val_len)
    val_plt = np.zeros((2,val_len))
    for i in range(val_len):
        val_plt[0,i] = val_loss[i][0]
        val_plt[1,i] = val_loss[i][1]

    plt.figure()
    plot_idx = np.arange(np.size(train_loss))
    plt.plot(plot_idx[5:-1],train_loss[5:-1],lw=2,label='training loss')
    plt.plot(val_plt[0,1:],val_plt[1,1:],lw=2,label='validation loss')
    plt.yscale("log")
    plt.xlabel('epoch')
    plt.ylabel('loss')
    plt.legend()

    # 在圖的右上角標示學習率
    plt.text(0.95, 0.95, f'Learning Rate: {lr:.6f}\nbatch size: {batch_size}', 
            transform=plt.gca().transAxes, fontsize=10, ha='right', va='top', color='red')




    # 儲存圖片到指定資料夾（需提供路徑）
    timestamp=datetime.now().strftime('%m%d%H%M')
    output_path = r'C:\Users\USER\Desktop\GNN_OPF_electricity_market-main\output_loss_fig'
    output_path = output_path + '/118ac_loss_fig_%s.png' % (timestamp)
    plt.savefig(output_path, dpi=300, bbox_inches='tight')  # 儲存為 PNG 格式，解析度 300 dpi


    plt.show(block=False)   

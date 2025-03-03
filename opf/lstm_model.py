import torch
import torch.nn as nn

class InferCell(nn.Module):
    def __init__(self, input_size, seq_len, hidden_size, num_layers_list, output_size):
        """
        LSTM-based InferCell 模型:
        - nodes=4
        - 具有不同結構的 LSTM 層
        - Fully Connected 層最後輸出
        
        :param input_size: LSTM 輸入維度 (預設為 1)
        :param hidden_size: LSTM 隱藏層維度 (預設為 64)
        :param num_layers_list: 每個 LSTM 層的層數
        :param output_size: 最終輸出維度
        """
        super(InferCell, self).__init__()
        self.seq_len = seq_len

        # 定義 LSTM 結構 (對應 InferCell 中的不同 LSTM)
        self.cells_structure = nn.ModuleList([
            nn.LSTM(input_size, input_size, num_layers=num_layers_list[0], batch_first=True),  # (I0-L0)
            nn.LSTM(input_size    , hidden_size, num_layers=num_layers_list[1], batch_first=True),  # (I0-L1, I1-L2)
            nn.LSTM(hidden_size, hidden_size, num_layers=num_layers_list[2], batch_first=True), # (I0-L3, I1-L4, I2-L5)
            nn.Identity(),  # 這裡的 Zero layer 用 Identity 代替
            nn.Identity(),  # Identity layer
            nn.LSTM(hidden_size, hidden_size, num_layers=num_layers_list[3], batch_first=True)  # 最後一層
        ])

        # 定義全連接層 (Fully Connected layers)
        self.fc_layers_structure = nn.Sequential(
            nn.Dropout(0.1),
            nn.Linear(hidden_size, hidden_size * input_size),  # 假設 LSTM 輸出展開後長度為 3072 (hidden_size * 48)
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_size * input_size, output_size)  # 最終輸出
        )
        self.expand_layer = nn.Linear(input_size, seq_len * input_size)

    def forward(self, x):
        x = self.expand_layer(x)
        x = x.view(x.size(0), self.seq_len, -1)

        lstm_outs = []
        for layer in self.cells_structure:
            if isinstance(layer, nn.LSTM):
                x, _ = layer(x)
            lstm_outs.append(x)
        
        x = lstm_outs[-1][:, -1, :]  # 取 LSTM 最後時間步的輸出
        x = self.fc_layers_structure(x)
        
        return x
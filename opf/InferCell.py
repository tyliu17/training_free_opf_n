import torch
import torch.nn as nn

# 定義了 InferCell 模型，它的結構由 NAS 搜索決定

# InferCell 模型的結構由 NAS 搜索決定
# 支持 LSTM, GRU, CNN, Transformer
class InferCell(nn.Module):
    def __init__(self, input_size, seq_len, hidden_size, output_size, genotype):
        """
        InferCell with NAS-based architecture:
        - Supports LSTM, GRU, CNN, Transformer
        - Uses NAS to dynamically adjust the model structure
        """
        super(InferCell, self).__init__()
        self.seq_len = seq_len
        self.cells_structure = nn.ModuleList()

        # Build network layers based on genotype
        for op_name, op_params in genotype:
            if op_name == 'LSTM':
                self.cells_structure.append(
                    nn.LSTM(
                        input_size if len(self.cells_structure) == 0 else hidden_size,
                        hidden_size,
                        num_layers=op_params['num_layers'],
                        batch_first=True
                    )
                )
            elif op_name == 'GRU':
                self.cells_structure.append(
                    nn.GRU(
                        input_size if len(self.cells_structure) == 0 else hidden_size,
                        hidden_size,
                        num_layers=op_params['num_layers'],
                        batch_first=True
                    )
                )
            elif op_name == 'CNN':
                self.cells_structure.append(
                    nn.Conv1d(
                        in_channels=input_size if len(self.cells_structure) == 0 else hidden_size,
                        out_channels=hidden_size,
                        kernel_size=op_params['kernel_size'],
                        stride=1,
                        padding=1
                    )
                )
            elif op_name == 'Transformer':
                self.cells_structure.append(
                    nn.TransformerEncoderLayer(
                        d_model=hidden_size,
                        nhead=op_params['nhead']
                    )
                )
            elif op_name == 'Identity':
                self.cells_structure.append(nn.Identity())

        # Fully Connected layers
        self.fc_layers_structure = nn.Sequential(
            nn.Dropout(0.1),
            nn.Linear(hidden_size, hidden_size * input_size),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_size * input_size, output_size)
        )
        self.expand_layer = nn.Linear(input_size, seq_len * input_size)

    def forward(self, x):
        print("=====================================")  
        print(x.shape)
        x = self.expand_layer(x)
        x = x.view(x.size(0), self.seq_len, -1)

        for layer in self.cells_structure:
            if isinstance(layer, nn.LSTM) or isinstance(layer, nn.GRU):
                x, _ = layer(x)
            elif isinstance(layer, nn.Conv1d):
                x = x.permute(0, 2, 1)
                x = layer(x)
                x = x.permute(0, 2, 1)
            elif isinstance(layer, nn.TransformerEncoderLayer):
                x = layer(x)

        x = x[:, -1, :]
        x = self.fc_layers_structure(x)
        return x

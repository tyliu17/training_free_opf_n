import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import xgboost as xgb
from sklearn.model_selection import train_test_split
import random
import joblib

def define_search_space():
    return {
        "num_lstm_layers": [1, 2, 3],
        "num_gru_layers": [1, 2, 3],
        "num_cnn_layers": [1, 2, 3],
        "num_transformer_layers": [1, 2],
        "hidden_size": [64, 128, 256],
        "dropout": [0.0, 0.2, 0.5],
    }

def sample_architecture(search_space):
    return {
        "num_lstm_layers": random.choice(search_space["num_lstm_layers"]),
        "num_gru_layers": random.choice(search_space["num_gru_layers"]),
        "num_cnn_layers": random.choice(search_space["num_cnn_layers"]),
        "num_transformer_layers": random.choice(search_space["num_transformer_layers"]),
        "hidden_size": random.choice(search_space["hidden_size"]),
        "dropout": random.choice(search_space["dropout"]),
    }

def generate_architectures(num_samples):
    search_space = define_search_space()
    return [sample_architecture(search_space) for _ in range(num_samples)]

class NeuralNet(nn.Module):
    def __init__(self, arch, input_size, output_size):
        super(NeuralNet, self).__init__()
        self.lstm = nn.LSTM(input_size, arch["hidden_size"], num_layers=arch["num_lstm_layers"], batch_first=True)
        self.gru = nn.GRU(input_size, arch["hidden_size"], num_layers=arch["num_gru_layers"], batch_first=True)
        self.fc = nn.Sequential(
            nn.Linear(arch["hidden_size"], output_size),
            nn.ReLU(),
            nn.Dropout(arch["dropout"])
        )
    
    def forward(self, x):
        x, _ = self.lstm(x)
        x, _ = self.gru(x)
        x = x[:, -1, :]
        return self.fc(x)

def train_and_evaluate(architectures, input_size, output_size):
    print("=============================================")
    print("Training and evaluating architectures...")
    results = []
    for arch in architectures:
        print(f"Training architecture: {arch}")
        
        model = NeuralNet(arch, input_size, output_size)
        optimizer = optim.Adam(model.parameters(), lr=0.001)
        criterion = nn.MSELoss()
        X_train = torch.rand(1000, 10, input_size)
        y_train = torch.rand(1000, output_size)
        for _ in range(5):
            optimizer.zero_grad()
            y_pred = model(X_train)
            loss = criterion(y_pred, y_train)
            loss.backward()
            optimizer.step()
        results.append({**arch, "MSE": loss.item()})
    return results

def save_benchmark(results, filename="benchmark.csv"):
    benchmark_df = pd.DataFrame(results)
    benchmark_df.to_csv(filename, index=False)

def train_surrogate_model(benchmark_filename="benchmark.csv"):
    benchmark_df = pd.read_csv(benchmark_filename)
    X = benchmark_df.drop(columns=["MSE"])
    y = benchmark_df["MSE"]
    X = pd.get_dummies(X)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = xgb.XGBRegressor(n_estimators=100, max_depth=5)
    model.fit(X_train, y_train)
    joblib.dump(model, "surrogate_model.pkl")

def predict_performance(arch, surrogate_model_filename="surrogate_model.pkl"):
    model = joblib.load(surrogate_model_filename)
    df = pd.DataFrame([arch])
    df = pd.get_dummies(df)
    missing_cols = set(X_train.columns) - set(df.columns)
    for col in missing_cols:
        df[col] = 0
    df = df[X_train.columns]
    return model.predict(df)[0]

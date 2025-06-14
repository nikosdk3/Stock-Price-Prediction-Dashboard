import numpy as np
import streamlit as st
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import torch.optim as optim
from sklearn.preprocessing import MinMaxScaler

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.classes.__path__ = []  # Fix error with torch and streamlit compatibility


class LSTMNet(nn.Module):
    def __init__(self, input_size=1, hidden_size=50, num_layers=3, dropout=0.2):
        super(LSTMNet, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0,
        )

        self.dropout = nn.Dropout(dropout)

        self.fc = nn.Linear(hidden_size, 1)

    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)

        out, _ = self.lstm(x, (h0, c0))

        out = self.dropout(out[:, -1, :])

        out = self.fc(out)

        return out


class LSTMModel:
    def __init__(self, lookback_period=60, hidden_size=50, num_layers=3, dropout=0.2):
        self.model = None
        self.scaler = MinMaxScaler(feature_range=(0, 1))
        self.lookback_period = lookback_period
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.dropout = dropout
        self.is_trained = False
        self.device = device

    def prepare_data(self, data, target_column="Close", fit_scaler=True):
        if fit_scaler:
            self.scaler.fit(data[[target_column]])
        scaled_data = self.scaler.transform(data[[target_column]])

        X, y = [], []

        for i in range(self.lookback_period, len(scaled_data)):
            X.append(scaled_data[i - self.lookback_period : i, 0])
            y.append(scaled_data[i, 0])

        X, y = np.array(X), np.array(y)

        return X, y

    def create_data_loaders(self, X, y, batch_size=32, train_split=0.8):
        train_size = int(len(X) * train_split)
        X_train, X_test = X[:train_size], X[train_size:]
        y_train, y_test = y[:train_size], y[train_size:]

        X_train = torch.FloatTensor(X_train).unsqueeze(-1)
        y_train = torch.FloatTensor(y_train).unsqueeze(-1)
        X_test = torch.FloatTensor(X_test).unsqueeze(-1)
        y_test = torch.FloatTensor(y_test).unsqueeze(-1)

        train_dataset = TensorDataset(X_train, y_train)
        test_dataset = TensorDataset(X_test, y_test)

        train_loader = DataLoader(
            dataset=train_dataset, batch_size=batch_size, shuffle=True
        )
        test_loader = DataLoader(
            dataset=test_dataset, batch_size=batch_size, shuffle=False
        )

        return train_loader, test_loader, (X_test, y_test)

    def train(
        self,
        data,
        target_column="Close",
        epochs=50,
        batch_size=32,
        learning_rate=0.001,
        teacher_forcing_ratio=0.5,
    ):
        X, y = self.prepare_data(data, target_column)
        train_loader, test_loader, test_data = self.create_data_loaders(
            X, y, batch_size
        )

        self.model = LSTMNet(
            input_size=1,
            hidden_size=self.hidden_size,
            num_layers=self.num_layers,
            dropout=self.dropout,
        ).to(self.device)

        criterion = nn.MSELoss()
        optimizer = optim.Adam(self.model.parameters(), lr=learning_rate)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, patience=10, factor=0.5
        )

        train_losses = []
        val_losses = []

        st.write("LSTM Model Progress")
        progress_bar = st.progress(0)
        best_val_loss = float("inf")
        for epoch in range(epochs):
            self.model.train()
            train_loss = 0.0
            for batch_X, batch_y in train_loader:
                batch_X, batch_y = batch_X.to(self.device), batch_y.to(self.device)

                optimizer.zero_grad()

                outputs = self.model(batch_X)

                # Ensure batch_y requires gradients
                batch_y.requires_grad = True

                # Teacher forcing logic
                if np.random.rand() < teacher_forcing_ratio:
                    outputs = (
                        batch_y.clone()
                    )  # Use ground truth as next input without detaching

                loss = criterion(outputs, batch_y)

                loss.backward()

                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)

                optimizer.step()

                train_loss += loss.item()

            self.model.eval()
            val_loss = 0.0
            with torch.no_grad():
                for batch_X, batch_y in test_loader:
                    batch_X, batch_y = batch_X.to(self.device), batch_y.to(self.device)
                    outputs = self.model(batch_X)
                    loss = criterion(outputs, batch_y)
                    val_loss += loss.item()

            avg_train_loss = train_loss / len(train_loader)
            avg_val_loss = val_loss / len(test_loader)

            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                torch.save(self.model.state_dict(), "model.pth")

            train_losses.append(avg_train_loss)
            val_losses.append(avg_val_loss)

            scheduler.step(avg_val_loss)

            print(
                f"Epoch [{epoch+1}/{epochs}], Train loss: {avg_train_loss:.6f}, Val loss: {avg_val_loss:.6f}"
            )
            progress_bar.progress((epoch + 1) / epochs)

        self.is_trained = True

        history = {"train_loss": train_losses, "val_loss": val_losses}

        return history, test_data

    def predict(self, data, steps=30):
        if not self.is_trained or self.model is None:
            raise ValueError("Model must be trained and initialized first")

        self.model.load_state_dict(torch.load("model.pth"))
        self.model.eval()

        scaled_data = self.scaler.transform(data[["Close"]])
        last_sequence = scaled_data[-self.lookback_period :]

        predictions = []
        current_sequence = torch.FloatTensor(last_sequence).unsqueeze(0).to(self.device)

        with torch.no_grad():
            for _ in range(steps):
                pred = self.model(current_sequence)
                predictions.append(pred.cpu().flatten().numpy()[0])

                new_pred = pred.unsqueeze(0)
                current_sequence = torch.cat(
                    [current_sequence[:, 1:, :], new_pred], dim=1
                )

        predictions = np.array(predictions).reshape(-1, 1)
        predictions = self.scaler.inverse_transform(predictions)

        return predictions.flatten()

    def eval_model(self, test_data):
        if not self.is_trained or self.model is None:
            raise ValueError("Model must be trained and initialized first")

        X_test, y_test = test_data
        X_test = X_test.to(self.device)

        self.model.load_state_dict(torch.load("model.pth"))
        self.model.eval()

        predictions = []

        with torch.no_grad():
            for i in range(len(X_test)):
                pred = self.model(X_test[i : i + 1])
                predictions.append(pred.item())

        y_test_inv = self.scaler.inverse_transform(y_test).flatten()
        predictions_inv = self.scaler.inverse_transform(
            np.array(predictions).reshape(-1, 1)
        ).flatten()

        return y_test_inv, predictions_inv

    def transfer_learning(
        self,
        new_data,
        target_column="Close",
        epochs=10,
        batch_size=32,
        learning_rate=0.0001,
    ):
        if not self.is_trained or self.model is None:
            raise ValueError("Model must be trained first before transfer learning")

        X, y = self.prepare_data(new_data, target_column, fit_scaler=False)
        train_loader, test_loader, test_data = self.create_data_loaders(
            X, y, batch_size
        )

        self.model.train()

        criterion = nn.MSELoss()
        optimizer = optim.Adam(self.model.parameters(), lr=learning_rate)

        for epoch in range(epochs):
            train_loss = 0.0
            for batch_X, batch_y in train_loader:
                batch_X, batch_y = batch_X.to(self.device), batch_y.to(self.device)

                optimizer.zero_grad()

                outputs = self.model(batch_X)
                loss = criterion(outputs, batch_y)

                loss.backward()
                optimizer.step()

                train_loss += loss.item()

            print(
                f"Transfer Learning Epoch [{epoch+1}/{epochs}], Train loss: {train_loss / len(train_loader):.6f}"
            )

        self.is_trained = True

        return test_data

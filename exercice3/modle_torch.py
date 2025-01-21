import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.metrics import mean_squared_error
from sklearn.decomposition import PCA
from tqdm import tqdm
import matplotlib.pyplot as plt
# RDKit
from rdkit import Chem
from rdkit.Chem import AllChem
import os

# Check if ROCm is available
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

print(torch.cuda.is_available())
print(torch.version.hip)

# Load and preprocess data
train = pd.read_csv('data/X_train.csv')
test = pd.read_csv('data/X_test.csv')
y_train = pd.read_csv('data/y_train.csv')['y']

# Convert SMILES to mols
train_mols = [AllChem.MolFromSmiles(smile) for smile in train['smiles']]
test_mols = [AllChem.MolFromSmiles(smile) for smile in test['smiles']]

# Convert Mol to fingerprints and save to CSV
train_fps_file = 'data/train_fps.csv'
test_fps_file = 'data/test_fps.csv'

if os.path.exists(train_fps_file) and os.path.exists(test_fps_file):
    train_fps = pd.read_csv(train_fps_file, index_col=0).values
    test_fps = pd.read_csv(test_fps_file, index_col=0).values
else:
    train_fps = np.array([AllChem.GetMorganFingerprintAsBitVect(mol, radius=2, nBits=2048) for mol in train_mols])
    test_fps = np.array([AllChem.GetMorganFingerprintAsBitVect(mol, radius=2, nBits=2048) for mol in test_mols])
    pd.DataFrame(train_fps).to_csv(train_fps_file)
    pd.DataFrame(test_fps).to_csv(test_fps_file)

print(f"The train test contains {train_fps.shape[0]} molecules encoded with {train_fps.shape[1]} bits.")
print(f"The test test contains {test_fps.shape[0]} molecules encoded with {test_fps.shape[1]} bits.")

"""
# PCA Variance explained
pca = PCA()
pca.fit(train_fps)
explained_variance = pca.explained_variance_ratio_
explained_variance_cum = np.cumsum(explained_variance)
print(f"Explained variance ratio: {explained_variance_cum}")

# Plot the explained variance
plt.figure(figsize=(10, 6))
plt.plot(range(1, len(explained_variance_cum) + 1), explained_variance_cum, marker='o', linestyle='--')
plt.xlabel('Number of Components')
plt.ylabel('Cumulative Explained Variance')
plt.title('Explained Variance vs. Number of Components')
plt.grid()
plt.show()
"""
# Reduce dimensionality with PCA using 400 components
pca = PCA(n_components=400)
X_train_pca = pca.fit_transform(train_fps)
X_test_pca = pca.transform(test_fps)

# Split the training data into training and validation sets
X_train, X_val, y_train, y_val = train_test_split(X_train_pca, y_train, test_size=0.2, random_state=42)

# Define the PyTorch model
class NeuralNetwork(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(NeuralNetwork, self).__init__()
        self.fc1 = nn.Linear(int(input_dim), int(hidden_dim))
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(int(hidden_dim), int(output_dim))

    def forward(self, x):
        out = self.fc1(x)
        out = self.relu(out)
        out = self.fc2(out)
        return out

class PyTorchRegressor(BaseEstimator, RegressorMixin):
    def __init__(self, input_dim, hidden_dim=32, output_dim=1, lr=0.01, epochs=100, batch_size=32):
        self.input_dim = int(input_dim)
        self.hidden_dim = int(hidden_dim)
        self.output_dim = int(output_dim)
        self.lr = lr
        self.epochs = epochs
        self.batch_size = batch_size
        self.model = NeuralNetwork(self.input_dim, self.hidden_dim, self.output_dim).to(device)
        self.criterion = nn.MSELoss()
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.lr)

    def fit(self, X, y):
        X_tensor = torch.tensor(X, dtype=torch.float32).to(device)
        y_tensor = torch.tensor(y.values, dtype=torch.float32).view(-1, 1).to(device)
        dataset = torch.utils.data.TensorDataset(X_tensor, y_tensor)
        dataloader = torch.utils.data.DataLoader(dataset, batch_size=self.batch_size, shuffle=True)

        self.model.train()
        for epoch in range(self.epochs):
            for batch_X, batch_y in dataloader:
                self.optimizer.zero_grad()
                outputs = self.model(batch_X)
                loss = self.criterion(outputs, batch_y)
                loss.backward()
                self.optimizer.step()
        return self

    def predict(self, X):
        self.model.eval()
        X_tensor = torch.tensor(X, dtype=torch.float32).to(device)
        with torch.no_grad():
            predictions = self.model(X_tensor).cpu().numpy()
        return predictions

# Define the parameter grid
param_grid = {
    'hidden_dim': [32, 64, 128],
    'lr': [0.001, 0.01, 0.1],
    'epochs': [50, 100],
    'batch_size': [16, 32, 64]
}

# Use GridSearchCV to find the best hyperparameters with a progress bar
grid_search = GridSearchCV(estimator=PyTorchRegressor(input_dim=400), param_grid=param_grid, cv=3, scoring='neg_mean_squared_error', verbose=1, n_jobs=-1)
grid_search.fit(X_train, y_train)

# Print the best parameters and the corresponding score
print(f"Best parameters: {grid_search.best_params_}")
print(f"Best score: {grid_search.best_score_}")

# Evaluate the best model on the validation set
best_model = grid_search.best_estimator_
y_pred = best_model.predict(X_val)
mse = mean_squared_error(y_val, y_pred)
print(f"Validation MSE: {mse}")
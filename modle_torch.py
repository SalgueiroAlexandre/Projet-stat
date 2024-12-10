import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import ParameterGrid
from sklearn.decomposition import PCA
from tqdm import tqdm
# RDKit
from rdkit import Chem
from rdkit.Chem import AllChem

# Define the PyTorch model
class NeuralNetwork(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(NeuralNetwork, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        out = self.fc1(x)
        out = self.relu(out)
        out = self.fc2(out)
        return out

# Create a custom estimator for scikit-learn
class PyTorchRegressor(BaseEstimator, RegressorMixin):
    def __init__(self, input_dim, hidden_dim=32, output_dim=1, lr=0.01, epochs=100, batch_size=32):
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.lr = lr
        self.epochs = epochs
        self.batch_size = batch_size
        self.model = NeuralNetwork(input_dim, hidden_dim, output_dim)
        self.criterion = nn.MSELoss()
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.lr)

    def fit(self, X, y):
        X_tensor = torch.tensor(X, dtype=torch.float32)
        y_tensor = torch.tensor(y, dtype=torch.float32).view(-1, 1)
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
        X_tensor = torch.tensor(X, dtype=torch.float32)
        with torch.no_grad():
            predictions = self.model(X_tensor).numpy()
        return predictions

# Load and preprocess data
train = pd.read_csv('data/X_train.csv')
test = pd.read_csv('data/X_test.csv')
y_train = pd.read_csv('data/y_train.csv')['y']

# Convert SMILES to mols
train_mols = [AllChem.MolFromSmiles(smile) for smile in train['smiles']]
test_mols = [AllChem.MolFromSmiles(smile) for smile in test['smiles']]

# Convert Mol to fingerprints
train_fps = np.array([AllChem.GetMorganFingerprintAsBitVect(mol, radius=2, nBits=2048) for mol in train_mols])
test_fps = np.array([AllChem.GetMorganFingerprintAsBitVect(mol, radius=2, nBits=2048) for mol in test_mols])

print(f"The train test contains {train_fps.shape[0]} molecules encoded with {train_fps.shape[1]} bits.")
print(f"The test test contains {test_fps.shape[0]} molecules encoded with {test_fps.shape[1]} bits.")

# Perform PCA
pca = PCA(n_components=100)  # Adjust the number of components as needed
X_train_pca = pca.fit_transform(train_fps)
X_test_pca = pca.transform(test_fps)

# Split the training data into training and validation sets
X_train, X_val, y_train, y_val = train_test_split(X_train_pca, y_train, test_size=0.2, random_state=42)

# Define the hyperparameter grid
param_grid = {
    'hidden_dim': [[32], [64], [128]],
    'lr': [[0.001], [0.01], [0.1]],
    'epochs': [[50], [100]],
    'batch_size': [[16], [32], [64]]
}

# Use GridSearchCV to find the best hyperparameters with a progress bar
param_list = list(ParameterGrid(param_grid))
best_score = float('inf')
best_params = None

for params in tqdm(param_list):
    model = PyTorchRegressor(input_dim=X_train.shape[1], **params)
    grid = GridSearchCV(estimator=model, param_grid=[params], cv=3, scoring='neg_mean_squared_error', n_jobs=-1)
    grid_result = grid.fit(X_train, y_train)
    if -grid_result.best_score_ < best_score:
        best_score = -grid_result.best_score_
        best_params = grid_result.best_params_

# Display the best hyperparameters
print(f"Best: {best_score} using {best_params}")

# Train the model with the best hyperparameters
best_model = PyTorchRegressor(input_dim=X_train.shape[1], **best_params)
best_model.fit(X_train, y_train)

# Make predictions on the test set
y_pred = best_model.predict(X_test_pca)

# Save the results to a CSV file
submission_df = pd.DataFrame()
submission_df['id'] = test['id']
submission_df['y'] = y_pred.flatten()
submission_df.to_csv('data/y_benchmark_nn_gridsearch.csv', index=False)
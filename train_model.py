# Linear algebra and data handling
import numpy as np
import pandas as pd

# RDKit
from rdkit import Chem
from rdkit.Chem import AllChem

# Machine learning
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestRegressor

# Progress bar
from tqdm import tqdm
from sklearn.model_selection import ParameterGrid

# Loading the train and test csv files
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

# Define the RandomForestRegressor model
model = RandomForestRegressor()

# Define the hyperparameter grid
param_grid = {
    'n_estimators': [[100], [200], [300]],
    'max_depth': [[None], [10], [20], [30]],
    'min_samples_split': [[2], [5], [10]]
}

# Use GridSearchCV to find the best hyperparameters with a progress bar
param_list = list(ParameterGrid(param_grid))
with tqdm(total=len(param_list)) as pbar:
    for params in param_list:
        grid = GridSearchCV(estimator=model, param_grid=[params], cv=3, n_jobs=-1, scoring='neg_mean_absolute_error')
        grid_result = grid.fit(X_train, y_train)
        pbar.update(1)

# Display the best hyperparameters
print(f"Best: {grid_result.best_score_} using {grid_result.best_params_}")

# Train the model with the best hyperparameters
best_model = grid_result.best_estimator_
best_model.fit(X_train, y_train)

# Make predictions on the test set
y_pred = best_model.predict(X_test_pca)

# Save the results to a CSV file
submission_df = pd.DataFrame()
submission_df['id'] = test['id']
submission_df['y'] = y_pred.flatten()
submission_df.to_csv('data/y_benchmark_rf_gridsearch.csv', index=False)
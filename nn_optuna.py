import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split, KFold
from sklearn.preprocessing import RobustScaler, StandardScaler
from sklearn.metrics import mean_squared_error
from sklearn.decomposition import PCA
from rdkit import Chem
from rdkit.Chem import AllChem, MACCSkeys
from optuna import create_study, Trial
import optuna
from torch.nn import functional as F
import os
from tqdm import tqdm

class AdvancedNeuralNetwork(nn.Module):
    def __init__(self, input_dim, layer_sizes, dropout_rates, use_batch_norm=True):
        super(AdvancedNeuralNetwork, self).__init__()
        self.layers = nn.ModuleList()
        self.batch_norms = nn.ModuleList()
        self.dropouts = nn.ModuleList()
        
        prev_size = input_dim
        for size, dropout_rate in zip(layer_sizes, dropout_rates):
            self.layers.append(nn.Linear(prev_size, size))
            if use_batch_norm:
                self.batch_norms.append(nn.BatchNorm1d(size))
            self.dropouts.append(nn.Dropout(dropout_rate))
            prev_size = size
            
        self.output_layer = nn.Linear(prev_size, 1)
        self.use_batch_norm = use_batch_norm

    def forward(self, x):
        for i, (layer, dropout) in enumerate(zip(self.layers, self.dropouts)):
            x = layer(x)
            if self.use_batch_norm:
                x = self.batch_norms[i](x)
            x = F.relu(x)
            x = dropout(x)
        return self.output_layer(x)

class ModelWrapper:
    def __init__(self, model, optimizer, criterion, scheduler=None):
        self.model = model
        self.optimizer = optimizer
        self.criterion = criterion
        self.scheduler = scheduler
        
    def train_step(self, X, y):
        self.model.train()
        self.optimizer.zero_grad()
        output = self.model(X)
        loss = self.criterion(output, y)
        loss.backward()
        self.optimizer.step()
        return loss.item()
        
    def validate(self, X, y):
        self.model.eval()
        with torch.no_grad():
            output = self.model(X)
            loss = self.criterion(output, y)
        return loss.item()

def create_fingerprints(smiles, method='morgan', radius=2, nBits=2048):
    mols = [Chem.MolFromSmiles(smile) for smile in smiles]
    fps = []
    
    for mol in mols:
        if mol is None:
            continue
            
        if method == 'morgan':
            fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius=radius, nBits=nBits)
        elif method == 'maccs':
            fp = MACCSkeys.GenMACCSKeys(mol)
        elif method == 'combined':
            # Utilisation de différents rayons pour plus de diversité
            fp1 = list(AllChem.GetMorganFingerprintAsBitVect(mol, radius=2, nBits=1024))
            fp2 = list(AllChem.GetMorganFingerprintAsBitVect(mol, radius=3, nBits=1024))
            fp3 = list(MACCSkeys.GenMACCSKeys(mol))
            fp = fp1 + fp2 + fp3
        else:
            raise ValueError(f"Unknown fingerprint method: {method}")
            
        fps.append(list(fp))
    
    return np.array(fps)

def objective(trial: Trial, X_train, y_train, X_val, y_val, input_dim):
    # Hyperparamètres de l'architecture
    n_layers = trial.suggest_int('n_layers', 2, 5)
    layer_sizes = []
    dropout_rates = []
    
    for i in range(n_layers):
        layer_sizes.append(trial.suggest_int(f'layer_{i}_size', 64, 512))
        dropout_rates.append(trial.suggest_float(f'dropout_{i}', 0.1, 0.5))
    
    # Hyperparamètres d'apprentissage
    lr = trial.suggest_float('lr', 1e-5, 1e-2, log=True)
    batch_size = trial.suggest_categorical('batch_size', [16, 32, 64, 128])
    use_batch_norm = trial.suggest_categorical('use_batch_norm', [True, False])
    
    # Création du modèle
    model = AdvancedNeuralNetwork(
        input_dim=input_dim,
        layer_sizes=layer_sizes,
        dropout_rates=dropout_rates,
        use_batch_norm=use_batch_norm
    ).to(device)
    
    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer,
        T_0=5,
        T_mult=2,
        eta_min=lr/10
    )
    
    wrapper = ModelWrapper(model, optimizer, criterion, scheduler)
    
    # Conversion des données
    X_train_tensor = torch.FloatTensor(X_train).to(device)
    y_train_tensor = torch.FloatTensor(y_train.values).reshape(-1, 1).to(device)
    X_val_tensor = torch.FloatTensor(X_val).to(device)
    y_val_tensor = torch.FloatTensor(y_val.values).reshape(-1, 1).to(device)
    
    dataset = torch.utils.data.TensorDataset(X_train_tensor, y_train_tensor)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)
    
    best_val_loss = float('inf')
    patience = 0
    max_patience = 5
    
    for epoch in range(50):
        # Entraînement
        epoch_losses = []
        for batch_X, batch_y in dataloader:
            loss = wrapper.train_step(batch_X, batch_y)
            epoch_losses.append(loss)
            
        # Validation
        val_loss = wrapper.validate(X_val_tensor, y_val_tensor)
        
        # Early stopping
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience = 0
        else:
            patience += 1
            
        if patience >= max_patience:
            break
            
        if wrapper.scheduler:
            wrapper.scheduler.step()
            
        # Report pour Optuna
        trial.report(val_loss, epoch)
        if trial.should_prune():
            raise optuna.TrialPruned()
    
    return best_val_loss

def main():
    # Configuration
    RANDOM_SEED = 42
    np.random.seed(RANDOM_SEED)
    torch.manual_seed(RANDOM_SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(RANDOM_SEED)
    
    # Chargement des données
    train = pd.read_csv('data/X_train.csv')
    test = pd.read_csv('data/X_test.csv')
    y_train = pd.read_csv('data/y_train.csv')['y']
    
    # Création des fingerprints combinés (Morgan + MACCS)
    print("Création des fingerprints...")
    train_fps = create_fingerprints(train['smiles'], method='combined')
    test_fps = create_fingerprints(test['smiles'], method='combined')
    
    # Prétraitement
    print("Prétraitement des données...")
    scaler = RobustScaler()
    X_train_scaled = scaler.fit_transform(train_fps)
    X_test_scaled = scaler.transform(test_fps)
    
    # PCA avec sélection automatique des composantes
    pca = PCA(n_components=0.99)  # Garde 99% de la variance
    X_train_transformed = pca.fit_transform(X_train_scaled)
    X_test_transformed = pca.transform(X_test_scaled)
    
    print(f"Dimensions après PCA: {X_train_transformed.shape[1]} composantes")
    
    # Split des données
    X_train, X_val, y_train_split, y_val = train_test_split(
        X_train_transformed, y_train, test_size=0.2, random_state=RANDOM_SEED
    )
    
    # Optimisation des hyperparamètres avec Optuna
    print("Début de l'optimisation des hyperparamètres...")
    study = create_study(
        direction="minimize",
        pruner=optuna.pruners.MedianPruner()
    )
    
    study.optimize(
        lambda trial: objective(
            trial, X_train, y_train_split, X_val, y_val, X_train_transformed.shape[1]
        ),
        n_trials=50,
        timeout=3600  # 1 heure maximum
    )
    
    print("Meilleurs hyperparamètres trouvés:", study.best_params)
    
    # Création de l'ensemble avec les meilleurs hyperparamètres
    print("Entraînement des modèles finaux...")
    models = []
    kf = KFold(n_splits=5, shuffle=True, random_state=RANDOM_SEED)
    
    for fold, (train_idx, val_idx) in enumerate(kf.split(X_train_transformed)):
        print(f"\nEntraînement du modèle {fold + 1}/5")
        X_fold_train = X_train_transformed[train_idx]
        y_fold_train = y_train.iloc[train_idx]
        
        # Création du modèle avec les meilleurs hyperparamètres
        model = AdvancedNeuralNetwork(
            input_dim=X_train_transformed.shape[1],
            layer_sizes=[study.best_params[f'layer_{i}_size'] for i in range(study.best_params['n_layers'])],
            dropout_rates=[study.best_params[f'dropout_{i}'] for i in range(study.best_params['n_layers'])],
            use_batch_norm=study.best_params['use_batch_norm']
        ).to(device)
        
        # Configuration de l'entraînement
        optimizer = optim.Adam(model.parameters(), lr=study.best_params['lr'])
        criterion = nn.MSELoss()
        wrapper = ModelWrapper(model, optimizer, criterion)
        
        # Entraînement
        X_tensor = torch.FloatTensor(X_fold_train).to(device)
        y_tensor = torch.FloatTensor(y_fold_train.values).reshape(-1, 1).to(device)
        dataset = torch.utils.data.TensorDataset(X_tensor, y_tensor)
        dataloader = torch.utils.data.DataLoader(
            dataset, 
            batch_size=study.best_params['batch_size'], 
            shuffle=True
        )
        
        best_loss = float('inf')
        for epoch in range(50):
            for batch_X, batch_y in dataloader:
                loss = wrapper.train_step(batch_X, batch_y)
                if loss < best_loss:
                    best_loss = loss
        
        models.append(wrapper)
    
    # Prédictions sur l'ensemble de test
    print("\nGénération des prédictions finales...")
    test_tensor = torch.FloatTensor(X_test_transformed).to(device)
    predictions = []
    
    for wrapper in models:
        wrapper.model.eval()
        with torch.no_grad():
            pred = wrapper.model(test_tensor).cpu().numpy().reshape(-1)
            predictions.append(pred)
    
    final_predictions = np.mean(predictions, axis=0)
    
    # Sauvegarde des prédictions
    submission = pd.DataFrame({
        'id': range(4400, 4400 + len(final_predictions)),
        'y': final_predictions
    })
    submission.to_csv('submission.csv', index=False)
    print("\nPrédictions sauvegardées dans submission.csv")

if __name__ == "__main__":
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    main()
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import Data, DataLoader
from torch_geometric.nn import GCNConv, global_mean_pool
import pandas as pd
import numpy as np
from rdkit import Chem
from tqdm import tqdm
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error

# Configuration
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {DEVICE}")

class MoleculeGNN(torch.nn.Module):
    def __init__(self, num_node_features):
        super(MoleculeGNN, self).__init__()
        # Couches de convolution
        self.conv1 = GCNConv(num_node_features, 128)
        self.conv2 = GCNConv(128, 64)
        self.conv3 = GCNConv(64, 32)
        
        # Couches fully connected pour la prédiction finale
        self.fc1 = nn.Linear(32, 16)
        self.fc2 = nn.Linear(16, 1)
        
        # Dropout pour régularisation
        self.dropout = nn.Dropout(0.2)

    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch
        
        # Convolutions sur le graphe
        x = F.relu(self.conv1(x, edge_index))
        x = self.dropout(x)
        x = F.relu(self.conv2(x, edge_index))
        x = self.dropout(x)
        x = F.relu(self.conv3(x, edge_index))
        
        # Pooling global
        x = global_mean_pool(x, batch)
        
        # Fully connected layers
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        
        return x

def mol_to_graph(mol):
    """Convertit une molécule RDKit en graph PyTorch Geometric."""
    # Features des atomes (one-hot encoding du type d'atome)
    atom_features = []
    for atom in mol.GetAtoms():
        features = []
        # One-hot encoding du type d'atome (C, N, O, F, etc.)
        atom_type = atom.GetSymbol()
        features.extend([
            atom.GetAtomicNum(),
            atom.GetDegree(),
            atom.GetFormalCharge(),
            atom.GetNumRadicalElectrons(),
            int(atom.GetIsAromatic()),
            atom.GetHybridization(),
        ])
        atom_features.append(features)
    
    # Construction des liaisons (edges)
    edges = []
    for bond in mol.GetBonds():
        i = bond.GetBeginAtomIdx()
        j = bond.GetEndAtomIdx()
        # Ajouter dans les deux sens car le graphe est non dirigé
        edges.append([i, j])
        edges.append([j, i])

    # Conversion en tenseurs PyTorch
    x = torch.tensor(atom_features, dtype=torch.float)
    edge_index = torch.tensor(edges, dtype=torch.long).t()
    
    return Data(x=x, edge_index=edge_index)

def prepare_data(smiles_list):
    """Prépare les données pour le GNN."""
    data_list = []
    for smiles in tqdm(smiles_list, desc="Préparation des données"):
        mol = Chem.MolFromSmiles(smiles)
        if mol is not None:
            data = mol_to_graph(mol)
            data_list.append(data)
    return data_list

def train_model(model, train_loader, optimizer, device):
    """Entraîne le modèle sur une époque."""
    model.train()
    total_loss = 0
    for data in train_loader:
        data = data.to(device)
        optimizer.zero_grad()
        out = model(data).squeeze()
        loss = F.mse_loss(out, data.y)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(train_loader)

def evaluate_model(model, loader, device):
    """Évalue le modèle."""
    model.eval()
    predictions = []
    targets = []
    with torch.no_grad():
        for data in loader:
            data = data.to(device)
            out = model(data)
            predictions.extend(out.cpu().numpy())
            targets.extend(data.y.cpu().numpy())
    return mean_squared_error(targets, predictions)

def main():
    # Chargement des données
    print("Chargement des données...")
    train_data = pd.read_csv('data/X_train.csv')
    test_data = pd.read_csv('data/X_test.csv')
    y_train = pd.read_csv('data/y_train.csv')['y']
    
    # Préparation des données
    print("Conversion des molécules en graphes...")
    train_graphs = prepare_data(train_data['smiles'])
    test_graphs = prepare_data(test_data['smiles'])
    
    # Ajout des valeurs cibles aux graphes d'entraînement
    for i, graph in enumerate(train_graphs):
        graph.y = torch.tensor([y_train.iloc[i]], dtype=torch.float).reshape(-1)
    
    # Paramètres d'entraînement
    BATCH_SIZE = 32
    EPOCHS = 100
    LEARNING_RATE = 0.001
    
    # Validation croisée
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    cv_scores = []
    fold_predictions = []
    
    # Créer des indices pour la validation croisée
    indices = np.arange(len(train_graphs))
    
    for fold, (train_idx, val_idx) in enumerate(kf.split(indices)):
        print(f"\nFold {fold + 1}/5")
        
        # Préparation des loaders
        train_loader = DataLoader([train_graphs[i] for i in train_idx], batch_size=BATCH_SIZE, shuffle=True)
        val_loader = DataLoader([train_graphs[i] for i in val_idx], batch_size=BATCH_SIZE)
        
        # Initialisation du modèle
        model = MoleculeGNN(num_node_features=train_graphs[0].x.shape[1]).to(DEVICE)
        optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=5)
        
        # Entraînement
        best_val_loss = float('inf')
        patience = 10
        patience_counter = 0
        
        for epoch in range(EPOCHS):
            train_loss = train_model(model, train_loader, optimizer, DEVICE)
            val_loss = evaluate_model(model, val_loader, DEVICE)
            
            scheduler.step(val_loss)
            
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
                # Sauvegarder le meilleur modèle
                torch.save(model.state_dict(), f'models/gnn_fold_{fold}.pt')
            else:
                patience_counter += 1
            
            if patience_counter >= patience:
                print(f"Early stopping à l'époque {epoch}")
                break
            
            if (epoch + 1) % 10 == 0:
                print(f"Époque {epoch+1}/{EPOCHS}, Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")
        
        # Charger le meilleur modèle
        model.load_state_dict(torch.load(f'models/gnn_fold_{fold}.pt'))
        final_val_loss = evaluate_model(model, val_loader, DEVICE)
        cv_scores.append(final_val_loss)
        print(f"Score final du fold {fold + 1}: {final_val_loss:.4f}")
    
    print(f"\nScore CV moyen: {np.mean(cv_scores):.4f} ± {np.std(cv_scores):.4f}")
    
    # Prédictions sur l'ensemble de test
    print("\nGénération des prédictions finales...")
    test_loader = DataLoader(test_graphs, batch_size=BATCH_SIZE)
    final_predictions = np.zeros(len(test_graphs))
    
    for fold in range(5):
        model = MoleculeGNN(num_node_features=train_graphs[0].x.shape[1]).to(DEVICE)
        model.load_state_dict(torch.load(f'models/gnn_fold_{fold}.pt'))
        model.eval()
        
        predictions = []
        with torch.no_grad():
            for data in test_loader:
                data = data.to(DEVICE)
                out = model(data)
                predictions.extend(out.cpu().numpy())
        
        final_predictions += np.array(predictions).flatten()
    
    final_predictions /= 5
    
    # Sauvegarde des prédictions
    submission = pd.DataFrame({
        'id': range(4400, 4400 + len(final_predictions)),
        'y': final_predictions
    })
    submission.to_csv('submission.csv', index=False)
    print("\nPrédictions sauvegardées dans submission.csv")

if __name__ == "__main__":
    main()
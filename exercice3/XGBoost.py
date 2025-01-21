import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import KFold
from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error
from rdkit import Chem
from rdkit.Chem import AllChem, MACCSkeys, Descriptors
import optuna
from tqdm import tqdm
import os
import joblib

def create_extended_fingerprints(smiles):
    """Crée des fingerprints enrichis avec des descripteurs moléculaires."""
    features_list = []
    for smile in tqdm(smiles):
        mol = Chem.MolFromSmiles(smile)
        if mol is None:
            continue

        # Morgan Fingerprints avec différents rayons
        fp1 = list(AllChem.GetMorganFingerprintAsBitVect(mol, radius=2, nBits=1024))
        fp2 = list(AllChem.GetMorganFingerprintAsBitVect(mol, radius=3, nBits=1024))
        
        # MACCS Keys
        fp3 = list(MACCSkeys.GenMACCSKeys(mol))
        
        # Descripteurs physico-chimiques
        descriptors = []
        descriptors.append(Descriptors.ExactMolWt(mol))
        descriptors.append(Descriptors.NumRotatableBonds(mol))
        descriptors.append(Descriptors.NumHAcceptors(mol))
        descriptors.append(Descriptors.NumHDonors(mol))
        descriptors.append(Descriptors.TPSA(mol))
        descriptors.append(Descriptors.MolLogP(mol))
        descriptors.append(Descriptors.NumAromaticRings(mol))
        
        # Combinaison de toutes les features
        features = fp1 + fp2 + fp3 + descriptors
        features_list.append(features)
    
    return np.array(features_list)

def objective_xgb(trial, X, y, X_val=None, y_val=None):
    params = {
        'max_depth': trial.suggest_int('max_depth', 3, 12),
        'learning_rate': trial.suggest_float('learning_rate', 0.001, 0.1, log=True),
        'n_estimators': trial.suggest_int('n_estimators', 100, 1000),
        'min_child_weight': trial.suggest_float('min_child_weight', 1, 10),
        'subsample': trial.suggest_float('subsample', 0.5, 1.0),
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.5, 1.0),
        'reg_alpha': trial.suggest_float('reg_alpha', 1e-8, 1.0, log=True),
        'reg_lambda': trial.suggest_float('reg_lambda', 1e-8, 1.0, log=True),
        'gamma': trial.suggest_float('gamma', 1e-8, 1.0, log=True)
    }
    
    model = XGBRegressor(**params, random_state=42)
    
    if X_val is not None:
        model.fit(X, y)
        pred = model.predict(X_val)
        score = mean_squared_error(y_val, pred)
        return score
    else:
        model.fit(X, y)
        return model

def main():
    # Création du dossier pour sauvegarder les modèles
    os.makedirs('models', exist_ok=True)
    
    # Chargement des données
    print("Chargement des données...")
    train = pd.read_csv('data/X_train.csv')
    test = pd.read_csv('data/X_test.csv')
    y_train = pd.read_csv('data/y_train.csv')['y']
    
    # Vérification des fichiers de features sauvegardés
    train_features_file = 'data/train_extended_features.joblib'
    test_features_file = 'data/test_extended_features.joblib'
    
    if os.path.exists(train_features_file) and os.path.exists(test_features_file):
        print("Chargement des features précalculées...")
        train_features = joblib.load(train_features_file)
        test_features = joblib.load(test_features_file)
    else:
        print("Création des fingerprints enrichis...")
        train_features = create_extended_fingerprints(train['smiles'])
        test_features = create_extended_fingerprints(test['smiles'])
        
        print("Sauvegarde des features...")
        joblib.dump(train_features, train_features_file)
        joblib.dump(test_features, test_features_file)
    
    # Normalisation
    print("Normalisation des données...")
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(train_features)
    X_test_scaled = scaler.transform(test_features)
    
    # Validation croisée
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    
    # Optimisation des hyperparamètres
    print("Optimisation des hyperparamètres...")
    xgb_models = []
    best_params = None
    best_score = float('inf')
    
    for fold, (train_idx, val_idx) in enumerate(kf.split(X_train_scaled)):
        print(f"\nFold {fold + 1}/5")
        X_fold_train, X_fold_val = X_train_scaled[train_idx], X_train_scaled[val_idx]
        y_fold_train, y_fold_val = y_train.iloc[train_idx], y_train.iloc[val_idx]
        
        # Optimisation XGBoost
        print("Optimisation XGBoost...")
        study = optuna.create_study(direction="minimize")
        study.optimize(
            lambda trial: objective_xgb(trial, X_fold_train, y_fold_train, X_fold_val, y_fold_val),
            n_trials=20
        )
        
        # Sauvegarder les meilleurs paramètres si ce fold donne un meilleur score
        if study.best_value < best_score:
            best_score = study.best_value
            best_params = study.best_params
        
        # Création du modèle XGBoost avec les meilleurs paramètres
        model = XGBRegressor(**study.best_params, random_state=42)
        model.fit(X_fold_train, y_fold_train)
        
        # Sauvegarde du modèle
        model_path = f'models/xgb_model_fold_{fold}.joblib'
        joblib.dump(model, model_path)
        print(f"Modèle sauvegardé: {model_path}")
        
        xgb_models.append(model)
        
        # Évaluation sur le fold de validation
        val_pred = model.predict(X_fold_val)
        val_score = mean_squared_error(y_fold_val, val_pred)
        print(f"Score MSE sur le fold de validation: {val_score:.4f}")
    
    # Sauvegarde des meilleurs paramètres
    joblib.dump(best_params, 'models/best_params.joblib')
    print("\nMeilleurs paramètres:", best_params)
    print("Meilleur score:", best_score)
    
    # Prédictions finales
    print("\nGénération des prédictions finales...")
    predictions = np.zeros(len(test_features))
    
    for i, model in enumerate(xgb_models):
        fold_predictions = model.predict(X_test_scaled)
        predictions += fold_predictions
        
    predictions /= len(xgb_models)
    
    # Sauvegarde des prédictions
    submission = pd.DataFrame({
        'id': range(4400, 4400 + len(predictions)),
        'y': predictions
    })
    submission.to_csv('submission.csv', index=False)
    print("\nPrédictions sauvegardées dans submission.csv")

if __name__ == "__main__":
    main()
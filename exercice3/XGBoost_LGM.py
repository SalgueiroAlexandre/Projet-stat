import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import KFold
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
from catboost import CatBoostRegressor
from sklearn.linear_model import ElasticNet
from sklearn.metrics import mean_squared_error
import optuna
from tqdm import tqdm
import os
import joblib

class StackingRegressor:
    def __init__(self, base_models, meta_model):
        self.base_models = base_models
        self.meta_model = meta_model
        self.base_predictions = None
        
    def fit(self, X, y, X_val=None, y_val=None):
        # Entraîner les modèles de base
        base_predictions = np.zeros((X.shape[0], len(self.base_models)))
        for i, model in enumerate(self.base_models):
            model.fit(X, y)
            base_predictions[:, i] = model.predict(X)
        
        # Entraîner le méta-modèle
        self.meta_model.fit(base_predictions, y)
        
    def predict(self, X):
        # Générer les prédictions des modèles de base
        base_predictions = np.zeros((X.shape[0], len(self.base_models)))
        for i, model in enumerate(self.base_models):
            base_predictions[:, i] = model.predict(X)
        
        # Prédiction finale avec le méta-modèle
        return self.meta_model.predict(base_predictions)

def objective(trial, X, y, X_val=None, y_val=None):
    # Paramètres XGBoost
    xgb_params = {
        'max_depth': trial.suggest_int('xgb_max_depth', 3, 12),
        'learning_rate': trial.suggest_float('xgb_lr', 0.001, 0.1, log=True),
        'n_estimators': trial.suggest_int('xgb_n_estimators', 100, 1000),
        'min_child_weight': trial.suggest_float('xgb_min_child_weight', 1, 10),
        'subsample': trial.suggest_float('xgb_subsample', 0.5, 1.0),
        'colsample_bytree': trial.suggest_float('xgb_colsample', 0.5, 1.0),
        'reg_alpha': trial.suggest_float('xgb_alpha', 1e-8, 1.0, log=True),
        'reg_lambda': trial.suggest_float('xgb_lambda', 1e-8, 1.0, log=True)
    }
    
    # Paramètres LightGBM
    lgb_params = {
        'num_leaves': trial.suggest_int('lgb_num_leaves', 20, 100),
        'learning_rate': trial.suggest_float('lgb_lr', 0.001, 0.1, log=True),
        'n_estimators': trial.suggest_int('lgb_n_estimators', 100, 1000),
        'min_child_samples': trial.suggest_int('lgb_min_child_samples', 5, 100),
        'subsample': trial.suggest_float('lgb_subsample', 0.5, 1.0),
        'colsample_bytree': trial.suggest_float('lgb_colsample', 0.5, 1.0)
    }
    
    # Paramètres CatBoost
    cat_params = {
        'learning_rate': trial.suggest_float('cat_lr', 0.001, 0.1, log=True),
        'depth': trial.suggest_int('cat_depth', 3, 12),
        'iterations': trial.suggest_int('cat_iterations', 100, 1000)
    }
    
    # Paramètres ElasticNet
    elastic_params = {
        'alpha': trial.suggest_float('elastic_alpha', 1e-5, 1.0, log=True),
        'l1_ratio': trial.suggest_float('elastic_l1_ratio', 0.0, 1.0)
    }
    
    # Création des modèles
    models = [
        XGBRegressor(**xgb_params, random_state=42),
        LGBMRegressor(**lgb_params, random_state=42),
        CatBoostRegressor(**cat_params, random_state=42, verbose=False)
    ]
    
    meta_model = ElasticNet(**elastic_params, random_state=42)
    
    # Création du stacking
    stacking = StackingRegressor(models, meta_model)
    
    if X_val is not None:
        stacking.fit(X, y)
        pred = stacking.predict(X_val)
        score = mean_squared_error(y_val, pred)
        return score
    else:
        stacking.fit(X, y)
        return stacking

def main():
    # Configuration des dossiers
    os.makedirs('models', exist_ok=True)
    
    # Chargement des données
    print("Chargement des données...")
    train_fps = pd.read_csv('data/train_fps.csv', index_col=0).values
    test_fps = pd.read_csv('data/test_fps.csv', index_col=0).values
    y_train = pd.read_csv('data/y_train.csv')['y']
    
    print(f"Dimensions des données : Train {train_fps.shape}, Test {test_fps.shape}")
    
    # Normalisation
    print("Normalisation des données...")
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(train_fps)
    X_test_scaled = scaler.transform(test_fps)
    
    # Validation croisée
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    stacking_models = []
    cv_scores = []
    
    for fold, (train_idx, val_idx) in enumerate(kf.split(X_train_scaled)):
        print(f"\nFold {fold + 1}/5")
        X_fold_train, X_fold_val = X_train_scaled[train_idx], X_train_scaled[val_idx]
        y_fold_train, y_fold_val = y_train.iloc[train_idx], y_train.iloc[val_idx]
        
        # Optimisation
        print("Optimisation des hyperparamètres...")
        study = optuna.create_study(direction="minimize")
        study.optimize(
            lambda trial: objective(trial, X_fold_train, y_fold_train, X_fold_val, y_fold_val),
            n_trials=30
        )
        
        # Entraînement avec les meilleurs paramètres
        best_stacking = objective(study.best_trial, X_fold_train, y_fold_train)
        stacking_models.append(best_stacking)
        
        # Évaluation
        val_pred = best_stacking.predict(X_fold_val)
        val_score = mean_squared_error(y_fold_val, val_pred)
        cv_scores.append(val_score)
        print(f"Score MSE sur le fold: {val_score:.4f}")
        
        # Sauvegarde du modèle
        joblib.dump(best_stacking, f'models/stacking_model_fold_{fold}.joblib')
    
    print(f"\nScore CV moyen: {np.mean(cv_scores):.4f} ± {np.std(cv_scores):.4f}")
    
    # Prédictions finales
    print("\nGénération des prédictions finales...")
    predictions = np.zeros(len(test_fps))
    
    for model in stacking_models:
        pred = model.predict(X_test_scaled)
        predictions += pred
    
    predictions /= len(stacking_models)
    
    # Sauvegarde des prédictions
    submission = pd.DataFrame({
        'id': range(4400, 4400 + len(predictions)),
        'y': predictions
    })
    submission.to_csv('submission.csv', index=False)
    print("\nPrédictions sauvegardées dans submission.csv")

if __name__ == "__main__":
    main()
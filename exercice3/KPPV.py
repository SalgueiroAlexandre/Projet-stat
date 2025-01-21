import numpy as np
import pandas as pd
from sklearn.neighbors import KNeighborsRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error
import warnings
warnings.filterwarnings('ignore')

def main():
    # Chargement des données
    print("Chargement des données...")
    train_fps = pd.read_csv('data/train_fps.csv', index_col=0)
    test_fps = pd.read_csv('data/test_fps.csv', index_col=0)
    y_train = pd.read_csv('data/y_train.csv')['y']
    
    print(f"Dimensions des données : Train {train_fps.shape}, Test {test_fps.shape}")
    
    # Normalisation des données
    print("Normalisation des données...")
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(train_fps)
    X_test_scaled = scaler.transform(test_fps)
    
    # Configuration de la validation croisée
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    k_values = [3, 5, 7, 10, 15]  # Différentes valeurs de k à tester
    cv_scores = {k: [] for k in k_values}
    
    # Évaluation des différentes valeurs de k
    print("\nÉvaluation des différentes valeurs de k...")
    for k in k_values:
        print(f"\nTest avec k={k}")
        for fold, (train_idx, val_idx) in enumerate(kf.split(X_train_scaled)):
            # Split des données
            X_fold_train, X_fold_val = X_train_scaled[train_idx], X_train_scaled[val_idx]
            y_fold_train, y_fold_val = y_train.iloc[train_idx], y_train.iloc[val_idx]
            
            # Création et entraînement du modèle
            knn = KNeighborsRegressor(n_neighbors=k, weights='distance')
            knn.fit(X_fold_train, y_fold_train)
            
            # Prédictions et score
            val_pred = knn.predict(X_fold_val)
            score = mean_squared_error(y_fold_val, val_pred)
            cv_scores[k].append(score)
            print(f"Fold {fold + 1}: MSE = {score:.4f}")
    
    # Affichage des résultats moyens pour chaque k
    print("\nRésultats moyens par valeur de k:")
    for k in k_values:
        mean_score = np.mean(cv_scores[k])
        std_score = np.std(cv_scores[k])
        print(f"k={k}: MSE = {mean_score:.4f} ± {std_score:.4f}")
    
    # Utilisation du meilleur k pour les prédictions finales
    best_k = min(cv_scores.keys(), key=lambda k: np.mean(cv_scores[k]))
    print(f"\nMeilleure valeur de k: {best_k}")
    
    # Entraînement du modèle final avec le meilleur k
    final_model = KNeighborsRegressor(n_neighbors=best_k, weights='distance')
    final_model.fit(X_train_scaled, y_train)
    
    # Prédictions sur l'ensemble de test
    test_predictions = final_model.predict(X_test_scaled)
    
    # Sauvegarde des prédictions
    submission = pd.DataFrame({
        'id': range(4400, 4400 + len(test_predictions)),
        'y': test_predictions
    })
    submission.to_csv('submission_knn.csv', index=False)
    print("\nPrédictions sauvegardées dans submission_knn.csv")

if __name__ == "__main__":
    main()
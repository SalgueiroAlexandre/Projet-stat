import pandas as pd
import numpy as np
from graphviz import Digraph


def normQ(x,q):
    """
    :param x:
    :param q:
    :return: Norme q-ième de x
    """
    return np.sum(np.abs(x)**q)**(1/q)

class Noeud:
    def __init__(self, seuil = None, colonne = None, gauche = None, droite = None, individus=None, moyenne=None):
        if individus is None:
            individus = []
        self.colonne = colonne
        self.seuil = seuil
        self.gauche = gauche
        self.droite = droite
        self.individus = individus
        self.moyenne = moyenne

    def est_feuille(self):
        return self.gauche is None and self.droite is None

    def __repr__(self, profondeur=0):
        if self.est_feuille():
            return f"{self.moyenne},{profondeur}"
        else:
            return f"Colonne : {self.colonne}\nSeuil : {self.seuil}\n" \
                   f"Gauche : {self.gauche.__repr__(profondeur+1)}\n" \
                   f"Droite : {self.droite.__repr__(profondeur+1)}"

class Arbre:
    def __init__(self, data, cible, profondeur_max=3):
        self.data = data
        self.cible = cible
        self.profondeur_max = profondeur_max
        self.racine = self.fit(data, cible, profondeur=0)  # Premier appel de fit avec profondeur=0


    def separation(self, data, q=2):
        meilleur_cout = float('inf')
        meilleur_seuil = None
        meilleur_col = None

        for col in self.data.columns:
            val = data[col].sort_values()  # Trier les valeurs de la colonne
            alphas = [(val.iloc[i] + val.iloc[i+1]) / 2 for i in range(len(val)-1)]  # Alphas

            for alpha in alphas:
                gauche = data[data[col] <= alpha]
                droite = data[data[col] > alpha]

                # récupérer les cibles
                cibles_gauche = self.cible[gauche.index]
                cibles_droite = self.cible[droite.index]

                # Calculer le coût : ici, norme q-ième
                cout_gauche = normQ(cibles_gauche, q)
                cout_droite = normQ(cibles_droite, q)
                cout_total = cout_gauche + cout_droite  # Coût total pour cette coupure

                if cout_total < meilleur_cout:  # Vérifier si c'est la meilleure coupure
                    meilleur_cout = cout_total
                    meilleur_seuil = alpha
                    meilleur_col = col

        #print(f"Meilleur coût : {meilleur_cout:.2f}, Colonne : {meilleur_col}, Seuil : {meilleur_seuil:.2f}")
        return meilleur_seuil, meilleur_col

    def variance(self, cible):
        return np.var(cible)

    def fit(self, data, cible, q=2, seuil_variance=0.1, profondeur=0):
        # condition d'arrêt
        if profondeur >= self.profondeur_max or self.variance(cible) < seuil_variance:
            print("noeud terminal Arrêt à la profondeur", profondeur)
            return Noeud(individus=cible, moyenne=np.mean(cible))

        # recherche de la meilleure coupure
        seuil, colonne = self.separation(data, q)

        if seuil is None:
            print("noeud terminal Aucune coupure trouvée à la profondeur", profondeur)
            return Noeud(individus=cible, moyenne=np.mean(cible))

        # séparation des données
        gauche = data[data[colonne] <= seuil]
        cibles_gauche = cible[gauche.index]

        droite = data[data[colonne] > seuil]
        cibles_droite = cible[droite.index]

        # Créer les sous-arbres récursivement
        noeud_gauche = self.fit(gauche, cibles_gauche, profondeur=profondeur + 1)  # Incrémenter la profondeur
        noeud_droite = self.fit(droite, cibles_droite, profondeur=profondeur + 1)  # Incrémenter la profondeur

        # Retourner le nœud avec la condition de coupure et les sous-arbres gauche et droite
        return Noeud(seuil=seuil, colonne=colonne,
                     gauche=noeud_gauche, droite=noeud_droite, individus=cible, moyenne=np.mean(cible))

if __name__ == '__main__':
    # Génération des données
    np.random.seed(42)  # Pour garantir la reproductibilité

    # Deux variables explicatives X1 et X2
    X1 = np.random.uniform(0, 10, 10)  # 10 valeurs aléatoires entre 0 et 10
    X2 = np.random.uniform(0, 10, 10)  # 10 valeurs aléatoires entre 0 et 10

    # Valeur cible Y en fonction de X1 et X2 avec un peu de bruit
    Y = 2 * X1 + 3 * X2 + np.random.normal(0, 1, 10)  # Relation linéaire avec bruit

    # Création d'un DataFrame
    data = pd.DataFrame({
        'X1': X1,
        'X2': X2,
    })

    cible = pd.Series(Y)

    # Création de l'arbre
    arbre = Arbre(data,cible, profondeur_max=2)
    print(arbre.racine)

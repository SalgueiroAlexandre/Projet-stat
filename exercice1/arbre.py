import pandas as pd
import numpy as np
from graphviz import Digraph
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from time import time
import matplotlib.pyplot as plt

def normQ(x,q):
    """
    :param x: donnée
    :param q: q-ième Norme
    :return: Norme q-ième de x
    """
    return np.sum(np.abs(x)**q)**(1/q)

def variance_pond(x):
    """
    Variance pondérée
    :param x: donnée
    :return: variance pondérée
    """
    return np.var(x) * len(x)

# Validation croisée
def validation_croisee(data, cible, profondeur_max, n_splits=5):
    """
    Validation croisée pour l'arbre de régression
    :param data: données d'entrée
    :param cible: valeurs cibles
    :param profondeur_max: profondeur maximale de l'arbre
    :param n_splits: nombre de plis
    :return: erreur quadratique moyenne
    """
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)
    erreurs_moyennes = []

    # Boucle sur chaque pli
    for train_index, test_index in kf.split(data):
        # Séparation des données en jeu d'entraînement et de test pour le pli actuel
        X_train, X_test = data.iloc[train_index], data.iloc[test_index]
        y_train, y_test = cible.iloc[train_index], cible.iloc[test_index]

        # Création et entraînement de l'arbre de régression sur les données d'entraînement
        arbre = Arbre(X_train, y_train, profondeur_max=profondeur_max)

        # Prédiction sur les données de test
        predictions_test = arbre.simulation(X_test)

        # Calcul de l'erreur quadratique moyenne pour ce pli
        erreur = mean_squared_error(y_test, predictions_test)
        erreurs_moyennes.append(erreur)
        print(f"Erreur quadratique moyenne pour ce pli : {erreur:.4f}")

    # Calcul de l'erreur moyenne sur tous les plis
    erreur_moyenne = np.mean(erreurs_moyennes)
    print(f"\nErreur quadratique moyenne sur les {n_splits} plis : {erreur_moyenne:.4f}")
    return erreur_moyenne

class Noeud:
    def __init__(self, seuil = None, colonne = None, gauche = None, droite = None, individus=None, moyenne=None):
        """
        Constructeur d'un objet nœud
        :param seuil: seuil de la coupure
        :param colonne: colonne concerné
        :param gauche: nœud à sa gauche
        :param droite: nœud à sa droite
        :param individus: liste des individus du nœud
        :param moyenne: moyenne des individus
        """
        if individus is None:
            individus = []
        self.colonne = colonne
        self.seuil = seuil
        self.gauche = gauche
        self.droite = droite
        self.individus = individus
        self.moyenne = moyenne

    def est_feuille(self):
        """
        Vérifie si le nœud est une feuille ou non
        :return: bool
        """
        return self.gauche is None and self.droite is None

    def nbr_individus(self):
        """
        :return: Nombre d'individus dans le nœud
        """
        return len(self.individus)

    def __repr__(self, profondeur=0):
        if self.est_feuille():
            return f"{self.moyenne},{profondeur}"
        else:
            return f"Colonne : {self.colonne}\nSeuil : {self.seuil}\n" \
                   f"Gauche : {self.gauche.__repr__(profondeur+1)}\n" \
                   f"Droite : {self.droite.__repr__(profondeur+1)}"


def variance(cible):
    """
    Variance au sein des données cible
    :param cible: donnée cible
    :return: variance
    """
    return np.var(cible)


class Arbre:
    def __init__(self, data, cible, profondeur_max=3, verbose=False,seuil_variance=1.5):
        """
        Constructeur d'un objet Arbre
        :param data: dataframe avec les features X
        :param cible: tableau avec les valeurs cible Y
        :param profondeur_max: profondeur max de l'arbre
        :param verbose: sortie écrite lors des fonctions
        :param seuil_variance: cas d'arrêt variance au sein d'un nœud de l'arbre
        """
        self.data = data
        self.cible = cible
        self.profondeur_max = profondeur_max
        self.verbose = verbose
        self.seuil_variance = seuil_variance
        self.racine = self.fit(data, cible, profondeur=0)  # Premier appel de fit avec profondeur=0
        self.predictions_test = self.simulation(self.data)
        self.score_test(self.predictions_test)


    def score_test(self,predictions_test):
        """
        Calcul l'erreur moyenne quadratique de l'arbre
        :param predictions_test: prediction sur l'ensemble d'entrainement
        :return: 0
        """
        print("Valeur test de l'arbre : "+str(np.mean((self.cible - predictions_test) ** 2)))

    def separation(self, data):
        """
        Recherche de la meilleure séparation des données
        :param data: donnée X et Y
        :param verbose: affichage ou non
        :return: meilleur_seuil, meilleur_colonne
        """
        meilleur_cout = float('inf')
        meilleur_seuil = None
        meilleur_col = None

        for col in self.data.columns:
            val = data[col].sort_values().unique()  # Trier les valeurs de la colonne
            alphas = [(val[i] + val[i+1]) / 2 for i in range(len(val)-1)]  # Alphas

            for alpha in alphas:
                gauche = data[data[col] <= alpha]
                droite = data[data[col] > alpha]

                # récupérer les cibles
                cibles_gauche = self.cible[gauche.index]
                cibles_droite = self.cible[droite.index]

                # Calculer le coût : ici, on utilise la variance pondérée comme exemple
                cout_gauche = variance_pond(cibles_gauche)
                cout_droite = variance_pond(cibles_droite)
                cout_total = cout_gauche + cout_droite  # Coût total pour cette coupure
                if cout_total < meilleur_cout:  # Vérifier si c'est la meilleure coupure
                    meilleur_cout = cout_total
                    meilleur_seuil = alpha
                    meilleur_col = col

        if self.verbose:
            # Affichage conditionnel si une coupure a été trouvée
            if meilleur_seuil is not None and meilleur_col is not None:
                print(f"Meilleur coût : {meilleur_cout:.2f}, Colonne : {meilleur_col}, Seuil : {meilleur_seuil:.2f}")
            else:
                print("Aucune coupure valide trouvée")
        return meilleur_seuil, meilleur_col

    def fit(self, data, cible, profondeur=0, seuil_variance = None):
        """
        Création de l'arbre complet avec les meilleures coupures
        :param data: valeur de X
        :param cible: valeur de Y
        :param seuil_variance: seuille de variance
        :param profondeur: profondeur actuel
        :return: arbre
        """
        if seuil_variance is None:
            seuil_variance = self.seuil_variance
        # condition d'arrêt
        if profondeur >= self.profondeur_max:
            if self.verbose:
                print("noeud terminal Arrêt à la profondeur max", profondeur)
            return Noeud(individus=cible, moyenne=np.mean(cible))

        if variance(cible) < seuil_variance:
            if self.verbose:
                print("nombre d'individu",len(cible))
                print("noeud terminal Arrêt à la profondeur à cause du seuil de variance", profondeur, variance(cible))
            return Noeud(individus=cible, moyenne=np.mean(cible))
        # recherche de la meilleure coupure
        seuil, colonne = self.separation(data)

        if seuil is None:
            if self.verbose:
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

    def simulation(self, nouvelle_donnee):
        """
        Simulation de nouveaux individus
        :param nouvelle_donnee: matrice d'individus
        :return: tableau des predictions
        """
        predictions = []
        for _,ligne in nouvelle_donnee.iterrows():
            predictions.append(self._simulation_individu(self.racine, ligne))
        return predictions

    def _simulation_individu(self, noeud, individu):
        if noeud.est_feuille():
            return noeud.moyenne

        if individu[noeud.colonne] <= noeud.seuil:
            return self._simulation_individu(noeud.gauche, individu)
        else:
            return self._simulation_individu(noeud.droite, individu)

    def visualiser_arbre(self):
        """
        Enregistre l'arbre de décision au format png
        """
        # Créer un objet Digraph pour l'arbre
        graph = Digraph()
        self._ajouter_noeud(self.racine, graph)
        graph.render("arbre_decision", format="png", cleanup=True)  # Sauvegarde en PNG

    def _ajouter_noeud(self, noeud, graph, parent_id=None, direction=None, profondeur=0):
        # Générer un identifiant unique pour chaque nœud
        node_id = str(id(noeud))

        # Déterminer l'étiquette du nœud
        if noeud.est_feuille():
            label = f"Moyenne : {noeud.moyenne:.2f}\nProfondeur : {profondeur} nombre individu : {noeud.nbr_individus()}"
        else:
            label = f"{noeud.colonne} <= {noeud.seuil:.2f}\nProfondeur : {profondeur}"

        # Ajouter le nœud au graphique
        graph.node(node_id, label=label)

        # Ajouter une arête depuis le parent si ce n'est pas la racine
        if parent_id is not None:
            graph.edge(parent_id, node_id, label=direction)

        # Appeler récursivement pour les sous-arbres gauche et droit
        if noeud.gauche:
            self._ajouter_noeud(noeud.gauche, graph, node_id, "Oui", profondeur + 1)
        if noeud.droite:
            self._ajouter_noeud(noeud.droite, graph, node_id, "Non", profondeur + 1)

    def __repr__(self):
        return self.racine.__repr__()


class GradientBoosting:
    def __init__(self, n_estimators=10, learning_rate=0.1, profondeur_max=3, seuil_variance=1.5, verbose=False):
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.profondeur_max = profondeur_max
        self.seuil_variance = seuil_variance
        self.verbose = verbose
        self.arbres = []  # Liste des arbres entraînés
        self.erreurs_entrainement = []  # Suivi des erreurs d'entraînement
        self.erreurs_validation = []  # Suivi des erreurs de validation

    def fit(self, data, cible, data_val=None, cible_val=None):
        """
        Entraîne les arbres séquentiellement.
        :param data: Données d'entraînement (features).
        :param cible: Valeurs cibles (target).
        :param data_val: Données de validation (features).
        :param cible_val: Valeurs cibles de validation (target).
        """
        residus = cible.copy()  # Initialiser les résidus avec la cible
        for i in range(self.n_estimators):
            if self.verbose:
                print(f"\nEntraînement de l'arbre {i+1}/{self.n_estimators}")

            # Entraîner un arbre sur les résidus
            arbre = Arbre(data, residus, profondeur_max=self.profondeur_max, verbose=self.verbose, seuil_variance=self.seuil_variance)
            self.arbres.append(arbre)

            # Prédire avec l'arbre actuel
            predictions_train = self.predict(data, jusqu_a=i+1)

            # Calcul de l'erreur d'entraînement
            erreur_entrainement = mean_squared_error(cible, predictions_train)
            self.erreurs_entrainement.append(erreur_entrainement)

            # Si des données de validation sont fournies, calculer l'erreur de validation
            if data_val is not None and cible_val is not None:
                predictions_val = self.predict(data_val, jusqu_a=i+1)
                erreur_validation = mean_squared_error(cible_val, predictions_val)
                self.erreurs_validation.append(erreur_validation)

            # Mettre à jour les résidus
            predictions = arbre.simulation(data)
            residus -= self.learning_rate * np.array(predictions)

            if self.verbose:
                print(f"Erreur d'entraînement après l'arbre {i+1} : {erreur_entrainement:.4f}")
                if data_val is not None:
                    print(f"Erreur de validation après l'arbre {i+1} : {erreur_validation:.4f}")

    def predict(self, data, jusqu_a=None):
        """
        Effectue des prédictions avec tous les arbres ou jusqu'à un certain arbre.
        :param data: Données d'entrée (features).
        :param jusqu_a: Nombre d'arbres à utiliser pour la prédiction (None pour tous les arbres).
        :return: Prédictions globales.
        """
        if jusqu_a is None:
            jusqu_a = len(self.arbres)

        predictions = np.zeros(len(data))  # Initialiser avec zéro
        for i in range(jusqu_a):
            predictions += self.learning_rate * np.array(self.arbres[i].simulation(data))
        return predictions

    def tracer_erreurs(self):
        """
        Trace l'évolution des erreurs d'entraînement et de validation.
        """
        plt.figure(figsize=(10, 6))
        plt.plot(self.erreurs_entrainement, label='Erreur d\'entraînement', marker='o')
        if self.erreurs_validation:
            plt.plot(self.erreurs_validation, label='Erreur de validation', marker='o')
        plt.title('Évolution des erreurs au fil des arbres')
        plt.xlabel('Nombre d\'arbres')
        plt.ylabel('Erreur quadratique moyenne')
        plt.legend()
        plt.grid(True)
        plt.show()



if __name__ == '__main__':
    # Paramètres
    tps_depart = time()
    r = 500  # Nombre d'échantillons (lignes dans X)
    p = 5  # Nombre de caractéristiques (colonnes dans X)
    s = 2  # Nombre de caractéristiques non-nulles dans chaque theta
    sigma = 0.5  # Écart-type du bruit gaussien

    # Générer la matrice X avec des valeurs de Rademacher (1 ou -1 avec probabilité 1/2)
    X = np.random.choice([-1, 1], size=(r, p))

    # Générer theta et Y
    theta_list = []
    Y = np.zeros(r)

    for j in range(r):
        # Générer theta^j avec exactement s valeurs non-nulles choisies aléatoirement parmi p
        theta = np.zeros(p)
        indices = np.random.choice(p, s, replace=False)  # Choisir s indices aléatoires
        theta[indices] = 1
        theta_list.append(theta)
        # Calculer Y^j = <X^j, theta^j> + epsilon
        epsilon = np.random.normal(0, sigma)
        Y[j] = X[j] @ theta + epsilon

    # Conversion en DataFrame pour utiliser avec l'arbre
    data = pd.DataFrame(X, columns=[f'X{i}' for i in range(p)])
    cible = pd.Series(Y, name="target")

    print("Données d'entrée (X):")
    print(data.head())
    print("\nValeurs cibles (Y):")
    print(cible.head())

    # couper les données pour avoir test et train
    X_train, X_test, y_train, y_test = train_test_split(data, cible, test_size=0.3, random_state=42)

    profondeur_max = 7
    # Instancier et entraîner l'arbre
    arbre = Arbre(X_train, y_train, profondeur_max=profondeur_max)
    # Visualisation de l'arbre
    arbre.visualiser_arbre()
    
    # Validation croisée
    n_splits = 5  # Par exemple, 5 plis
    # Dans la validation croisée, essaie différentes profondeurs et seuils
    profondeurs_possibles = [5, 6, 7]

    for profondeur in profondeurs_possibles:
        print(f"\nProfondeur max : {profondeur}")
        validation_croisee(data, cible, profondeur_max=profondeur, n_splits=5)
    tps_fin = time()
    print("duree programme : " + str(tps_fin - tps_depart))
    
    # Prédiction sur la nouvelle donnée
    predictions = arbre.simulation(X_test)
    # graphique des prédictions et des valeurs réelles
    X = np.arange(len(predictions))
    plt.scatter(X,y_test, label='Valeurs réelles')
    plt.scatter(X,predictions,label='Prédictions')
    plt.legend()
    plt.show()

    # valeur de l'erreur
    print("Erreur quadratique moyenne : ",mean_squared_error(y_test, predictions))

    # Entraîner le modèle de Gradient Boosting
    data_train, data_val, cible_train, cible_val = train_test_split(data, cible, test_size=0.2, random_state=42)
    gb = GradientBoosting(n_estimators=50, learning_rate=0.1, profondeur_max=5, verbose=False)
    gb.fit(data_train, cible_train, data_val=data_val, cible_val=cible_val)

    # Tracer l'évolution des erreurs
    gb.tracer_erreurs()

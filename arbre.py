import pandas as pd
import numpy as np
from graphviz import Digraph


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
        """
        Vérifie si le nœud est une feuille ou non
        :return: bool
        """
        return self.gauche is None and self.droite is None

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
    def __init__(self, data, cible, profondeur_max=3, verbose=False):
        self.data = data
        self.cible = cible
        self.profondeur_max = profondeur_max
        self.verbose = verbose
        self.racine = self.fit(data, cible, profondeur=0)  # Premier appel de fit avec profondeur=0
        predictions_test = self.simulation(self.data)
        self.score_test(predictions_test)


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
            val = data[col].sort_values()  # Trier les valeurs de la colonne
            alphas = [(val.iloc[i] + val.iloc[i+1]) / 2 for i in range(len(val)-1)]  # Alphas

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

    def fit(self, data, cible, seuil_variance=0.1, profondeur=0):
        """
        Création de l'arbre complet avec les meilleures coupures
        :param data: valeur de X
        :param cible: valeur de Y
        :param seuil_variance: seuille de variance
        :param profondeur: profondeur actuel
        :return: arbre
        """
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
            label = f"Moyenne : {noeud.moyenne:.2f}\nProfondeur : {profondeur}"
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

if __name__ == '__main__':
    # Paramètres
    r = 100  # Nombre d'échantillons (lignes dans X)
    p = 10  # Nombre de caractéristiques (colonnes dans X)
    s = 5  # Nombre de caractéristiques non-nulles dans chaque theta
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

    # Instancier et entraîner l'arbre
    arbre = Arbre(data, cible, profondeur_max=6)
    # Visualisation de l'arbre
    arbre.visualiser_arbre()


import numpy as np
import pandas as pd
import os
from sklearn.neighbors import NearestNeighbors
from csv import writer

from scipy.sparse.csgraph import shortest_path

def lire_fichiers_specifies(fichier_txt):
    """
    Lit un fichier .txt contenant la liste des fichiers à traiter, un par ligne.

    Arguments:
        fichier_txt (str): Chemin vers le fichier .txt.

    Retourne:
        list: Liste des fichiers spécifiés.
    """
    try:
        with open(fichier_txt, 'r') as f:
            fichiers = [ligne.strip() for ligne in f if ligne.strip()]
        return fichiers
    except FileNotFoundError:
        print(f"Le fichier {fichier_txt} n'existe pas.")
        return []


def normaliser_preserver_ratio(data):
    """
    Normalise les données entre 0 et 1 tout en préservant le ratio initial des dimensions.

    Args:
        data (numpy array ou pandas dataframe): Données à normaliser.

    Returns:
        numpy array: Données normalisées entre 0 et 1 avec respect du ratio initial.
    """
    if not isinstance(data, np.ndarray):
        data = data.to_numpy()

    # Calculer les min et max pour chaque dimension
    min_vals = data.min(axis=0)
    max_vals = data.max(axis=0)
    ranges = max_vals - min_vals

    # Identifier la plus grande étendue (dimension dominante)
    max_range = ranges.max()

    # Normaliser toutes les dimensions pour que la dimension dominante soit dans [0, 1]
    normalized_data = (data - min_vals) / max_range

    return normalized_data


def distance_geodesique_optimisee(HD, k):
    """
    Utilise la fonction k_plus_proches_voisins pour calculer la matrice des distances géodésiques.

    Args:
        HD (numpy array ou pandas dataframe): Données en haute dimension.
        k (int): Nombre de voisins pour construire le graphe.

    Returns:
        numpy array: Matrice des distances géodésiques.
    """
    if not isinstance(HD, np.ndarray):
        HD = HD.to_numpy()

    # Utilisation de la fonction existante pour obtenir les voisins
    indices_voisins = k_plus_proches_voisins(HD, k)
    n_points = HD.shape[0]

    # Construire une matrice d'adjacence initialisée à l'infini
    matrice_adjacence = np.full((n_points, n_points), np.inf)
    for pointA in range(n_points):
        for pointB in indices_voisins[pointA]:
            distance = np.linalg.norm(HD[pointA] - HD[pointB])
            matrice_adjacence[pointA, pointB] = distance
            matrice_adjacence[pointB, pointA] = distance  # Graphe non dirigé

    # Calculer les distances géodésiques avec l'algorithme de plus court chemin
    distances_geodesiques = shortest_path(matrice_adjacence, directed=False)
    return distances_geodesiques


def distorsion_geodesique(HD, BD, k):
    """
    Calcule la distorsion géodésique entre les espaces haute et basse dimensions.

    Args:
        HD (numpy array ou pandas dataframe): Données en haute dimension.
        BD (numpy array): Données en basse dimension.
        k (int): Nombre de voisins pour construire le graphe dans X_HD.

    Returns:
        float: La distorsion géodésique moyenne.
    """
    if not isinstance(HD, np.ndarray):
        HD = HD.to_numpy()

    # Calcul des distances géodésiques dans l'espace HD
    distances_HD = distance_geodesique_optimisee(HD, k)

    # Calcul des distances euclidiennes dans l'espace BD
    distances_BD = np.linalg.norm(BD[:, np.newaxis] - BD[np.newaxis, :], axis=-1)

    # Comparaison des distances
    n_points = distances_HD.shape[0]
    distorsion_totale = 0
    n_comparaisons = 0

    for i in range(n_points):
        for j in range(i + 1, n_points):
            if distances_HD[i, j] < np.inf:  # Ignorer les paires non connectées
                distorsion = abs(distances_BD[i, j] - distances_HD[i, j]) / distances_HD[i, j]
                distorsion_totale += distorsion
                n_comparaisons += 1

    return distorsion_totale / n_comparaisons if n_comparaisons > 0 else np.inf


def matrice_distances(X, normalize=True):
    """
    Calcul matrice des distances entre points d'un jeu de données en haute dimension.
    distances[pointA][pointB] retourne la distance entre les points A et B.

    Arguments:
        X (pandas dataframe ou numpy array): Jeu de données en haute dimension.
        normalize (bool): Si True, applique une normalisation qui préserve le ratio.

    Returns:
        numpy array: Matrice des distances.
    """
    if not isinstance(X, np.ndarray):
        X = X.to_numpy()

    if normalize:
        X = normaliser_preserver_ratio(X)

    n_points = X.shape[0]
    distances = np.full((n_points, n_points), np.inf)
    for pointA in range(n_points):
        x_i = X[pointA]
        for pointB in range(pointA + 1, n_points):
            x_j = X[pointB]
            distances[pointA][pointB] = np.linalg.norm(x_i - x_j)
            distances[pointB][pointA] = distances[pointA][pointB]  # Symétrie
    return distances


def matrice_poids(d, alpha):
    """
    Calcul matrice de poids basée sur matrice de distances et constante alpha.
    poids[pointA][pointB] retourne le poids associé à la distance entre les points A et B.

    Arguments:
        d (numpy array): Matrice des distances.
        alpha (float): Constante.
                       - Si alpha >= 1 : Favorise les petites distances.
                       - Si 0 < alpha < 1 : Favorise les grandes distances.
                       - Si alpha = 0 : Tous les poids sont égaux à 1.

    Retourne:
        numpy array: Matrice des poids.
    """
    n_points = d.shape[0]
    poids = -np.ones((n_points, n_points))
    for pointA in range(n_points):
        for pointB in range(pointA + 1, n_points):
            if d[pointA][pointB] != 0:
                poids[pointA][pointB] = 1 / d[pointA][pointB] ** alpha
            else:
                poids[pointA][pointB] = 1
    return poids

def stress_majoration(HD, BD, alpha=2.0):
    """
    Calcul stress majoré normalisé entre données haute dimension HD
    et données réduites en basse dimension BD.

    Arguments:
        HD (pandas dataframe): Données en haute dimension.
        BD (numpy array): Données réduites en basse dimension.
        alpha (float): Constante utilisée pour calculer la matrice de poids (défaut 2.0).

    Retourne:
        float: Stress majoré normalisé.
    """
    d = matrice_distances(HD)
    p = matrice_poids(d, alpha)
    n_points = d.shape[0]
    n_paires = 0
    stress = 0
    for pointA in range(n_points):
        for pointB in range(pointA + 1, n_points):
            n_paires += 1
            stress += p[pointA][pointB] * (np.linalg.norm(BD[pointA] - BD[pointB]) - d[pointA][pointB]) ** 2
    return stress / n_paires

def k_plus_proches_voisins(X, k):
    """
    Pour chaque point du jeu de données, trouve les k plus proches voisins.

    Arguments:
        X (numpy array ou pandas dataframe): Données (haute ou basse dimension).
        k (int): Nombre de voisins à considérer.

    Retourne:
        numpy array: Tableau contenant indices des k plus proches voisins pour chaque point.
    """
    if not isinstance(X, np.ndarray):
        X = X.to_numpy()
    voisins = NearestNeighbors(n_neighbors=k + 1, algorithm='kd_tree').fit(X)
    indices = voisins.kneighbors(X)[1]
    return indices

def indice_jaccard(HD, BD, k=7):
    """
    Calcul indice de Jaccard des k plus proches voisins pour chaque point, et renvoie la moyenne.

    Arguments:
        HD (pandas dataframe): Données en haute dimension.
        BD (numpy array): Données réduites en basse dimension.
        k (int): Nombre de voisins à considérer (défaut 7).

    Retourne:
        float: Moyenne de l'indice de Jaccard (en pourcentage).
    """
    voisins_BD = k_plus_proches_voisins(BD, k)
    voisins_HD = k_plus_proches_voisins(HD, k)
    liste_jaccard = []
    for i in range(voisins_HD.shape[0]):
        intersection = np.intersect1d(voisins_BD[i], voisins_HD[i])
        union = np.union1d(voisins_BD[i], voisins_HD[i])
        indice = len(intersection) / len(union)
        liste_jaccard.append(indice)
    return sum(liste_jaccard) * 100 / len(liste_jaccard)



def get_files(directory):
    if not os.path.exists(directory):
        raise FileNotFoundError(f"Le répertoire spécifié n'existe pas : {directory}")
    # Filter files based on the expected naming pattern
    return [f for f in os.listdir(directory) if f.endswith('.npy') and len(f.split('_')) >= 3]


# Fonction pour extraire les paramètres depuis le nom de fichier
def extraire_parametres(nom_fichier):
    elements = nom_fichier[:-4].split('_')
    params = {'mode': elements[0], 'distribution': elements[2]}

    if params['mode'] == 'reduit':
        if params['distribution'] == 'uniforme':
            params.update({
                'k': int(elements[1]),
                'limite_inf': float(elements[3]),
                'limite_sup': float(elements[4]),
                'nb_echantillons': int(elements[5]),
                'nb_dmensions': int(elements[6]),
                'num_repetition': int(elements[7])
            })
        else:
            params.update({
                'k': int(elements[1]),
                'nb_echantillons': int(elements[3]),
                'ecart_type': float(elements[4]),
                'nb_centre': int(elements[5]),
                'nb_dmensions': int(elements[6]),
                'num_repetition': int(elements[7])
            })
    else:
        if params['distribution'] == 'uniforme':
            params.update({
                'limite_inf': float(elements[1]),
                'limite_sup': float(elements[2]),
                'nb_echantillons': int(elements[3]),
                'nb_dmensions': int(elements[4]),
                'num_repetition': int(elements[5])
            })
        else:
            params.update({
                'nb_echantillons': int(elements[1]),
                'ecart_type': float(elements[2]),
                'nb_centre': int(elements[3]),
                'nb_dmensions': int(elements[4]),
                'num_repetition': int(elements[5])
            })
    return params

# Fonction pour calculer les métriques de réduction
def calculer_metrics_reduction(fichiers_reduction, repertoire_haute_dimension, dossier_sortie,nom_sous_dossier):
    """
    Calcule les métriques de réduction sur les fichiers spécifiés dans la liste.
    
    Arguments:
        fichiers_reduction (list): Liste des noms des fichiers à traiter.
        repertoire_haute_dimension (str): Chemin vers le répertoire des fichiers haute dimension.
        dossier_sortie (str): Chemin vers le dossier où sauvegarder les fichiers CSV.
    """
    if not os.path.exists(dossier_sortie):
        os.makedirs(dossier_sortie)

    if not fichiers_reduction:
        print("Aucun fichier à traiter.")
        return

    # Utilisation du premier fichier pour extraire les paramètres
    params = extraire_parametres(fichiers_reduction[0])
    colonnes = list(params.keys()) + ['stress_majoration', 'indice_jaccard', 'distorsion_geodesique']
    fichier_csv = f"{dossier_sortie}/metrics_{nom_sous_dossier}.csv"
    df_metrics = pd.DataFrame(columns=colonnes)
    df_metrics.to_csv(fichier_csv, index=False)

    for fichier in fichiers_reduction:
        # Vérifier si le fichier existe réellement
        if not os.path.exists(f"{repertoire_reduction}/{fichier}"):
            print(f"Le fichier de réduction {fichier} est introuvable.")
            continue

        params = extraire_parametres(fichier)
        ligne = list(params.values())

        BD = np.load(f"{repertoire_reduction}/{fichier}")
        fichier_HD = "_".join(fichier.split('_')[2:])
        chemin_HD = f"{repertoire_haute_dimension}/{fichier_HD}"

        if not os.path.exists(chemin_HD):
            print(f"Fichier haute dimension introuvable : {chemin_HD}")
            continue

        HD = np.load(chemin_HD)

        # Calcul des métriques
        stress = stress_majoration(HD, BD)
        jaccard = indice_jaccard(HD, BD)
        distorsion = distorsion_geodesique(HD, BD, k=7)  # k peut être ajusté
        ligne.extend([stress, jaccard, distorsion])

        # Sauvegarder les résultats dans le fichier CSV
        with open(fichier_csv, 'a', newline='') as f:
            writer(f).writerow(ligne)


############A EXECUTER 9 FOIS 

# 1
repertoire_reduction = "./donnees_reduites/n_neighbors_5" 
calculer_metrics_reduction(lire_fichiers_specifies("./donnees_reduites/fichier_sortie.txt"), "./resultats_uniforme","./metriques","uni_5")
# 2
repertoire_reduction = "./donnees_reduites_clusters/n_neighbors_5" 
calculer_metrics_reduction(lire_fichiers_specifies("./donnees_reduites_clusters/fichier_sortie.txt"), "./resultats_Clusters","./metriques","clust_5")
# 3
repertoire_reduction = "./donnees_reduites/n_neighbors_10" 
calculer_metrics_reduction(lire_fichiers_specifies("./donnees_reduites/fichier_sortie_10.txt"), "./resultats_uniforme","./metriques","uni_10")
# 4
repertoire_reduction = "./donnees_reduites_clusters/n_neighbors_10" 
calculer_metrics_reduction(lire_fichiers_specifies("./donnees_reduites_clusters/fichier_sortie_10.txt"), "./resultats_Clusters","./metriques","clust_10")
# 5
repertoire_reduction = "./donnees_reduites_gaussienne/n_neighbors_5" 
calculer_metrics_reduction(lire_fichiers_specifies("./donnees_reduites_gaussienne/fichier_sortie.txt"), "./resultats_Gaussienne","./metriques","gauss_5")

# 6
repertoire_reduction = "./donnees_reduites/n_neighbors_70" 
calculer_metrics_reduction(lire_fichiers_specifies("./donnees_reduites/fichier_sortie_70.txt"), "./resultats_uniforme","./metriques","uni_70")
# 7
# repertoire_reduction = "./donnees_reduites_gaussienne/n_neighbors_10" 
# calculer_metrics_reduction(lire_fichiers_specifies("./donnees_reduites_gaussienne/fichier_sortie.txt"), "./resultats_Gaussienne","./metriques","gauss_10")
# 8
# repertoire_reduction = "./donnees_reduites_gaussienne/n_neighbors_70" 
# calculer_metrics_reduction(lire_fichiers_specifies("./donnees_reduites_gaussienne/fichier_sortie.txt"), "./resultats_Gaussienne","./metriques","gauss_70")
# 9
repertoire_reduction = "./donnees_reduites_clusters2/n_neighbors_70" 
calculer_metrics_reduction(lire_fichiers_specifies("./donnees_reduites_clusters/fichier_sortie_70.txt"), "./resultats_Clusters","./metriques","clust_70")
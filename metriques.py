import numpy as np
import pandas as pd
import os
from sklearn.neighbors import NearestNeighbors
from csv import writer
from scipy.sparse.csgraph import shortest_path

def normaliser_preserver_ratio(data):
    """
    Normalise les données d'un jeu de données tout en préservant le ratio initial des dimensions.

    Args:
    data (pandas.DataFrame ou numpy.ndarray): Jeu de données à normaliser.

    Returns:
    normalized_data (numpy.ndarray) : Jeu de données normalisé où les dimensions sont comprises entre 0 et 1.
    """
    if not isinstance(data, np.ndarray):
        data = data.to_numpy()

    min_vals = data.min(axis=0)
    max_vals = data.max(axis=0)
    ranges = max_vals - min_vals
    max_range = ranges.max()
    normalized_data = (data - min_vals) / max_range

    return normalized_data

def matrice_distances(X, normalize=True):
    """
    Calcule la matrice des distances entre les points d'un jeu de données en haute dimension.

    Args:
    X (pandas.DataFrame ou numpy.ndarray): Jeu de données à partir duquel les distances seront calculées.
    normalize (bool): Indique si les données doivent être normalisées avant le calcul des distances (Défaut : True).

    Retourne:
    distances (numpy.ndarray) : Matrice de distances où chaque élément (a, b) représente la distance entre les points a et b.
    """
    if not isinstance(X, np.ndarray):
        X = X.to_numpy()

    if normalize:
        X = normaliser_preserver_ratio(X)

    n_points = X.shape[0]
    distances = np.full((n_points, n_points), np.inf)
    for pointA in range(n_points):
        x_a = X[pointA]
        for pointB in range(pointA + 1, n_points):
            x_b = X[pointB]
            distances[pointA][pointB] = np.linalg.norm(x_a - x_b)
            distances[pointB][pointA] = distances[pointA][pointB]
    return distances


def matrice_poids(d, alpha):
    """
    Calcule la matrice de poids basée sur la matrice de distances et la constante alpha.

    Args:
    d (numpy.ndarray): Matrice des distances entre les points. 
    alpha (float): Constante utilisée pour ajuster les poids en fonction des distances (Défaut : 2.0)
    
    Retourne:
    poids (numpy.ndarray) : Matrice des poids où chaque élément (a, b) représente le poids associé à la distance entre les points a et b.
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


def stress(HD, BD, alpha=2.0):
    """
    Calcule le stress normalisé entre les données en haute dimension et les données réduites en basse dimension.

    Args:
    HD (numpy.ndarray): Jeu de données en haute dimension, où chaque ligne représente un point.
    BD (numpy.ndarray): Jeu de données en basse dimension, où chaque ligne représente un point réduit.
    alpha (float): Paramètre de pondération pour le calcul des poids entre points. (Défaut : 2.0).

    Retourne:
    float: Le stress majoré normalisé, représentant l'écart total entre les distances de points dans les deux espaces.
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
    Trouve les k plus proches voisins pour chaque point d'un jeu de données.

    Args:
    X (numpy.ndarray ou pandas.DataFrame): Jeu de données d'entrée, où chaque ligne représente un point dans l'espace multidimensionnel.
    k (int): Le nombre de voisins à identifier pour chaque point.

    Retourne:
    indices (numpy.ndarray) : Un tableau 2D de forme (n_points, k), où n_points est le nombre de points dans le jeu de données,
    et chaque ligne contient les indices des k plus proches voisins du point correspondant.
    """
    if not isinstance(X, np.ndarray):
        X = X.to_numpy()
    voisins = NearestNeighbors(n_neighbors=k + 1, algorithm='kd_tree').fit(X)
    indices = voisins.kneighbors(X)[1]
    return indices


def indice_jaccard(HD, BD, k=7):
    """
    Calcule l'indice de Jaccard des k plus proches voisins pour chaque point.

    Args:
    HD (numpy.ndarray ou pandas.DataFrame): Le jeu de données d'origine en haute dimension.
    BD (numpy.ndarray ou pandas.DataFrame): Le jeu de données réduit en basse dimension.
    k (int): Le nombre de voisins à considérer pour le calcul de l'indice de Jaccard (Défaut : 7).

    Retourne:
    float: La moyenne des indices de Jaccard calculés pour chaque point.
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

def distance_geodesique_optimisee(HD, k):
    """
    Calcule la matrice des distances géodésiques entre les points d'un jeu de données en haute dimension en utilisant les k plus proches voisins.

    Args:
    HD (numpy.ndarray ou pandas.DataFrame): Le jeu de données d'origine en haute dimension.
    k (int): Le nombre de voisins à considérer pour calculer les distances géodésiques.

    Retourne:
    distances_geodesiques (numpy.ndarray) : Une matrice carrée des distances géodésiques entre tous les points du jeu de données.
    """
    if not isinstance(HD, np.ndarray):
        HD = HD.to_numpy()

    indices_voisins = k_plus_proches_voisins(HD, k)
    n_points = HD.shape[0]

    matrice_adjacence = np.full((n_points, n_points), np.inf)
    for pointA in range(n_points):
        for pointB in indices_voisins[pointA]:
            distance = np.linalg.norm(HD[pointA] - HD[pointB])
            matrice_adjacence[pointA, pointB] = distance
            matrice_adjacence[pointB, pointA] = distance

    distances_geodesiques = shortest_path(matrice_adjacence, directed=False)
    return distances_geodesiques


def distorsion_geodesique(HD, BD, k):
    """
    Calcule la distorsion géodésique entre les espaces en haute et basse dimension.

    Args:
    HD (numpy.ndarray ou pandas.DataFrame): Le jeu de données en haute dimension.
    BD (numpy.ndarray): Le jeu de données réduit en basse dimension.
    k (int): Le nombre de voisins à considérer pour le calcul des distances géodésiques.

    Retourne:
    float: La distorsion géodésique moyenne entre les deux espaces. Si aucune comparaison n'a pu être faite, renvoie np.inf.
    """
    if not isinstance(HD, np.ndarray):
        HD = HD.to_numpy()

    distances_HD = distance_geodesique_optimisee(HD, k)

    distances_BD = np.linalg.norm(BD[:, np.newaxis] - BD[np.newaxis, :], axis=-1)

    n_points = distances_HD.shape[0]
    distorsion_totale = 0
    n_comparaisons = 0

    for i in range(n_points):
        for j in range(i + 1, n_points):
            if distances_HD[i, j] < np.inf: 
                distorsion = abs(distances_BD[i, j] - distances_HD[i, j]) / distances_HD[i, j]
                distorsion_totale += distorsion
                n_comparaisons += 1

    return distorsion_totale / n_comparaisons if n_comparaisons > 0 else np.inf


def calculer_metrics_reduction(repertoire_haute_dimension, repertoire_reduction, dossier_sortie, nom_sous_dossier, k=7):
    """
    Calcule les métriques de réduction sur les fichiers haute et basse dimension.
    
    Arguments:
        repertoire_haute_dimension (str): Chemin vers le répertoire des fichiers haute dimension.
        repertoire_reduction (str): Chemin vers le répertoire des fichiers de réduction.
        dossier_sortie (str): Chemin vers le dossier où sauvegarder les fichiers CSV.
        nom_sous_dossier (str): Nom du sous-dossier à ajouter dans le fichier CSV.
    """
    if not os.path.exists(dossier_sortie):
        os.makedirs(dossier_sortie)

    # Parcours des fichiers haute et basse dimension
    fichiers_reduction = [f for f in os.listdir(repertoire_reduction) if f.endswith('.npy')]
    colonnes = ['stress', 'indice_jaccard', 'distorsion_geodesique']
    fichier_csv = f"{dossier_sortie}/metrics_{nom_sous_dossier}.csv"
    df_metrics = pd.DataFrame(columns=colonnes)
    df_metrics.to_csv(fichier_csv, index=False)

    for fichier in fichiers_reduction:
        # Chargement du fichier de réduction
        BD = np.load(f"{repertoire_reduction}/{fichier}")
        fichier_HD = "_".join(fichier.split('_')[2:])
        chemin_HD = f"{repertoire_haute_dimension}/{fichier_HD}"

        if not os.path.exists(chemin_HD):
            print(f"Fichier haute dimension introuvable : {chemin_HD}")
            continue

        HD = np.load(chemin_HD)

        stress = stress(HD, BD)
        jaccard = indice_jaccard(HD, BD)
        distorsion = distorsion_geodesique(HD, BD, k=k)

        with open(fichier_csv, 'a', newline='') as f:
            writer(f).writerow([stress, jaccard, distorsion])

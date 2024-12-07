import numpy as np
import pandas as pd
import os
from sklearn.neighbors import NearestNeighbors
from csv import writer
from scipy.sparse.csgraph import shortest_path

def normaliser_preserver_ratio(donnees):
    """
    Normalise les données d'un jeu de données tout en préservant le ratio initial des dimensions.

    Args:
    donnees (pandas.DataFrame ou numpy.ndarray): Jeu de données à normaliser.

    Returns:
    donnees_normalisees (numpy.ndarray) : Jeu de données normalisé où les dimensions sont comprises entre 0 et 1.
    """
    if not isinstance(donnees, np.ndarray):
        donnees = donnees.to_numpy()

    min_vals = donnees.min(axis=0)
    max_vals = donnees.max(axis=0)
    ranges = max_vals - min_vals
    max_range = ranges.max()
    donnees_normalisees = (donnees - min_vals) / max_range

    return donnees_normalisees

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

    if normaliser:
        X = normaliser_preserver_ratio(X)

    n_points = X.shape[0]
    distances = np.full((n_points, n_points), np.inf)
    for point_a in range(n_points):
        x_a = X[point_a]
        for point_b in range(point_a + 1, n_points):
            x_b = X[point_b]
            distances[point_a][point_b] = np.linalg.norm(x_a - x_b)
            distances[point_b][point_a] = distances[point_a][point_b]
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
    for point_a in range(n_points):
        for point_b in range(point_a + 1, n_points):
            if d[point_a][point_b] != 0:
                poids[point_a][point_b] = 1 / d[point_a][point_b] ** alpha
            else:
                poids[point_a][point_b] = 1
    return poids


def stress(hd, bd, alpha=2.0):
    """
    Calcule le stress normalisé entre les données en haute dimension et les données réduites en basse dimension.

    Args:
    hd (numpy.ndarray): Jeu de données en haute dimension, où chaque ligne représente un point.
    bd (numpy.ndarray): Jeu de données en basse dimension, où chaque ligne représente un point réduit.
    alpha (float): Paramètre de pondération pour le calcul des poids entre points. (Défaut : 2.0).

    Retourne:
    float: Le stress majoré normalisé, représentant l'écart total entre les distances de points dans les deux espaces.
    """
    d = matrice_distances(hd)
    p = matrice_poids(d, alpha)
    n_points = d.shape[0]
    n_paires = 0
    stress = 0
    for point_a in range(n_points):
        for point_b in range(point_a + 1, n_points):
            n_paires += 1
            stress += p[point_a][point_b] * (np.linalg.norm(bd[point_a] - bd[point_b]) - d[point_a][point_b]) ** 2
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


def indice_jaccard(hd, bd, k=7):
    """
    Calcule l'indice de Jaccard des k plus proches voisins pour chaque point.

    Args:
    hd (numpy.ndarray ou pandas.DataFrame): Le jeu de données d'origine en haute dimension.
    bd (numpy.ndarray ou pandas.DataFrame): Le jeu de données réduit en basse dimension.
    k (int): Le nombre de voisins à considérer pour le calcul de l'indice de Jaccard (Défaut : 7).

    Retourne:
    float: La moyenne des indices de Jaccard calculés pour chaque point.
    """
    voisins_bd = k_plus_proches_voisins(bd, k)
    voisins_hd = k_plus_proches_voisins(hd, k)
    liste_jaccard = []
    for i in range(voisins_hd.shape[0]):
        intersection = np.intersect1d(voisins_bd[i], voisins_hd[i])
        union = np.union1d(voisins_bd[i], voisins_hd[i])
        indice = len(intersection) / len(union)
        liste_jaccard.append(indice)
    return sum(liste_jaccard) * 100 / len(liste_jaccard)

def distance_geodesique_optimisee(hd, k):
    """
    Calcule la matrice des distances géodésiques entre les points d'un jeu de données en haute dimension en utilisant les k plus proches voisins.

    Args:
    hd (numpy.ndarray ou pandas.DataFrame): Le jeu de données d'origine en haute dimension.
    k (int): Le nombre de voisins à considérer pour calculer les distances géodésiques.

    Retourne:
    distances_geodesiques (numpy.ndarray) : Une matrice carrée des distances géodésiques entre tous les points du jeu de données.
    """
    if not isinstance(hd, np.ndarray):
        hd = hd.to_numpy()

    indices_voisins = k_plus_proches_voisins(hd, k)
    n_points = hd.shape[0]

    matrice_adjacence = np.full((n_points, n_points), np.inf)
    for point_a in range(n_points):
        for point_b in indices_voisins[point_a]:
            distance = np.linalg.norm(hd[point_a] - hd[point_b])
            matrice_adjacence[point_a, point_b] = distance
            matrice_adjacence[point_b, point_a] = distance

    distances_geodesiques = shortest_path(matrice_adjacence, directed=False)
    return distances_geodesiques


def distorsion_geodesique(hd, bd, k):
    """
    Calcule la distorsion géodésique entre les espaces en haute et basse dimension.

    Args:
    hd (numpy.ndarray ou pandas.DataFrame): Le jeu de données en haute dimension.
    bd (numpy.ndarray): Le jeu de données réduit en basse dimension.
    k (int): Le nombre de voisins à considérer pour le calcul des distances géodésiques.

    Retourne:
    float: La distorsion géodésique moyenne entre les deux espaces. Si aucune comparaison n'a pu être faite, renvoie np.inf.
    """
    if not isinstance(hd, np.ndarray):
        hd = hd.to_numpy()

    distances_hd = distance_geodesique_optimisee(hd, k)

    distances_bd = np.linalg.norm(bd[:, np.newaxis] - bd[np.newaxis, :], axis=-1)

    n_points = distances_hd.shape[0]
    distorsion_totale = 0
    n_comparaisons = 0

    for i in range(n_points):
        for j in range(i + 1, n_points):
            if distances_hd[i, j] < np.inf: 
                distorsion = abs(distances_bd[i, j] - distances_hd[i, j]) / distances_hd[i, j]
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

    fichiers_reduction = [f for f in os.listdir(repertoire_reduction) if f.endswith('.npy')]
    colonnes = ['stress', 'indice_jaccard', 'distorsion_geodesique']
    fichier_csv = f"{dossier_sortie}/metrics_{nom_sous_dossier}.csv"
    df_metriques = pd.DataFrame(columns=colonnes)
    df_metriques.to_csv(fichier_csv, index=False)

    for fichier in fichiers_reduction:
        bd = np.load(f"{repertoire_reduction}/{fichier}")
        fichier_hd = "_".join(fichier.split('_')[2:])
        chemin_hd = f"{repertoire_haute_dimension}/{fichier_hd}"

        if not os.path.exists(chemin_hd):
            print(f"Fichier haute dimension introuvable : {chemin_hd}")
            continue

        hd = np.load(chemin_hd)

        stress = stress(hd, bd)
        jaccard = indice_jaccard(hd, bd)
        distorsion = distorsion_geodesique(hd, bd, k=k)

        with open(fichier_csv, 'a', newline='') as f:
            writer(f).writerow([stress, jaccard, distorsion])

import numpy as np
import pandas as pd
import os
from sklearn.neighbors import NearestNeighbors
from csv import writer
from scipy.sparse.csgraph import shortest_path


def normaliser_preserver_ratio(data):
    """
    Normalise les données entre 0 et 1 tout en préservant le ratio initial des dimensions.
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
    Calcule la matrice des distances géodésiques.
    """
    if not isinstance(HD, np.ndarray):
        HD = HD.to_numpy()

    # Utilisation de la fonction k_plus_proches_voisins pour obtenir les voisins
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
    """
    if not isinstance(X, np.ndarray):
        X = X.to_numpy()
    voisins = NearestNeighbors(n_neighbors=k + 1, algorithm='kd_tree').fit(X)
    indices = voisins.kneighbors(X)[1]
    return indices


def indice_jaccard(HD, BD, k=7):
    """
    Calcul indice de Jaccard des k plus proches voisins pour chaque point, et renvoie la moyenne.
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
    colonnes = ['stress_majoration', 'indice_jaccard', 'distorsion_geodesique']
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

        # Chargement des données haute dimension
        HD = np.load(chemin_HD)

        # Calcul des métriques
        stress = stress_majoration(HD, BD)
        jaccard = indice_jaccard(HD, BD)
        distorsion = distorsion_geodesique(HD, BD, k=k)

        # Sauvegarder les résultats dans le fichier CSV
        with open(fichier_csv, 'a', newline='') as f:
            writer(f).writerow([stress, jaccard, distorsion])
import numpy as np
import pandas as pd
import os
import itertools
from sklearn.datasets import make_blobs
from sklearn.preprocessing import MinMaxScaler

def generation_donnees(nb_echantillons, ecart_type, nb_centres, nb_caracteristiques, graine_aleatoire):
    """Génère des données synthétiques normalisées avec une distribution de clusters
    ou une distribution gaussienne (en fixant le nombre de centres à 1).

    Args:
        nb_echantillons (int): nombre d'échantillons (répartis également entre les clusters)
        ecart_type (float): écart type des clusters
        nb_centres (int): nombre de clusters
        nb_caracteristiques (int): nombre de caractéristiques pour chaque échantillon
        graine_aleatoire (int): graine pour la génération aléatoire

    Returns:
        numpy array: tableau Xnorm des échantillons générés et normalisés
        numpy array: tableau y des étiquettes de cluster pour chaque échantillon
    """
    X, y = make_blobs(n_samples=nb_echantillons, cluster_std=ecart_type, centers=nb_centres, 
                      n_features=nb_caracteristiques, random_state=graine_aleatoire, return_centers=False)
    scaler = MinMaxScaler()
    Xnorm = scaler.fit_transform(X) 
    return Xnorm, y

def noms_colonnes(nb_caracteristiques):
    """Définit les noms des colonnes pour le fichier CSV des données générées.
    Chaque colonne représente une caractéristique, et une colonne finale spécifie le cluster de chaque échantillon.

    Args:
        nb_caracteristiques (int): nombre de caractéristiques pour chaque échantillon

    Returns:
        list: liste des noms de colonnes à inclure dans le fichier CSV généré
    """
    colonnes_caracteristiques = [f'caracteristique_{i}' for i in range(1, nb_caracteristiques + 1)]
    return colonnes_caracteristiques + ['cluster']

def nom_fichier(dossiers_sortie, nb_echantillons, ecart_type, nb_centres, nb_caracteristiques, df, repetition):
    """Crée un fichier CSV contenant les données générées.

    Args:
        dossiers_sortie (list): chemins vers les répertoires pour sauvegarder les données
        nb_echantillons (int): nombre d'échantillons
        ecart_type (float): écart type des clusters
        nb_centres (int): nombre de clusters
        nb_caracteristiques (int): nombre de caractéristiques
        df (pandas dataframe): données à sauvegarder
        repetition (int): numéro de répétition
    """
    if nb_centres == 1: 
        chemin_fichier = os.path.join(dossiers_sortie[0], f"gaussienne_{nb_echantillons}_{ecart_type}_{nb_centres}_{nb_caracteristiques}_{repetition}.csv")
    else:
        chemin_fichier = os.path.join(dossiers_sortie[1], f"cluster_{nb_echantillons}_{ecart_type}_{nb_centres}_{nb_caracteristiques}_{repetition}.csv")
    df.to_csv(chemin_fichier, index=False)

def produit_cartesien(nb_echantillons, ecarts_types, nb_centres, nb_caracteristiques):
    """Retourne le produit cartésien des paramètres.

    Args:
        nb_echantillons (list): liste des tailles d'échantillons
        ecarts_types (list): liste des écarts types
        nb_centres (list): liste des nombres de clusters
        nb_caracteristiques (list): liste des nombres de caractéristiques

    Returns:
        set: ensemble des combinaisons de paramètres
    """
    return set(itertools.product(nb_echantillons, ecarts_types, nb_centres, nb_caracteristiques))

def test_parametres(nb_echantillons, ecarts_types, nb_centres, nb_caracteristiques, graine_aleatoire, dossiers_sortie, repetitions=5):
    """Génère les données pour toutes les combinaisons possibles de paramètres
    et les sauvegarde dans des fichiers CSV.

    Args:
        nb_echantillons (list): liste des tailles d'échantillons
        ecarts_types (list): liste des écarts types
        nb_centres (list): liste des nombres de clusters
        nb_caracteristiques (list): liste des nombres de caractéristiques
        graine_aleatoire (int): graine pour la génération aléatoire
        dossiers_sortie (list): chemins des répertoires où sauvegarder les données
        repetitions (int): nombre de répétitions pour chaque combinaison
    """
    combinaisons = produit_cartesien(nb_echantillons, ecarts_types, nb_centres, nb_caracteristiques)
    for rep in range(repetitions):
        for parametres in combinaisons:
            echantillons, ecart, centres, caracteristiques = parametres
            X, y = generation_donnees(echantillons, ecart, centres, caracteristiques, graine_aleatoire)
            donnees = np.concatenate([X, y.reshape(-1, 1).astype(int)], axis=1)
            noms = noms_colonnes(caracteristiques)
            df = pd.DataFrame(donnees, columns=noms)
            nom_fichier(dossiers_sortie, echantillons, ecart, centres, caracteristiques, df, rep)

def generateur_gaussien_cluster(nb_echantillons=[i for i in range(100, 4001, 100)], 
                                ecarts_types=[0.0, 0.5, 1.0], 
                                nb_centres=[1, 4, 10], 
                                nb_caracteristiques=[2**j for j in range(2, 7)], 
                                graine_aleatoire=0):
    """Fonction principale pour générer les données et les sauvegarder.

    Args:
        nb_echantillons (list): tailles des échantillons
        ecarts_types (list): écarts types des clusters
        nb_centres (list): nombres de clusters
        nb_caracteristiques (list): nombres de caractéristiques
        graine_aleatoire (int): graine pour la génération aléatoire
    """
    dossier_gaussien = os.path.join(os.environ.get("CHEMIN_GENERATION_DONNEES", ""), "resultats_Gaussienne")
    if not os.path.exists(dossier_gaussien):
        os.makedirs(dossier_gaussien)

    dossier_clusters = os.path.join(os.environ.get("CHEMIN_GENERATION_DONNEES", ""), "resultats_Clusters")
    if not os.path.exists(dossier_clusters):
        os.makedirs(dossier_clusters)

    dossiers_sortie = [dossier_gaussien, dossier_clusters]
    test_parametres(nb_echantillons, ecarts_types, nb_centres, nb_caracteristiques, graine_aleatoire, dossiers_sortie)




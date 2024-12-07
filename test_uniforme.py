import numpy as np
import pandas as pd
import os
import itertools

def generation_donnees(borne_inf, borne_sup, taille):
    """Génère des données synthétiques normalisées avec une distribution uniforme.

    Arguments:
        borne_inf (float): limite inférieure de l'intervalle des valeurs générées
        borne_sup (float): limite supérieure de l'intervalle des valeurs générées
        taille (tuple de longueur 2): tuple (nombre d'échantillons, nombre de caractéristiques) définissant la taille des données

    Renvoi:
        numpy array: tableau des échantillons générés et normalisés
    """
    tableau_uniforme = np.random.uniform(low=borne_inf, high=borne_sup, size=taille)
    return tableau_uniforme

def noms_colonnes(nb_caracteristiques):
    """Définit les noms des colonnes pour le fichier CSV des données générées.
    
    Arguments:
        nb_caracteristiques (int): nombre de caractéristiques par échantillon

    Renvoi:
        list: liste des noms de colonnes à inclure dans le fichier CSV
    """
    colonnes_caracteristiques = []
    for i in range(1, nb_caracteristiques + 1):   
        colonnes_caracteristiques.append(f'caracteristique_{i}')
    return colonnes_caracteristiques

def nom_fichier(dossier_sortie, borne_inf, borne_sup, nb_echantillons, nb_caracteristiques, df, repetition):
    """Crée un fichier CSV contenant les données générées.

    Arguments:
        dossier_sortie (str): chemin vers le répertoire où sauvegarder les données
        borne_inf (float): limite inférieure des valeurs générées
        borne_sup (float): limite supérieure des valeurs générées
        nb_echantillons (int): nombre d'échantillons
        nb_caracteristiques (int): nombre de caractéristiques par échantillon
        df (pandas dataframe): dataframe contenant les données générées
        repetition (int): numéro de répétition pour différencier les fichiers
    """
    chemin_fichier = os.path.join(dossier_sortie, f"uniforme_{borne_inf}_{borne_sup}_{nb_echantillons}_{nb_caracteristiques}_{repetition}.csv")
    df.to_csv(chemin_fichier, index=False)

def produit_cartesien(liste_echantillons, liste_caracteristiques):
    """Retourne le produit cartésien des tailles d'échantillons et des nombres de caractéristiques.

    Arguments:
        liste_echantillons (list): liste des tailles d'échantillons
        liste_caracteristiques (list): liste des nombres de caractéristiques

    Renvoi:
        set: ensemble des combinaisons possibles
    """
    return set(itertools.product(liste_echantillons, liste_caracteristiques))

def test_parametres(borne_inf, borne_sup, liste_echantillons, liste_caracteristiques, dossier_sortie, repetitions=5):
    """Génère des données pour toutes les combinaisons possibles de paramètres et les sauvegarde dans des fichiers CSV.

    Arguments:
        borne_inf (float): limite inférieure des valeurs générées
        borne_sup (float): limite supérieure des valeurs générées
        liste_echantillons (list): liste des tailles d'échantillons
        liste_caracteristiques (list): liste des nombres de caractéristiques
        dossier_sortie (str): répertoire où sauvegarder les fichiers
        repetitions (int): nombre de répétitions pour chaque combinaison
    """
    combinaisons = produit_cartesien(liste_echantillons, liste_caracteristiques)
    for rep in range(repetitions):
        for params in combinaisons: 
            nb_echantillons, nb_caracteristiques = params
            taille = (nb_echantillons, nb_caracteristiques)
            donnees_uniformes = generation_donnees(borne_inf, borne_sup, taille)
            noms = noms_colonnes(nb_caracteristiques)
            df = pd.DataFrame(donnees_uniformes, columns=noms)
            nom_fichier(dossier_sortie, borne_inf, borne_sup, nb_echantillons, nb_caracteristiques, df, rep)

def generateur_uniforme(borne_inf=0.0, borne_sup=1.0, 
                        liste_echantillons=[i for i in range(100, 4001, 100)], 
                        liste_caracteristiques=[2**j for j in range(2, 8)]):
    """Fonction principale qui crée les données et les sauvegarde dans des fichiers CSV.

    Arguments:
        borne_inf (float): limite inférieure des valeurs générées (par défaut, 0.0)
        borne_sup (float): limite supérieure des valeurs générées (par défaut, 1.0)
        liste_echantillons (list): liste des tailles d'échantillons (par défaut, de 100 à 4000 avec un pas de 100)
        liste_caracteristiques (list): liste des nombres de caractéristiques (par défaut, 4, 8, 16, 32, 64)
    """
    dossier_sortie = os.path.join(os.environ.get("CHEMIN_GENERATION_DONNEES", ""), "resultats_uniforme")
    if not os.path.exists(dossier_sortie):
        os.makedirs(dossier_sortie)

    test_parametres(borne_inf, borne_sup, liste_echantillons, liste_caracteristiques, dossier_sortie)

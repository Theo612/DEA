import os
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.manifold import Isomap
import shutil

def reduire_et_sauvegarder(fichier_entree, fichier_sortie, nb_voisins, nb_composantes):
    """Applique une réduction dimensionnelle avec l'algorithme Isomap à un fichier CSV et 
    sauvegarde les données réduites dans un nouveau fichier CSV.

    Args:
    fichier_entree (str) : Chemin du fichier CSV d'entrée contenant les données à réduire. 
    fichier_sortie (str) : Chemin du fichier CSV de sortie.
    nb_voisins (int) : Nombre de voisins à utiliser pour la construction du graphe dans Isomap.
    nb_composantes (int) : Nombre de dimensions souhaitées dans l'espace réduit.
    """
    donnees = pd.read_csv(fichier_entree)
    
    if donnees.shape[1] < 2:
        print(f"Le fichier {fichier_entree} doit contenir au moins 2 colonnes de caractéristiques.")
        return

    X = donnees.values 
    normaliseur = StandardScaler()
    X_normalise = normaliseur.fit_transform(X)

    isomap = Isomap(n_neighbors=nb_voisins, n_components=nb_composantes)
    X_reduit = isomap.fit_transform(X_normalise)

    donnees_reduites = pd.DataFrame(X_reduit, columns=[f"Dimension_{i+1}" for i in range(nb_composantes)])
    donnees_reduites.to_csv(fichier_sortie, index=False)
    print(f"Fichier réduit sauvegardé sous {fichier_sortie}")

def nettoyer_dossier_sortie(dossier_sortie):
    """
    Supprime tous les fichiers et sous-dossiers dans un répertoire de sortie existant.
    Si le répertoire n'existe pas, un message est affiché indiquant qu'il sera créé.

    Args :
    dossier_sortie (str) : Chemin du répertoire de sortie à nettoyer.
    """

    if os.path.exists(dossier_sortie):
        for nom_fichier in os.listdir(dossier_sortie):
            chemin_fichier = os.path.join(dossier_sortie, nom_fichier)
            if os.path.isdir(chemin_fichier):
                shutil.rmtree(chemin_fichier) 
            else:
                os.remove(chemin_fichier) 
        print(f"Dossier '{dossier_sortie}' nettoyé.")
    else:
        print(f"Dossier '{dossier_sortie}' n'existe pas, il sera créé.")

def traiter_dossier(dossier_entree, dossier_sortie_base, valeurs_voisins, nb_composantes=2):
    """
    Traite tous les fichiers CSV dans un répertoire donné, applique la réduction dimensionnelle 
    Isomap pour chaque valeur de nb_voisins, sauvegarde les résultats dans des sous-dossiers spécifiques.

    Args :
    dossier_entree (str) : Chemin du répertoire contenant les fichiers CSV à traiter.
    dossier_sortie_base (str) : Chemin du répertoire principal où les resultats seront sauvegardés.
    valeurs_voisins (list) : Liste des differentes valeurs de nb_voisins à utiliser pour Isomap.
    nb_composantes (int) : Nombre de dimensions souhaitees dans l'espace reduit (défaut 2).

    """

    if not os.path.exists(dossier_sortie_base):
        os.makedirs(dossier_sortie_base)
        print(f"Dossier principal '{dossier_sortie_base}' créé.")
    else:
        nettoyer_dossier_sortie(dossier_sortie_base)

    for nb_voisins in valeurs_voisins:
        dossier_sortie = os.path.join(dossier_sortie_base, f"nb_voisins_{nb_voisins}")
        
        if not os.path.exists(dossier_sortie):
            os.makedirs(dossier_sortie)
            print(f"Sous-dossier '{dossier_sortie}' créé.")

        for nom_fichier in os.listdir(dossier_entree):
            if nom_fichier.endswith(".csv"):
                fichier_entree = os.path.join(dossier_entree, nom_fichier)
                fichier_sortie = os.path.join(dossier_sortie, f"reduit_{nb_voisins}_{nom_fichier}")
                reduire_et_sauvegarder(fichier_entree, fichier_sortie, nb_voisins, nb_composantes)

def traiter_plusieurs_dossiers(dossier_entrees, dossier_sortie_base, valeurs_voisins, nb_composantes=2):
    """
    Traite plusieurs répertoires d'entrée en appliquant la réduction dimensionnelle Isomap 
    sur les fichiers CSV contenus dans chacun, et organise les résultats dans des sous-dossiers.

    Args : 
    dossier_entrees (list) : Liste des chemins des répertoires d'entrée à traiter.
    dossier_sortie_base (str) : Chemin du répertoire principal.
    valeurs_voisins (list) : Liste des différentes valeurs de nb_voisins à utiliser pour Isomap.
    nb_composantes (int) : Nombre de dimensions souhaitées dans l'espace réduit (par défaut 2).
    """

    if not os.path.exists(dossier_sortie_base):
        os.makedirs(dossier_sortie_base)
        print(f"Dossier principal '{dossier_sortie_base}' créé.")
    
    for dossier_entree in dossier_entrees:
        sous_dossier_sortie = os.path.join(dossier_sortie_base, os.path.basename(dossier_entree))
        
        if not os.path.exists(sous_dossier_sortie):
            os.makedirs(sous_dossier_sortie)
            print(f"Sous-dossier '{sous_dossier_sortie}' créé.")
            
        traiter_dossier(dossier_entree, sous_dossier_sortie, valeurs_voisins, nb_composantes)

import os
from test_gaussien import generateur_gaussien_cluster
from test_uniforme import generateur_uniforme
from isomap import traiter_plusieurs_dossiers
from metriques import calculer_metrics_reduction 

def main():
    """
    Fonction principale pour générer les données, appliquer la réduction de dimension
    avec Isomap et calculer les métriques associées.
    """
    generateur_uniforme()
    generateur_gaussien_cluster()

    repertoire_courant = os.getcwd()
    dossiers_entree = [
        os.path.join(repertoire_courant, 'resultats_Gaussienne'), 
        os.path.join(repertoire_courant, 'resultats_Clusters'),  
        os.path.join(repertoire_courant, 'resultats_uniforme')     
    ]
    repertoire_sortie = os.path.join(repertoire_courant, 'donnees_reduites')
    
    valeurs_voisins = [5, 10, 70]
    traiter_plusieurs_dossiers(
        dossiers_entree, 
        repertoire_sortie, 
        valeurs_voisins=valeurs_voisins, 
        nb_composantes=2
    )

    dossier_sortie_metrics = os.path.join(repertoire_courant, 'metriques_reduction')
    for dossier_hd in dossiers_entree:
        nom_sous_dossier = os.path.basename(dossier_hd)
        for nb_voisins in valeurs_voisins :
            dossier_bd = os.path.join(
                repertoire_sortie, nom_sous_dossier, f"nb_voisins_{nb_voisins}")
            print(f"Calcul des métriques pour {nom_sous_dossier} avec nb_voisins={nb_voisins}.")
            calculer_metrics_reduction(
                repertoire_haute_dimension=dossier_hd,
                repertoire_reduction=dossier_bd,
                dossier_sortie=dossier_sortie_metrics,
                nom_sous_dossier=f"{nom_sous_dossier}_nb_voisins{nb_voisins}"
            )

if __name__ == "__main__":
    main()



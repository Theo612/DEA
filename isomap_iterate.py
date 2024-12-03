import os
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.manifold import Isomap
import shutil

def reduce_and_save(input_file, output_file, n_neighbors, n_components):
    """Applique une réduction dimensionnelle avec l'algorithme Isomap à un fichier CSV et 
    sauvegarde les données réduites dans un nouveau fichier CSV.

    Args:
    input_file (str) : Chemin du fichier CSV d'entrée contenant les données à réduire. 
    output_file (str) : Chemin du fichier CSV de sortie.
    n_neighbors (int) : Nombre de voisins à utiliser pour la construction du graphe dans Isomap.
    n_components (int) : Nombre de dimensions souhaitées dans l'espace réduit.
    """
    data = pd.read_csv(input_file)
    
    if data.shape[1] < 2:
        print(f"Le fichier {input_file} doit contenir au moins 2 colonnes de caractéristiques.")
        return

    X = data.values 

    scaler = StandardScaler()
    X_normalized = scaler.fit_transform(X)

    isomap = Isomap(n_neighbors=n_neighbors, n_components=n_components)
    X_reduced = isomap.fit_transform(X_normalized)

    reduced_data = pd.DataFrame(X_reduced, columns=[f"Dimension_{i+1}" for i in range(n_components)])

    reduced_data.to_csv(output_file, index=False)
    print(f"Fichier réduit sauvegardé sous {output_file}")

def clear_output_dir(output_dir):
    """
    Supprime tous les fichiers et sous-dossiers dans un répertoire de sortie existant.
    Si le répertoire n'existe pas, un message est affiché indiquant qu'il sera créé.

    Args :
    output_dir (str) : Chemin du répertoire de sortie à nettoyer.
    """

    if os.path.exists(output_dir):
        for filename in os.listdir(output_dir):
            file_path = os.path.join(output_dir, filename)
            if os.path.isdir(file_path):
                shutil.rmtree(file_path) 
            else:
                os.remove(file_path) 
        print(f"Dossier '{output_dir}' nettoyé.")
    else:
        print(f"Dossier '{output_dir}' n'existe pas, il sera créé.")

def process_directory(input_dir, base_output_dir, n_neighbors_values, n_components=2):
    """
    Traite tous les fichiers CSV dans un répertoire donné, applique la réduction dimensionnelle 
    Isomap pour chaque valeur de n_neighbors, sauvegarde les résultats dans des sous-dossiers spécifiques.

    Args :
    input_dir (str) : Chemin du répertoire contenant les fichiers CSV à traiter.
    base_output_dir (str) : Chemin du répertoire principal où les resultats seront sauvegardés.
    n_neighbors_values (list) : Liste des differentes valeurs de n_neighbors à utiliser pour Isomap.
    n_components (int) : Nombre de dimensions souhaitees dans l'espace reduit (défaut 2).

    """

    if not os.path.exists(base_output_dir):
        os.makedirs(base_output_dir)
        print(f"Dossier principal '{base_output_dir}' créé.")
    else:
        clear_output_dir(base_output_dir)

    for n_neighbors in n_neighbors_values:
        output_dir = os.path.join(base_output_dir, f"n_neighbors_{n_neighbors}")
        
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
            print(f"Sous-dossier '{output_dir}' créé.")

        for filename in os.listdir(input_dir):
            if filename.endswith(".csv"):
                input_file = os.path.join(input_dir, filename)
                output_file = os.path.join(output_dir, f"reduit_{n_neighbors}_{filename}")
                reduce_and_save(input_file, output_file, n_neighbors, n_components)

def process_multiple_dirs(input_dirs, base_output_dir, n_neighbors_values, n_components=2):
    """
    Traite plusieurs répertoires d'entrée en appliquant la réduction dimensionnelle Isomap 
    sur les fichiers CSV contenus dans chacun, et organise les résultats dans des sous-dossiers.

    Args : 
    input_dirs (list) : Liste des chemins des répertoires d'entrée à traiter.
    base_output_dir (str) : Chemin du répertoire principal.
    n_neighbors_values (list) : Liste des différentes valeurs de n_neighbors à utiliser pour Isomap.
    n_components (int) : Nombre de dimensions souhaitées dans l'espace réduit (par défaut 2).
    """

    if not os.path.exists(base_output_dir):
        os.makedirs(base_output_dir)
        print(f"Dossier principal '{base_output_dir}' créé.")
    
    for input_dir in input_dirs:
        output_sub_dir = os.path.join(base_output_dir, os.path.basename(input_dir))
        
        if not os.path.exists(output_sub_dir):
            os.makedirs(output_sub_dir)
            print(f"Sous-dossier '{output_sub_dir}' créé.")
            
        process_directory(input_dir, output_sub_dir, n_neighbors_values, n_components)

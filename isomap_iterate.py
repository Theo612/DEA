import os
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.manifold import Isomap
import shutil

def reduce_and_save(input_file, output_file, n_neighbors, n_components):
    """Applique Isomap à un fichier CSV et sauvegarde les résultats réduits dans un nouveau fichier CSV."""
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
    """Supprime tous les fichiers et sous-dossiers dans un répertoire de sortie existant."""
    if os.path.exists(output_dir):
        for filename in os.listdir(output_dir):
            file_path = os.path.join(output_dir, filename)
            if os.path.isdir(file_path):
                shutil.rmtree(file_path)  # Supprimer un sous-dossier
            else:
                os.remove(file_path)  # Supprimer un fichier
        print(f"Dossier '{output_dir}' nettoyé.")
    else:
        print(f"Dossier '{output_dir}' n'existe pas, il sera créé.")

def process_directory(input_dir, base_output_dir, n_neighbors_values, n_components=2):
    """Traite tous les fichiers CSV d'un répertoire et applique Isomap pour chaque valeur de n_neighbors."""
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
    """Traite plusieurs répertoires d'entrée (Gaussienne, Clusters, Uniforme) en appliquant Isomap sur chacun."""
    if not os.path.exists(base_output_dir):
        os.makedirs(base_output_dir)
        print(f"Dossier principal '{base_output_dir}' créé.")
    
    for input_dir in input_dirs:
        output_sub_dir = os.path.join(base_output_dir, os.path.basename(input_dir))
        
        if not os.path.exists(output_sub_dir):
            os.makedirs(output_sub_dir)
            print(f"Sous-dossier '{output_sub_dir}' créé.")
            
        process_directory(input_dir, output_sub_dir, n_neighbors_values, n_components)

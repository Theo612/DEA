import os
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.manifold import Isomap
import shutil

# Fonction pour appliquer Isomap et générer un fichier Numpy (.npy)
def reduce_and_save(input_file, output_file, n_neighbors, n_components):
    # Charger les données (au lieu de pd.read_csv, on utilise np.load pour les fichiers .npy)
    data = np.load(input_file)
    
    # Vérifier que les données ont min 2 colonnes pour les caractéristiques
    if data.shape[1] < 2:
        print(f"Le fichier {input_file} doit contenir au moins 2 colonnes de caractéristiques.")
        return

    # Normalisation des données
    scaler = StandardScaler()
    X_normalized = scaler.fit_transform(data)

    # Application d'Isomap
    isomap = Isomap(n_neighbors=n_neighbors, n_components=n_components)
    X_reduced = isomap.fit_transform(X_normalized)

    # Sauvegarder les données réduites dans un fichier .npy
    np.save(output_file, X_reduced)
    print(f"Fichier réduit sauvegardé sous {output_file}")

# Fonction pour nettoyer un dossier avant d'y ajouter des fichiers
def clear_output_dir(output_dir):
    # Vérifier si le dossier existe
    if os.path.exists(output_dir):
        # Supprimer tout le contenu du dossier (fichiers et sous-dossiers)
        for filename in os.listdir(output_dir):
            file_path = os.path.join(output_dir, filename)
            if os.path.isdir(file_path):
                shutil.rmtree(file_path)  # Supprime un sous-dossier
            else:
                os.remove(file_path)  # Supprime un fichier
        print(f"Dossier '{output_dir}' nettoyé.")
    else:
        print(f"Dossier '{output_dir}' n'existe pas, il sera créé.")

# Fonction pour traiter tous les fichiers d'un dossier d'entrée et créer un dossier de sortie pour chaque n_neighbors
def process_directory(input_dir, base_output_dir, n_neighbors_values, n_components=2):
    # Vérifier si le dossier de base de sortie existe, sinon le créer
    if not os.path.exists(base_output_dir):
        os.makedirs(base_output_dir)
        print(f"Dossier principal '{base_output_dir}' créé.")
    else:
        # Nettoyer le dossier de sortie s'il existe déjà
        clear_output_dir(base_output_dir)

    # Parcourir toutes les valeurs de n_neighbors
    for n_neighbors in n_neighbors_values:
        # Créer un sous-dossier spécifique pour chaque valeur de n_neighbors
        output_dir = os.path.join(base_output_dir, f"n_neighbors_{n_neighbors}")
        
        # Vérifier si le sous-dossier pour n_neighbors existe, sinon le créer
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
            print(f"Sous-dossier '{output_dir}' créé.")

        # Parcourir tous les fichiers .npy dans le dossier d'entrée
        for filename in os.listdir(input_dir):
            if filename.endswith(".npy"):
                input_file = os.path.join(input_dir, filename)
                output_file = os.path.join(output_dir, f"reduit_{n_neighbors}_{filename}")
                
                # Appliquer la réduction et sauvegarder le fichier
                reduce_and_save(input_file, output_file, n_neighbors, n_components)

# Exemple d'utilisation
current_directory = os.getcwd()  # Obtenir le répertoire actuel où le script est exécuté
input_directory = os.path.join(current_directory, 'resultats_Clusters')  # Dossier d'entrée dans le répertoire courant
output_directory = os.path.join(current_directory, 'donnees_reduites_clusters')  # Dossier de sortie dans le répertoire courant

# Appeler la fonction pour traiter tous les fichiers avec différentes valeurs de n_neighbors
n_neighbors_list = [5, 10, 70]  # Liste des valeurs de n_neighbors à tester
process_directory(input_directory, output_directory, n_neighbors_values=n_neighbors_list, n_components=2)

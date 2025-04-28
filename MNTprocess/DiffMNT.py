import rasterio
import numpy as np
import matplotlib.pyplot as plt
import os # Pour gérer les chemins de fichiers

def compare_mnts(mnt_t1_filepath: str, mnt_t2_filepath: str, output_diff_filepath: str = None):
    """
    Compare deux fichiers MNT GeoTIFF pour calculer la différence d'altitude (érosion/dépôt).

    Args:
        mnt_t1_filepath (str): Chemin vers le fichier MNT le plus ancien (Temps 1).
        mnt_t2_filepath (str): Chemin vers le fichier MNT le plus récent (Temps 2).
        output_diff_filepath (str, optional): Chemin pour sauvegarder la carte de différence
                                              au format GeoTIFF. Si None, ne sauvegarde pas.

    Returns:
        tuple: (diff_map, profile) ou (None, None) en cas d'erreur.
               diff_map (np.ndarray): Array 2D des différences d'altitude (T2 - T1).
                                      Positif = Dépôt net, Négatif = Érosion nette.
               profile (dict): Métadonnées du raster pour la sauvegarde éventuelle.
    """
    print(f"Comparaison des MNTs :\n  T1: {mnt_t1_filepath}\n  T2: {mnt_t2_filepath}")

    # Vérifier l'existence des fichiers
    if not os.path.exists(mnt_t1_filepath):
        print(f"Erreur : Fichier MNT T1 introuvable : {mnt_t1_filepath}")
        return None, None
    if not os.path.exists(mnt_t2_filepath):
        print(f"Erreur : Fichier MNT T2 introuvable : {mnt_t2_filepath}")
        return None, None

    try:
        # Ouvrir les deux fichiers MNT
        with rasterio.open(mnt_t1_filepath) as src_t1, rasterio.open(mnt_t2_filepath) as src_t2:

            # --- Vérifications de Cohérence (Crucial !) ---
            print("Vérification de la cohérence des MNTs...")
            if src_t1.crs != src_t2.crs:
                print(f"Erreur : Les Systèmes de Coordonnées de Référence (SCR) diffèrent !")
                print(f"  SCR T1: {src_t1.crs}")
                print(f"  SCR T2: {src_t2.crs}")
                print("Veuillez reprojeter les MNTs dans le même SCR avant comparaison.")
                return None, None

            if src_t1.transform != src_t2.transform:
                # La transform contient la résolution et l'origine. Une légère différence peut être ok,
                # mais une grosse différence indique un problème.
                print(f"Avertissement : Les transformations affines diffèrent légèrement.")
                print(f"  Transform T1: {src_t1.transform}")
                print(f"  Transform T2: {src_t2.transform}")
                # On pourrait ajouter une tolérance ici si nécessaire
                if not np.allclose(np.array(src_t1.res), np.array(src_t2.res)):
                     print(f"Erreur critique: Les résolutions diffèrent! ({src_t1.res} vs {src_t2.res})")
                     return None, None
                print("-> Continuation, mais soyez prudent avec l'interprétation.")


            if src_t1.shape != src_t2.shape:
                print(f"Erreur : Les dimensions des MNTs diffèrent !")
                print(f"  Shape T1: {src_t1.shape}")
                print(f"  Shape T2: {src_t2.shape}")
                print("Veuillez redécouper/rééchantillonner les MNTs aux mêmes dimensions.")
                return None, None

            print("Cohérence vérifiée (SCR, Dimensions).")

            # Lire les données d'altitude (première bande)
            mnt_t1 = src_t1.read(1)
            mnt_t2 = src_t2.read(1)

            # Gérer les valeurs NoData (les remplacer par NaN pour les calculs)
            nodata_t1 = src_t1.nodata
            nodata_t2 = src_t2.nodata

            mask_t1 = (mnt_t1 == nodata_t1) if nodata_t1 is not None else np.zeros(mnt_t1.shape, dtype=bool)
            mask_t2 = (mnt_t2 == nodata_t2) if nodata_t2 is not None else np.zeros(mnt_t2.shape, dtype=bool)

            # Mettre NaN là où il y a NoData dans l'un OU l'autre des MNTs
            valid_mask = ~(mask_t1 | mask_t2) # Pixels valides dans les DEUX MNTs

            # Convertir en float pour permettre les NaN
            mnt_t1 = mnt_t1.astype(float)
            mnt_t2 = mnt_t2.astype(float)
            mnt_t1[mask_t1 | mask_t2] = np.nan # Mettre NaN si une des valeurs est nodata
            mnt_t2[mask_t1 | mask_t2] = np.nan # Mettre NaN si une des valeurs est nodata


            # --- Calcul de la Différence ---
            print("Calcul de la différence d'altitude (T2 - T1)...")
            diff_map = mnt_t2 - mnt_t1
            # diff_map contiendra NaN là où au moins un des MNT avait NoData

            print(f"Calcul terminé. Plage de différence : Min={np.nanmin(diff_map):.3f}, Max={np.nanmax(diff_map):.3f}")

            # --- Préparer les métadonnées pour la sauvegarde ---
            profile = src_t1.profile # Utiliser le profil du premier MNT comme base
            profile.update(
                dtype=rasterio.float32, # Sauvegarder en float pour garder les décimales
                count=1,
                nodata=np.nan # Utiliser NaN comme valeur NoData dans le fichier de sortie
            )
            # Optionnel: Mettre à jour la compression
            # profile['compress'] = 'lzw'

            # --- Sauvegarde Optionnelle ---
            if output_diff_filepath:
                # S'assurer que le dossier de sortie existe
                output_dir = os.path.dirname(output_diff_filepath)
                if output_dir and not os.path.exists(output_dir):
                    os.makedirs(output_dir)

                print(f"Sauvegarde de la carte de différence vers : {output_diff_filepath}")
                with rasterio.open(output_diff_filepath, 'w', **profile) as dst:
                    dst.write(diff_map.astype(rasterio.float32), 1)
                print("Sauvegarde terminée.")

            return diff_map, profile

    except rasterio.RasterioIOError as e:
        print(f"Erreur d'entrée/sortie Rasterio : {e}")
        return None, None
    except Exception as e:
        print(f"Une erreur inattendue est survenue : {e}")
        return None, None

def plot_difference_map(diff_map, title="Différence d'Altitude (T2 - T1)", cmap='RdBu'):
    """Affiche la carte de différence avec une colormap divergente."""
    if diff_map is None:
        print("Aucune carte de différence à afficher.")
        return

    print("Affichage de la carte de différence...")
    # Calculer les limites pour la colormap (centrée autour de 0)
    max_abs_diff = np.nanmax(np.abs(diff_map))
    if max_abs_diff == 0: max_abs_diff = 1.0 # Eviter vmin=vmax si aucune différence

    plt.figure(figsize=(10, 8))
    im = plt.imshow(diff_map, cmap=cmap, vmin=-max_abs_diff, vmax=max_abs_diff)
    plt.colorbar(im, label='Différence Altitude (m)\n(Positif=Dépôt, Négatif=Érosion)')
    plt.title(title)
    plt.xlabel("Pixels (X)")
    plt.ylabel("Pixels (Y)")
    # Mettre les Y croissants vers le haut (origine en bas à gauche)
    # plt.gca().invert_yaxis() # Dépend si l'array est lu 'correctement'
    plt.show()

# --- Exemple d'Utilisation ---
if __name__ == "__main__":
    # --- !!! METTRE ICI LES CHEMINS VERS VOS FICHIERS MNT !!! ---
    mnt_t1_file = "C:\\Users\\franc\\Downloads\\Ile1\\Ile1_epsg4269.tif"  # Exemple: 'data/alps_srtm_2000.tif'
    mnt_t2_file = "C:\\Users\\franc\\Downloads\\Ile2\\output_USGS30m.tif"   # Exemple: 'data/alps_copdem_2015.tif'

    # --- Chemin optionnel pour sauvegarder la carte de différence ---
    output_file = "resultats/difference_mnt.tif" # Mettre None pour ne pas sauvegarder

    # Appeler la fonction de comparaison
    difference_array, meta_profile = compare_mnts(mnt_t1_file, mnt_t2_file, output_file)

    # Afficher la carte résultante si le calcul a réussi
    if difference_array is not None:
        plot_difference_map(difference_array)
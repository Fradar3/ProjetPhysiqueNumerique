import rasterio
from rasterio.warp import calculate_default_transform, reproject, Resampling
import numpy as np
import os

# --- Chemins des fichiers ---
mnt_t1_filepath = r"C:\\Users\\franc\\Downloads\\Ile1\\output_SRTMGL1.tif"
mnt_t2_filepath = r"C:\\Users\\franc\\Downloads\\Ile2\\output_USGS30m.tif"
# Chemin pour le fichier T1 reprojeté
mnt_t1_reprojected_filepath = r"C:\Users\franc\Documents\FrkWork\Ile1_epsg4269.tif"
# Chemin optionnel pour le fichier T1 reprojeté ET rééchantillonné (si nécessaire ensuite)
# mnt_t1_reproj_resamp_filepath = r"C:\Users\franc\Documents\FrkWork\srtm_reproj_resamp.tif"

# --- SCR Cible ---
target_crs = 'EPSG:4269' # Ou rasterio.crs.CRS.from_epsg(4269)

try:
    print(f"Reprojection de {mnt_t1_filepath} (EPSG:4326) vers {target_crs}...")

    # Ouvrir le raster source (T1)
    with rasterio.open(mnt_t1_filepath) as src_t1:
        source_crs = src_t1.crs
        source_nodata = src_t1.nodata
        source_dtype = src_t1.read(1).dtype # Lire le type de données original

        # Calculer la transformation et les dimensions pour le raster reprojeté
        # Garder la résolution approximative de la source lors de la reprojection initiale
        transform, width, height = calculate_default_transform(
            source_crs, target_crs, src_t1.width, src_t1.height, *src_t1.bounds)

        # Mettre à jour le profil pour la sortie reprojetée
        profile_out = src_t1.profile.copy()
        profile_out.update({
            'crs': target_crs,
            'transform': transform,
            'width': width,
            'height': height,
            'nodata': source_nodata if source_nodata is not None else -9999, # Définir une valeur nodata si absente
            'dtype': source_dtype # Garder le type de données si possible
        })

        print(f"  Nouvelle Transform: {transform}")
        print(f"  Nouvelles Dimensions: ({width}, {height})")

        # Effectuer la reprojection
        print("  Exécution de la reprojection...")
        with rasterio.open(mnt_t1_reprojected_filepath, 'w', **profile_out) as dst:
            reproject(
                source=rasterio.band(src_t1, 1),
                destination=rasterio.band(dst, 1),
                src_transform=src_t1.transform,
                src_crs=src_t1.crs,
                src_nodata=source_nodata,
                dst_transform=transform,
                dst_crs=target_crs,
                dst_nodata=profile_out['nodata'],
                resampling=Resampling.bilinear # Méthode de rééchantillonnage pendant la reprojection
            )
        print("Reprojection terminée.")

    # --- Étape suivante potentielle : Rééchantillonnage ---
    # Maintenant, mnt_t1_reprojected_filepath (T1') et mnt_t2_filepath (T2) sont dans le même SCR (EPSG:4269).
    # MAIS ils n'ont probablement pas la même résolution/dimensions.
    # Il faut rééchantillonner T1' à la résolution de T2 (ou l'inverse).
    # Le code de rééchantillonnage de la réponse précédente peut être adapté ici,
    # en utilisant mnt_t1_reprojected_filepath comme source et mnt_t2_filepath comme cible de référence.
    # (Code de rééchantillonnage omis ici pour la clarté, mais il faut le faire)

    # --- Une fois T1' reprojeté ET rééchantillonné à la grille de T2 ---
    # Supposons que le résultat final est dans 'mnt_t1_final_for_comparison.tif'
    # mnt_t1_final_file = 'chemin/vers/mnt_t1_final_for_comparison.tif'
    # output_diff_file = r"C:\Users\franc\Documents\FrkWork\resultats\difference_mnt_final.tif"

    # print("\nComparaison des MNTs après reprojection et rééchantillonnage...")
    # difference_array, meta_profile = compare_mnts(mnt_t1_final_file, mnt_t2_filepath, output_diff_file)

    # if difference_array is not None:
    #     plot_difference_map(difference_array)


except ImportError:
     print("Erreur: La bibliothèque rasterio ou une de ses dépendances (GDAL) semble manquer ou mal configurée.")
except Exception as e:
    print(f"Une erreur est survenue pendant la reprojection : {e}")
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import convolve2d
import time # Pour le suivi du temps

# --- Paramètres de Simulation ---
GRID_SIZE = 100      # Taille de la grille (plus grand = meilleure stat, mais plus long)
N_SIMULATIONS_PER_P = 5 # Nombre de simulations pour chaque P_TREE (pour moyenner)
MAX_STEPS = GRID_SIZE * 2 # Nombre max d'étapes par simulation (sécurité)

# --- Paramètres d'Analyse ---
P_TREE_VALUES = np.linspace(0.01, 1.0, 50) # Plage de densités d'arbres à tester

# --- États ---
EMPTY = 0
TREE = 1
FIRE = 2

# --- Noyau de Convolution (Voisinage Moore) ---
kernel = np.array([[1, 1, 1],
                   [1, 0, 1],
                   [1, 1, 1]], dtype=int)

# --- Fonction de Simulation Unique (sans animation) ---
def run_single_fire_simulation(grid_size, p_tree, max_steps):
    """
    Exécute une simulation de feu de forêt pour un p_tree donné
    et retourne la fraction de la zone *initialement boisée* qui a brûlé.
    """
    # 1. Initialisation de la grille
    initial_grid = np.random.choice(
        [EMPTY, TREE],
        size=(grid_size, grid_size),
        p=[1 - p_tree, p_tree]
    )
    initial_trees = (initial_grid == TREE)
    n_initial_trees = np.sum(initial_trees)

    # Si pas d'arbres, rien ne brûle
    if n_initial_trees == 0:
        return 0.0

    grid = initial_grid.copy()

    # 2. Démarrer le feu (patch au centre)
    center = grid_size // 2
    fire_size = 2
    start_fire_mask = np.zeros_like(grid, dtype=bool)
    start_fire_mask[center-fire_size//2 : center+fire_size//2 +1,
                    center-fire_size//2 : center+fire_size//2 +1] = True
    ignition_mask = start_fire_mask & initial_trees # Seulement les arbres dans la zone prennent feu
    grid[ignition_mask] = FIRE

    # 3. Boucle de Simulation
    for step in range(max_steps):
        is_fire = (grid == FIRE)
        # Si plus de feu, arrêter la simulation
        if not np.any(is_fire):
            break

        # Calculer la prochaine grille
        new_grid = grid.copy()
        burning_neighbors = convolve2d(is_fire, kernel, mode='same', boundary='wrap')
        trees_catching_fire = (grid == TREE) & (burning_neighbors > 0)
        new_grid[trees_catching_fire] = FIRE
        new_grid[is_fire] = EMPTY # Le feu s'éteint

        grid = new_grid

    # 4. Calculer les résultats
    final_empty = (grid == EMPTY)
    # Les cellules brûlées sont celles qui étaient des arbres ET qui sont maintenant vides
    burned_mask = initial_trees & final_empty
    n_burned = np.sum(burned_mask)

    fraction_burned = n_burned / n_initial_trees if n_initial_trees > 0 else 0.0
    return fraction_burned

# --- Boucle Principale d'Analyse ---
results_fraction_burned = []
std_dev_fraction_burned = []

print(f"Début de l'analyse pour {len(P_TREE_VALUES)} valeurs de P_TREE...")
start_time = time.time()

for i, p_tree in enumerate(P_TREE_VALUES):
    current_p_fractions = []
    for sim in range(N_SIMULATIONS_PER_P):
        fraction = run_single_fire_simulation(GRID_SIZE, p_tree, MAX_STEPS)
        current_p_fractions.append(fraction)

    avg_fraction = np.mean(current_p_fractions)
    std_fraction = np.std(current_p_fractions)
    results_fraction_burned.append(avg_fraction)
    std_dev_fraction_burned.append(std_fraction)

    # Affichage de la progression
    if (i + 1) % 5 == 0:
        elapsed = time.time() - start_time
        print(f"  Terminé {i+1}/{len(P_TREE_VALUES)} (P_TREE={p_tree:.3f}) - Temps écoulé: {elapsed:.1f}s")


total_time = time.time() - start_time
print(f"Analyse terminée en {total_time:.1f} secondes.")

# --- Affichage des Résultats ---
plt.figure(figsize=(10, 6))

# Convertir results_fraction_burned en un tableau NumPy pour l'indexation
results_np = np.array(results_fraction_burned)
std_dev_np = np.array(std_dev_fraction_burned) # Faire de même pour l'écart-type

# Utiliser les tableaux NumPy pour le tracé et l'analyse
plt.errorbar(P_TREE_VALUES, results_np, yerr=std_dev_np,
             fmt='-o', capsize=5, markersize=4, label='Fraction moyenne brûlée')

# Estimation visuelle de p_c
# Trouver les indices où P_TREE est dans la région critique
critical_region_indices = np.where((P_TREE_VALUES > 0.45) & (P_TREE_VALUES < 0.75))[0] # Ajuster la plage si besoin

# Vérifier qu'il y a bien des points dans la région critique
if len(critical_region_indices) > 1: # Besoin d'au moins 2 points pour calculer diff
    # Sélectionner les résultats DANS la région critique en utilisant le tableau NumPy
    results_in_region = results_np[critical_region_indices]

    # Calculer la différence entre les points consécutifs DANS la région
    diffs_in_region = np.diff(results_in_region)

    # Trouver l'indice de la plus grande différence *relativement à la sous-liste 'results_in_region'*
    max_diff_local_index = np.argmax(diffs_in_region)

    # Trouver l'indice global correspondant dans P_TREE_VALUES
    # C'est l'indice du *premier* des deux points qui ont donné la plus grande différence
    approx_pc_global_index = critical_region_indices[max_diff_local_index]

    # Obtenir la valeur de P_TREE à cet indice
    approx_pc = P_TREE_VALUES[approx_pc_global_index]

    plt.axvline(approx_pc, color='r', linestyle='--', label=f'Seuil critique approx. $p_c \\approx {approx_pc:.3f}$')
elif len(critical_region_indices) == 1:
     print("Avertissement : Un seul point trouvé dans la région critique, impossible d'estimer pc.")
else:
     print("Avertissement : Aucun point trouvé dans la région critique, impossible d'estimer pc.")


plt.xlabel("Densité initiale d'arbres ($P_{TREE}$)")
plt.ylabel("Fraction moyenne de la forêt brûlée")
plt.title(f"Analyse de Percolation du Feu de Forêt (Grille {GRID_SIZE}x{GRID_SIZE}, {N_SIMULATIONS_PER_P} sims/p)")
plt.grid(True, which='both', linestyle='--', linewidth=0.5)
plt.legend()
plt.ylim(0, 1.05) # Assure que l'axe Y va de 0 à 1
plt.show()
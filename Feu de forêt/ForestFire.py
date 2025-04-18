import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import matplotlib.colors as mcolors
from scipy.signal import convolve2d

# --- Paramètres ---
GRID_SIZE = 100    # Taille de la grille (GRID_SIZE x GRID_SIZE)
FRAMES = 150       # Nombre d'étapes/frames dans l'animation
INTERVAL = 100     # Millisecondes entre les frames (ajuste la vitesse)
P_TREE = 0.60      # Probabilité initiale qu'une cellule soit un arbre
IGNITION_ROW = GRID_SIZE // 2 # Ligne pour démarrer le feu (ou choisir un point)

SAVE_ANIMATION = True # Mettre à True pour sauvegarder en GIF
FILENAME = 'forest_fire.gif'

# --- États ---
EMPTY = 0
TREE = 1
FIRE = 2

# --- Initialisation de la Grille ---
# 1. Créer une grille avec des arbres et des espaces vides
grid = np.random.choice(
    [EMPTY, TREE],
    size=(GRID_SIZE, GRID_SIZE),
    p=[1 - P_TREE, P_TREE]
)

# 2. Démarrer le feu
# Option A: Une ligne de feu sur un bord (assurez-vous qu'il y a des arbres!)
# grid[IGNITION_ROW, :] = np.where(grid[IGNITION_ROW, :] == TREE, FIRE, grid[IGNITION_ROW, :])
# Option B: Un petit patch de feu au centre (plus commun)
center = GRID_SIZE // 2
fire_size = 2 # Taille du patch initial
grid[center-fire_size//2 : center+fire_size//2 +1,
     center-fire_size//2 : center+fire_size//2 +1] = np.where(
         grid[center-fire_size//2 : center+fire_size//2 +1,
              center-fire_size//2 : center+fire_size//2 +1] == TREE,
         FIRE,
         grid[center-fire_size//2 : center+fire_size//2 +1,
              center-fire_size//2 : center+fire_size//2 +1]
     )


# --- Logique de la Simulation ---
# Noyau de convolution pour détecter les voisins en feu (Moore)
kernel = np.array([[1, 1, 1],
                   [1, 0, 1], # Le centre est 0 car on ne se compte pas soi-même
                   [1, 1, 1]], dtype=int)

def update_grid(frameNum, img, grid, N):
    """Calcule la prochaine génération de la grille de feu de forêt."""
    # Crée une copie pour stocker le nouvel état
    new_grid = grid.copy()

    # 1. Trouver les arbres qui vont prendre feu
    #    Trouver les emplacements actuels du feu
    is_fire = (grid == FIRE)
    # Compter les voisins en feu pour chaque cellule
    burning_neighbors = convolve2d(is_fire, kernel, mode='same', boundary='wrap')
    # Identifier les arbres avec au moins un voisin en feu
    trees_catching_fire = (grid == TREE) & (burning_neighbors > 0)
    # Mettre ces arbres en feu dans la nouvelle grille
    new_grid[trees_catching_fire] = FIRE

    # 2. Trouver les feux qui s'éteignent
    #    Les cellules qui étaient en feu dans l'ancienne grille deviennent vides
    new_grid[is_fire] = EMPTY

    # 3. Mettre à jour la grille et l'image affichée
    img.set_data(new_grid)
    grid[:] = new_grid[:] # Met à jour la grille globale pour la prochaine itération
    return img,

# --- Mise en place de l'Animation ---
fig, ax = plt.subplots(figsize=(8, 8))

# Créer une colormap personnalisée
# Noir/Marron foncé pour EMPTY, Vert pour TREE, Rouge/Orange pour FIRE
colors = ['#4d331a', 'green', 'red'] # Vous pouvez ajuster les couleurs
cmap = mcolors.ListedColormap(colors)
bounds = [EMPTY, TREE, FIRE, FIRE + 1] # Définir les limites pour les couleurs
norm = mcolors.BoundaryNorm(bounds, cmap.N)

img = ax.imshow(grid, cmap=cmap, norm=norm, interpolation='nearest')

# Cacher les axes et ajouter un titre
ax.set_xticks([])
ax.set_yticks([])
ax.set_title(f"Forest Fire Simulation (p_tree = {P_TREE:.2f})")

# Créer l'objet animation
ani = animation.FuncAnimation(fig, update_grid, fargs=(img, grid, GRID_SIZE),
                              frames=FRAMES, interval=INTERVAL, blit=True,
                              repeat=False) # repeat=False arrête l'animation

# --- Sauvegarde ou Affichage ---
if SAVE_ANIMATION:
    print(f"Sauvegarde de l'animation en cours ({FRAMES} frames)...")
    try:
        ani.save(FILENAME, writer='pillow', fps=1000/INTERVAL)
        print(f"Animation sauvegardée sous '{FILENAME}'")
    except Exception as e:
        print(f"Erreur lors de la sauvegarde : {e}")
        print("Assurez-vous que 'pillow' est installé (`pip install pillow`)")
else:
    plt.show()

print("Terminé.")
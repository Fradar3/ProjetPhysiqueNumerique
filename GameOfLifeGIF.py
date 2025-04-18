import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from scipy.signal import convolve2d

# --- Paramètres ---
GRID_SIZE = 100  # Taille de la grille (GRID_SIZE x GRID_SIZE)
FRAMES = 200     # Nombre d'étapes/frames dans l'animation
INTERVAL = 100   # Millisecondes entre les frames (ajuste la vitesse)
SAVE_ANIMATION = True # Mettre à True pour sauvegarder en GIF
FILENAME = 'game_of_life.gif'

# --- Initialisation de la Grille ---
# Choisissez une méthode d'initialisation:
# 1. Aléatoire
grid = np.random.choice([0, 1], size=(GRID_SIZE, GRID_SIZE), p=[0.75, 0.25])

# 2. Ou ajoutez des structures spécifiques (décommentez pour tester)
# grid = np.zeros((GRID_SIZE, GRID_SIZE), dtype=int)
# # Ajouter un Glider
# grid[1, 2] = 1
# grid[2, 3] = 1
# grid[3, 1:4] = 1
# # Ajouter un R-pentomino (évolue longtemps)
# mid = GRID_SIZE // 2
# grid[mid, mid+1] = 1
# grid[mid, mid+2] = 1
# grid[mid+1, mid] = 1
# grid[mid+1, mid+1] = 1
# grid[mid+2, mid+1] = 1

# --- Logique du Jeu de la Vie ---
# Noyau de convolution pour compter les voisins (Moore)
kernel = np.array([[1, 1, 1],
                   [1, 0, 1],
                   [1, 1, 1]], dtype=int)

def update_grid(frameNum, img, grid, N):
    """Calcule la prochaine génération de la grille."""
    # 1. Compter les voisins vivants pour chaque cellule
    #    'wrap' gère les conditions aux limites périodiques
    live_neighbors = convolve2d(grid, kernel, mode='same', boundary='wrap')

    # 2. Appliquer les règles du Jeu de la Vie
    #    Conditions pour qu'une cellule soit vivante à la prochaine étape:
    #    - Une cellule vivante (grid == 1) survit si elle a 2 ou 3 voisins vivants.
    #    - Une cellule morte (grid == 0) devient vivante si elle a exactement 3 voisins vivants.
    new_grid = (
        ((grid == 1) & ((live_neighbors == 2) | (live_neighbors == 3))) |
        ((grid == 0) & (live_neighbors == 3))
    ).astype(int) # Convertir le résultat booléen en 0 ou 1

    # 3. Mettre à jour la grille et l'image affichée
    img.set_data(new_grid)
    grid[:] = new_grid[:] # Met à jour la grille globale pour la prochaine itération
    return img,

# --- Mise en place de l'Animation ---
fig, ax = plt.subplots(figsize=(8, 8)) # Ajuster la taille si besoin
img = ax.imshow(grid, cmap='binary', interpolation='nearest') # 'binary' -> noir/blanc

# Cacher les axes pour une meilleure visualisation
ax.set_xticks([])
ax.set_yticks([])
ax.set_title("Conway's Game of Life")

# Créer l'objet animation
# blit=True optimise le rendu (redessine seulement ce qui change)
ani = animation.FuncAnimation(fig, update_grid, fargs=(img, grid, GRID_SIZE),
                              frames=FRAMES, interval=INTERVAL, blit=True,
                              repeat=False) # repeat=False arrête l'animation après FRAMES

# --- Sauvegarde ou Affichage ---
if SAVE_ANIMATION:
    print(f"Sauvegarde de l'animation en cours ({FRAMES} frames)... Cela peut prendre un moment.")
    try:
        ani.save(FILENAME, writer='pillow', fps=1000/INTERVAL)
        print(f"Animation sauvegardée sous '{FILENAME}'")
    except Exception as e:
        print(f"Erreur lors de la sauvegarde : {e}")
        print("Assurez-vous que 'pillow' est installé (`pip install pillow`)")
else:
    plt.show() # Affiche l'animation interactivement

print("Terminé.")
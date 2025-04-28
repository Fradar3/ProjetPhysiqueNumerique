import matplotlib.pyplot as plt
import numpy as np

# Supposons que agent_data est déjà obtenu avec :
agent_data = model.datacollector.get_agent_vars_dataframe()

# On spécifie que l'on travaille avec "Altitude"
variable_name = "Altitude"

# Récupérer les données à step 0 et step 50
step0 = agent_data.xs(0, level="Step")
step50 = agent_data.xs(50, level="Step")  # Change à 100 si tu as 100 steps

# Créer une grille vide pour stocker la différence
difference_grid = np.zeros((model.height, model.width))

# Calculer la différence Altitude_step50 - Altitude_step0
for index, row in step0.iterrows():
    x, y = row["Pos"]
    altitude_step0 = row[variable_name]
    altitude_step50 = step50.loc[index][variable_name]
    difference = altitude_step50 - altitude_step0
    difference_grid[y, x] = difference

# Afficher la différence
plt.figure(figsize=(8, 6))
plt.imshow(difference_grid, origin='lower', cmap='coolwarm')  # Color map pour voir où ça augmente/baisse
plt.colorbar(label="Δ Altitude (Step 50 - Step 0)")
plt.title("Changement d'Altitude entre Step 0 et Step 50")
plt.xlabel("X Coordinate")
plt.ylabel("Y Coordinate")
plt.gca().invert_yaxis()
plt.show()

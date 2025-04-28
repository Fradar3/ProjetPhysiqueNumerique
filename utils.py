import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os
import imageio
import rasterio

def generate_dummy_dem(width: int, height: int) -> np.ndarray:
    dummy_dem = np.zeros((height, width))

    # Create a sloped surface with hills and noise
    for y in range(height):
        for x in range(width):
            # Basic slope (eastward and downward)
            base_height = 120 + x * 0.8 + (height - 1 - y) * 0.5

            # Add small undulations (hills)
            hills = np.sin(x / 5) * 5 + np.sin(y / 7) * 5

            # Larger landscape-scale hills
            large_hills = np.sin(x / 15) * 15 + np.sin(y / 20) * 15

            # Random small noise
            noise = np.random.normal(0, 0.5)

            dummy_dem[y, x] = base_height + hills + large_hills + noise

    # Add multiple basins (depressions)
    basins = [(width // 2, height // 2, 155), (width // 5, 4 * height // 5, 100), (4 * width // 5, height // 5, 120)]
    for x0, y0, depth in basins:
        for dy in range(-5, 6):
            for dx in range(-5, 6):
                xx = x0 + dx
                yy = y0 + dy
                if 0 <= xx < width and 0 <= yy < height:
                    dist = np.sqrt(dx**2 + dy**2)
                    if dist <= 5:
                        dummy_dem[yy, xx] -= (depth / (1 + dist))  # Smooth depression

    # Add a ridge along a line
    for i in range(width):
        if width // 3 <= i <= 2 * width // 3:
            ridge_y = height // 3
            if 0 <= ridge_y < height:
                dummy_dem[ridge_y, i] -= 50
                dummy_dem[ridge_y+1, i] -= 50
                dummy_dem[ridge_y-1, i] -= 50

    return dummy_dem

def pluviogram(steps):
    # rain = [0.1] * 20 + [0.2]*10 + [0.4]*10 + [0.5]*10 + [0.2]*30 + [0.4]*20
    rain = np.linspace(0, 4*np.pi, steps*2)
    rain = 0.4*np.sin(rain) + 1
    return list(rain)

def create_spatial_gif(
    agent_data_df: pd.DataFrame,
    model_width: int,
    model_height: int,
    variable_name: str,
    output_filename: str = "animation.gif",
    fps: int = 10,
    cmap: str = 'gray_r', # Colormap for the plot
    vmin: float = None, # Minimum value for colorbar (fixed across frames)
    vmax: float = None, # Maximum value for colorbar (fixed across frames)
    title_prefix: str = "" # Prefix for the title
):
    """
    Creates a GIF animation of a spatial variable over time.

    Args:
        agent_data_df: DataFrame from model.datacollector.get_agent_vars_dataframe().
        model_width: Width of the model grid.
        model_height: Height of the model grid.
        variable_name: The name of the variable column to visualize (e.g., "WaterDepth").
        output_filename: The name of the output GIF file.
        fps: Frames per second for the GIF.
        cmap: Matplotlib colormap to use.
        vmin: Optional. Minimum value for the colorbar range. If None, calculated from data.
        vmax: Optional. Maximum value for the colorbar range. If None, calculated from data.
        title_prefix: Optional. String prefix for the plot title.
    """
    if variable_name not in agent_data_df.columns:
        print(f"Error: Variable '{variable_name}' not found in collected data.")
        print(f"Available variables: {list(agent_data_df.columns)}")
        return

    # Get all unique steps from the index
    all_steps = agent_data_df.index.get_level_values("Step").unique()

    if len(all_steps) == 0:
        print("Error: No steps found in the agent data.")
        return

    # Determine fixed vmin and vmax for consistent colorbar across frames
    if vmin is None or vmax is None:
        # Filter out potential non-finite values if necessary
        variable_values = agent_data_df[variable_name].dropna()
        if not variable_values.empty:
             vmin = vmin if vmin is not None else variable_values.min()
             vmax = vmax if vmax is not None else variable_values.max()
        else:
            # Default range if no valid data is found
            vmin = 0
            vmax = 1

    if vmin == vmax: # Handle cases where the variable is constant
        vmax = vmin + 1e-6 # Add a small offset to make colorbar work

    print(f"Creating GIF for '{variable_name}'. Vmin={vmin}, Vmax={vmax}")

    # --- Generate frames ---
    image_files = []
    temp_dir = "temp_frames"
    os.makedirs(temp_dir, exist_ok=True) # Create a directory for temporary frames

    plt.ioff() # Turn off interactive plotting

    for step in all_steps:
        try:
            step_data = agent_data_df.loc[step]
        except KeyError:
            print(f"Warning: Data for step {step} not found. Skipping frame.")
            continue

        grid_data = np.full((model_height, model_width), np.nan) # Initialize grid with NaN

        for index, row in step_data.iterrows():
            x, y = row["Pos"]
            if 0 <= x < model_width and 0 <= y < model_height:
                grid_data[y, x] = row[variable_name]
            # else: Warning already printed in plot_spatial_variable

        fig, ax = plt.subplots(figsize=(6, 6)) # Create figure and axes

        # Use imshow to display the grid data
        im = ax.imshow(grid_data, origin='lower', cmap=cmap, interpolation='nearest',
                       vmin=vmin, vmax=vmax) # Apply fixed vmin/vmax

        ax.set_title(f"{title_prefix}{variable_name} at Step {step}")
        ax.set_xlabel("X Coordinate")
        ax.set_ylabel("Y Coordinate")

        # Add colorbar - specify mappable (im)
        cbar = fig.colorbar(im, ax=ax, label=variable_name)

        # Save the figure to a temporary file
        # Use a format string like {step:04d} to ensure files are sorted correctly
        filename = os.path.join(temp_dir, f"{variable_name}_frame_{step:04d}.png")
        fig.savefig(filename, bbox_inches='tight', dpi=100) # Save tightly with reasonable dpi
        image_files.append(filename)

        plt.close(fig) # Close the figure to free memory

    # --- Create GIF from frames ---
    if not image_files:
        print("No frames were generated to create a GIF.")
        return

    print(f"Creating GIF '{output_filename}' from {len(image_files)} frames...")
    try:
        with imageio.get_writer(output_filename, mode='I', fps=fps) as writer:
            for filename in image_files:
                image = imageio.imread(filename)
                writer.append_data(image)
        print(f"GIF saved as '{output_filename}'")
    except Exception as e:
        print(f"Error creating GIF: {e}")
    finally:
        # --- Clean up temporary files ---
        print(f"Cleaning up temporary frame files in '{temp_dir}'...")
        for filename in image_files:
            try:
                os.remove(filename)
            except OSError as e:
                print(f"Error removing file {filename}: {e}")
        # Optional: Remove the temporary directory if empty
        try:
            os.rmdir(temp_dir)
            print(f"Temporary directory '{temp_dir}' removed.")
        except OSError:
             # Directory might not be empty if cleanup failed partially
             print(f"Temporary directory '{temp_dir}' not empty, skipping removal.")

    plt.ion() # Turn interactive plotting back on
    

def load_dem_from_tif(tif_filepath: str) -> np.ndarray:
    with rasterio.open(tif_filepath) as dataset:
        return dataset.read(1) # En assumant que l'altitude se trouve dans la première bande
    
# alt = load_dem_from_tif("MNTprocess/MNTs/srtm_reprojected_epsg4269.tif")
# plt.imshow(alt, origin="lower")
# plt.show()
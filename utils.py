import numpy as np
import matplotlib.pyplot as plt

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
    rain = [0.1] * 20 + [0.2]*10 + [0.4]*10 + [0.5]*10 + [0.2]*30 + [0.4]*20
    return rain
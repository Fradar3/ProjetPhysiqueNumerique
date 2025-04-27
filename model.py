# scavatu_model.py

import mesa
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from agent import TerrainAgent

class SCAVATUModel(mesa.Model):
    def __init__(self,
                 width: int,
                 height: int,
                 dem_array: np.ndarray,
                 veg_array: np.ndarray,
                 soil_capacity_array: np.ndarray,
                 soil_infil_thresh_array: np.ndarray,
                 soil_loss_thresh_array: np.ndarray,
                 initial_soil_water_content_array: np.ndarray,
                 pTol: float,  # Lower outflow threshold for erosion
                 pTou: float,  # Upper outflow threshold for erosion
                 pEmax: float, # Max erosion per cell per step
                 pTr: float,   # Run-up normalization threshold for erosion
                 pTvd: float,  # Vegetation density normalization threshold for erosion
                 pTrm: float,  # Run-up threshold of motion (transport/deposition)
                 ptc: float,   # Transport capacity (as fraction of water depth, e.g., 0.2 for 20%)
                 pf: float,    # Head loss due to friction
                 pRr: float,   # Water flow relaxation rate (0 < pRr <= 1)
                 ptmax: float, # Max sediment transport *out* per cell per step
                 rainfall_per_step: list[float] = None
                 ):
        super().__init__()

        self.width = width
        self.height = height
        self.pTol = pTol
        self.pTou = pTou
        self.pEmax = pEmax
        self.pTr = pTr
        self.pTvd = pTvd
        self.pTrm = pTrm
        self.ptc = ptc
        self.pf = pf
        self.pRr = pRr
        self.ptmax = ptmax
        self.rainfall_data = rainfall_per_step if rainfall_per_step is not None else []
        self.steps = 0
        self.grid = mesa.space.SingleGrid(self.width, self.height, torus=False)
        agent_id_counter = 0
        for x in range(self.width):
            for y in range(self.height):
                pos = (x, y)
                altitude = dem_array[y, x]
                vegetation_density = veg_array[y, x]
                infiltration_params = (
                    soil_capacity_array[y, x],
                    soil_infil_thresh_array[y, x],
                    soil_loss_thresh_array[y, x],
                    initial_soil_water_content_array[y, x]
                )
                agent = TerrainAgent(
                    model=self,
                    pos=pos,
                    altitude=altitude,
                    vegetation_density=vegetation_density,
                    infiltration_params=infiltration_params
                )
                agent_id_counter += 1
        for agent in self.agents:
            agent.cache_neighbors()

        self.datacollector = mesa.DataCollector(
            agent_reporters={
                "WaterDepth": "water_depth",
                "RunUp": "run_up",
                "DepositedMaterial": "deposited_material",
                "SedimentInWater": "sediment_in_water",
                "SoilWaterContent": "soil_water_content",
                "Pos" : "pos"
            }
            # rajouter des trackers pour l'érosion totale et les sédiments
        )

        self.datacollector.collect(self)

    def get_rain_at_pos(self, pos: tuple[int, int]) -> float:
        if self.steps < len(self.rainfall_data):
            return self.rainfall_data[self.steps]
        else:
            return 0.0

    def step(self):
        self.agents.do("step")
        self.agents.do("advance")
        self.datacollector.collect(self)
        self.steps += 1
        
    def plot_spatial_variable(self, agent_data_df: pd.DataFrame, step: int, variable_name: str):
        if variable_name not in agent_data_df.columns:
            print(f"Error: Variable '{variable_name}' not found in collected data.")
            print(f"Available variables: {list(agent_data_df.columns)}")
            return

        try:
            step_data = agent_data_df.xs(step, level="Step")
        except KeyError:
            print(f"Error: Data for step {step} not found.")
            print(f"Available steps: {agent_data_df.index.get_level_values('Step').unique().tolist()}")
            return

        grid_data = np.zeros((self.height, self.width))

        for index, row in step_data.iterrows():
            x, y = row["Pos"]
            grid_data[y, x] = row[variable_name]

        plt.figure(figsize=(6, 6))
        plt.imshow(grid_data, origin='lower', cmap='viridis')
        plt.colorbar(label=variable_name)
        plt.title(f"{variable_name} at Step {step}")
        plt.xlabel("X Coordinate")
        plt.ylabel("Y Coordinate")
        plt.gca().invert_yaxis()
        plt.show()

    def plot_temporal_average(self, agent_data_df: pd.DataFrame, variable_name: str):
        if variable_name not in agent_data_df.columns:
            print(f"Error: Variable '{variable_name}' not found in collected data.")
            print(f"Available variables: {list(agent_data_df.columns)}")
            return
        average_data = agent_data_df.groupby("Step")[variable_name].mean()
        plt.figure(figsize=(10, 6))
        average_data.plot()
        plt.title(f"Average {variable_name} Over Time")
        plt.xlabel("Step")
        plt.ylabel(f"Average {variable_name}")
        plt.grid(True)
        plt.show()



if __name__ == '__main__':
    WIDTH = 50
    HEIGHT = 50
    dummy_dem = np.random.rand(HEIGHT, WIDTH) * 100 + 100
    dummy_veg = np.random.rand(HEIGHT, WIDTH) * 0.8
    dummy_soil_cap = np.full((HEIGHT, WIDTH), 50.0)
    dummy_soil_infil = np.full((HEIGHT, WIDTH), 5.0)
    dummy_soil_loss = np.full((HEIGHT, WIDTH), 1.0)
    dummy_soil_wc = np.full((HEIGHT, WIDTH), 10.0)

    for y in range(HEIGHT):
         for x in range(WIDTH):
              dummy_dem[y, x] = 100 + x + (HEIGHT-1-y) * 0.5 + np.sin(x/5) * 5 + np.sin(y/5) * 5
              if 20 < x < 30 and 20 < y < 30:
                   dummy_dem[y, x] -= 20 # A little basin
    params = {
        "pTol": 0.001,
        "pTou": 0.01,
        "pEmax": 0.1,
        "pTr": 0.01,
        "pTvd": 0.5,
        "pTrm": 0.005,
        "ptc": 0.5,
        "pf": 0.001,
        "pRr": 0.1,
        "ptmax": 0.5
    }

    rainfall_scenario = [0.1] * 10 + [0.05] * 10 + [0.0] * 30 # 10 steps light rain, 10 steps lighter rain, then dry

    model = SCAVATUModel(
        width=WIDTH,
        height=HEIGHT,
        dem_array=dummy_dem,
        veg_array=dummy_veg,
        soil_capacity_array=dummy_soil_cap,
        soil_infil_thresh_array=dummy_soil_infil,
        soil_loss_thresh_array=dummy_soil_loss,
        initial_soil_water_content_array=dummy_soil_wc,
        rainfall_per_step=rainfall_scenario,
        **params
    )

    NUM_STEPS = 50
    print(f"Running SCAVATU model for {NUM_STEPS} steps...")
    for i in range(NUM_STEPS):
        model.step()
        if (i + 1) % 10 == 0:
             print(f"Step {i+1} complete.")

    print("\nSimulation complete.")
    agent_data = model.datacollector.get_agent_vars_dataframe()
    model.plot_spatial_variable(agent_data, NUM_STEPS - 1, "DepositedMaterial")
    model.plot_temporal_average(agent_data, "WaterDepth")
    model.plot_temporal_average(agent_data, "DepositedMaterial")
    model.plot_temporal_average(agent_data, "SedimentInWater")
    model.plot_temporal_average(agent_data, "SoilWaterContent")
    plt.show()
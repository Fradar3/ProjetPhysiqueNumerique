import mesa
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from agent import TerrainAgent
from utils import *

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
                 pTol: float,
                 pTou: float,
                 pEmax: float,
                 pTr: float,
                 pTvd: float,
                 pTrm: float,
                 ptc: float,
                 pf: float,
                 pRr: float,
                 ptmax: float,
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
                self.grid.place_agent(agent, pos)
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
                "Pos": "pos",
                "Altitude": "altitude"
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
        plt.imshow(grid_data, origin='lower', cmap='gray_r')
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
    # WIDTH = 100
    # HEIGHT = 100
    alt = load_dem_from_tif("MNTprocess/MNTs/srtm_reprojected_epsg4269.tif")
    HEIGHT, WIDTH = alt.shape
    dummy_dem = alt
    dummy_veg = np.random.rand(HEIGHT, WIDTH) * 0.5
    dummy_soil_cap = np.full((HEIGHT, WIDTH), 3.1)
    dummy_soil_infil = np.full((HEIGHT, WIDTH), 3.0)
    dummy_soil_loss = np.full((HEIGHT, WIDTH), 1.1)
    dummy_soil_wc = np.full((HEIGHT, WIDTH), 5)

    params = {
        "pTol": 0.01,
        "pTou": 0.5,
        "pEmax": 25e-1,
        "pTr": 0.01,
        "pTvd": 0.5,
        "pTrm": 0.005,
        "ptc": 0.2,
        "pf": 0.5,
        "pRr": 0.09,
        "ptmax": 25e-1
    }
    NUM_STEPS = 50
    # rainfall_scenario = pluviogram(NUM_STEPS)
    rainfall_scenario = [1.5] * NUM_STEPS*2
    
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

    print(f"Running SCAVATU model for {NUM_STEPS} steps...")
    for i in range(NUM_STEPS):
        model.step()
        if (i + 1) % 10 == 0:
             print(f"Step {i+1} complete.")

    print("\nSimulation complete.")
    agent_data = model.datacollector.get_agent_vars_dataframe()

    variable_name = "Altitude"
    step0 = agent_data.xs(0, level="Step")
    step50 = agent_data.xs(NUM_STEPS-1, level="Step")

    difference_grid = np.zeros((model.height, model.width))

    for index, row in step0.iterrows():
        x, y = row["Pos"]
        altitude_step0 = row[variable_name]
        altitude_step50 = step50.loc[index][variable_name]
        difference = altitude_step50 - altitude_step0
        difference_grid[y, x] = difference

    plt.figure(figsize=(8, 6))
    plt.imshow(difference_grid, origin='lower', cmap='coolwarm')
    plt.colorbar(label="Δ Altitude (Step 50 - Step 0)")
    plt.title("Changement d'Altitude entre Step 0 et Step 50")
    plt.xlabel("X Coordinate")
    plt.ylabel("Y Coordinate")
    plt.gca().invert_yaxis()
    plt.show()
    create_spatial_gif(
        agent_data_df=agent_data,
        model_width=WIDTH,
        model_height=HEIGHT,
        variable_name="Altitude",
        output_filename="Altitude_1.gif",
        fps=15,
        cmap="gray_r",
        title_prefix=""
    )
    # s = [5, 21, 51, 71, 91]
    # for i in s:
    #     model.plot_spatial_variable(agent_data, i, "WaterDepth")
    # model.plot_temporal_average(agent_data, "WaterDepth")
    # model.plot_temporal_average(agent_data, "Altitude")
    # plt.show()
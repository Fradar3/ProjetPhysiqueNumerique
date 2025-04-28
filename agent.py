import numpy as np
from mesa import Agent, Model

class TerrainAgent(Agent):
    def __init__(self, model, pos, altitude, vegetation_density, infiltration_params):
        """
        Initializes a new TerrainAgent.

        Args:
            model (mesa.Model): The simulation model.
            pos (tuple): The agent's position on the grid (x, y).
            altitude (float): The initial altitude of the cell.
            vegetation_density (float): The initial vegetation density (0.0 to 1.0).
            infiltration_params (tuple): Tuple containing (capacity, infil_thresh, loss_thresh, initial_water_content).
        """
        super().__init__(model)
        self.pos = pos
        self.neighbors = []
        self.altitude = altitude
        self.vegetation_density = vegetation_density
        self.water_depth = 0.0
        self.run_up = 0.0
        self.sediment_in_water = 0.0
        self.deposited_material = 0.0
        self.soil_capacity, self.soil_infil_thresh, self.soil_loss_thresh, self.soil_water_content = infiltration_params
        self.water_outflow_to = {}
        self.sediment_outflow_to = {}
        self.water_inflow_from = {}
        self.sediment_inflow_from = {}
        self.next_state = {}
        self._current_step_eroded_material = 0.0

    def cache_neighbors(self):
        self.neighbors = self.model.grid.get_neighbors(
            self.pos,
            moore=False,       # Von Neumann neighbors
            include_center=False
        )

    def get_total_head(self):
        return self.altitude + self.run_up

    def calculate_infiltration(self):
        # T1
        receptivity = self.soil_capacity - self.soil_water_content
        infiltration = max(0, min(self.water_depth, self.soil_infil_thresh, receptivity))
        water_loss = max(0, min(self.soil_water_content, self.soil_loss_thresh))
        delta_wd_infiltration = -infiltration
        delta_wc = infiltration - water_loss

        return delta_wd_infiltration, delta_wc

    def calculate_erosion(self, total_outflow):
        # T2
        E = 0.0
        O = total_outflow
        if O > self.model.pTou:
            E = self.model.pEmax
        elif O > self.model.pTol:
            On = (O / self.model.pTou) * (np.pi / 2) if self.model.pTou > 0 else np.pi / 2
            if self.run_up > self.model.pTr:
                 rn = np.pi / 2
            elif self.model.pTr > 0:
                 rn = (self.run_up / self.model.pTr) * (np.pi / 2)
            else:
                 rn = 0
            if self.vegetation_density > self.model.pTvd:
                 vdn = np.pi / 2
            elif self.model.pTvd > 0:
                 vdn = (self.vegetation_density / self.model.pTvd) * (np.pi / 2)
            else:
                 vdn = 0
            On = np.clip(On, 0, np.pi / 2)
            rn = np.clip(rn, 0, np.pi / 2)
            vdn = np.clip(vdn, 0, np.pi / 2)
            E = self.model.pEmax * np.sin(On) * np.sin(rn) * np.cos(vdn)
            E = max(0.0, E)
        return E

    def calculate_transport_deposition(self, water_depth_after_flows, eroded_material, incoming_sediment):
        # T3
        available_sediment = self.deposited_material + eroded_material + incoming_sediment
        transport_capacity = self.model.ptc * water_depth_after_flows
        sediment_potential_for_outflow = 0.0
        next_deposited_material = self.deposited_material
        if self.run_up < self.model.pTrm:
            next_deposited_material = available_sediment
            sediment_potential_for_outflow = 0.0
        else:
            if available_sediment <= transport_capacity:
                next_deposited_material = 0.0
                sediment_potential_for_outflow = available_sediment
            else:
                next_deposited_material = available_sediment - transport_capacity
                sediment_potential_for_outflow = transport_capacity

        return sediment_potential_for_outflow, next_deposited_material


    def minimize_H(self):
        central_total_head = self.get_total_head()
        potential_recipients = [n for n in self.neighbors if n.get_total_head() <= central_total_head]
        if not potential_recipients:
            return {}

        remaining_neighbors = list(potential_recipients)
        equilibrium_head = None
        while True:
            if not remaining_neighbors:
                 break

            head_sum = central_total_head + sum(n.get_total_head() for n in remaining_neighbors)
            num_cells = 1 + len(remaining_neighbors)
            avg_head = head_sum / num_cells

            next_remaining_neighbors = [n for n in remaining_neighbors if n.get_total_head() <= avg_head]

            if len(next_remaining_neighbors) == len(remaining_neighbors):
                equilibrium_head = avg_head
                break
            else:
                remaining_neighbors = next_remaining_neighbors

        delta_H_map = {}
        if equilibrium_head is not None:
            for neighbor in remaining_neighbors:
                 delta_h_neighbor = equilibrium_head - neighbor.get_total_head()
                 delta_H_map[neighbor] = max(0.0, delta_h_neighbor)

        return delta_H_map


    def calculate_water_outflows(self, delta_wd_infiltration):
        # I1 pour l'eau
        delta_H_map = self.minimize_H()
        wd_after_rain = self.water_depth + self.model.get_rain_at_pos(self.pos)
        wd_available_for_outflow = wd_after_rain + delta_wd_infiltration
        wd_available_for_outflow = max(0.0, wd_available_for_outflow)
        water_outflow_to = {}
        total_water_outflow = 0.0
        r0 = self.run_up
        total_delta_H = sum(delta_H_map.values())

        if r0 > 1e-9 and total_delta_H > 1e-9 and wd_available_for_outflow > 1e-9:
            for neighbor, delta_h in delta_H_map.items():
                 outflow_j = wd_available_for_outflow * (delta_h / r0) * self.model.pRr
                 water_outflow_to[neighbor] = outflow_j
                 total_water_outflow += outflow_j

        total_water_outflow = min(total_water_outflow, wd_available_for_outflow)

        if total_water_outflow < sum(water_outflow_to.values()) and sum(water_outflow_to.values()) > 1e-9:
             scale_factor = total_water_outflow / sum(water_outflow_to.values())
             for n in water_outflow_to:
                 water_outflow_to[n] *= scale_factor

        wd_residual = wd_available_for_outflow - total_water_outflow
        wd_residual = max(0.0, wd_residual)

        return water_outflow_to, total_water_outflow, wd_residual


    def calculate_sediment_outflows(self, sediment_available_for_outflow, water_outflow_to):
        # I1 pour les sédiments
        potential_total_sediment_outflow = 0.0
        potential_sediment_outflows = {}
        total_water_outflow = sum(water_outflow_to.values())

        if total_water_outflow > 1e-9 and sediment_available_for_outflow > 1e-9:
            for neighbor, water_outflow in water_outflow_to.items():
                pot_sed_out = sediment_available_for_outflow * (water_outflow / total_water_outflow)
                potential_sediment_outflows[neighbor] = pot_sed_out
                potential_total_sediment_outflow += pot_sed_out

        actual_total_sediment_outflow = min(potential_total_sediment_outflow,
                                            sediment_available_for_outflow,
                                            self.model.ptmax)

        scaling_factor = 1.0
        if potential_total_sediment_outflow > 1e-9:
            scaling_factor = actual_total_sediment_outflow / potential_total_sediment_outflow

        final_sediment_outflow_to = {}
        total_sediment_outflow = 0.0
        for neighbor, pot_sed_out in potential_sediment_outflows.items():
            final_sed_out = pot_sed_out * scaling_factor
            final_sediment_outflow_to[neighbor] = final_sed_out
            total_sediment_outflow += final_sed_out

        return final_sediment_outflow_to, total_sediment_outflow


    def calculate_new_runup(self, wd_residual):
        # I2
        water_inflow_from = {}
        sediment_inflow_from = {}
        total_water_inflow_q = 0.0

        for neighbor in self.neighbors:
             inflow_w = neighbor.water_outflow_to.get(self, 0.0)
             inflow_s = neighbor.sediment_outflow_to.get(self, 0.0)

             water_inflow_from[neighbor] = inflow_w
             sediment_inflow_from[neighbor] = inflow_s
             total_water_inflow_q += inflow_w

        next_water_depth = wd_residual + total_water_inflow_q
        next_water_depth = max(0.0, next_water_depth)
        wdr = wd_residual
        H_current = self.get_total_head()
        f = self.model.pf
        sum_qH = 0.0
        for neighbor, q_i in water_inflow_from.items():
             sum_qH += q_i * neighbor.get_total_head()

        denominator = next_water_depth
        calculated_r_new = 0.0
        if denominator > 1e-9:
             avg_head_term = (wdr * H_current + sum_qH) / denominator
             calculated_r_new = avg_head_term - self.altitude - f

        next_run_up = max(0.0, calculated_r_new)
        next_run_up = max(next_run_up, next_water_depth)

        return next_water_depth, next_run_up, water_inflow_from, sediment_inflow_from


    def step(self):
        self.water_outflow_to = {}
        self.sediment_outflow_to = {}
        self.water_inflow_from = {}
        self.sediment_inflow_from = {}
        self._current_step_eroded_material = 0.0
        self.next_state = {}
        self.next_state['altitude'] = self.altitude
        delta_wd_infiltration, delta_wc = self.calculate_infiltration()
        next_soil_water_content = max(0.0, self.soil_water_content + delta_wc)
        self.next_state['soil_water_content'] = next_soil_water_content
        water_outflow_to, total_water_outflow, wd_residual = self.calculate_water_outflows(delta_wd_infiltration)
        self.water_outflow_to = water_outflow_to
        eroded_material = self.calculate_erosion(total_water_outflow)
        self._current_step_eroded_material = eroded_material
        next_water_depth, next_run_up, water_inflow_from, sediment_inflow_from = self.calculate_new_runup(wd_residual)
        self.next_state['water_depth'] = next_water_depth
        self.next_state['run_up'] = next_run_up
        self.water_inflow_from = water_inflow_from
        self.sediment_inflow_from = sediment_inflow_from
        total_incoming_sediment = sum(self.sediment_inflow_from.values())
        sediment_potential_for_outflow, next_deposited_material = self.calculate_transport_deposition(
             next_water_depth,
             self._current_step_eroded_material,
             total_incoming_sediment
        )
        self.next_state['deposited_material'] = next_deposited_material
        net_mass_change = (self.next_state["deposited_material"]-self.deposited_material-self._current_step_eroded_material)
        delta_altitude = net_mass_change / 1.3 # C'est la densité du sol en kg/m³
        next_altitude = self.altitude + delta_altitude
        self.next_state["altitude"] = next_altitude
        
        
        sediment_outflow_to, total_sediment_outflow = self.calculate_sediment_outflows(
            sediment_potential_for_outflow,
            water_outflow_to
        )
        self.sediment_outflow_to = sediment_outflow_to
        self.next_state['sediment_in_water'] = sediment_potential_for_outflow - total_sediment_outflow
        self.next_state['sediment_in_water'] = max(0.0, self.next_state['sediment_in_water'])

    def advance(self):
        self.water_depth = self.next_state.get('water_depth', self.water_depth)
        self.run_up = self.next_state.get('run_up', self.run_up)
        self.sediment_in_water = self.next_state.get('sediment_in_water', self.sediment_in_water)
        self.deposited_material = self.next_state.get('deposited_material', self.deposited_material)
        self.soil_water_content = self.next_state.get('soil_water_content', self.soil_water_content)
        self.altitude = self.next_state.get('altitude', self.altitude)
        self.next_state = {}
        self._current_step_eroded_material = 0.0

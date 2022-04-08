import logging

from sage.all import RIF
import sage.all as sg
import numpy as np
from influxdb_client import InfluxDBClient
from influxdb_client.client.write_api import SYNCHRONOUS
from oomodelling import ModelSolver

from digital_twin.data_access.dbmanager.incubator_data_conversion import convert_to_results_db
from incubator.models.plant_models.model_functions import create_lookup_table
from verified_twin.incubator_models import SwitchingFourParameterModel
from verified_twin.controllers import SignalArraySwitchedController
from verified_twin.simulators import HybridSimulator

class VerifiedPlantSimulator4Params:
    def run_simulation(self, timespan_seconds, initial_box_temperature, initial_heat_temperature,
                       room_temperature, heater_on,
                       C_air, G_box, C_heater, G_heater):

        timetable = np.array(timespan_seconds)
        # Need to feed room temp into box as signal: this will force a small timestep
        # but I will leave it for now for testing purposes!
        heater_on_arr = np.array(heater_on)

        model = SwitchingFourParameterModel([RIF(timespan_seconds[0]), initial_heat_temperature, initial_box_temperature],
            T_R=room_temperature[0],
            C_air=C_air, G_box=G_box, C_heater=C_heater, G_heater=G_heater)

        controller = SignalArraySwitchedController(
            {'heater_on': False},
            timetable,
            {'heater_on': heater_on_arr},
        )

        simulator = HybridSimulator(model, controller,
            controller_input_map=(lambda x: x[0]),
            controller_output_map=(lambda xin, x: xin),
        )

        trace = list(simulator.run(time_limit=RIF(timespan_seconds[-1])))

        return trace, model

from typing import List, Tuple

from sage.all import RIF
import sage.all as sg
import numpy as np
from copy import deepcopy
from concurrent.futures import ProcessPoolExecutor

from verified_twin.incubator_models import SwitchingFourParameterModel
from verified_twin.controllers import SignalArraySwitchedController
from verified_twin.simulators import HybridSimulator, VerifiedContinuousSimulator
from verified_twin.traces import VerifiedHybridTrace
from verified_twin.incubator_models import fourpincubator
from verified_twin.lbuc import Signal


class VerifiedPlantMonitor4Params:
    def __init__(self, properties):
        self.executor = ProcessPoolExecutor()
        self.properties = properties

    def verified_monitoring_results(self,
            timespan_seconds, initial_box_temperature, initial_heat_temperature,
            room_temperature, heater_on,
            C_air, G_box, C_heater, G_heater) -> \
            List[Signal]:
        fut = self.executor.submit(
            self.run_and_monitor,
            self.properties,
            timespan_seconds,
            initial_box_temperature,
            initial_heat_temperature,
            room_temperature,
            heater_on,
            C_air,
            G_box,
            C_heater,
            G_heater,
        )
        return fut.result()

    @classmethod
    def run_and_monitor(cls, properties,
        timespan_seconds, initial_box_temperature,
        initial_heat_temperature,
        room_temperature, heater_on,
        C_air, G_box, C_heater, G_heater) -> List[Signal]:
        trace, model = cls.run_simulation(
            timespan_seconds, initial_box_temperature,
            initial_heat_temperature, room_temperature,
            heater_on,
            C_air, G_box, C_heater, G_heater,
        )
        monitoring_results = [
            prop.signal(trace).G(-timespan_seconds[0])
                for prop in properties
        ]
        return monitoring_results

    @staticmethod
    def run_simulation(
            timespan_seconds, initial_box_temperature, initial_heat_temperature,
            room_temperature, heater_on,
            C_air, G_box, C_heater, G_heater) -> \
            Tuple[VerifiedHybridTrace, SwitchingFourParameterModel]:
        timetable = np.array(deepcopy(timespan_seconds))
        # Need to feed room temp into box as signal: this will force a small timestep
        # but I will leave it for now for testing purposes!
        heater_on_arr = np.array(deepcopy(heater_on))

        model = SwitchingFourParameterModel([RIF(deepcopy(timespan_seconds[0])), deepcopy(initial_heat_temperature), deepcopy(initial_box_temperature)],
             T_R=deepcopy(room_temperature[0]),
             C_air=deepcopy(C_air), G_box=deepcopy(G_box),
             C_heater=deepcopy(C_heater), G_heater=deepcopy(G_heater))

        controller = SignalArraySwitchedController(
            {'heater_on': False},
            timetable,
            {'heater_on': heater_on_arr},
        )

        simulator = HybridSimulator(model, controller,
            controller_input_map=(lambda x: x[0]),
            controller_output_map=(lambda xin, x: xin),
        )

        import time
        t1 = time.time()
        trace = simulator.run(time_limit=(RIF(deepcopy(timespan_seconds[-1])) - RIF(deepcopy(timespan_seconds[0]))))
        t2 = time.time()
        print(f"ran verified simulation in {t2 - t1} sec")
        # from time import sleep
        # sleep(30)

        return trace, model # type: ignore
        # return None, model # type: ignore

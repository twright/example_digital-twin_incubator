from curses.ascii import ctrl
from typing import List, Optional, Tuple

from sage.all import RIF
import sage.all as sg
import numpy as np
from copy import deepcopy
from concurrent.futures import ProcessPoolExecutor

from verified_twin.incubator_models import SwitchingFourParameterModelCAGB
from verified_twin.controllers import PeriodicOpenLoopController, SignalArraySwitchedController
from verified_twin.simulators import HybridSimulator, VerifiedContinuousSimulator
from verified_twin.traces import VerifiedHybridTrace
from verified_twin.incubator_models import fourpincubator
from verified_twin.lbuc import Signal, Logic


class VerifiedPlantMonitor4Params:
    def __init__(self, properties: List[Logic]):
        self.executor = ProcessPoolExecutor(max_workers=1)
        self.properties = properties

    def verified_monitoring_results(self,
            tstart, tend, initial_box_temperature, initial_heat_temperature,
            initial_room_temperature, ctrl_step_size, n_samples_period,
            n_samples_heating,
            C_air, G_box, C_heater, G_heater) -> \
            List[Signal]:
        prop_duration = max(prop.duration for prop in self.properties)
        fut = self.executor.submit(
            self.run_and_monitor,
            self.properties,
            tstart,
            tend + prop_duration,
            initial_box_temperature, initial_heat_temperature,
            initial_room_temperature, ctrl_step_size, n_samples_period,
            n_samples_heating,
            C_air, G_box, C_heater, G_heater,
        )
        return fut.result()

    @classmethod
    def run_and_monitor(cls, properties,
        tstart, tend, initial_box_temperature, initial_heat_temperature,
            initial_room_temperature, ctrl_step_size, n_samples_period,
            n_samples_heating,
            C_air, G_box, C_heater, G_heater) -> List[Tuple[Optional[bool], Signal]]:
        trace, _ = cls.run_simulation(
            tstart, tend, initial_box_temperature, initial_heat_temperature,
            initial_room_temperature, ctrl_step_size, n_samples_period,
            n_samples_heating,
            C_air, G_box, C_heater, G_heater,
        )
        signals = [
            prop.signal(trace)
                   for prop in properties
        ]
        monitoring_results = [
            (sig(0), sig.G(-tstart)) for sig in signals
        ]
        return monitoring_results

    @staticmethod
    def run_simulation(
            tstart, tend, initial_box_temperature, initial_heat_temperature,
            initial_room_temperature, ctrl_step_size, n_samples_period,
            n_samples_heating,
            C_air, G_box, C_heater, G_heater) -> \
            Tuple[VerifiedHybridTrace, SwitchingFourParameterModelCAGB]:
        # Need to feed room temp into box as signal: this will force a small timestep
        # but I will leave it for now for testing purposes!
        # Better yet, can we make this an interval?

        model = SwitchingFourParameterModelCAGB([RIF(tstart),
            RIF(initial_heat_temperature),
            RIF(initial_box_temperature),
            RIF(C_air), RIF(G_box)],
             T_R=RIF(initial_room_temperature),
             C_H=RIF(C_heater), G_H=RIF(G_heater))

        controller = PeriodicOpenLoopController(
            ctrl_step_size,
            n_samples_period,
            n_samples_heating,
        )

        simulator = HybridSimulator(model, controller,
            controller_input_map=(lambda x: x[0]),
            controller_output_map=(lambda xin, x: xin),
        )

        import time
        t1 = time.time()
        trace = simulator.run(time_limit=(RIF(tend) - RIF(tstart)))
        t2 = time.time()
        print(f"ran verified simulation in {t2 - t1} sec")
        # from time import sleep
        # sleep(30)

        return trace, model # type: ignore
        # return None, model # type: ignore

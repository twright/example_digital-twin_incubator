import logging
from concurrent.futures import ProcessPoolExecutor

import numpy as np
from scipy.optimize import minimize
from sage.all import RIF
import sage.all as sg

from incubator.interfaces.database import IDatabase
from verified_twin.incubator_models import SwitchingFourParameterModelCAGB
from verified_twin.controllers import SignalArraySwitchedController
from verified_twin.simulators import HybridSimulator


class UncertaintyCalibrationSystem:
    def __init__(self, times, T_A, T_H, C_air, G_box, C_heater, G_heater, room_T, ctrl_signal):
        self.controller = SignalArraySwitchedController(
            {'heater_on': False},
            times,
            {'heater_on': ctrl_signal},
        )
        self.tstart = RIF(times[0])
        self.tend = RIF(times[-1])
        self.T_A = RIF(T_A)
        self.T_H = RIF(T_H)
        self.C_air = RIF(C_air)
        self.G_box = RIF(G_box)
        self.C_heater = RIF(C_heater)
        self.G_heater = RIF(G_heater)
        self.T_R = RIF(room_T[0])

    def spread_parameters(self, T_H_spread, T_A_spread, C_air_spread, G_box_spread):
        return (
            self.T_H + RIF(-T_H_spread,T_H_spread),
            self.T_A + RIF(-T_A_spread,T_A_spread),
            self.C_air + RIF(-C_air_spread, C_air_spread),
            self.G_box + RIF(-G_box_spread, G_box_spread),
        )

    def model(self, T_H_spread, T_A_spread, C_air_spread, G_box_spread):
        return SwitchingFourParameterModelCAGB(
            [
                self.tstart,
                *self.spread_parameters(T_H_spread, T_A_spread, C_air_spread,
                                        G_box_spread),
            ],
            T_R=self.T_R,
            C_H=self.C_heater,
            G_H=self.G_heater,
        )

    def simulator(self, T_H_spread, T_A_spread, C_air_spread, G_box_spread):
        return HybridSimulator(
            self.model(T_H_spread, T_A_spread, C_air_spread, G_box_spread),
            self.controller,
            controller_input_map=(lambda x: x[0]),
            controller_output_map=(lambda xin, x: xin),
        )

    def verified_trace(self, T_H_spread, T_A_spread, C_air_spread, G_box_spread):
        return self.simulator(T_H_spread, T_A_spread, C_air_spread, G_box_spread).run(
            start_time=self.tstart,
            time_limit=(self.tend - self.tstart),
        )


def euclidian_norm(xs):
    return sum(x**2 for x in xs)


def inner_dist_from_int(x, I):
    print(f"computing inner dist of {x} and {I.str(style='brackets')}")
    if x > I.upper():
        return (RIF(x) - I)
    elif x < I.lower():
        return (I - RIF(x))
    else:
        return RIF(0)


def eval_trace(tr, t):
    y = None
    t0 = tr.domain.edges()[0]
    
    for r in tr.values:
        print(f"t0 = {t0.str(style='brackets')}, t = {t.str(style='brackets')}, r.time = {r.time}")
        if t.overlaps(t0 + RIF(0, r.time)):
            print("overlap!")
            y = tr.interval_list_union(r(t - t0), y)
            print(f"y = {[yi.str(style='brackets') for yi in y]}")
        t0 += RIF(r.time)

    return y


def violation_degree(sol, trace):
    times = sol.y[0]
    durations = [RIF(t1) for t1 in times]
    intervals = [eval_trace(trace.continuous_part, duration) for duration in durations]
    #print(f"intervals = {[(i.str(style='brackets') if i else None) for i in intervals]}")
    try:
        return sum(
            euclidian_norm(inner_dist_from_int(y, interval[i])
                for y, interval in zip(ys, intervals))
            for i, ys in enumerate(reversed(sol.y[1:]), 1)
        ).lower()
    except TypeError:
        return 1


class UncertaintyCalibrationProblem:
    def __init__(self, sol, system: UncertaintyCalibrationSystem):
        self.sol = sol
        self.system = system

    def cost(self, p):
        T_H_spread, T_A_spread, C_air_spread, G_box_spread = p
        trace = self.system.verified_trace(*p)
        
        return (2**5*violation_degree(self.sol, trace)
            + euclidian_norm(trace(trace.domain.edges()[1])[1:3]).upper())

    def solution_raw(self, method='Nelder-Mead',
            options={'maxiter': 10, 'xatol': 0.1, 'fatol': 1}):
        return minimize(self.cost, np.array([1, 1, 1, 1]), method=method,
            options=options)

    def solution(self, *args, **kwargs):
        return self.system.spread_parameters(*self.solution_raw(*args, **kwargs).x)


class UncertaintyCalibrator:
    def __init__(self, database: IDatabase, plant_simulator, conv_xatol, conv_fatol, max_iterations):
        self._l = logging.getLogger("Calibrator")
        self.executor = ProcessPoolExecutor()
        self.database = database
        self.plant_simulator = plant_simulator
        self.conv_xatol = conv_xatol
        self.conv_fatol = conv_fatol
        self.max_iterations = max_iterations

    @staticmethod
    def run_calibration(problem: UncertaintyCalibrationProblem):
        return problem.solution()

    def calibrate(self, tstart, tend, C_air, G_box, C_heater, G_heater):
        # Get simulation data from the calibration database
        signals, t_start_idx, t_end_idx = self.database.get_plant_signals_between(tstart, tend)
        times = signals["time"][t_start_idx:t_end_idx]
        reference_T = signals["T"][t_start_idx:t_end_idx]
        ctrl_signal = signals["in_heater_on"][t_start_idx:t_end_idx]
        reference_T_heater = signals["T_heater"][t_start_idx:t_end_idx]
        room_T = signals["in_room_temperature"][t_start_idx:t_end_idx]

        # Get a reference sol by running the plant model  
        sol, model = self.plant_simulator.run_simulation(
            times, reference_T[0], reference_T_heater[0], room_T, ctrl_signal,
            C_air, G_box, C_heater, G_heater)

        # Define the UncertaintyCalibrationProblem
        prob = UncertaintyCalibrationProblem(
            sol,
            UncertaintyCalibrationSystem(
                times, reference_T[0], reference_T_heater[0],
                C_air, G_box, C_heater, G_heater, room_T, ctrl_signal,
            )
        )
        
        # Use executor to run the actual simulation and return the result
        fut = self.executor.submit(
            self.run_calibration,
            prob,
        )
        return fut.result()


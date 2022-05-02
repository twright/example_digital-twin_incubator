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
from lbuc import FlowstarFailedException


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

    def verified_trace(self, T_H_spread, T_A_spread, C_air_spread, G_box_spread, extra_time=RIF(0.0)):
        return self.simulator(T_H_spread, T_A_spread, C_air_spread, G_box_spread).run(
            start_time=self.tstart,
            time_limit=(self.tend - self.tstart + extra_time),
        )


def euclidian_norm(xs):
    return sum(x**2 for x in xs)


def inner_dist_from_int(x, I):
    # print(f"computing inner dist of {x} and {I.str(style='brackets')}")
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
        # print(f"t0 = {t0.str(style='brackets')}, t = {t.str(style='brackets')}, r.time = {r.time}")
        if t.overlaps(t0 + RIF(0, r.time)):
            # print("overlap!")
            y = tr.interval_list_union(r(t - t0), y)
            # print(f"y = {[yi.str(style='brackets') for yi in y]}")
        t0 += RIF(r.time)

    return y


def violation_degree(data, trace):
    times = data[0]
    durations = [RIF(t1) for t1 in times]
    try:
        intervals = [eval_trace(trace.continuous_part, duration) for duration in durations]
    #print(f"intervals = {[(i.str(style='brackets') if i else None) for i in intervals]}")
        return sum(
            euclidian_norm(inner_dist_from_int(y, interval[i])
                for y, interval in zip(ys, intervals))
            for i, ys in enumerate(reversed(data[1:]), 1)
        ).lower()
    except TypeError:
        return 1


class UncertaintyCalibrationProblem:
    def __init__(self, data, system: UncertaintyCalibrationSystem):
        self.data = data
        self.system = system

    def cost(self, p):
        try:
            trace = self.system.verified_trace(*p)
        except FlowstarFailedException:
            return 2**10

        # return violation_degree(self.data, trace)
        return (2**5*violation_degree(self.data, trace)
              + euclidian_norm(trace(trace.domain.edges()[1] + RIF("[-0.01,0.01]"))[1:3]).upper())

    def solution_raw(self, x0=np.array([0.2,0.2,0.2,0.2]), method='Nelder-Mead',
            options={'maxiter': 10, 'xatol': 0.1, 'fatol': 1}):
        return minimize(self.cost, x0, method=method,
            options=options)

    def solution(self, *args, **kwargs):
        return self.system.spread_parameters(*self.solution_raw(*args, **kwargs).x)

    def solution_final(self, *args, **kwargs):
        # Solution for model starting after the end of the calibration period

        # Solution for calibration period
        spreads = self.solution_raw(*args, **kwargs).x
        (T_H0, T_A0, C_air, G_box) = self.system.spread_parameters(*spreads)

        # Run again to find final values
        tr = self.system.verified_trace(*spreads, extra_time=RIF(6.0))
        T_H, T_A = eval_trace(
            tr.continuous_part,
            tr.domain.edges()[1] + RIF("[-0.01,0.01]")
        )[1:3]
        
        return (T_H0, T_A0, T_H, T_A, C_air, G_box)


class UncertaintyCalibrator:
    def __init__(self, database: IDatabase):
        self._l = logging.getLogger("Calibrator")
        self.executor = ProcessPoolExecutor(max_workers=1)
        self.database = database

    @staticmethod
    def run_calibration(problem: UncertaintyCalibrationProblem):
        return problem.solution_final()

    def calibrate(self, tstart, tend, C_air, G_box, C_heater, G_heater):
        # Get simulation data from the calibration database
        signals, t_start_idx, t_end_idx = self.database.get_plant_signals_between(tstart, tend)
        times = signals["time"][t_start_idx:t_end_idx]
        reference_T = signals["T"][t_start_idx:t_end_idx]
        ctrl_signal = signals["in_heater_on"][t_start_idx:t_end_idx]
        reference_T_heater = signals["T_heater"][t_start_idx:t_end_idx]
        room_T = signals["in_room_temperature"][t_start_idx:t_end_idx]

        # Define the UncertaintyCalibrationProblem
        prob = UncertaintyCalibrationProblem(
            (times, reference_T, reference_T_heater),
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


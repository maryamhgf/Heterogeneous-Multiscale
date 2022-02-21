from .solvers import FixedGridODESolver
from .rk_common import rk4_alt_step_func
from .misc import Perturb
import torch
import numpy as np
import time


class Euler(FixedGridODESolver):
    order = 1

    def _step_func(self, func, t0, dt, t1, y0):
        f0 = func(t0, y0, perturb=Perturb.NEXT if self.perturb else Perturb.NONE)
        return dt * f0, f0


class Midpoint(FixedGridODESolver):
    order = 2

    def _step_func(self, func, t0, dt, t1, y0):
        half_dt = 0.5 * dt
        f0 = func(t0, y0, perturb=Perturb.NEXT if self.perturb else Perturb.NONE)
        y_mid = y0 + f0 * half_dt
        return dt * func(t0 + half_dt, y_mid), f0


class RK4(FixedGridODESolver):
    order = 4

    def _step_func(self, func, t0, dt, t1, y0):
        f0 = func(t0, y0, perturb=Perturb.NEXT if self.perturb else Perturb.NONE)
        return rk4_alt_step_func(func, t0, dt, t1, y0, f0=f0, perturb=self.perturb), f0

#for fast var
def grid_constructor(step_size, func, y0, start_time, end_time):

    niters = torch.ceil((end_time - start_time) / step_size + 1).item()
    #print(niters)
    #print("---", (end_time - start_time))
    #print("---2: ", (end_time - start_time) / step_size + 1)
    t_infer = torch.arange(0, niters, dtype=start_time.dtype, device=start_time.device) * step_size + start_time
    t_infer[-1] = end_time
    #print("t_infer: ", t_infer)
    #print("-----end")
    return t_infer

class HMM(FixedGridODESolver):
    order = 1
    #fast and slow var have common t not value
    def _step_func(self, func, t0, dt, t1, y0, func_fast=None, dt_fast=None, y0_fast=None, t_fast_interval=None, 
        batch_time=None):
        def _fast_step_func(func, t0, dt, t1, y0_fast, y0):
            half_dt = 0.5 * dt
            f0 = func(t0, y0_fast, y0)
            y_mid = y0_fast.reshape(len(y0_fast), 1) + f0 * half_dt
            return dt *  func(t0 + half_dt, y_mid, y0), f0
        
        def _fast_integrate(y0_fast):
            start = time.time()
            #start point of slow and fast var are the same.
            #start and end time for fast variable
            time_grid = t_fast_interval
            solution_fast = torch.empty(len(time_grid), *(y0_fast.T)[0].shape, dtype=y0_fast.dtype, device=y0_fast.device)
            solution_fast[0] = (y0_fast.T)[0]
            _y0 = y0_fast.T[0]
            j = 1
            #solving fast part
            fast_sample_num = 0
            for t0_sample, t1_sample in zip(time_grid[:-1], time_grid[1:]):
                dt = t1_sample - t0_sample

                dy, f0 = _fast_step_func(func_fast, t0_sample, dt, t1_sample, _y0, y0)
                dy = dy.reshape(len(dy))
                y1 = _y0 + dy
                solution_fast[j] =  y1
                j += 1
                _y0 = y1
                fast_sample_num += 1
            timing = time.time() - start
            return solution_fast, timing
            
        solution_fast, timing = _fast_integrate(y0_fast)
        f = 0
        
        avg_solution = torch.sum(solution_fast, 0)/int(solution_fast.shape[0])
        f0 = func(t0, y0, avg_solution)
        half_dt = 0.5 * dt
        y_mid = y0.reshape(len(y0), 1) + f0 * half_dt
        f = f + f0
        return dt * func(t0+half_dt, y_mid, avg_solution), f0, solution_fast, timing
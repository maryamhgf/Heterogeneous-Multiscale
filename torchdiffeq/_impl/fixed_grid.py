from .solvers import FixedGridODESolver
from .rk_common import rk4_alt_step_func
from .misc import Perturb
import torch
import numpy as np


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
    t_infer = torch.arange(0, niters, dtype=start_time.dtype, device=start_time.device) * step_size + start_time
    t_infer[-1] = end_time
    return t_infer

class HMM(FixedGridODESolver):
    order = 1
    #fast and slow var have common t not value
    def _step_func(self, func, t0, dt, t1, y0, func_fast=None, dt_fast=None, y0_fast=None):
        
        def _fast_step_func(func, t0, dt, t1, y0_fast, y0):
            f0 = func(t0, y0_fast, y0)
            return dt * f0, f0
        
        def _fast_integrate(y0_fast):
            #start point of slow and fast var are the same.
            #start and end time for fast variable
            t0_fast = t0
            t1_fast = t0_fast +  dt_fast
            time_grid = grid_constructor(dt_fast, func_fast, y0_fast, t0_fast, t1_fast)
            assert time_grid[0] == t0_fast and time_grid[-1] == t1_fast
            solution_fast = torch.empty(len(time_grid), *y0_fast.shape, dtype=y0_fast.dtype, device=y0_fast.device)
            solution_fast[0] = y0_fast
            _y0 = y0_fast
            j = 1
            #solving fast part
            for t0_sample, t1_sample in zip(time_grid[:-1], time_grid[1:]):
                dt = t1_sample - t0_sample
                dy, f0 = _fast_step_func(func_fast, t0_fast, dt, t1_fast, _y0, y0)
                dy = dy.reshape(len(dy))
                y1 = _y0 + dy
                '''
                if self.interp == "linear":
                    solution_fast[j] = self._linear_interp(t0, t1, _y0, y1, t1)
                elif self.interp == "cubic":
                    f1 = self.func(t1, y1)
                    solution_fast[j] = self._cubic_hermite_interp(t0, _y0, f0, t1, y1, f1, t1)
                '''
                solution_fast[j] =  y1
                j += 1
                _y0 = y1
            return solution_fast
        '''
        def fast_var_avg(solution, k=None):
            number_of_fast_points = len(solution)
            if(k == None):
                k = torch.randn(solution.shape)
            average = int(torch.sum(k * solution)) / len(solution)
            return average
        '''
        solution_fast = _fast_integrate(y0)
        #TO DO: add average as the input
        f = 0
        for i in solution_fast:
            f0 = func(t0, y0, i)
            f = f + f0

        
        return dt * (f/len(solution_fast)), f0/len(solution_fast), solution_fast
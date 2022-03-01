from .solvers import FixedGridODESolver
from .rk_common import rk4_alt_step_func
from .misc import Perturb
import torch
import numpy as np
<<<<<<< HEAD
from scipy.ndimage import gaussian_filter1d
=======
import time
>>>>>>> master


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
<<<<<<< HEAD
def grid_constructor(step_size, start_time, sampling_rate):
    # niters = torch.ceil((end_time - start_time) / step_size + 1).item()
    t_infer = torch.arange(0, sampling_rate, dtype=start_time.dtype, device=start_time.device) * step_size + start_time
    # t_infer[-1] = end_time
=======
def grid_constructor(step_size, func, y0, start_time, end_time):

    niters = torch.ceil((end_time - start_time) / step_size + 1).item()
    #print(niters)
    #print("---", (end_time - start_time))
    #print("---2: ", (end_time - start_time) / step_size + 1)
    t_infer = torch.arange(0, niters, dtype=start_time.dtype, device=start_time.device) * step_size + start_time
    t_infer[-1] = end_time
    #print("t_infer: ", t_infer)
    #print("-----end")
>>>>>>> master
    return t_infer

class HMM(FixedGridODESolver):
    order = 1
    #fast and slow var have common t not value
<<<<<<< HEAD
    def _step_func(self, func, t0, dt, t1, y0, func_fast=None, dt_fast=None, y0_fast=None, sampling_rate=3, kernel='gaussian'):
        
=======
    def _step_func(self, func, t0, dt, t1, y0, func_fast=None, dt_fast=None, y0_fast=None, t_fast_interval=None, 
        batch_time=None):
>>>>>>> master
        def _fast_step_func(func, t0, dt, t1, y0_fast, y0):
            half_dt = 0.5 * dt
            f0 = func(t0, y0_fast, y0)
            y_mid = y0_fast.reshape(len(y0_fast), 1) + f0 * half_dt
            return dt *  func(t0 + half_dt, y_mid, y0), f0
        
<<<<<<< HEAD
        def _fast_integrate(y0_fast, sampling_rate):
            #start point of slow and fast var are the same.
            #start and end time for fast variable
            nfe_fast = 0
            t0_fast = t0
            t1_fast = t0_fast + (sampling_rate - 1) * dt_fast
            # time_grid = grid_constructor(dt_fast, func_fast, y0_fast, t0_fast, t1_fast)
            time_grid = grid_constructor(dt_fast, t0_fast, sampling_rate)
            assert time_grid[0] == t0_fast and abs(time_grid[-1] - t1_fast) <= 0.001
            solution_fast = torch.empty(len(time_grid), *y0_fast.shape, dtype=y0_fast.dtype, device=y0_fast.device)
            solution_fast[0] = y0_fast
            _y0 = y0_fast
=======
        def _fast_integrate(y0_fast):
            start = time.time()
            #start point of slow and fast var are the same.
            #start and end time for fast variable
            time_grid = t_fast_interval
            solution_fast = torch.empty(len(time_grid), *(y0_fast.T)[0].shape, dtype=y0_fast.dtype, device=y0_fast.device)
            solution_fast[0] = (y0_fast.T)[0]
            _y0 = y0_fast.T[0]
>>>>>>> master
            j = 1
            #solving fast part
            fast_sample_num = 0
            for t0_sample, t1_sample in zip(time_grid[:-1], time_grid[1:]):
                dt = t1_sample - t0_sample
<<<<<<< HEAD
                dy, f0 = _fast_step_func(func_fast, t0_fast, dt, t1_fast, _y0, y0)
                nfe_fast += 1
=======

                dy, f0 = _fast_step_func(func_fast, t0_sample, dt, t1_sample, _y0, y0)
>>>>>>> master
                dy = dy.reshape(len(dy))
                y1 = _y0 + dy
                solution_fast[j] =  y1
                j += 1
                _y0 = y1
<<<<<<< HEAD
            return solution_fast, nfe_fast
        '''
        def fast_var_avg(solution, k=None):
            number_of_fast_points = len(solution)
            if(k == None):
                k = torch.randn(solution.shape)
            average = int(torch.sum(k * solution)) / len(solution)
            return average
        '''
        solution_fast, nfe_fast = _fast_integrate(y0, sampling_rate)
        self.nfe_fast += nfe_fast
        #TO DO: add average as the input
        self.nfe_slow += len(solution_fast)

        f = 0
        if kernel == 'uniform':
            for i in solution_fast:
                f0 = func(t0, y0, i)
                f = f + f0

            return dt * (f/len(solution_fast)), f0/len(solution_fast), solution_fast[0]
        elif kernel == 'gaussian':
            f_samples = [func(t0, y0, i) for i in solution_fast]
            tmp = np.zeros(sampling_rate)
            tmp[sampling_rate // 2] = 1
            sigma = 1
            gaussian_filer = gaussian_filter1d(tmp, sigma)
            for i in range(sampling_rate):
                f += gaussian_filer[i] * f_samples[i]

            return dt * f, f, solution_fast[0]
=======
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
>>>>>>> master

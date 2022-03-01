#run export PYTHONPATH="${PYTHONPATH}:/home/mhaghifam/Documents/Research/Neural-ODE/Code/torchdiffeq/torchdiffeq" to be able to import torchdiffeq
import sys 
sys.path.insert(0, '..')
import os
import argparse
import time
import numpy as np
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt

import os
import psutil
import tracemalloc

# inner psutil function
def process_memory():
    process = psutil.Process(os.getpid())
    mem_info = process.memory_info()
    return mem_info.rss
 
# decorator function
def profile(func):
    def wrapper(*args, **kwargs):
 
        mem_before = process_memory()
        result = func(*args, **kwargs)
        mem_after = process_memory()
        print("{}:consumed memory: {:,}".format(
            func.__name__,
            mem_before, mem_after, mem_after - mem_before))
 
        return result
    return wrapper

torch.pi = torch.acos(torch.zeros(1)).item() * 2

parser = argparse.ArgumentParser('ODE demo')
parser.add_argument('--method', type=str, choices=['dopri5', 'adams'], default='dopri5')
parser.add_argument('--data_size', type=int, default=500)
parser.add_argument('--batch_time', type=int, default=100)
parser.add_argument('--fast_batch_time', type=int, default=100)
parser.add_argument('--batch_size', type=int, default=64)
parser.add_argument('--niters', type=int, default=1000)
parser.add_argument('--test_freq', type=int, default=40)
parser.add_argument('--fast_step', type=int, default=10)
parser.add_argument('--viz', action='store_true')
parser.add_argument('--gpu', type=int, default=0)
parser.add_argument('--adjoint', action='store_true')
args = parser.parse_args()
print(args.adjoint)
if args.adjoint:
    from torchdiffeq import odeint_adjoint as odeint
else:
    from torchdiffeq import odeint


# device = torch.device('cuda:' + str(args.gpu) if torch.cuda.is_available() else 'cpu')
device = torch.device('cpu')


def generate_stellar_orbits(a=2, b=3, epslion=0.009):
    ts = torch.linspace(0., 0.03, args.data_size).to(device)
    data = []
    t_prev = ts[0]
    A = torch.tensor([[0., a, 0., 0.], [-a, 0., 0., 0.], [0., 0., 0., b], [0., 0., -b, 0.]])
    y_0 = torch.tensor([[1., 0., 1., 0]])
    y = y_0
    for t in ts:
        dt = t - t_prev
        f = torch.tensor([[0., y[0][1]**2/a, 0., (2*y[0][0]*y[0][1])/b]])
        y = y + dt * ((1/epslion) * A@y.T + f.T).T
        t_prev = t
        data.append(y)
    return data, ts

def extract_modes(data, index_slow, index_fast):
    slow = [item[0][index_slow] for item in data]
    fast = [item[0][index_fast] for item in data]
    return slow, fast

def time_series_sampling(slow, fast, data_t, slow_step_size=None, fast_step_size=None, fast_size=args.fast_step):
    dt = data_t[1] - data_t[0]
    if(slow_step_size == None):
        slow_step_size = dt * 12
    if(fast_step_size == None):
        fast_step_size = dt/3
    slow_step = int(slow_step_size/dt)
    fast_step = int(fast_step_size/dt)
    if(slow_step == 0):
        slow_step = 1
    if(fast_step == 0):
        fast_step = 1
    t_slow = [data_t[i * slow_step] for i in range(int(len(data_t)/slow_step))]
    t_fast = [[data_t[data_t.index(sample_t) + i*fast_step] for i in range(min(fast_size, int(len(data_t) - data_t.index(sample_t)/fast_step)))] for sample_t in t_slow]
    data_slow = [slow[i * slow_step] for i in range(int(len(slow)/slow_step))]
    data_fast = [[fast[data_t.index(sample_t) + i*fast_step] for i in range(min(fast_size, int(len(fast) - data_t.index(sample_t)/fast_step)))] for sample_t in t_slow]
    print(len(t_slow))
    for i in range(len(data_fast)):
        if(len(data_fast[i]) != fast_size):
            for j in range(fast_size - len(data_fast[i])):
                data_fast[i].append(data_fast[i][-1])
                t_fast[i].append(t_fast[i][-1])
    return t_slow, t_fast, torch.tensor(data_slow), torch.tensor(data_fast)

def convert_to_float(lst):
    output = []
    for element in lst:
        output.append(float(element))
    return output

true_y_0 = [torch.tensor([[1., 0., 1., 0]])]
true_y0_slow, true_y0_fast = extract_modes(true_y_0, 0, 3)
data, ts = generate_stellar_orbits()
data_slow, data_fast = extract_modes(data, 0, 2)
t_slow, t_fast, slow, fast = time_series_sampling(data_slow, data_fast, convert_to_float(ts))

def plot_data(fast, slow, t_slow, t_fast, title=None, xlabel=None, ylabel=None):
    plt.figure()
    slow_int = convert_to_float(slow)
    if(len(fast.shape) == 2):
        fast_one_dim = fast[:, :1]
        fast_one_dim.reshape([len(fast)])
        fast = fast_one_dim
    fast_int = convert_to_float(fast)
    plt.plot(t_slow, slow_int, '--go', label="slow")
    plt.plot(t_slow, fast_int, '--bo', label="fast")
    plt.legend(loc="upper left")
    if(title != None):
        plt.title(title)
    if(xlabel != None):
        plt.xlabel(xlabel)
    if(ylabel != None):
        plt.ylabel(ylabel)
    plt.savefig(title+".png")

plot_data(fast, slow, t_slow, t_fast, title="input")
def get_batch(true_y_slow, true_y_fast, ts, ts_fast):
    s = torch.from_numpy(np.random.choice(np.arange(len(ts) - args.batch_time, dtype=np.int64), args.batch_size, replace=False))
    print("true y_fast", true_y_fast.shape, s.shape)
    batch_y0_slow = true_y_slow[s]  # (M, D)
    batch_y0_fast = true_y_fast[s]  # (M, D)
    batch_t = ts[:args.batch_time]  # (T)
    batch_t_fast = ts_fast[:args.batch_time]
    batch_y_slow = torch.stack([true_y_slow[s + i] for i in range(args.batch_time)], dim=0)  # (T, M, D)
    batch_y_fast = torch.stack([true_y_fast[s + i] for i in range(args.fast_batch_time)], dim=0)  # (T, M, D)
    return batch_y0_slow.to(device), batch_y_slow.to(device), batch_y0_fast.to(device), batch_y_fast.to(device), torch.tensor(batch_t).to(device), torch.tensor(batch_t_fast).to(device)

def makedirs(dirname):
    if not os.path.exists(dirname):
        os.makedirs(dirname)


if args.viz:
    makedirs('png')
    import matplotlib.pyplot as plt
    fig = plt.figure(figsize=(12, 4), facecolor='white')
    ax_traj = fig.add_subplot(131, frameon=False)
    ax_phase = fig.add_subplot(132, frameon=False)
    ax_vecfield = fig.add_subplot(133, frameon=False)
    plt.show(block=False)


#f_theta (slow ODE)
class ODEFunc_fast(nn.Module):

    def __init__(self, dim_in=3, dim_out=1):
        super(ODEFunc_fast, self).__init__()

        self.net = nn.Sequential(
            nn.Linear(dim_in, 50),
            nn.Tanh(),
            nn.Linear(50, dim_out),
        )
        self.nfe = 0
        for m in self.net.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, mean=0, std=0.1)
                nn.init.constant_(m.bias, val=0)

    def forward(self, t, y, y_other_dim):
        if(y.shape == torch.Size([])):
            dim0 = 1
        else:
            dim0 = len(y)
        y = y.reshape([dim0, 1])
        y_other_dim = y_other_dim.reshape([len(y_other_dim), 1])
        input = torch.concat([y, y_other_dim, torch.full(y.shape, t)], 1)
        self.nfe += 1
        return self.net(torch.sin(input))

#f_theta (slow ODE)
class ODEFunc_slow(nn.Module):

    def __init__(self, dim_in=3, dim_out=1):
        super(ODEFunc_slow, self).__init__()

        self.net = nn.Sequential(
            nn.Linear(dim_in, 50),
            nn.Tanh(),
            nn.Linear(50, dim_out),
        )

        for m in self.net.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, mean=0, std=0.1)
                nn.init.constant_(m.bias, val=0)
        self.nfe = 0
    def forward(self, t, y, y_other_dim):
        y = y.reshape([len(y), 1])
        y_other_dim = y_other_dim.reshape([len(y_other_dim), 1])
        input = torch.concat([y, y_other_dim, torch.full(y.shape, t)], 1)
        self.nfe += 1
        return self.net(torch.sin(input))
       

class RunningAverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self, momentum=0.99):
        self.momentum = momentum
        self.reset()

    def reset(self):
        self.val = None
        self.avg = 0

    def update(self, val):
        if self.val is None:
            self.avg = val
        else:
            self.avg = self.avg * self.momentum + val * (1 - self.momentum)
        self.val = val

def train(niters=2000, lr=1e-3, stepsize=0.01, sampling_rate=5, to_print=True):
    ii = 0
    _dt_fast = stepsize
    func = ODEFunc_slow().to(device)
    func_fast = ODEFunc_fast().to(device)
    
    optimizer = optim.Adam(func.parameters(), lr=1e-1)
    optimizer_fast = optim.Adam(func_fast.parameters(), lr=1e-2)
    
    
    start, prev = time.time(), time.time()

    # time_meter = RunningAverageMeter(0.97)
    
    #loss_meter = RunningAverageMeter(0.97)
    #loss_meter_fast = RunningAverageMeter(0.97)
    losses_fast = []
    losses_slow = []
    times = []
    slow_intg_time = []
    fast_intg_time = []
    nfes = []
    nfes_fast = []
    for itr in range(1, args.niters + 1):
        print("----", itr)
        func.nfe = 0
        func_fast.nfe = 0
        start = time.time()
        optimizer.zero_grad()
        optimizer_fast.zero_grad()
        batch_y0_slow, batch_y_slow, batch_y0_fast, batch_y_fast, batch_t, batch_t_fast = get_batch(slow, fast, t_slow, t_fast)
        tracemalloc.start()
        
        pred_y, pred_y_fast, timing, fast_timing = odeint(func, batch_y0_slow, batch_t, batch_t_fast, func_fast=func_fast, y0_fast=batch_y0_fast, dt_fast=_dt_fast)
        current, peak = tracemalloc.get_traced_memory()
        print(f"Current memory usage is {current / 10**6}MB; Peak was {peak / 10**6}MB")
        tracemalloc.stop()
        slow_intg_time.append(timing)
        fast_intg_time.append(fast_timing)
        loss = torch.nn.functional.binary_cross_entropy_with_logits(pred_y, batch_y_slow)
        loss_fast = torch.nn.functional.binary_cross_entropy_with_logits(pred_y_fast, batch_y_fast)
        
        loss.backward(retain_graph=True)
        loss_fast.backward(retain_graph=True)
        
        optimizer.step()
        optimizer_fast.step()

        #loss_meter.update(loss.item())
        #loss_meter_fast.update(loss_fast.item())
        print("loss (slow): ", float(loss), "loss (fast): ", float(loss_fast))
        if(float(loss_fast) >= 2):
            print("outlier")
            if(len(losses_fast) == 0):
                losses_fast.append(0)
            else:
                losses_fast.append(losses_fast[len(losses_fast) - 1])
        else:
            losses_fast.append(float(loss_fast))
        losses_slow.append(float(loss))
        print("NFE: ", func.nfe, func_fast.nfe)
        nfes.append(func.nfe)
        nfes_fast.append(func_fast.nfe)

        
        if itr % args.test_freq == 0:
            print("----------------------------------------")
            with torch.no_grad():
                if(len(true_y0_fast) == 1):
                    true_y0_fast_lst = args.fast_step * [true_y0_fast]
                    true_y0_fast = true_y0_fast_lst
                pred_y, pred_y_fast, timing, fast_timing = odeint(func, torch.tensor([true_y0_slow]), torch.tensor(t_slow), torch.tensor(t_fast), func_fast=func_fast, y0_fast=torch.tensor(true_y0_fast).T, dt_fast=_dt_fast)
                plt.figure()
                plt.plot(pred_y.reshape(len(pred_y), 1), label = "predicted")
                plt.plot(slow, label = "true value")
                plt.legend("upper right")
                plt.title("prediction" + str(itr))
                plt.savefig("prediction" + str(itr))
                loss = torch.nn.functional.binary_cross_entropy_with_logits(pred_y.reshape(len(pred_y)), slow)
                loss_fast = torch.nn.functional.binary_cross_entropy_with_logits(pred_y_fast.reshape(fast.shape), fast)
                print('[slow] Iter {:04d} | Total Loss {:.6f}'.format(itr, loss.item()))
                print('[fast] Iter {:04d} | Total Loss {:.6f}'.format(itr, loss_fast.item()))
                ii += 1
        
        times.append(time.time() - end)
        end = time.time()
        

        
    plot_data(torch.tensor(losses_fast), torch.tensor(losses_slow), t_slow, t_fast, xlabel="interation", ylabel="loss", title="loss")
    #plot_data(torch.tensor(nfes_fast), torch.tensor(nfes), xlabel="iteration", ylabel="nfe")

    print("timing avg: ", sum(times)/len(times))
    print("slow intg abg time: ", sum(slow_intg_time)/len(slow_intg_time))
    print("fast intg abg time: ", sum(fast_intg_time)/len(fast_intg_time))
    print("NFE for fast ode: ", func.nfe)
    print("NFE for slow ode: ", func_fast.nfe)


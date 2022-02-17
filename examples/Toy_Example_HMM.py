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

torch.pi = torch.acos(torch.zeros(1)).item() * 2

parser = argparse.ArgumentParser('ODE demo')
parser.add_argument('--method', type=str, choices=['dopri5', 'adams'], default='dopri5')
parser.add_argument('--data_size', type=int, default=1000)
parser.add_argument('--batch_time', type=int, default=10)
parser.add_argument('--batch_size', type=int, default=20)
parser.add_argument('--niters', type=int, default=8000)
parser.add_argument('--test_freq', type=int, default=20)
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


t = torch.linspace(0., 0.1, args.data_size).to(device)

freq_slow = 1
freq_fast = 4

class Lambda_slow(nn.Module):
    def forward(self, t, y, y_fast):
        return torch.sin(freq_slow * t) + 1/(y_fast + 0.01)


class Lambda_fast(nn.Module):
    def forward(self, t, y, y_slow):
        return torch.sin(freq_fast * t)**2 + torch.cos(y_slow)**2

lambda_slow = Lambda_slow()
lambda_fast = Lambda_fast()


def generate_stellar_orbits(a=2, b=3, epslion=0.01):
    # t = t.to('cpu')
    data = []
    t_prev = t[0]
    A = torch.tensor([[0., a, 0., 0.], [-a, 0., 0., 0.], [0., 0., 0., b], [0., 0., -b, 0.]]).to(device)
    y_0 = torch.tensor([[1., 0., 1., 0]]).to(device)
    y = y_0
    for t_sample in t:
        dt = t_sample - t_prev
        f = torch.tensor([[0., y[0][1]**2/a, 0., (2*y[0][0]*y[0][1])/b]]).to(device)
        # print(y.device, dt.device, A.device, f.device)
        y = y + dt * ((1/epslion) * A@y.T + f.T).T
        t_prev = t_sample
        data.append(y)
    return data

def extract_modes(data, index_slow, index_fast):
    slow = [item[0][index_slow] for item in data]
    fast = [item[0][index_fast] for item in data]
    return torch.tensor(slow).to(device), torch.tensor(fast).to(device)
    
def generate_time_series(true_y0_fast, true_y0_slow):
    data_slow = []
    data_fast = []
    t_prev = t[0]
    y_slow = true_y0_slow
    y_fast = true_y0_fast
    
    for t_sample in t:
        dt = t_sample - t_prev
        y_slow_prev = y_slow
        y_fast_prev = y_fast
        y_slow = y_slow + dt * lambda_slow.forward(t_sample, y_slow_prev, y_fast_prev)
        y_fast = y_fast + dt * lambda_fast.forward(t_sample, y_fast_prev, y_slow_prev)

        t_prev = t_sample

        data_slow.append(y_slow)
        data_fast.append(y_fast)
    return torch.tensor(data_slow), torch.tensor(data_fast)

true_y_0 = [torch.tensor([[1., 0., 1., 0]]).to(device)]
true_y0_slow, true_y0_fast = extract_modes(true_y_0, 0, 3)
#slow, fast = generate_time_series(true_y0_fast, true_y0_slow)
data = generate_stellar_orbits()
slow, fast = extract_modes(data, 0, 3)

def convert_to_int(lst):
    output = []
    for element in lst:
        output.append(int(element))
    return output

def plot_data(fast, slow):
    slow_int = convert_to_int(slow)
    fast_int = convert_to_int(fast)
    plt.plot(slow_int, label="slow")
    plt.plot(fast_int, label="fast")
    plt.legend(loc="upper left")
    plt.show()

#plot_data(fast, slow)
def get_batch(true_y_slow, true_y_fast):
    s = torch.from_numpy(np.random.choice(np.arange(args.data_size - args.batch_time, dtype=np.int64), args.batch_size, replace=False))
    batch_y0_slow = true_y_slow[s]  # (M, D)
    batch_y0_fast = true_y_fast[s]  # (M, D)
    batch_t = t[:args.batch_time]  # (T)
    batch_y_slow = torch.stack([true_y_slow[s + i] for i in range(args.batch_time)], dim=0)  # (T, M, D)
    batch_y_fast = torch.stack([true_y_fast[s + i] for i in range(args.batch_time)], dim=0)  # (T, M, D)
    return batch_y0_slow.to(device), batch_t.to(device), batch_y_slow.to(device), batch_y0_fast.to(device), batch_y_fast.to(device)

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


def visualize(true_y, pred_y, odefunc, itr):

    if args.viz:

        ax_traj.cla()
        ax_traj.set_title('Trajectories')
        ax_traj.set_xlabel('t')
        ax_traj.set_ylabel('x,y')
        ax_traj.plot(t.cpu().numpy(), true_y.cpu().numpy()[:, 0, 0], t.cpu().numpy(), true_y.cpu().numpy()[:, 0, 1], 'g-')
        ax_traj.plot(t.cpu().numpy(), pred_y.cpu().numpy()[:, 0, 0], '--', t.cpu().numpy(), pred_y.cpu().numpy()[:, 0, 1], 'b--')
        ax_traj.set_xlim(t.cpu().min(), t.cpu().max())
        ax_traj.set_ylim(-2, 2)
        ax_traj.legend()

        ax_phase.cla()
        ax_phase.set_title('Phase Portrait')
        ax_phase.set_xlabel('x')
        ax_phase.set_ylabel('y')
        ax_phase.plot(true_y.cpu().numpy()[:, 0, 0], true_y.cpu().numpy()[:, 0, 1], 'g-')
        ax_phase.plot(pred_y.cpu().numpy()[:, 0, 0], pred_y.cpu().numpy()[:, 0, 1], 'b--')
        ax_phase.set_xlim(-2, 2)
        ax_phase.set_ylim(-2, 2)

        ax_vecfield.cla()
        ax_vecfield.set_title('Learned Vector Field')
        ax_vecfield.set_xlabel('x')
        ax_vecfield.set_ylabel('y')

        y, x = np.mgrid[-2:2:21j, -2:2:21j]
        dydt = odefunc(0, torch.Tensor(np.stack([x, y], -1).reshape(21 * 21, 2)).to(device)).cpu().detach().numpy()
        mag = np.sqrt(dydt[:, 0]**2 + dydt[:, 1]**2).reshape(-1, 1)
        dydt = (dydt / mag)
        dydt = dydt.reshape(21, 21, 2)

        ax_vecfield.streamplot(x, y, dydt[:, :, 0], dydt[:, :, 1], color="black")
        ax_vecfield.set_xlim(-2, 2)
        ax_vecfield.set_ylim(-2, 2)

        fig.tight_layout()
        plt.savefig('png/{:03d}'.format(itr))
        plt.draw()
        plt.pause(0.001)

#f_theta (slow ODE)
class ODEFunc_fast(nn.Module):

    def __init__(self, dim_in=1, dim_out=1):
        super(ODEFunc_fast, self).__init__()

        self.net = nn.Sequential(
            nn.Linear(dim_in, 50),
            nn.Tanh(),
            nn.Linear(50, dim_out),
        )

        for m in self.net.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, mean=0, std=0.1)
                nn.init.constant_(m.bias, val=0)

    def forward(self, t, y, y_other_dim):
        return self.net(y.reshape(len(y), 1)**3)

#f_theta (slow ODE)
class ODEFunc_slow(nn.Module):

    def __init__(self, dim_in=1, dim_out=1):
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

    def forward(self, t, y, y_oter_dim):
        return self.net(y.reshape(len(y), 1)**3)
       

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
    
    optimizer = optim.RMSprop(func.parameters(), lr)
    optimizer_fast = optim.RMSprop(func_fast.parameters(), lr)
    
    start, prev = time.time(), time.time()

    # time_meter = RunningAverageMeter(0.97)
    
    loss_meter = RunningAverageMeter(0.97)
    loss_meter_fast = RunningAverageMeter(0.97)
    

    for itr in range(1, niters + 1):
        optimizer.zero_grad()
        optimizer_fast.zero_grad()
        batch_y0_slow, batch_t, batch_y_slow, batch_y0_fast, batch_y_fast = get_batch(slow, fast)
        
        pred_y, pred_y_fast, _, _ = odeint(func, batch_y0_slow, batch_t, func_fast=func_fast, y0_fast=batch_y0_fast, dt_fast=_dt_fast, sampling_rate=sampling_rate)
        
        loss = torch.mean(torch.abs(pred_y - batch_y_slow))
        loss_fast = torch.mean(torch.abs(pred_y_fast - batch_y_fast[0:2]))


        loss.backward(retain_graph=True)
        loss_fast.backward(retain_graph=True)
        
        optimizer.step()
        optimizer_fast.step()
        
        # time_meter.update(time.time() - prev)
        loss_meter.update(loss.item())
        loss_meter_fast.update(loss_fast.item())

        if itr % args.test_freq == 0:
            with torch.no_grad():
                # print(true_y0_slow.device, t.device, true_y0_fast.device)
                pred_y, pred_y_fast, nfe_slow, nfe_fast = odeint(func, true_y0_slow, t, func_fast=func_fast, y0_fast=true_y0_fast, dt_fast=_dt_fast, sampling_rate=sampling_rate)
                loss = torch.mean(torch.abs(pred_y - slow))
                loss_fast = torch.mean(torch.abs(pred_y_fast - fast))
                if to_print:
                    print('[slow] Iter {:04d} | Total Loss {:.6f} | nfe {}'.format(itr, loss.item(), nfe_slow))
                    print('[fast] Iter {:04d} | Total Loss {:.6f} | nfe {}'.format(itr, loss_fast.item(), nfe_fast))
                    print('[time] Iter {:04d} | Takes {:.2f} seconds'.format(itr, time.time() - prev))
                visualize(slow, pred_y, func, ii)
                ii += 1
        prev = time.time()
    else:
        if to_print:
            print(f'The whole training takes {prev - start:.2f} seconds')

        return  [loss.item(), loss_fast.item(), prev - start]

def test_sampling_rate():
    sampling_rate_list = [3, 5, 7, 9, 11, 13, 15]
    loss_slow, loss_fast = [], []
    for i in tqdm(range(len(sampling_rate_list))):
        res = train(sampling_rate=sampling_rate_list[i], niters=5000, to_print=False)
        loss_slow.append(res[0])
        loss_fast.append(res[1])
    plt.figure(figsize=(10, 10))
    plt.xlabel('sampling rate')
    plt.ylabel('loss')
    plt.plot(sampling_rate_list, loss_slow, label='slow variable')
    plt.plot(sampling_rate_list, loss_fast, label='fast variable')
    plt.legend()
    plt.show()

def test_stepsize():
    stepsize_list = np.linspace(0.01, 0.1, 5)
    loss_slow, loss_fast = [], []
    for i in tqdm(range(len(stepsize_list))):
        res = train(stepsize=stepsize_list[i], niters=2000, sampling_rate=5, to_print=False)
        loss_slow.append(res[0])
        loss_fast.append(res[1])
    plt.figure(figsize=(10, 10))
    plt.xlabel('step size')
    plt.ylabel('loss')
    plt.plot(stepsize_list, loss_slow, label='slow variable')
    plt.plot(stepsize_list, loss_fast, label='fast variable')
    plt.legend()
    plt.show()



if __name__ == '__main__':

    train(sampling_rate=25, niters=5000, to_print=True)
    # test_sampling_rate()
    

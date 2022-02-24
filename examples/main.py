# ==================================================================
# modified from tutorials/module1-neuralde/m1a_neural_ode_cookbook.ipynb
# ==================================================================

import argparse
import matplotlib.pyplot as plt
import torch
import torch.utils.data as data
import torch.nn as nn
import sys
import pickle 
from torchdyn.core import NeuralODE
from torchdyn.datasets import *
from torchdyn.utils import *

import pandas as pd 
#from torchdyn.dataset_utils.data_utils import get_dataloader
import numpy as np
import time
#from dmd_utils import *
#from utils_dataset import *

torch.manual_seed(0)
torch.set_default_dtype(torch.float64)

parser = argparse.ArgumentParser()
parser.add_argument('--rerun', type=eval, default=False, choices=[True, False])
parser.add_argument('--dmd', type=eval, default=False, choices=[True, False])
parser.add_argument('--fro', type=eval, default=False, choices=[True, False])
parser.add_argument('--nepochs', type=int, default=50)
parser.add_argument('--fro_steps', type=int, default=2)
parser.add_argument('--cutoff', type=int, default=2000)
parser.add_argument('--freq', type=int, default=20)
parser.add_argument('--data', type=str, default='multiscale')
parser.add_argument('--fast_epochs', type=int, default=2)
parser.add_argument('--fast_samples', type=int, default=10)
parser.add_argument('--length_of_intervals', type=int, default=50)


args = parser.parse_args()
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def get_kernel(shape):
    return torch.full(shape, 1)

def get_estimation(network, fast_series, slow_value):
    values = []
    for x in fast_series:
        x = x.unsqueeze(0)
        features = torch.cat([slow_value, x], dim=1)
        values = values + [network(features)]
    values = torch.cat(values).clone()
    kernels = get_kernel(values.shape).clone()
    estimation = torch.mean(kernels * values, 0).unsqueeze(0)
    return estimation
    

# the dynamical system
if args.data == 'multiscale':
    def multi_res_fun(t, x, ep):
        a = 1
        b = 2
        A = [0, a, 0, 0,
             -a, 0, 0, 0,
             0, 0, 0, b,
             0, 0, -b, 0],
        A = np.array(A, dtype='float64')
        A = A.reshape(4, 4)
        fx = [0, x[1]**2/a, 0, 2*x[0]*x[1]/b]
        fx = np.array(fx, dtype='float64')
        fx = fx.reshape(4,1)
        dx = (A @ x)/ep + fx
        return dx

    T = 0.1
    dim_org = 4
    dim = dim_org
    r = dim
    step = 1e-6
    t = np.arange(0, T, step)
    dt = t[1] - t[0]
    Niter = len(t)
    x = np.array([1, 0, 1, 0])
    x = x.reshape(4,1)
    EP = np.array([0.001, 0.002, 0.004, 0.008, 0.01, 0.1])
    alpha = np.array([1e3, 2e3, 4e3, 1e4, 2e4], dtype = 'int32')
    print("Made the data...")
    if args.rerun:
        print("Rerunning...")
        X_dict = {}
        for ep in EP:
            X = [x]
            for i in range(len(t)):
                dx = multi_res_fun(t[i], x, ep)
                x = x + dt * dx
                X.append(x)
            X = np.concatenate(X, 1)
            X = X[:, 0:len(t)]
            X_dict[ep] = X

        a_file = open("data.pkl", "wb")
        pickle.dump(X_dict, a_file)
        a_file.close()
        print('Input file created and saved.')
    else:
        a_file = open("data.pkl", "rb")
        X_dict = pickle.load(a_file)
        print('Input file loaded.')
        print("X_dict: ", len(X_dict), X_dict.keys(), len(X_dict[0.001]), X_dict[0.001].shape)

    # plot original data
    fig, axs = plt.subplots(4)
    # plt.rcParams['text.usetex'] = True
    fig.suptitle('original data for ep = 0.001')
    axs[0].plot(t, X_dict[0.001][0,:])
    axs[0].set_ylabel('x1')
    axs[1].plot(t, X_dict[0.001][1,:])
    axs[1].set_ylabel('x2')

    axs[2].plot(t, X_dict[0.001][2,:])
    axs[2].set_ylabel('x3')

    axs[3].plot(t, X_dict[0.001][3,:])
    axs[3].set_ylabel('x4')
    plt.show()


    # pick one for training
    X = X_dict[0.001]
    X = torch.Tensor(X).to(device)
    t_span = torch.Tensor(t).to(device)
    cutoff = args.cutoff
    X = X[:,:cutoff]
    t_span = t_span[:cutoff]
    print("X after picking: ", X.shape, "t: ", t_span.shape)

print("Already have the data...")

# define neural net
dim_fast = 2
dim_slow = 2
net_slow = nn.Sequential(
    #features: the init value and the x_fast_est
    nn.Linear(dim_slow + dim_fast, 50, bias = False),
    nn.ReLU(inplace=False),
    nn.Linear(50, dim_slow, bias = False))

net_fast = nn.Sequential(
    nn.Linear(dim_fast, 50, bias = False),
    nn.ReLU(inplace=False),
    nn.Linear(50, dim_fast, bias = False))

X_slow = X[[0, 1], :]
X_fast = X[[2, 3], :]

# Neural ODE and optimizer
neuralDE_slow = NeuralODE(net_slow, sensitivity='interpolated_adjoint', solver='euler').to(device)
neuralDE_fast = NeuralODE(net_fast, sensitivity='interpolated_adjoint', solver='euler').to(device)

optim_slow = torch.optim.Adam(net_slow.parameters(), lr=1)
optim_fast = torch.optim.Adam(neuralDE_fast.parameters(), lr=1)

scheduler_slow = torch.optim.lr_scheduler.MultiStepLR(optim_slow, milestones=[900,1300], gamma=2)
scheduler_fast = torch.optim.lr_scheduler.MultiStepLR(optim_fast, milestones=[900,1300], gamma=2)
loss_fn_slow = nn.MSELoss()
loss_fn_fast = nn.MSELoss()
#initial points of all modes:
#Two first modes are slow variables and two last modes are fast variables.
x0_slow = X[[0, 1], 0]
x0_fast = X[[2, 3], 0]
print("X: ", X.shape)
print("t_span: ", t_span[0:args.length_of_intervals].shape)
print("x0 fast: ", x0_fast.shape, "x0_slow: ", x0_slow.shape)
x0_fast = torch.unsqueeze(x0_fast, 0)
x0_slow = torch.unsqueeze(x0_slow, 0)
print("after squeez x0 fast: ", x0_fast.shape, "x0_slow: ", x0_slow.shape)

dt = t_span[args.length_of_intervals] - t_span[0] 

for iter in range(args.nepochs):
    predicted_slow_series = []
    t_slow_eval = []
    print("iters: ", iter)
    for i in range(int(len(t_span)/args.length_of_intervals)):#t_sapan_slow
        t_start = i*args.length_of_intervals
        t_interval = t_span[t_start: t_start+args.length_of_intervals]
        # optim.zero_grad()
        optim_slow.zero_grad()
        optim_fast.zero_grad()
        #for the fast series part

        for j in range(args.fast_epochs):
            t_eval_fast, pred_fast = neuralDE_fast(x0_fast, t_span[t_start: t_start+args.fast_samples])
            pred_fast = torch.squeeze(pred_fast)                                                  
            t_fast_eval = np.arange(t_start,(t_start+args.fast_samples)).tolist()
            X_fast_eval = X_fast[:, t_fast_eval]
            loss_fast = loss_fn_fast(pred_fast.T, X_fast_eval)
            loss_fast.backward(retain_graph=True)
            print("loss fast, iter["+str(iter)+"]", float(loss_fast))
            optim_fast.step()
            scheduler_fast.step()
    with torch.autograd.set_detect_anomaly(True):
        estimation = get_estimation(net_slow, pred_fast, x0_slow)
        predicted_slow = x0_slow + dt*estimation
        x0_slow = predicted_slow
        predicted_slow_series = predicted_slow_series + [predicted_slow]
        t_slow_eval = t_slow_eval +[(i + 1)*args.length_of_intervals - 1]
        pred_slow = torch.cat(predicted_slow_series)
        print("slow...")
        X_slow_eval = X_slow[:, t_slow_eval]
        loss_slow = loss_fn_slow(pred_slow.T, X_slow_eval)
        loss_slow.backward(retain_graph=True)
        print("loss slow, iter["+str(iter)+"]", float(loss_slow))
        optim_slow.step()
        scheduler_slow.step()
        
'''
if i%args.freq == 0:
    with torch.no_grad():
        # fix this part as well, this is for plotting
        t_plot = t_eval_slow[1:]
        # L = len(t_plot)
        real = X[0, 0:L].detach().numpy()
        prediction = pred_[0:L, 0].detach().numpy()
        plt.plot(t_plot, prediction)
        plt.plot(t_plot, real)
        plt.legend(['predict train', 'real train'])
        plt.show()
'''
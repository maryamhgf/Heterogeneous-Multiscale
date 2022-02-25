# ==================================================================
# modified from tutorials/module1-neuralde/m1a_neural_ode_cookbook.ipynb
# ==================================================================

import argparse
from random import uniform
import matplotlib.pyplot as plt
import torch
import torch.utils.data as data
import torch.nn as nn
import sys
import pickle 
from torchdyn.core import NeuralODE
from torchdyn.datasets import *
from torchdyn.utils import *
import os

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
parser.add_argument('--nepochs', type=int, default=100)
parser.add_argument('--fro_steps', type=int, default=2)
parser.add_argument('--cutoff', type=int, default=3000)
parser.add_argument('--freq', type=int, default=50)
parser.add_argument('--data', type=str, default='multiscale')
parser.add_argument('--fast_epochs', type=int, default=10)
parser.add_argument('--fast_samples', type=int, default=10)
parser.add_argument('--length_of_intervals', type=int, default=200)
parser.add_argument('--kernel_method', type=str, default='exp')


args = parser.parse_args()
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

dirName = "./slow_step" + str(args.length_of_intervals)+"_fast_step"+str(args.fast_samples)\
    +"_kernel"+args.kernel_method+"_cutoff"+str(args.cutoff)+"_nepoch"+str(args.nepochs)+"_fast_nepoch"+str(args.fast_epochs)

if not os.path.exists(dirName):
    os.mkdir(dirName)
    print("Directory " , dirName ,  " Created ")
else:    
    print("Directory " , dirName ,  " already exists")

result_dir = os.path.join(dirName, 'Results/')

def get_kernel(shape, method='uniform', t=None, C=0.003):
    if method == 'uniform':
        return torch.full(shape, 1)
    
    if method == 'exp':
        assert t != None
        t_cat = t.repeat(shape[1]).reshape(shape)
        return 1/ torch.exp(5 / (t_cat**2 - 1))
    
    if method == 'cos':
        assert t != None
        t_cat = t.repeat(shape[1]).reshape(shape)
        return 0.5 * (1 + torch.cos(np.pi * t_cat))

def get_estimation(network, fast_series, slow_value, shape, t_eval_fast):
    kernels = get_kernel(fast_series.shape, method=args.kernel_method, t=t_eval_fast)
    slow_value_squeezes = torch.squeeze(slow_value, 0)
    slow_calue_spanned =slow_value_squeezes.repeat(len(fast_series)).reshape(shape)
    features = torch.cat([slow_calue_spanned, fast_series], dim=1)
    results = network(features.clone().detach())

    estimation = torch.mean(kernels * results, 0)
    estimation_unsq = torch.unsqueeze(estimation, 0)
    return estimation_unsq

def get_fast_prediction(t_span, x0_fast_, x0_slow, fast_epochs, length_of_intervals, t_start):
    dt = t_span[1] - t_span[0]
    x0_fast = x0_fast_
    predicted_series = [x0_fast]
    t_fast_eval = [t_start]
    features = torch.concat((x0_fast, x0_slow), dim=1)
    for i in range(length_of_intervals):
        predicted = x0_fast.clone().detach() + dt*net_fast(features)
        x0_fast = predicted
        features = torch.concat((x0_fast, x0_slow), dim=1)
        predicted_series = predicted_series + [predicted]
        t_fast_eval = t_fast_eval +[t_fast_eval[-1] + 1]
    pred_fast = torch.cat(predicted_series[0: len(predicted_series) - 1])

    return pred_fast, t_fast_eval[0: len(t_fast_eval)-1]


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
    nn.Linear(dim_fast + dim_slow, 50, bias = False),
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
x0_fast = torch.unsqueeze(x0_fast, 0)
x0_slow = torch.unsqueeze(x0_slow, 0)

dt = t_span[args.length_of_intervals] - t_span[0] 
losses = []
for iter in range(args.nepochs):
    predicted_slow_series = [x0_slow.clone().detach()]
    t_slow_eval = [0]
    optim_slow.zero_grad()
    optim_fast.zero_grad()
    print("-----------iters: ", iter)
    for i in range(int(len(t_span)/args.length_of_intervals)):#t_sapan_slow
        t_start = i*args.length_of_intervals
        t_interval = t_span[t_start: t_start+args.length_of_intervals]
        t_interval_fast = t_span[t_start: t_start+args.fast_samples]
        x0_fast = X[[2, 3], t_start]
        x0_fast = torch.unsqueeze(x0_fast, 0)
        #for the fast series part
        pred_fast, t_fast_eval = get_fast_prediction(t_interval_fast, x0_fast.clone().detach(), x0_slow.clone().detach(), args.fast_epochs, args.fast_samples, t_start)
    
        X_fast_eval = X_fast[:, t_fast_eval]
        loss_fast = loss_fn_slow(pred_fast.T, X_fast_eval)
        loss_fast.backward(retain_graph=True)
        optim_fast.step()
        scheduler_fast.step()
        print(str(i) + "th point: loss fast, iter["+str(iter)+"]", float(loss_fast))
        estimation = get_estimation(net_slow, pred_fast.clone().detach(), x0_slow.clone().detach(), pred_fast.shape, torch.tensor(t_fast_eval))
        predicted_slow = x0_slow.clone().detach() + dt*estimation
        x0_slow = predicted_slow
        
        predicted_slow_series = predicted_slow_series + [predicted_slow]
        t_slow_eval = t_slow_eval +[(i + 1)*args.length_of_intervals - 1]
    pred_slow = torch.cat(predicted_slow_series[0:len(predicted_slow_series)-1])
    X_slow_eval = X_slow[:, t_slow_eval[0:len(t_slow_eval)-1]]
    loss_slow = loss_fn_slow(pred_slow.T, X_slow_eval)
    loss_slow.backward(retain_graph=True)
    print("loss slow, iter["+str(iter)+"]", float(loss_slow))
    losses.append(loss_slow.item())
    optim_slow.step()
    scheduler_slow.step()
    if iter%args.freq == 0:
        with torch.no_grad():
            t_plot = t_slow_eval
            # L = len(t_plot)
            real = X_slow_eval[0].detach().numpy()
            print("real: ", real.shape)
            print(pred_slow.shape)
            prediction = pred_slow.T
            prediction = prediction[0].numpy()
            plt.figure()
            plt.plot(prediction, label ="predicted")
            plt.plot(real, label="true value")
            plt.legend("upper right")
            plt.title("prediction (mode=1)" + str(iter))
            plt.savefig(dirName+"/prediction" + str(iter)+"mode 0")
            real = X_slow[1, t_slow_eval].detach().numpy()
            prediction = pred_slow.T
            prediction = prediction[1].numpy()
            plt.figure()
            plt.plot(prediction, label="predicted")
            plt.plot(real, label="true value")
            plt.legend("upper right")
            plt.title("prediction (mode=2)" + str(iter))
            plt.savefig(dirName+"/prediction" + str(iter)+"mode 1")


plt.figure()
plt.plot(losses)
plt.savefig(dirName+"/losses.png")

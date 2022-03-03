import argparse
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import pickle
from torchdyn.core import NeuralODE
from torchdyn.datasets import *
from torchdyn.utils import *
import time
import os
import tracemalloc
import pandas as pd
import numpy as np
import time
import pytorch_lightning as pl
import sys
sys.path.insert(0, '..')

torch.manual_seed(0)
torch.set_default_dtype(torch.float64)

parser = argparse.ArgumentParser()
parser.add_argument('--rerun', type=eval, default=False, choices=[True, False])
parser.add_argument('--solver', type=str, default='euler')
parser.add_argument('--baseline', type=eval,
                    default=False, choices=[True, False])
parser.add_argument('--fro', type=eval, default=False, choices=[True, False])
parser.add_argument('--nepochs', type=int, default=400)
parser.add_argument('--sesitivity', type=str, default="autograd")
parser.add_argument('--cutoff', type=int, default=3000)
parser.add_argument('--test', type=eval, default=False, choices=[True, False])
parser.add_argument('--freq', type=int, default=50)
parser.add_argument('--data', type=str, default='multiscale')
parser.add_argument('--fast_epochs', type=int, default=2)
parser.add_argument('--fast_samples', type=int, default=5)
parser.add_argument('--fast_length_of_intervals', type=int, default=1)
parser.add_argument('--length_of_intervals', type=int, default=100)
parser.add_argument('--kernel_method', type=str, default='exp')
parser.add_argument('--adjoint', type=eval,
                    default=False, choices=[True, False])


args = parser.parse_args()
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

if args.baseline:
    folder = './NODE'
else:
    folder = './NODE_HMM'

if args.adjoint:
    from torchdiffeq import odeint_adjoint as odeint
else:
    from torchdiffeq import odeint

if not os.path.exists(folder):
    os.mkdir(folder)
    print("Directory ", folder,  " Created ")
else:
    print("Directory ", folder,  " already exists")

dirName = folder + "/slow_step" + str(args.length_of_intervals)+"_fast_step"+str(args.fast_samples)\
    + "_kernel"+args.kernel_method+"_cutoff"+str(args.cutoff)\
    + "_nepoch"+str(args.nepochs)+"_fast_nepoch"+str(args.fast_epochs)+"_fast_step_length"+str(args.fast_length_of_intervals)\
    + "_sensitivity"+args.sesitivity+"_solver"+args.solver

if not os.path.exists(dirName):
    os.mkdir(dirName)
    print("Directory ", dirName,  " Created ")
else:
    print("Directory ", dirName,  " already exists")


result_dir = os.path.join(dirName, 'Results/')


def get_kernel(shape, method='uniform', t=None, C=0.003):
    if method == 'uniform':
        return torch.full(shape, 1)

    if method == 'exp':
        assert t != None
        t_cat = t.repeat(shape[1]).reshape(shape)
        return 1 / torch.exp(5 / (t_cat**2 - 1))

    if method == 'cos':
        assert t != None
        t_cat = t.repeat(shape[1]).reshape(shape)
        return 0.5 * (1 + torch.cos(np.pi * t_cat))


def store_data(loss, memory, time):
    data = {
        'loss': loss,
        'memory': memory,
        'time': time
    }
    dataframe = pd.DataFrame(data)
    dataframe.to_csv(dirName+"/train_inf.csv")


def get_estimation(network, fast_series, slow_value, shape, t_eval_fast):
    kernels = get_kernel(
        fast_series.shape, method=args.kernel_method, t=t_eval_fast)
    slow_value_squeezes = torch.squeeze(slow_value, 0)
    slow_calue_spanned = slow_value_squeezes.repeat(
        len(fast_series)).reshape(shape)
    features = torch.cat([slow_calue_spanned, fast_series], dim=1)
    results = network(None, features.clone().detach())[:, 2:]
    # print(results.shape)

    estimation = torch.mean(kernels * results, 0)
    estimation_unsq = torch.unsqueeze(estimation, 0)
    return estimation_unsq


def get_fast_prediction(t_span, x0_fast_, x0_slow, fast_epochs, length_of_intervals, t_start, fast_step=1):
    dt = t_span[fast_step] - t_span[0]
    # print(t_span.shape, t_span, length_of_intervals, t_start, x0_fast_)
    # print(t_span, length_of_intervals, t_start, dt.shape)
    x0_fast = x0_fast_
    predicted_series = [x0_fast]
    t_fast_eval = [t_start]
    features = torch.concat((x0_slow, x0_fast), dim=1)
    if args.solver == 'euler':
        for _ in range(length_of_intervals - 1):
            x0_fast = x0_fast.clone().detach() + dt * \
                net_fast(None, features)[:, 2:]
            predicted_series = predicted_series + [x0_fast]
            features = torch.concat((x0_slow, x0_fast), dim=1)
            t_fast_eval = t_fast_eval + [t_fast_eval[-1] + fast_step]
        pred_fast = torch.cat(predicted_series[0: len(predicted_series) - 1])
    if args.solver == 'dopri5':
        pred_fast = odeint(net_fast, features, t_span).squeeze(1)[:, 2:]
        for _ in range(length_of_intervals):
            t_fast_eval = t_fast_eval + [t_fast_eval[-1] + fast_step]
    # print(pred_fast.shape,  t_fast_eval[0: len(t_fast_eval)-1])
    return pred_fast, t_fast_eval[0: len(t_fast_eval)-1]


'''
def reverse_mode_derivetive(neural_net, parameters, t_start, t_end, final_state, grad_outputs):
    s0 = (final_state, grad_outputs, 0)
    def aug_dynamics(state, adjoint_param, t, parameters):
        return neural_net(neural_net(state), -adjoint_param.T * )
'''

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
        fx = fx.reshape(4, 1)
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
    x = x.reshape(4, 1)
    EP = np.array([0.001, 0.002, 0.004, 0.008, 0.01, 0.1])
    alpha = np.array([1e3, 2e3, 4e3, 1e4, 2e4], dtype='int32')
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
        a_file = open("../data.pkl", "rb")
        X_dict = pickle.load(a_file)
        print('Input file loaded.')
        print("X_dict: ", len(X_dict), X_dict.keys(),
              len(X_dict[0.001]), X_dict[0.001].shape)

    # plot original data
    fig, axs = plt.subplots(4)
    # plt.rcParams['text.usetex'] = True
    fig.suptitle('original data for ep = 0.001')
    axs[0].plot(t, X_dict[0.001][0, :])
    axs[0].set_ylabel('x1')
    axs[1].plot(t, X_dict[0.001][1, :])
    axs[1].set_ylabel('x2')

    axs[2].plot(t, X_dict[0.001][2, :])
    axs[2].set_ylabel('x3')

    axs[3].plot(t, X_dict[0.001][3, :])
    axs[3].set_ylabel('x4')
    # plt.show()

    # pick one for training
    X = X_dict[0.001]
    X = torch.Tensor(X).to(device)
    t_span = torch.Tensor(t).to(device)
    # print(t_span)
    cutoff = args.cutoff
    X = X[:, :cutoff]
    t_span = t_span[:cutoff]
    print("X after picking: ", X.shape, "t: ", t_span.shape)

print("Already have the data...")
dim_fast = 2
dim_slow = 2
if args.baseline == False:
    # define neural net
    class ODEfunc(nn.Module):
        def __init__(self, out_dim, mode):
            # store stores the slow mode if the func is for fast mode, vice versa
            super(ODEfunc, self).__init__()
            self.out_dim = out_dim
            self.mode = mode
            self.linear_in = nn.Linear(dim_slow + dim_fast, 50, bias=False)
            self.relu = nn.ReLU(inplace=False)
            self.linear_out = nn.Linear(50, out_dim, bias=False)

        def forward(self, t, x):
            # tt = torch.ones((1, 1)) * t
            # ttx = torch.cat([tt, x], 1)
            slow, fast = x[:, :2].clone().detach(), x[:, 2:].clone().detach()
            out = self.linear_in(x)
            out = self.relu(out)
            out = self.linear_out(out)
            if self.mode == 'fast':
                return torch.cat((slow, out), dim=1)
            else:
                return torch.cat((out, fast), dim=1)

    net_slow = ODEfunc(dim_slow, 'slow')
    net_fast = ODEfunc(dim_fast, 'fast')

    # net_slow = nn.Sequential(
    #     #features: the init value and the x_fast_est
    #     nn.Linear(dim_slow + dim_fast, 50, bias = False),
    #     nn.ReLU(inplace=False),
    #     nn.Linear(50, dim_slow, bias = False))

    # net_fast = nn.Sequential(
    #     nn.Linear(dim_fast + dim_slow, 50, bias = False),
    #     nn.ReLU(inplace=False),
    #     nn.Linear(50, dim_fast, bias = False))

    X_slow = X[[0, 1], :]
    X_fast = X[[2, 3], :]

    optim_slow = torch.optim.Adam(net_slow.parameters(), lr=0.1)
    optim_fast = torch.optim.Adam(net_fast.parameters(), lr=0.01)

    scheduler_slow = torch.optim.lr_scheduler.MultiStepLR(
        optim_slow, milestones=[60, 100], gamma=2)
    scheduler_fast = torch.optim.lr_scheduler.MultiStepLR(
        optim_fast, milestones=[60, 100], gamma=2)
    loss_fn_slow = nn.MSELoss()
    loss_fn_fast = nn.MSELoss()
    # initial points of all modes:
    # Two first modes are slow variables and two last modes are fast variables.
    x0_slow = X[[0, 1], 0]
    x0_fast = X[[2, 3], 0]
    x0_fast = torch.unsqueeze(x0_fast, 0)
    x0_slow = torch.unsqueeze(x0_slow, 0)

    dt = t_span[args.length_of_intervals] - t_span[0]
    losses = []
    times = []
    memory = []
    for iter in range(args.nepochs):
        x0_slow = X[[0, 1], 0]
        x0_fast = X[[2, 3], 0]
        x0_fast = torch.unsqueeze(x0_fast, 0)
        x0_slow = torch.unsqueeze(x0_slow, 0)
        tracemalloc.start()
        predicted_slow_series = [x0_slow.clone().detach()]
        t_slow_eval = [0]
        optim_slow.zero_grad()
        optim_fast.zero_grad()
        print("-----------iters: ", iter)
        start = time.time()
        for i in range(int(len(t_span)/args.length_of_intervals)):  # t_sapan_slow
            t_start = i*args.length_of_intervals
            t_interval_fast = t_span[t_start: t_start +
                                     args.fast_samples * args.fast_length_of_intervals]
            x0_fast = X[[2, 3], t_start]
            x0_fast = torch.unsqueeze(x0_fast, 0)
            # for the fast series part
            pred_fast, t_fast_eval = get_fast_prediction(t_interval_fast, x0_fast.clone().detach(), x0_slow.clone().detach(),
                                                         args.fast_epochs, args.fast_samples, t_start, fast_step=args.fast_length_of_intervals)

            X_fast_eval = X_fast[:, t_fast_eval]
            loss_fast = loss_fn_slow(pred_fast.T, X_fast_eval)
            loss_fast.backward(retain_graph=True)
            optim_fast.step()
            scheduler_fast.step()
            print(
                str(i) + "th point: loss fast, iter["+str(iter)+"]", float(loss_fast))
            estimation = get_estimation(net_slow, pred_fast.clone().detach(
            ), x0_slow.clone().detach(), pred_fast.shape, torch.tensor(t_fast_eval))
            predicted_slow = x0_slow.clone().detach() + dt*estimation
            x0_slow = predicted_slow
            predicted_slow_series = predicted_slow_series + [predicted_slow]
            t_slow_eval = t_slow_eval + [(i + 1)*args.length_of_intervals - 1]

        pred_slow = torch.cat(
            predicted_slow_series[0:len(predicted_slow_series)])
        X_slow_eval = X_slow[:, t_slow_eval]
        loss_slow = loss_fn_slow(pred_slow.T, X_slow_eval)
        loss_slow.backward(retain_graph=True)
        optim_slow.step()
        scheduler_slow.step()
        time_iter = time.time() - start
        current, peak = tracemalloc.get_traced_memory()
        memory.append(current / 10**6)
        losses.append(loss_slow.item())
        times.append(time_iter)
        print("loss slow, iter["+str(iter)+"]", float(loss_slow))
        if (args.test == True) and ((iter % args.freq == 0 and iter != 0) or iter == args.nepochs - 1):
            with torch.no_grad():
                t_plot = t_slow_eval
                # L = len(t_plot)
                real0 = X_slow[0, t_slow_eval].detach().numpy()
                prediction = pred_slow.T
                prediction0 = prediction[0].numpy()
                plt.figure()
                plt.plot(prediction0, label="predicted")
                plt.plot(real0, label="true value")
                plt.legend()
                plt.title("prediction (mode=1)" + str(iter))
                plt.savefig(dirName+"/prediction" + str(iter)+"mode 0")
                real1 = X_slow[1, t_slow_eval].detach().numpy()
                prediction1 = prediction[1].numpy()
                plt.figure()
                plt.plot(prediction1, label="predicted")
                plt.plot(real1, label="true value")
                plt.legend()
                plt.title("prediction (mode=2)" + str(iter))
                plt.savefig(dirName+"/prediction" + str(iter)+"mode 1")

    plt.figure()
    plt.plot(losses)
    plt.title("train loss")
    plt.xlabel("epoch")
    plt.ylabel("loss")
    plt.savefig(dirName+"/train_losses.png")

    store_data(losses, memory, times)
    if args.test:
        torch.save(real0, dirName+"/real0_data")
        torch.save(real1, dirName+"/real1_data")
        torch.save(prediction0, dirName+"/prediction0_data")
        torch.save(prediction1, dirName+"/prediction1_data")

else:
    print("base NODE")
    # NODE
    neural_net = nn.Sequential(
        nn.Linear(dim_slow, 50, bias=False),
        nn.ReLU(inplace=False),
        nn.Linear(50, dim_slow, bias=False))

    # Neural ODE
    neural_ODE = NeuralODE(neural_net, solver='euler',
                           atol=1e-3, rtol=1e-3).to(device)
    optim = torch.optim.Adam(neural_ODE.parameters(), lr=0.01)
    scheduler = torch.optim.lr_scheduler.MultiStepLR(
        optim, milestones=[50, 100], gamma=2)
    loss_fn = nn.MSELoss()
    losses = []
    times = []
    memory = []
    X_slow = X[[0, 1], :]
    t_span_model = t_span[np.arange(0, len(t_span), args.length_of_intervals)]
    for iter in range(args.nepochs):
        tracemalloc.start()
        start = time.time()
        optim.zero_grad()
        x0 = X[[0, 1], 0]
        x0 = torch.unsqueeze(x0, 0)
        t_eval, pred = neural_ODE(x0, t_span_model)
        t_eval_indx = [t_span.tolist().index(t) for t in t_eval]
        t_eval_indx = [0] + t_eval_indx[0:len(t_eval_indx)-1]
        X_eval = X_slow[:, t_eval_indx]
        pred_squeezed = torch.squeeze(pred, 1)
        loss = loss_fn(pred_squeezed.T, X_eval)
        print('[' + str(iter) + '] train loss: ' + str(loss))
        loss.backward()
        optim.step()
        scheduler.step()
        time_iter = time.time() - start
        current, peak = tracemalloc.get_traced_memory()
        memory.append(current / 10**6)
        print("loss, iter["+str(iter)+"]", float(loss))
        losses.append(loss.item())
        times.append(time_iter)
        if (args.test == True) and ((iter % args.freq == 0 and iter != 0) or iter == args.nepochs - 1):
            with torch.no_grad():
                t_plot = t_eval
                # L = len(t_plot)
                real0 = X_eval[0, :].detach().numpy()
                prediction = pred.T
                prediction0 = torch.squeeze(prediction[0], 0)
                prediction0 = prediction0.numpy()
                plt.figure()
                print("real0.shape: ", real0.shape)
                print("prediction0.shape: ", prediction0.shape)
                plt.plot(prediction0, label="predicted")
                plt.plot(real0, label="true value")
                plt.legend()
                plt.title("prediction (mode=1)" + str(iter))
                plt.savefig(dirName+"/prediction" + str(iter)+"mode 0")
                real1 = X_eval[1, :].detach().numpy()
                prediction1 = torch.squeeze(prediction[1], 0)
                prediction1 = prediction1.numpy()
                plt.figure()
                plt.plot(prediction1, label="predicted")
                plt.plot(real1, label="true value")
                plt.legend()
                plt.title("prediction (mode=2)" + str(iter))
                plt.savefig(dirName+"/prediction" + str(iter)+"mode 1")

    plt.figure()
    plt.plot(losses)
    plt.title("train loss")
    plt.xlabel("epoch")
    plt.ylabel("loss")
    plt.savefig(dirName+"/train_losses.png")

    store_data(losses, memory, times)
    if args.test:
        torch.save(real0, dirName+"/real0_data")
        torch.save(real1, dirName+"/real1_data")
        torch.save(prediction0, dirName+"/prediction0_data")
        torch.save(prediction1, dirName+"/prediction1_data")

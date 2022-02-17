import torch
import torch.nn as nn
import matplotlib.pyplot as plt

a = torch.tensor([[1, 0, 4]])
b = torch.tensor([[2, 4, 8]])

print(a)
print(a.shape)

print(b)
print(b.shape)

c = a * b
print(c)
print(c.shape)

print(int(torch.sum(c))/3)

d =  torch.randn(c.shape)
print(d)


print("dataset test:")

true_y0 = torch.tensor([[2., 0.]])
print("true y0: ", true_y0)
data_size = 12
t = torch.linspace(0., 25., data_size)
true_A = torch.tensor([[-0.1, 2.0], [-2.0, -0.1]])
print("true A: ", true_A)
print("-----------")
print(torch.sin(torch.tensor([1])).shape)

class Lambda_slow(nn.Module):
    def forward(self, t, y, y_fast):
        return torch.sin(1*t) + torch.cos(y_fast)
print(Lambda_slow.forward(torch.tensor([2]), torch.tensor([3]), torch.tensor([4]), torch.tensor([5])))


true_y0_fast = torch.tensor([0])
true_y0_slow = torch.tensor([0])
class Lambda_slow(nn.Module):
    def forward(self, t, y, y_fast):
        print("inside")
        return torch.sin(1 * t) + torch.cos(y_fast)


class Lambda_fast(nn.Module):
    def forward(self, t, y, y_slow):
        return torch.sin(2 * t) + y_slow**2

ll = Lambda_slow()
y_slow = true_y0_slow
y_fast = true_y0_fast
t = torch.tensor([1])
y_slow_prev = y_slow
y_fast_prev = y_fast
y_slow = y_slow + ll.forward(t=t, y=y_slow_prev,y_fast= y_fast_prev)


ts = torch.linspace(0., 25., 102)

t_prev = ts[0]

for t in ts:
    dt = t - t_prev
    print(dt)
    t_prev = t
    
import numpy as np
ll = list(np.arange(0, 100))
s = torch.from_numpy(np.random.choice(np.arange(100 - 10, dtype=np.int64), 10, replace=False))
print(s)
print(list(s))
ll = np.array(ll)
print(ts[s])


print(ts.reshape(102, 1))

'''
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
        print("inside")
        return self.net(y**3)

f = ODEFunc_slow()
print(type(f))
f.forward(torch.tensor([1]), torch.tensor([2]), torch.tensor([3]))
'''
a = 2.
b = 3.

A = torch.tensor([[0., a, 0., 0.], [-a, 0., 0., 0.], [0., 0., 0., b], [0., 0., -1*b, 0.]])

print(A.T)
print(A)

t = torch.linspace(0., 0.1, 100)

def generate_stellar_orbits(a=2, b=3, epslion=0.01):
    data = []
    t_prev = t[0]
    A = torch.tensor([[0., a, 0., 0.], [-a, 0., 0., 0.], [0., 0., 0., b], [0., 0., -b, 0.]])
    y_0 = torch.tensor([[1., 0., 1., 0]])
    y = y_0
    for t_sample in t:
        dt = t_sample - t_prev
        print(dt)
        f = torch.tensor([[0., y[0][1]**2/a, 0., (2*y[0][0]*y[0][1])/b]])
        y = y + dt * ((1/epslion) * A@y.T + f.T).T
        t_prev = t_sample
        data.append(y)
    return data
def convert_to_int(lst):
    output = []
    for element in lst:
        output.append(int(element))
    return output

def plot_data(fast, slow, t):
    slow_int = convert_to_int(slow)
    fast_int = convert_to_int(fast)
    plt.plot(t, slow_int, label="slow")
    plt.plot(t, fast_int, label="fast")
    plt.legend(loc="upper left")
    plt.show()
    
    

data = generate_stellar_orbits()
print(data[0].shape)
slow = [item[0][0] for item in data]
fast = [item[0][3] for item in data]

#plot_data(fast, slow, t)

def extract_modes(data, index_slow, index_fast):
    slow = [item[0][index_slow] for item in data]
    fast = [item[0][index_fast] for item in data]
    return slow, fast

true_y_0 = [torch.tensor([[1., 0., 1., 0]])]

true_y0_slow, true_y0_fast = extract_modes(true_y_0, 0, 3)
print(true_y0_slow, true_y0_fast)

print("------------------------")
data_t = [10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 110, 120, 130, 140, 150, 160]
batch_y_slow =[data_t[i * int(len(data_t)/5)] for i in range(5)]  # (T, M, D)
print(batch_y_slow)
for i in range(len(batch_y_slow) - 1):
    element = batch_y_slow[i]
    next_elem = batch_y_slow[i + 1]
    index = data_t.index(element)
    next_index = data_t.index(next_elem)
    temp = [data_t[index + i * 1] for i in range(2)]
    print(temp)

print(data_t[1:10:2])

def generate_stellar_orbits(a=2, b=3, epslion=0.01, fast_dataset_size=5):
    ts = torch.linspace(0., 0.1, 1000)
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

true_y_0 = [torch.tensor([[1., 0., 1., 0]])]
true_y0_slow, true_y0_fast = extract_modes(true_y_0, 0, 3)
data, ts = generate_stellar_orbits()
slow, fast = extract_modes(data, 0, 3)

def convert_to_float(lst):
    output = []
    for element in lst:
        output.append(float(element))
    return output

print(convert_to_float(ts))
def time_series_sampling(slow, fast, data_t, slow_step_size=None, fast_step_size=None, fast_size=4):
    dt = data_t[1] - data_t[0]
    print("dt: ", dt)
    if(slow_step_size == None):
        slow_step_size = dt * 3
    if(fast_step_size == None):
        fast_step_size = dt
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
    return t_slow, t_fast, data_slow, data_fast
t_slow, t_fast, data_slow, data_fast = time_series_sampling(slow, fast, convert_to_float(ts))
#print(len(t_slow), len(t_fast), len(data_slow), len(data_fast))
#print(data_fast[0])
    
s = torch.from_numpy(np.random.choice(np.arange(333 - 10, dtype=np.int64), 20, replace=False))
print(s)

print(t_fast[0])
time_grid = t_fast[0]

for t0_sample, t1_sample in zip(time_grid[:-1], time_grid[1:]):

a = torch.randn(6, 4)
print(torch.sum(a, 0))


a = torch.tensor([0])
b = torch.tensor(0)
print(a.shape[0])
print(b.shape)

true_y_0 = [torch.tensor([[1., 0., 1., 0]])]
true_y0_slow, true_y0_fast = extract_modes(true_y_0, 0, 3)
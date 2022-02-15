import torch
import torch.nn as nn

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
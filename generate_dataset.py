import torch
import dynamics as dyn
import numpy as np

## Config
M = 2.4
m = 0.24
g = 9.81
l = 0.18

num_samples = 50000 ## Number of samples

xdata = np.array([np.zeros(5)]) #0:theta, 1:thetadot, 2:pos, 3:xdot, 4:u
ydata = np.array([dyn.dynamics(xdata[-1][:4],xdata[-1][4],M,m,g,l)])
for i in range(0,num_samples):
    ## Generate Random Samples
    x0 = np.random.uniform(low=0,high=2*np.pi)
    x1 = np.random.uniform(low=-1,high=1)
    x2 = np.random.uniform(low=-10,high=10)
    x3 = np.random.uniform(low=-1,high=1)
    u = np.random.uniform(low=-1,high=1)

    xdata = np.append(xdata,[np.array([x0,x1,x2,x3,u])],axis=0)
    y = dyn.dynamics(xdata[-1][:4],xdata[-1][4],M,m,g,l)
    ydata = np.append(ydata,[y],axis=0)
    if i % 10000 == 0:
        print(str(i) + "\n")

xdata = torch.tensor(xdata)
torch.save(xdata,'xdata_test.pt')
ydata = torch.tensor(ydata)
torch.save(ydata,'ydata_test.pt')
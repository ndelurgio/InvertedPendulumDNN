import torch
import dynamics as dyn
import numpy as np

## Config
M = 2.4
m = 0.24
g = 9.81
l = 0.18

num_samples = 200000 ## Number of samples

xdata = np.array([np.zeros(4)]) #0:theta, 1:thetadot, 2:posdot, 3:u, 4:t
ydata = np.array([dyn.propogate(np.array([0,0,0,0]),1,0.001,0,M,m,g,l)])
for i in range(0,num_samples):
    ## Generate Random Samples
    x0 = np.random.uniform(low=0,high=2*np.pi)
    x1 = np.random.uniform(low=-1,high=1)
    x3 = np.random.uniform(low=-0.5,high=0.5)
    u = np.random.uniform(low=-1,high=1)
    # t = np.random.uniform(low=0,high=5)

    xdata = np.append(xdata,[np.array([x0,x1,x3,u])],axis=0)
    y = dyn.propogate(np.array([x0,x1,0,x3]), 1, 0.001, u, M,m,g,l)
    ydata = np.append(ydata,[y],axis=0)
    if i % 10000 == 0:
        print(str(i) + "\n")

xdata = torch.tensor(xdata)
torch.save(xdata,'xdata_ode.pt')
ydata = torch.tensor(ydata)
torch.save(ydata,'ydata_ode.pt')
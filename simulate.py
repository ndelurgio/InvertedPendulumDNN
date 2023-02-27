import dynamics as dyn
import numpy as np
import matplotlib.pyplot as plt
import dnn
import torch

M = 2.4
m = 0.24
g = 9.81
l = 0.18
u = 0

x0 = np.array([np.pi/8,0,0,0])
dt = 0.001
t0 = 0
tf = 1
tarr = np.arange(dt,tf,dt)
x = np.array([x0])
xtorch = np.array([x0])
xdot = np.zeros(len(x0))
xdot_prev = np.zeros(len(x0))
xdottorch = np.zeros(len(x0))
xdottorch_prev = np.zeros(len(x0))

model = dnn.DNN(input_size=5,hidden_size=512,output_size=4)
model.load_state_dict(torch.load('model_weights.pth'))

for t in tarr[0:-1]:
    # print(x[-1])
    # print(np.array([u]))
    # tarr = np.append(tarr,t)
    xdot = dyn.dynamics(x[-1],u,M,m,g,l)
    xtorchin = torch.concatenate((torch.tensor(x[-1],dtype=torch.float32),torch.tensor([u],dtype=torch.float32)))
    xdottorch = model(xtorchin)
    x = np.append(x,[dyn.integrate(x[-1],xdot_prev,xdot,dt)],axis=0)
    xtorch = np.append(xtorch,[dyn.integrate(xtorch[-1],xdottorch_prev,xdottorch,dt)],axis=0)
    xdot_prev = xdot
    xdottorch_prev = xdottorch


fig, axs = plt.subplots(4)
fig.suptitle("True Dynamics")
axs[0].plot(tarr,x.T[0])
axs[0].plot(tarr,xtorch.T[0])
axs[0].set(ylabel="Theta")
axs[1].plot(tarr,x.T[1])
axs[1].plot(tarr,xtorch.T[1])
axs[1].set(ylabel="Theta dot")
axs[2].plot(tarr,x.T[2])
axs[2].plot(tarr,xtorch.T[2])
axs[2].set(ylabel="Pos")
axs[3].plot(tarr,x.T[3])
axs[3].plot(tarr,xtorch.T[3])
axs[3].set(ylabel="Pos dot")

plt.show(block=False)
plt.pause(0.1) # Pause for interval seconds.
input("hit[enter] to end.")
plt.close('all')
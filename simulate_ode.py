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
t = 1

xf = dyn.propogate(x0,t,dt,u,M,m,g,l)
print(xf)
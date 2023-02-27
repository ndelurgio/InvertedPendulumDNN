import numpy as np
import torch
import dynamics as dyn
import dnn

M = 2.4
m = 0.24
g = 9.81
l = 0.18

x0 = np.random.uniform(low=0,high=2*np.pi)
x1 = np.random.uniform(low=-1,high=1)
x2 = np.random.uniform(low=-10,high=10)
x3 = np.random.uniform(low=-1,high=1)
u = np.random.uniform(low=-1,high=1)

x0 = np.pi/8
x1 = 0
x2 = 0
x3 = 0
u = 0

xanalytic = np.array([x0,x1,x2,x3])
xtorch = torch.tensor([[x0,x1,x2,x3,u]])

model = dnn.DNN(input_size=5,hidden_size=512,output_size=4)
model.load_state_dict(torch.load('model_weights.pth'))

print("Analytic Output: " + str(dyn.dynamics(xanalytic,u,M,m,g,l)))
print("DNN Output: " + str(model(xtorch)))
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
x3 = np.random.uniform(low=-0.5,high=0.5)
u = np.random.uniform(low=-1,high=1)
t = 1

# xanalytic = np.array([x0,x1,x2,x3])
xtorch = torch.tensor([[x0,x1,x3,u]])

model = dnn.DNN(input_size=4,hidden_size=512,output_size=4)
model.load_state_dict(torch.load('model_weights_ode.pth'))

print("Analytic Output: " + str(dyn.propogate(np.array([x0,x1,0,x3]),t,0.001,u,M,m,g,l)))
print("DNN Output: " + str(model(xtorch)))
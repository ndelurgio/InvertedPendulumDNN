import numpy as np
import torch
import dnn

model = dnn.DNN(input_size=4,hidden_size=512,output_size=4)
model.load_state_dict(torch.load('model_weights.pth'))
testindata = torch.load("xdata_test.pt").to(dtype=torch.float32)
testoutdata = torch.load("ydata_test.pt").to(dtype=torch.float32)
output = model(testindata)
diff = (output - testoutdata).detach().numpy()
diff_vec = np.reshape(diff, (diff.shape[0], -1))
rel_error = np.mean(np.linalg.norm(diff_vec, axis=1)[1:] / np.linalg.norm(np.reshape(testoutdata.detach().numpy(), (testoutdata.shape[0], -1)), axis=1)[1:]* 100)

print(rel_error)
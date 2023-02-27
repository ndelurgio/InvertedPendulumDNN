import torch
import numpy as np
import dnn

model = dnn.DNN(input_size=5,hidden_size=512,output_size=4)

criterion = torch.nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
# print(model.parameters)

## Load Data
x = torch.load("xdata.pt").to(dtype=torch.float32)
y = torch.load("ydata.pt").to(dtype=torch.float32)

# import pdb
# pdb.set_trace()
# print(x)

# Tracking loss
loss_list = []
# Training the model
for epoch in range(1000):
    # Forward pass
    outputs = model(x)
    loss = criterion(outputs, y)
    # IMPORTANT: Always remember to detach a tensor from the computation graph if you want to use it outside of the graph
    loss_list.append(loss.detach().numpy())


    # Backward pass and optimization
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    
    # Printing the loss every 100 epochs
    # if (epoch + 1) % 100 == 0:
    print(f'Epoch [{epoch + 1}/1000], Loss: {loss.item()}')
    # if loss.item() < 0.05:
    #     break

torch.save(model.state_dict(),"model_weights.pth")
import numpy as np
import torch 
import torch.nn as nn
import torch.nn.functional as f
import matplotlib.pyplot as plt
import pandas as pd
import fdm

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class pid_mod(nn.Module):
    def __init__(self, input_dim, pid_heads, hidden_layers, cycle, output_dim):
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.layers = nn.ModuleList()
        self.tok_head = fdm.PID_LAYER(pid_heads, input_dim, output_dim, cycle)
        for _ in range(hidden_layers):
            self.layers.append(fdm.PID_LAYER(pid_heads, input_dim, output_dim, cycle))
    def forward(self, x):
        for l in self.layers:
            x = l.forward(x)
        x = self.tok_head.state_to_token(x)
        return x

    
model = pid_mod(3072,2,2,5,10).to(device)

csv_training_path = "/home/marsbase/fdm/archive/train.csv"
csv_test_path = "/home/marsbase/fdm/archive/test.csv"

my_train_data = pd.read_csv(csv_training_path)
my_test_data = pd.read_csv(csv_test_path)

x_train = my_train_data.drop("label", axis=1).to_numpy(dtype=np.float32)
y_train = my_train_data["label"].to_numpy(dtype=np.float32)

'''x_test = my_test_data.drop("7", axis=1).to_numpy(dtype=np.float32)
y_test = my_test_data["7"].to_numpy(dtype=np.float32)'''

x_train = torch.FloatTensor(x_train).to(device)
y_train = torch.LongTensor(y_train).to(device)

print(x_train.shape)

'''x_test = torch.FloatTensor(x_test).reshape(9999, 784).to(device)
y_test = torch.LongTensor(y_test).to(device)'''

epochs = 2
loss = []
test_loss = []

critereon = nn.CrossEntropyLoss()
optimiser = torch.optim.Adamax(model.parameters(), lr = 1e-4)

bad_examples = []

torch.autograd.set_detect_anomaly(True, check_nan=False)

for i in range(epochs):
    x = 0
    b = 0
    for x_t,y_t in zip(x_train, y_train):
        y_t = y_t.reshape(1)
        y = model.forward(x_t)

        Loss = critereon(y, y_t)

        Loss.backward()
        optimiser.step()

        loss.append(Loss.item())
        if Loss.item() >1.4:
            b += 1
        if x %100 == 0:
            print(f"epoch: {i}, game: {x}, ratio of bad examples: {b}/1000")
            print(f"result: {torch.argmax(y)}, answer: {y_t}")
            b = 0
        x += 1

    z = 0

    '''if i % 5 == 0:
            for x_t, y_t in zip(x_test, y_test):
                y = model.forward(x_t)
                y_t = y_t.reshape(1)

                if torch.argmax(y) != y_t:
                    z+=1
                    print(z)
            with open("bad_examples.txt", "a") as writer:
                writer.write(f"innauracy_score: {z}",)

    print(f"{z}/{9999} accuracy score on test set")'''



plt.plot(range(60000 * epochs), loss)
plt.yscale("log")
plt.xlabel("epoch")
plt.ylabel("error")

plt.savefig("model_loss.png")



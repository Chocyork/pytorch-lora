import torch
import torch.nn as nn

class oriNet(nn.Module):
    def __init__(self, hiden_size_1 = 1000, hiden_size_2 = 2000):
        super(oriNet,self).__init__()
        self.linear1 = nn.Linear(28*28, hiden_size_1)
        self.linear2 = nn.Linear(hiden_size_1, hiden_size_2)
        self.linear3 = nn.Linear(hiden_size_2, 10)
        self.relu = nn.ReLU()

    def forward(self, img):
        x = img.view(-1, 28*28)
        x = self.relu(self.linear1(x))
        x = self.relu(self.linear2(x))
        x = self.relu(self.linear3(x))
        return x
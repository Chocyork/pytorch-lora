import torch
import torchvision.datasets as datasets 
import torchvision.transforms as transforms
import torch.nn as nn
import matplotlib.pyplot as plt
from tqdm import tqdm
from train_orin.train_utils import train
from train_orin.data_utils import load_mnist_dataset, create_dataloader, get_mnist_transform
from train_orin.model import oriNet

custom_transform = get_mnist_transform(mean=(0.5,), std=(0.5,))  

train_dataset = load_mnist_dataset(transform=custom_transform)
test_dataset = load_mnist_dataset(train=False)

train_loader = create_dataloader(train_dataset, batch_size=64, shuffle=True)
test_loader = create_dataloader(test_dataset, batch_size=64, shuffle=False)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

net = oriNet()
net.load_state_dict(torch.load('checkpoints/train_original_model.pth', map_location=device))
net.to(device)
net.eval()

def test():
    correct = 0
    total = 0

    wrong_counts = [0 for i in range(10)]

    with torch.no_grad():
        for data in tqdm(test_loader, desc='Testing'):
            x, y = data
            x = x.to(device)
            y = y.to(device)
            output = net(x.view(-1, 784))
            for idx, i in enumerate(output):
                if torch.argmax(i) == y[idx]:
                    correct +=1
                else:
                    wrong_counts[y[idx]] +=1
                total +=1
    print(f'Accuracy: {round(correct/total, 3)}')
    for i in range(len(wrong_counts)):
        print(f'wrong counts for the digit {i}: {wrong_counts[i]}')

test()
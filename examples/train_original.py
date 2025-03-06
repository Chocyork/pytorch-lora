import torch
import torchvision.datasets as datasets 
import torchvision.transforms as transforms
import torch.nn as nn
import matplotlib.pyplot as plt
from tqdm import tqdm
from train_orin.train_utils import train
from train_orin.data_utils import load_mnist_dataset, create_dataloader, get_mnist_transform
from train_orin.model import oriNet

_ = torch.manual_seed(0)


custom_transform = get_mnist_transform(mean=(0.5,), std=(0.5,))  

train_dataset = load_mnist_dataset(transform=custom_transform)
test_dataset = load_mnist_dataset(train=False)

train_loader = create_dataloader(train_dataset, batch_size=64, shuffle=True)
test_loader = create_dataloader(test_dataset, batch_size=64, shuffle=False)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

net = oriNet()
net.to(device)

train(
    train_loader= train_loader,
    net= net,
    epochs= 10,
    device= device,
    lr = 1e-3,
)
torch.save(net.state_dict(), 'checkpoints/train_original_model.pth')
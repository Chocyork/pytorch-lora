import torch
import torchvision.datasets as datasets 
import torchvision.transforms as transforms
import torch.nn as nn
import matplotlib.pyplot as plt
from tqdm import tqdm
from train_orin.train_utils import train
from train_orin.data_utils import load_mnist_dataset, create_dataloader, get_mnist_transform
from train_orin.model import oriNet
from train_orin.lora import LoRAParametrization, linear_layer_parameterization, enable_disable_lora
import torch.nn.utils.parametrize as parametrize

_ = torch.manual_seed(0)


custom_transform = get_mnist_transform(mean=(0.5,), std=(0.5,))  

train_dataset = load_mnist_dataset(transform=custom_transform)
exclude_indices = train_dataset.targets == 9
train_dataset.data = train_dataset.data[exclude_indices]
train_dataset.targets = train_dataset.targets[exclude_indices]

train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=10, shuffle=True)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

net = oriNet()
net.load_state_dict(torch.load('checkpoints/train_original_model.pth', map_location=device))
net.to(device)
original_weights = {}
for name, param in net.named_parameters():
    original_weights[name] = param.clone().detach()

# add lora
parametrize.register_parametrization(
    net.linear1, "weight", linear_layer_parameterization(net.linear1, device)
)
parametrize.register_parametrization(
    net.linear2, "weight", linear_layer_parameterization(net.linear2, device)
)
parametrize.register_parametrization(
    net.linear3, "weight", linear_layer_parameterization(net.linear3, device)
)

# The parameters count
# total_parameters_lora = 0
# total_parameters_non_lora = 0
# for index, layer in enumerate([net.linear1, net.linear2, net.linear3]):
#     total_parameters_lora += layer.parametrizations["weight"][0].lora_A.nelement() + layer.parametrizations["weight"][0].lora_B.nelement()
#     total_parameters_non_lora += layer.weight.nelement() + layer.bias.nelement()
#     print(
#         f'Layer {index+1}: W: {layer.weight.shape} + B: {layer.bias.shape} + Lora_A: {layer.parametrizations["weight"][0].lora_A.shape} + Lora_B: {layer.parametrizations["weight"][0].lora_B.shape}'
#     )

# assert total_parameters_non_lora == 2807010 # total_parameters_original
# print(f'Total number of parameters (original): {total_parameters_non_lora:,}')
# print(f'Total number of parameters (original + LoRA): {total_parameters_lora + total_parameters_non_lora:,}')
# print(f'Parameters introduced by LoRA: {total_parameters_lora:,}')
# parameters_incremment = (total_parameters_lora / total_parameters_non_lora) * 100
# print(f'Parameters incremment: {parameters_incremment:.3f}%')

# freeze the original weights
for name, param in net.named_parameters():
    if 'lora' not in name:
        print(f'Freezing non-LoRA parameter {name}')
        param.requires_grad = False

# train lora
train(train_loader, net, epochs=10)

# eval
# # Check that the frozen parameters are still unchanged by the finetuning
# assert torch.all(net.linear1.parametrizations.weight.original == original_weights['linear1.weight'])
# assert torch.all(net.linear2.parametrizations.weight.original == original_weights['linear2.weight'])
# assert torch.all(net.linear3.parametrizations.weight.original == original_weights['linear3.weight'])

# enable_disable_lora(net, enabled=True)
# # The new linear1.weight is obtained by the "forward" function of our LoRA parametrization
# # The original weights have been moved to net.linear1.parametrizations.weight.original
# # More info here: https://pytorch.org/tutorials/intermediate/parametrizations.html#inspecting-a-parametrized-module
# assert torch.equal(net.linear1.weight, net.linear1.parametrizations.weight.original + (net.linear1.parametrizations.weight[0].lora_B @ net.linear1.parametrizations.weight[0].lora_A) * net.linear1.parametrizations.weight[0].scale)

# enable_disable_lora(net, enabled=False)
# # If we disable LoRA, the linear1.weight is the original one
# assert torch.equal(net.linear1.weight, original_weights['linear1.weight'])
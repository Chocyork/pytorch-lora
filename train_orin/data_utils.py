# data_utils.py
from torchvision import transforms, datasets
from torch.utils.data import DataLoader

def get_mnist_transform(mean=(0.1307,), std=(0.3081,)):
    """返回 MNIST 的标准化转换操作"""
    return transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean, std)
    ])

def load_mnist_dataset(root='./data', train=True, transform=None):
    """加载 MNIST 数据集"""
    if transform is None:
        transform = get_mnist_transform()  # 默认使用 MNIST 的标准化参数
    return datasets.MNIST(
        root=root,
        train=train,
        download=True,
        transform=transform
    )

def create_dataloader(dataset, batch_size=10, shuffle=True):
    """创建 DataLoader"""
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle
    )
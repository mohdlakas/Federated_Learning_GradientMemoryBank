import torch
import torchvision.datasets as datasets

# This will show you where MNIST gets downloaded
train_dataset = datasets.MNIST(root='./data', train=True, download=True)
print(f"Dataset root: {train_dataset.root}")
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import numpy as np

transform = transforms.ToTensor()
dataset = datasets.ImageFolder('./dataset/cifar10_dataset', transform=transform)
loader = DataLoader(dataset, batch_size=64, shuffle=False)

mean = 0.0
std = 0.0
total_samples = 0
for images, _ in loader:
    batch_samples = images.size(0)  # batch size (number of images in this batch)
    images = images.view(batch_samples, images.size(1), -1)  # Flatten height and width
    mean += images.mean(2).sum(0)
    std += images.std(2).sum(0)
    total_samples += batch_samples

mean /= total_samples
std /= total_samples
print(f"Mean: {mean}, Std: {std}")
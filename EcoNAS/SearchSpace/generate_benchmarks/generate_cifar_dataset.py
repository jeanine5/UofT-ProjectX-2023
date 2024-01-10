"""

"""
from torchvision.datasets import CIFAR10
import torchvision.transforms as transforms
from torch.utils.data import DataLoader

from EcoNAS.SearchSpace.search_spaces.search_spaces import BenchmarkDataset
import ssl

ssl._create_default_https_context = ssl._create_unverified_context

transform = transforms.ToTensor()

train_dataset = CIFAR10(root='../data', train=True, transform=transform, download=True)
test_dataset = CIFAR10(root='../data', train=False, transform=transform)

batch_size = 128

train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False)

benchmark_dataset = BenchmarkDataset()
architectures = benchmark_dataset.generate_architectures(max_hidden_layers=15, max_hidden_size=128, data_name='CIFAR')
benchmark_dataset.evaluate_architectures(architectures, train_loader, test_loader)
benchmark_dataset.store_results_to_csv(filename='../precomputed_datasets/trained_cifar10_dataset.csv')
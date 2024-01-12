"""
This contains the code for the neural network architecture. The neural network is a fully connected neural network
with a variable number of hidden layers. The number of hidden layers and the number of hidden units per layer are
hyperparameters. The neural network is used for the evolutionary search algorithms. Note, ther are no convolutional
layers in this neural network.
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from sklearn.metrics import accuracy_score
from thop import profile

device = "mps" if torch.backends.mps.is_available() else "cpu"
device = torch.device(device)
print(f"Using device: {device}")


class NeuralNetwork(nn.Module):
    """
    A fully connected neural network with a variable number of hidden layers
    If testing with the MNIST dataset, the input size is 784 (28x28), and the output size is 10 (10 classes)
    If testing with the CIFAR-10 dataset, the input size is 3072 (32x32x3), and the output size is 10 (10 classes)
    If testing with the Statlog dataset, the input size is 21, and the output size is 2 (2 classes)
    """

    def __init__(self, input_size, output_size, hidden_sizes):
        super().__init__()

        self.input_size = input_size
        self.output_size = output_size

        self.hidden_layers = nn.ModuleList()
        for hidden_size in hidden_sizes:
            hidden_layer = nn.Linear(input_size, hidden_size)
            self.hidden_layers.append(hidden_layer)
            input_size = hidden_size

        self.output_layer = nn.Linear(input_size, self.output_size)

    def forward(self, x):
        batch_size = len(x)
        x = x.view(batch_size, self.input_size).to(device)

        for layer in self.hidden_layers:
            h = layer(x)
            x = F.relu(h)

        output = self.output_layer(x)
        return output


class NeuralArchitecture:
    """
    A wrapper class for the NeuralNetwork class. This class is used for the evolutionary search algorithms
    """

    def __init__(self, input_size, output_size, hidden_sizes):
        self.model = NeuralNetwork(input_size, output_size, hidden_sizes).to(device)
        self.input_size = input_size
        self.output_size = output_size
        self.hidden_sizes = hidden_sizes
        self.activation = nn.ReLU()
        self.objectives = {
            'accuracy': 0.0,
            'introspectability': 0.0,
            'flops': 0.0
        }
        self.nondominated_rank = 0
        self.crowding_distance = 0.0

    def introspectability_metric(self, loader):
        """
        Metric for evaluating the interpretability of a neural network. This is based on the paper
        https://arxiv.org/pdf/2112.08645.pdf
        :param loader: Data loader for the dataset
        :return: Returns the introspectability of the neural network
        """

        self.model.eval()

        class_activations = {}

        with torch.no_grad():
            for inputs, targets in loader:
                inputs, targets = inputs.to(device), targets.to(device)
                outputs = self.model(inputs)

                # convert targets and outputs to numpy arrays
                targets_np = targets.cpu().numpy()  # Use .cpu() to copy to host memory
                outputs_np = F.relu(outputs).cpu().numpy()

                for i in range(len(targets_np)):
                    target_class = targets_np[i]

                    if target_class not in class_activations:
                        class_activations[target_class] = []

                    class_activations[target_class].append(outputs_np[i])

        # calculate mean activations for each class
        mean_activations = {cls: np.mean(np.array(acts), axis=0) for cls, acts in class_activations.items()}

        # cosine distance
        introspectability = 0.0
        num_classes = len(mean_activations)

        for cls1 in mean_activations:
            for cls2 in mean_activations:
                if cls1 < cls2:
                    # cosine distance between mean activations of two classes
                    cosine_distance = F.cosine_similarity(
                        torch.FloatTensor(mean_activations[cls1]).to(device),  # Move to GPU
                        torch.FloatTensor(mean_activations[cls2]).to(device),
                        dim=0
                    ).item()

                    introspectability += cosine_distance

        # normalize by the number of unique class pairs
        introspectability /= num_classes * (num_classes - 1) / 2

        self.objectives['introspectability'] = introspectability

        return introspectability

    def flops_estimation(self, input_size=(1, 3, 32, 32)):
        """
        Estimates the number of FLOPs for the neural network
        :param input_size: The input size of the neural network. For MNIST, this is (1, 1, 28, 28), for CIFAR-10, this
        is (1, 3, 32, 32).
        :return: Returns the number of FLOPs
        """

        self.model.eval()

        dummy_input = torch.randn(input_size).to(device)
        flops, params = profile(self.model, inputs=(dummy_input,))

        self.objectives['flops'] = flops

        return flops

    def accuracy(self, outputs, labels):
        """
        This function calculates the accuracy of the neural network. It is used for training and evaluating the
        neural network.
        :param outputs: The outputs of the neural network. Data type is torch.Tensor
        :param labels: The labels of the data. Data type is torch.Tensor
        :return: Returns the accuracy of the neural network
        """
        predictions = outputs.argmax(-1)
        # correct = torch.sum(labels == predictions).item()
        correct = accuracy_score(predictions.detach().numpy(), labels)
        return correct

    def evaluate_accuracy(self, loader):
        """
        Evaluates the accuracy of the neural network
        :param loader: Data loader for the dataset
        :return: Returns the accuracy of the neural network
        """
        # loss function
        criterion = nn.CrossEntropyLoss()

        self.model.eval()
        acc = 0
        loss = 0
        n_samples = 0

        with torch.no_grad():
            for inputs, targets in loader:
                inputs, targets = inputs.to(device), targets.to(device)
                outputs = self.model(inputs)
                loss += criterion(outputs, targets).item() * len(targets)

                acc += self.accuracy(outputs, targets) * len(targets)
                n_samples += len(targets)

        self.objectives['accuracy'] = acc / n_samples

        return acc / n_samples

    def evaluate_all_objectives(self, loader):
        """
        Evaluates all the objectives of the neural network at once
        :param loader: Data loader for the dataset
        :return: Returns the loss, accuracy, interpretability, and FLOPs of the neural network
        """

        interpretable = self.introspectability_metric(loader)
        flops = self.flops_estimation()

        return interpretable, flops

    def train(self, train_loader, test_loader, epochs=20):
        """
        Trains the neural network. Optimizer is Adam, learning rate is 1e-4, and loss function is CrossEntropyLoss
        (can change if needed)
        :param epochs: Number of epochs (rounds) to train the neural network for
        :param loader: Data loader for the dataset
        :return: Returns the loss and accuracy of the neural network
        """
        criterion = nn.CrossEntropyLoss()
        lr = 1e-3  # The learning rate is a hyperparameter
        optimizer = optim.Adam(self.model.parameters(), lr=lr)

        train_accuracies = []
        test_accuracies = []
        int_values = []
        flops_values = []

        for epoch in range(epochs):

            train_loss = 0
            train_acc = 0
            n_samples = 0
            self.model.train()

            for inputs, targets in train_loader:
                inputs, targets = inputs.to(device), targets.to(device)
                optimizer.zero_grad()
                outputs = self.model(inputs)
                loss = criterion(outputs, targets)
                loss.backward()
                optimizer.step()

                train_loss += loss.detach().item() * len(targets)
                train_acc += self.accuracy(outputs, targets) * len(targets)
                n_samples += len(targets)

            train_accuracies.append(train_acc / n_samples)

            test_acc = self.evaluate_accuracy(test_loader)
            test_accuracies.append(test_acc)

            interpretable, flops = self.evaluate_all_objectives(test_loader)
            int_values.append(interpretable)
            flops_values.append(flops)

        return test_accuracies[-1], int_values[-1], flops_values[-1]

    def clone(self):
        return NeuralArchitecture(self.input_size, self.output_size, self.hidden_sizes.copy())


def Dataloading(x, y, batch_size, train):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    x = torch.Tensor(x).to(device)
    y = torch.LongTensor(y.values).to(device)

    dataset = torch.utils.data.TensorDataset(x, y)
    if train:
        return torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)
    return torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=False)

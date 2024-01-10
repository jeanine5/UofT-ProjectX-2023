"""
The main function of the EcoNAS algorithm. This is the function that is called when the user wants to run the
algorithm.
"""
import math

import numpy as np
from matplotlib import pyplot as plt

from EcoNAS.EA.NSGA import NSGA_II
from calculate_means import *


def plot_pareto_frontier(architectures):
    accuracies = np.array([arch.objectives['accuracy'] for arch in architectures])
    introspectabilities = np.array([arch.objectives['introspectability'] for arch in architectures])
    flops_values = np.array([arch.objectives['flops'] for arch in architectures])

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    # Scatter plot
    ax.scatter(accuracies, introspectabilities, flops_values, c=flops_values, cmap='viridis', label='Architectures')

    # Labeling
    ax.set_xlabel('Accuracy')
    ax.set_ylabel('Introspectability')
    ax.set_zlabel('FLOPS')
    ax.set_title('Pareto Frontier of Architectures')
    ax.legend()
    ax.view_init(elev=20, azim=30)  # Set the view angle for a better perspective

    # Show the plot
    plt.show()


def run_algorithm(dataset: str):
    """

    :param dataset:
    :return:
    """

    # modify this before you run the algorithm below
    nsga = NSGA_II(population_size=150, generations=14, crossover_factor=0.5, mutation_factor=0.5)
    architectures = nsga.evolve(hidden_layers=15, hidden_size=128, data_name=dataset)

    min_acc = math.inf
    max_acc = -math.inf

    min_interpretability = math.inf
    max_interpretability = -math.inf

    min_flops = math.inf
    max_flops = -math.inf

    for arch in architectures:
        acc, interpretability, flops = arch.objectives['accuracy'], arch.objectives['introspectability'], \
            arch.objectives['flops']

        min_acc = min(min_acc, acc)
        max_acc = max(max_acc, acc)

        min_interpretability = min(min_interpretability, interpretability)
        max_interpretability = max(max_interpretability, interpretability)

        min_flops = min(min_flops, flops)
        max_flops = max(max_flops, flops)

    print(
        f'Average Accuracy: {calculate_average_accuracy(architectures)}, Min Accuracy: {min_acc}, Max Accuracy: {max_acc}')
    print(
        f'Average Interpretability: {calculate_average_interpretability(architectures)}, Min Interpretability: {min_interpretability}, Max Interpretability: {max_interpretability}')
    print(f'Average FLOPs: {calculate_average_flops(architectures)}, Min FLOPs: {min_flops}, Max FLOPs: {max_flops}')

    plot_pareto_frontier(architectures)


# RUN THE ALGORITHM HERE
run_algorithm('CIFAR')

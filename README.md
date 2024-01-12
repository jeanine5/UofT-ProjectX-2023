# ProjectX-2023
## Introduction: EcoNAS
This project was built enitrley during the timeline of Uoft AI's ProjectX machine learning research competition. This was a solo project focused on reducing computational complexity of machine learning models. In particular, I used neural architecture search to explore interpretable and computationally efficient fully-connected, feed forwad DNNs trained and evaluated on the MNIST and CIFAR-10 datasets. To perform the NAS technique, I used the NSGA-2 algorithm as my search strategy and multi-objective optimization. This benchmark is titled EcoNAS and produced very convincing results. To read more about the procees, see the reserach paper.

## Running the Program

### Requirements
Please ensure you installed all the libraries and framworks in the requirements.txt. They are qucik and easy to install!

### Understadning the DIrectories
**************************************

#### EA
This directory contains the files for performing the evolutionary algorith, NSGA-2. I define the DNN architectures and paerto and genetic functions relevant to executing the evolve function in the NSGA-_II.py file.

#### Search Space
This directory contains several important aspects the generate the search space and compute/predict oobjective values. The precomputed objectives datasets work in combination with the predictor files to predict the values of hundreds of architectures using decision tree regession models.

#### Training
Here, you will find the main file, where you can run the entire program. PLease follow the below steps in order to run this without error:

1. Adjust the arguments in NSGA_II() and evolve() the run_algorithm function according to the reserach paper:
   a. For MNIST, hidden layers = 20, hidden size = 200
   b. For CIFAR-10, hidden layers = 15, hidden size = 128
   c. ONLY adjust the generatiosn, crossover and mutation arguments in the NSGA_II() function (ensure population_size is set to 250)
2. Find the flops_estimation() function in Architectures.py file and adjust the input argument:
   a. For MNIST, this is (1, 1, 28, 28), for CIFAR-10, this is (1, 3, 32, 32)
3. Run the algorithm as either run_algorithm('MNIST') or run_algorithm('CIFAR')
4. view the console for results!


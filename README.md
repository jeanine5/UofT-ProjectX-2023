# ProjectX-2023
## Introduction: EcoNAS
This project was built enitrley during the timeline of Uoft AI's ProjectX machine learning research competition. This was a solo project focused on reducing computational complexity of machine learning models. In particular, I used neural architecture search to explore interpretable and computationally efficient fully-connected, feed forwad DNNs trained and evaluated on the MNIST and CIFAR-10 datasets. To perform the NAS technique, I used the NSGA-2 algorithm as my search strategy and multi-objective optimization. This benchmark is titled EcoNAS and produced very convincing results. To read more about the procees, see the reserach paper.

## Running the Program

### Requirements
Please ensure you installed all the libraries and framworks in the requirements.txt. They are qucik and easy to install!

### Understadning the DIrectories

#### EA
This directory contains the files for performing the evolutionary algorith, NSGA-2. I define the DNN architectures and paerto and genetic functions relevant to executing the evolve function in the NSGA-_II.py file.

#### Search Space
This directory

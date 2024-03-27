# CUDA Advanced Libraries
## Project Description

In this project, we first implemented a handwritten classification of the MNIST dataset with one fully connected layer. The code is heavily based on the bool "Learning Deep Learning" by Markus Ekman. We then tweaked the parameters to see the influence of the neuron numbers on the learning curves. The pictures are train10.png, train25.png and train50.png, where it is clearly seen that increasing the neuron number leads to better results. The code was not run on GPUs. In order to run on GPU's we implemented a Convolution Neural Network heavily based on the book "Machine learning with Python" by Cuantum.
We obtained a similar graph cnn.png, as well as the verification that Keras was run on GPUs.

## Code Organization

```bin/```
This folder should hold all binary/executable code that is built automatically or manually. Executable code should have use the .exe extension or programming language-specific extension.

```data/```
This folder contains the MNITS handwritten classification data.

```lib/```
The folder is empty.

```src/```
The source code contains the two python files mentioned before, namely classication.py and cnn.py.

```README.md```
This file should hold the description of the project so that anyone cloning or deciding if they want to clone this repository can understand its purpose to help with their decision.

```INSTALL```
This file should hold the human-readable set of instructions for installing the code so that it can be executed. If possible it should be organized around different operating systems, so that it can be done by as many people as possible with different constraints.

```Makefile or CMAkeLists.txt or build.sh```
The project was made in python so there is no makefile.

```run.sh```
run.sh runs the cnn example.

# Data_Science_Python

Python is a great general-purpose programming language on its own, but with the help of a few popular libraries (numpy, scipy, matplotlib, tensorflow) it becomes a powerful environment for scientific computing and data analysis.

## Introduction to Python 

Python is a high-level, dynamically typed multi-paradigm programming language. Python code is often said to be almost like pseudocode, since it allows you to express very powerful ideas in very few lines of code while being very readable.
Here is the formal web of Python: https://www.python.org, you can find a lot of useful learning materials here.
Python Versions

There are currently two different supported versions of Python, 2.7.x version, and 3.5.x/3.6.x version. Somewhat confusingly, Python 3.x.x introduced many backwards-incompatible changes to the language, so code written for 2.7 may not work under 3.x and vice versa. For this class, all codes will use Python 3.6 on the most popular python platform Anaconda for data science.

## Install Anaconda

The reason why we use Anaconda instead of the original Python is because that the open source Anaconda Distribution is the easiest way to do Python data science and machine learning. 
It includes hundreds of popular data science packages, the conda package and virtual environment manager for Windows, Linux and MacOS. Conda makes it quick and easy to install, run and upgrade complex data science and machine learning environments like scikit-learn, TensorFlow and SciPy. Anaconda Distribution is the foundation of millions of data science projects as well as Amazon Web Services' Machine Learning AMIs and Anaconda for Microsoft on Azure and Windows.

- Step 1: Download Anaconda Distribution (Anaconda 5.0.1) Python 3.6 64-Bit version (Windows, MacOS or Linux): https://www.anaconda.com/download/.
- Step 2: Install Anaconda 5.0.1 with the default settings (unless you know what you are doing.)
- Step 3: Open Anaconda Navigator, you will see similar GUI as below, which means you have successfully installed Anaconda, congratulations! 



## Basic Anaconda 

One good feature about Anaconda is its virtual environment manager, it avoids the confusion of different version python interpreters mess your computer default or previous projects. In the follow section, we will learn the basic operations about using environments.
There are two ways to start the Conda Command Line Window:
- 1. On Windows, go to Start > Anaconda3(64bit) > Anaconda Prompt.
- 2. On the Anaconda Navigator, go to Environments >  Anaconda3 > Open Terminal.

### Conda Basics
In the Conda Command Window, type the following commands to get familiar with the Anaconda.
Verify conda is installed, check version number:
```python
conda info
```
Update conda to the current version:
```python
conda update conda
```
Command line help:
```python
conda install --help
```

### Using Environments

Python is still a developing programming language with an active community. The downside of this is that some predeveloped packages may not work in the newer python version. Yes, it is not always a good choice to use the most advantage technology in real project. 
As we know the current python version in `Environments>base(root)` is version 3.6.3. Assume that you join a team where everyone is working on a project in Python 3.5,  we need to create an independent developing environment to work with the team members without intervened by any issues caused by the Python version. 
The following steps will help you to install Python 3.5 in an Anaconda environment.
There are two ways to set up the environment of Python 3.5, one is in the Conda Command Window and the other one is on the Anaconda Navigator.

#### Conda Command Window
In the Conda Command Window, create a new environment named “py35”, install Python 3.5, by invoking the following command::
```
conda create --name py35 python=3.5
```
The anaconda will suggest that several packages will by installed, type “y” and wait the installation done.
Activate the conda environment by issuing the following command:
```
activate py35 
```
**LINUX, macOS: source activate py35**
Then, you can check your python version by type:
```
python --version 
```
You will get the results like: Python 3.5.4: Anaconda, Inc. This means that this environment is Python 3.5.

#### Installing and updating packages
Now, we fist need to list all packages and versions installed in active environment, enter the following command:
```
conda list
```
Assume we want to install a new package in the active environment (py35), enter the command:
```
conda install numpy
```
and then enter “y” to proceed.

Check the package Numpy is properly installed by running “conda list” again.Or you can try the following code to use the Numpy package.
```python
import numpy as np
print(np.version.version)
```
Update a package in the current environment, for example
```
conda update numpy
```
Deactivate the current environment:
```
deactivate
``` 
**LINUX, macOS: source deactivate**.







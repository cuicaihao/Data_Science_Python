# Data_Science_Python

Lastest Update: Dec 2020

Started: 25-April-2018

![Logo](./images/Logo.png)

Python is a great general-purpose programming language on its own, but with the help of a few popular libraries (numpy, scipy, matplotlib, tensorflow, pytorch) it becomes a powerful environment for scientific computing and data analysis.

I started the notebook repo as a beginner tutorial in 2018, then I updated it with more and more interesting topics around the python programming and applications.

This repo is designed with the following structure:

    ├── 00.Unsorted
    ├── 01.Python
    │ ├── Learn_Python_3_in_15mins.py
    │ ├── Python_01_Basic.ipynb
    │ ├── Python_02_Numpy.ipynb
    │ ├── Python_03_Read_and_Write_Files.ipynb
    │ ├── Python_04_Matplotlib_Data_Visualization\ .ipynb
    │ └── Python_05_Pandas.ipynb
    ├── 02.TensorFlow
    │ ├── Keras
    │ ├── TF1.x
    │ └── TF2.x
    ├── 03.PyTorch
    │ ├── PyTorch_01_Basics.ipynb
    │ ├── PyTorch_02_autograd_tutorial.ipynb
    │ ├── PyTorch_03_neural_networks_tutorial.ipynb
    │ ├── PyTorch_04_cifar10_tutorial.ipynb
    │ ├── PyTorch_05_data_loading_tutorial.ipynb
    │ ├── PyTorch_06_saving_loading_models.ipynb
    │ ├── PyTorch_07_transfer_learning_tutorial.ipynb
    │ ├── PyTorch_08_neural_style_tutorial.ipynb
    │ ├── PyTorch_09_deploy_seq2seq_hybrid_frontend_tutorial.ipynb
    │ └── nn_tutorial.ipynb
    ├── 04.ComputerVision
    ├── 05.NLP_TextMining
    │ ├── 00.intro_tensorflow_text.ipynb
    │ ├── 00.text_tensorflow_datasets.ipynb
    │ ├── 01.word_embeddings.ipynb
    │ ├── 01_Trained_Word_Embeddings_with_Gensim.ipynb
    │ ├── 02_Embedding_Songs_to_Make_Recommendations.ipynb
    │ ├── 03_Sentence_Classification_with_BERT.ipynb
    │ ├── 04.Neural\ machine\ translation\ with\ attention.ipynb
    │ ├── NLTK
    │ ├── Transformers_huggingface.ipynb
    │ ├── image_captioning.ipynb
    │ ├── jieba_text.ipynb
    │ ├── meta.tsv
    │ ├── text_classification_rnn.ipynb
    │ ├── text_generation.ipynb
    │ ├── transformer.ipynb
    │ └── vecs.tsv
    ├── 06.MachineLearning
    │ ├── Create\ your\ own\ Deep\ Learning\ framework\ using\ Numpy.ipynb
    │ ├── catboost_tutorial.ipynb
    │ ├── classifier.py
    │ ├── clustering.py
    │ ├── model.json
    │ ├── multiclass_model.json
    │ ├── plot_cluster_comparison.ipynb
    │ ├── plot_release_highlights_0_23_0.ipynb
    │ ├── regression.py
    │ └── regression.svg
    ├── 07.Data_Visualization
    │ ├── 01_Python_Bokeh.ipynb
    │ ├── Folium_Map_Melbourne_Index.html
    │ ├── Python\ Folium\ Map\ Demo.ipynb
    │ ├── example.py
    │ ├── lines.html
    │ └── python_progressbar.ipynb
    ├── 08.PySpark
    │ └── Demo.py
    ├── 09.WebService
    │ └── Tornado
    ├── 10.Algorithms_Data_Structure
    │ ├── Data\ Structures.ipynb
    │ ├── Dynamic\ Programming.ipynb
    │ ├── Graph\ Algorithm.ipynb
    │ ├── Prime\ Numbers\ and\ Prime\ Factorization.ipynb
    │ ├── Probability.ipynb
    │ ├── ReadMe.md
    │ ├── Searching_n_Sorting
    │ ├── **pycache**
    │ ├── longest-palindromic-substring.py
    │ └── playground.py
    ├── MIT\ License.txt
    ├── README.md
    ├── data
    └── images

## Introduction to Python

Python is a high-level, dynamically typed multi-paradigm programming language. Python code is often said to be almost like pseudocode, since it allows you to express very powerful ideas in very few lines of code while being very readable.
Here is the formal web of Python: https://www.python.org, you can find a lot of useful learning materials here.
Python Versions

There are currently two different supported versions of Python, 2.7.x version, and 3.5.x/3.6.x version. Somewhat confusingly, Python 3.x.x introduced many backwards-incompatible changes to the language, so code written for 2.7 may not work under 3.x and vice versa. For this class, all codes will use Python 3.6 on the most popular python platform Anaconda for data science.

## Install Anaconda

The reason why we use Anaconda instead of the original Python is because that the open source Anaconda Distribution is the easiest way to do Python data science and machine learning.
It includes hundreds of popular data science packages, the conda package and virtual environment manager for Windows, Linux and MacOS. Conda makes it quick and easy to install, run and upgrade complex data science and machine learning environments like scikit-learn, TensorFlow and SciPy. Anaconda Distribution is the foundation of millions of data science projects as well as Amazon Web Services' Machine Learning AMIs and Anaconda for Microsoft on Azure and Windows.

- Step 1: Download Anaconda Distribution (Anaconda) Python 3.6 (7.x, 8.x) 64-Bit version (Windows, MacOS or Linux): https://www.anaconda.com/download/.
- Step 2: Install Anaconda with the default settings (unless you know what you are doing.)
- Step 3: Open Anaconda Navigator, you will see similar GUI as below, which means you have successfully installed Anaconda, congratulations!

## Install TensorFlow

Follow the official documentation to install the TensorFlow package: https://www.tensorflow.org/install/ (TF 2.x)

After installation, type the following commands:

```python
import tensorflow as tf
print(tf.__version__) # this should be tensorflow >2.0
```

## Install PyTorch

Follow the official documentation to install the TensorFlow package: https://pytorch.org/get-started/locally/
After installation, type the following commands:

```python
import torch
print(torch.__version__) # this should be torch > 1.5.0
```

## Basic Anaconda

One good feature about Anaconda is its virtual environment manager, it avoids the confusion of different version python interpreters mess your computer default or previous projects. In the follow section, we will learn the basic operations about using environments.
There are two ways to start the Conda Command Line Window:

- 1. On Windows, go to Start > Anaconda3(64bit) > Anaconda Prompt.
- 2. On the Anaconda Navigator, go to Environments > Anaconda3 > Open Terminal.

### Conda Basics

In the Conda Command Window, type the following commands to get familiar with the Anaconda.
Verify conda is installed, check version number:

```
conda info
```

Update conda to the current version:

```
conda update conda
```

Command line help:

```
conda install --help
```

### Using Environments

Python is still a developing programming language with an active community. The downside of this is that some predeveloped packages may not work in the newer python version. Yes, it is not always a good choice to use the most advantage technology in real project.
As we know the current python version in `Environments>base(root)` is version 3.6.3. Assume that you join a team where everyone is working on a project in Python 3.5, we need to create an independent developing environment to work with the team members without intervened by any issues caused by the Python version.
The following steps will help you to install Python 3.5 in an Anaconda environment.
There are two ways to set up the environment of Python 3.5, one is in the Conda Command Window and the other one is on the Anaconda Navigator.

#### Conda Command Window

In the Conda Command Window, create a new environment named “py35”, install Python 3.6, by invoking the following command::

```
conda create --name py36 python=3.6
```

The anaconda will suggest that several packages will by installed, type “y” and wait the installation done.
Activate the conda environment by issuing the following command:

```
activate py36
```

**LINUX, macOS: source activate py36**
Then, you can check your python version by type:

```
python --version
```

You will get the results like: Python 3.6.X: Anaconda, Inc. This means that this environment is Python 3.6.

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

License
This **Data_Science_Python** is distributed under the MIT license (see MIT LICENCE.txt).

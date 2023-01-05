# Data_Science_Python
![Logo](./images/Logo.png)

Datetime:
- Started: 25-April-2018
- Revised: 05-Jan-2023

Time: Nov 2021:
- Review Pytorch 1.19. 
- Add PySpark with Scala and Python.
- Explore FastAPI features.

Python is a great general-purpose programming language on its own, but with the help of a few popular libraries (numpy, scipy, matplotlib, tensorflow, pytorch) it becomes a powerful environment for scientific computing and data analysis.

I started the notebook repo as a beginner tutorial in 2018, then I updated it with more and more interesting topics around the python programming and applications.

## Introduction to Python

Python is a high-level, dynamically typed multi-paradigm programming language. Python code is often said to be almost like pseudocode, since it allows you to express very powerful ideas in very few lines of code while being very readable.
Here is the formal web of Python: https://www.python.org, you can find a lot of useful learning materials here.
Python Versions

There are currently two different supported versions of Python, 2.7.x version, and 3.5.x/3.6.x version. Somewhat confusingly, Python 3.x.x introduced many backwards-incompatible changes to the language, so code written for 2.7 are NOT work under 3.x and vice versa. For this class, all codes will use Python 3.6 on the most popular python platform Anaconda for data science.

## Install Anaconda

The reason why we use Anaconda instead of the original Python is because that the open source Anaconda Distribution is the easiest way to do Python data science and machine learning.
It includes hundreds of popular data science packages, the conda package and virtual environment manager for Windows, Linux and MacOS. Conda makes it quick and easy to install, run and upgrade complex data science and machine learning environments like scikit-learn, TensorFlow and SciPy. Anaconda Distribution is the foundation of millions of data science projects as well as Amazon Web Services' Machine Learning AMIs and Anaconda for Microsoft on Azure and Windows.

- Step 1: Download Anaconda Distribution (Anaconda) Python 3.6 (7.x, 8.x, 9) 64-Bit version (Windows, MacOS or Linux): https://www.anaconda.com/download/.
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




This repo is designed with the following structure:
```bash
    ❯ tree 
    .
    ├── 00.Unsorted
    │   ├── 01_Python_Datashader_NYC taxi trips.ipynb
    │   ├── CircleProblem.py
    │   ├── Gibbs_Sampling.py
    │   ├── Lightbgm_demo.ipynb
    │   ├── Metropolis_Hastings_Sampling.py
    │   ├── Metropolis_Hastings_Sampling2.py
    │   ├── Python_with_Database_SQLite.ipynb
    │   ├── Unsorted Functions.ipynb
    │   ├── busyloop.py
    │   ├── busyloop.svg
    │   ├── checkpoint
    │   ├── deploy_seq2seq_hybrid_frontend_tutorial (1).py
    │   ├── edward_supervised_regression.ipynb
    │   ├── embedding_in_tk_sgskip.py
    │   ├── embedding_in_wx5_sgskip.py
    │   ├── finetuning_torchvision_models_tutorial.ipynb
    │   ├── finetuning_torchvision_models_tutorial.py
    │   ├── food_count.py
    │   ├── food_question.py
    │   ├── gpu_add.py
    │   ├── gpu_print.py
    │   ├── hellopython.py
    │   ├── helloworld.py
    │   ├── neural_style_tutorial.py
    │   ├── python-sql-database-schema.png
    │   ├── python-sql-database-schema.webp
    │   ├── python.thread.py
    │   ├── python.thread.svg
    │   ├── recursive.py
    │   ├── recursive.svg
    │   ├── scikit_plot_gpr_noisy_targets.ipynb
    │   ├── sm_app.sqlite
    │   ├── thread_names.py
    │   └── tkMatplot.py
    ├── 01.Python
    │   ├── LearnPython.py
    │   ├── Learn_Python_3_in_15mins.py
    │   ├── Python_01_Basic.ipynb
    │   ├── Python_02_Numpy.ipynb
    │   ├── Python_03_Read_and_Write_Files.ipynb
    │   ├── Python_04_Matplotlib_Data_Visualization .ipynb
    │   ├── Python_05_Pandas.ipynb
    │   ├── Python_06_Probability.ipynb
    │   ├── Python_07_Path.ipynb
    │   ├── Python_08_Numba.ipynb
    │   ├── Python_breakpoint.py
    │   ├── renameFileViaModificatedTime.py
    │   └── renameFiles.py
    ├── 02.TensorFlow
    │   ├── Keras
    │   │   ├── TensorFlow_Keras_01 Keras API.ipynb
    │   │   ├── TensorFlow_Keras_02 Basic classification.ipynb
    │   │   ├── TensorFlow_Keras_03 Predict house prices regression.ipynb
    │   │   ├── TensorFlow_Keras_04 Save and restore models.ipynb
    │   │   └── TensorFlow_Keras_05 Text classification with movie reviews.ipynb
    │   ├── TF1.x
    │   │   ├── TensorFlow_01_Graph_and_Session.ipynb
    │   │   ├── TensorFlow_02_Linear_Regression_with_TF.ipynb
    │   │   ├── TensorFlow_03_Saving and Restoring Models.ipynb
    │   │   ├── TensorFlow_04_Visualizing the Graph and Training Curves Using TensorBoard.ipynb
    │   │   ├── TensorFlow_05_Neural_Networks.ipynb
    │   │   ├── TensorFlow_06_Training_Deep_Nets.ipynb
    │   │   ├── TensorFlow_07_Transfer_Learning_(Reusing Pretrained Layers or Models).ipynb
    │   │   ├── TensorFlow_08_Faster_Optimizers_for_Building_Deep_Nets.ipynb
    │   │   ├── TensorFlow_09_Avoiding_Overfitting_by_Regularizations.ipynb
    │   │   ├── TensorFlow_10_Distributed_TF_Computation (Draft).ipynb
    │   │   ├── TensorFlow_11_Convolutional_Neural_Networks_(Deep ConvNets).ipynb
    │   │   ├── TensorFlow_12_Autoencoders.ipynb
    │   │   ├── Tensorflow_EX_01_Convolutional_Variational_Autoencoder.ipynb
    │   │   └── cvae.gif
    │   ├── TF2.x
    │   │   ├── 00.TF_basic_eager.ipynb
    │   │   ├── 01.TF_basic_tensor.ipynb
    │   │   ├── A0.beginner.ipynb
    │   │   ├── A1.classification.ipynb
    │   │   ├── ConceptualGraph.ipynb
    │   │   ├── README.md
    │   │   ├── TF.2x.Classification.ipynb
    │   │   ├── TF.2x.Overfit_and_Underfit.ipynb
    │   │   ├── TF.2x.Regression.ipynb
    │   │   ├── TF.2x.Save_and_Load.ipynb
    │   │   ├── TF.2x.Text_Classification.ipynb
    │   │   ├── TF.2x.Text_Classification_with_hub.ipynb
    │   │   ├── graphs.ipynb
    │   │   ├── saved_model
    │   │   │   └── my_model
    │   │   │       ├── assets
    │   │   │       ├── keras_metadata.pb
    │   │   │       ├── my_model.h5
    │   │   │       ├── saved_model.pb
    │   │   │       └── variables
    │   │   │           ├── variables.data-00000-of-00001
    │   │   │           └── variables.index
    │   │   ├── test.py
    │   │   ├── weights.data-00000-of-00001
    │   │   └── weights.index
    │   ├── TFJS
    │   │   ├── TFJS_Prediction2D
    │   │   │   ├── TFJS_Steps.md
    │   │   │   ├── index.html
    │   │   │   └── script.js
    │   │   ├── ml5.js
    │   │   │   └── hello-ml5
    │   │   │       ├── images
    │   │   │       │   ├── bird.png
    │   │   │       │   └── panda.jpeg
    │   │   │       ├── index.html
    │   │   │       ├── index_start.html
    │   │   │       └── sketch.js
    │   │   └── new.js
    │   └── TF_Cases
    │       └── TrafficSignClassification
    │           ├── Chris_TrafficSignClassifier-ApplyModel.ipynb
    │           ├── Chris_TrafficSignClassifier-GenerateModel.ipynb
    │           └── README.md
    ├── 03.PyTorch
    │   ├── PyTorch_00_quickstart_tutorial.ipynb
    │   ├── PyTorch_01_Basics.ipynb
    │   ├── PyTorch_02_autograd_tutorial.ipynb
    │   ├── PyTorch_03_neural_networks_tutorial.ipynb
    │   ├── PyTorch_04_cifar10_tutorial.ipynb
    │   ├── PyTorch_05_data_loading_tutorial.ipynb
    │   ├── PyTorch_06_saving_loading_models.ipynb
    │   ├── PyTorch_07_transfer_learning_tutorial.ipynb
    │   ├── PyTorch_08_neural_style_tutorial.ipynb
    │   ├── PyTorch_09_deploy_seq2seq_hybrid_frontend_tutorial.ipynb
    │   ├── data
    │   │   ├── FashionMNIST
    │   │   │   ├── processed
    │   │   │   │   ├── test.pt
    │   │   │   │   └── training.pt
    │   │   │   └── raw
    │   │   │       ├── t10k-images-idx3-ubyte
    │   │   │       ├── t10k-images-idx3-ubyte.gz
    │   │   │       ├── t10k-labels-idx1-ubyte
    │   │   │       ├── t10k-labels-idx1-ubyte.gz
    │   │   │       ├── train-images-idx3-ubyte
    │   │   │       ├── train-images-idx3-ubyte.gz
    │   │   │       ├── train-labels-idx1-ubyte
    │   │   │       └── train-labels-idx1-ubyte.gz
    │   │   └── mnist
    │   │       └── mnist.pkl.gz
    │   ├── data_tutorial.ipynb
    │   ├── helloPytorch.py
    │   ├── nn_tutorial.ipynb
    │   ├── tensorboard
    │   │   └── pytorch_tensorboard.ipynb
    │   └── torchvision_finetuning_instance_segmentation.ipynb
    ├── 04.ComputerVision
    │   ├── metrics_iou_dice.png
    │   ├── metrics_iou_dice_soft.png
    │   ├── segmentation_metrics.py
    │   └── segmentation_metrics_playground.ipynb
    ├── 05.NLP_TextMining
    │   ├── 00.intro_tensorflow_text.ipynb
    │   ├── 00.text_tensorflow_datasets.ipynb
    │   ├── 01.word_embeddings.ipynb
    │   ├── 01_Trained_Word_Embeddings_with_Gensim.ipynb
    │   ├── 02_Embedding_Songs_to_Make_Recommendations.ipynb
    │   ├── 03_Sentence_Classification_with_BERT.ipynb
    │   ├── 04.Neural machine translation with attention.ipynb
    │   ├── NLTK
    │   │   └── 00.language_processing_and_python.ipynb
    │   ├── Transformers_huggingface.ipynb
    │   ├── Word2Vec.ipynb
    │   ├── image_captioning.ipynb
    │   ├── jieba_text.ipynb
    │   ├── meta.tsv
    │   ├── text_classification_rnn.ipynb
    │   ├── text_generation.ipynb
    │   ├── transformer.ipynb
    │   └── vecs.tsv
    ├── 06.MachineLearning
    │   ├── Create your own Deep Learning framework using Numpy.ipynb
    │   ├── LightGBM_tutorial.ipynb
    │   ├── Scikit-Learn
    │   │   ├── classifier.py
    │   │   ├── clustering.py
    │   │   ├── plot_cluster_comparison.ipynb
    │   │   ├── plot_gradient_boosting_regression.ipynb
    │   │   ├── plot_kmeans_digits.ipynb
    │   │   ├── plot_release_highlights_0_23_0.ipynb
    │   │   ├── plot_release_highlights_0_24_0.ipynb
    │   │   ├── regression.py
    │   │   ├── rf_iris.onnx
    │   │   └── scikit_onnx.ipynb
    │   ├── catboost_tutorial.ipynb
    │   ├── model.json
    │   ├── multiclass_model.json
    │   └── regression.svg
    ├── 07.Data_Visualization
    │   ├── 01_Python_Bokeh.ipynb
    │   ├── Folium_Map_Melbourne_Index.html
    │   ├── Python Folium Map Demo.ipynb
    │   ├── SimpleTFboard_embeddings - Chris Check.ipynb
    │   ├── Tutorial_Dash_Layout
    │   │   ├── app.Button.py
    │   │   ├── app.CallBack.py
    │   │   ├── app.ChainCallback.py
    │   │   ├── app.DataSelection.py
    │   │   ├── app.GenericCrossfilterRecipe.py
    │   │   ├── app.InteractiveVIS.py
    │   │   ├── app.Markdown.py
    │   │   ├── app.MulltInput.py
    │   │   ├── app.MultOutput.py
    │   │   ├── app.PreventUpdate.py
    │   │   ├── app.PreventUpdate2.py
    │   │   ├── app.UpdateGraphsHover.py
    │   │   ├── app.py
    │   │   ├── app.scatter.py
    │   │   ├── app.slider.py
    │   │   └── app.table.py
    │   ├── example.py
    │   ├── lines.html
    │   ├── plot_all_scaling.ipynb
    │   └── python_progressbar.ipynb
    ├── 08.PySpark
    │   ├── 01.quickstart_Spark.ipynb
    │   ├── 02.Apache_Arrow.ipynb
    │   ├── BostonHousing.csv
    │   ├── C01.Colab_spark_basic_example.ipynb
    │   ├── C02.PySpark_Regression_Analysis.ipynb
    │   ├── Classification_Iris.py
    │   ├── Demo.py
    │   ├── Iris.csv
    │   ├── Regression_BostonHousing.py
    │   ├── WordCount.py
    │   ├── WordCount.scala
    │   ├── WordCount03.scala
    │   ├── WordCountDebug.scala
    │   ├── WordCount_Acc.scala
    │   ├── WordCount_BC.scala
    │   ├── bar.parquet
    │   │   ├── _SUCCESS
    │   │   ├── part-00000-2ba30f5f-bbc7-4539-8891-070f05679026-c000.snappy.parquet
    │   │   ├── part-00001-2ba30f5f-bbc7-4539-8891-070f05679026-c000.snappy.parquet
    │   │   ├── part-00002-2ba30f5f-bbc7-4539-8891-070f05679026-c000.snappy.parquet
    │   │   ├── part-00003-2ba30f5f-bbc7-4539-8891-070f05679026-c000.snappy.parquet
    │   │   ├── part-00004-2ba30f5f-bbc7-4539-8891-070f05679026-c000.snappy.parquet
    │   │   └── part-00005-2ba30f5f-bbc7-4539-8891-070f05679026-c000.snappy.parquet
    │   ├── foo.csv
    │   │   ├── _SUCCESS
    │   │   ├── part-00000-b61fcafb-58f4-44f3-8da5-e75419c08609-c000.csv
    │   │   ├── part-00001-b61fcafb-58f4-44f3-8da5-e75419c08609-c000.csv
    │   │   ├── part-00002-b61fcafb-58f4-44f3-8da5-e75419c08609-c000.csv
    │   │   ├── part-00003-b61fcafb-58f4-44f3-8da5-e75419c08609-c000.csv
    │   │   ├── part-00004-b61fcafb-58f4-44f3-8da5-e75419c08609-c000.csv
    │   │   └── part-00005-b61fcafb-58f4-44f3-8da5-e75419c08609-c000.csv
    │   ├── spark-warehouse
    │   ├── wikiOfSpark.txt
    │   └── zoo.orc
    │       ├── _SUCCESS
    │       ├── part-00000-0b8ab0b2-7cf9-4d5e-90b3-1075fe84b53b-c000.snappy.orc
    │       ├── part-00001-0b8ab0b2-7cf9-4d5e-90b3-1075fe84b53b-c000.snappy.orc
    │       ├── part-00002-0b8ab0b2-7cf9-4d5e-90b3-1075fe84b53b-c000.snappy.orc
    │       ├── part-00003-0b8ab0b2-7cf9-4d5e-90b3-1075fe84b53b-c000.snappy.orc
    │       ├── part-00004-0b8ab0b2-7cf9-4d5e-90b3-1075fe84b53b-c000.snappy.orc
    │       └── part-00005-0b8ab0b2-7cf9-4d5e-90b3-1075fe84b53b-c000.snappy.orc
    ├── 09.WebService
    │   ├── Tornado
    │   │   ├── chat
    │   │   │   ├── chatdemo.py
    │   │   │   ├── static
    │   │   │   │   ├── chat.css
    │   │   │   │   └── chat.js
    │   │   │   └── templates
    │   │   │       ├── index.html
    │   │   │       └── message.html
    │   │   └── helloTornado.py
    │   └── fastapi
    │       ├── __pycache__
    │       │   └── main.cpython-38.pyc
    │       └── main.py
    ├── 10.Algorithms_Data_Structure
    │   ├── 01.Intro_BinarySearch
    │   │   ├── __pycache__
    │   │   │   ├── binary_search.cpython-38.pyc
    │   │   │   └── test_binary_search.cpython-38-pytest-6.1.1.pyc
    │   │   ├── binary_search.py
    │   │   ├── items.json
    │   │   └── test_binary_search.py
    │   ├── 02.SelectionSort
    │   │   └── 01_selection_sort.py
    │   ├── 03.Recursion
    │   │   ├── 01_countdown.py
    │   │   ├── 02_greet.py
    │   │   ├── 03_factorial.py
    │   │   ├── 04_count.py
    │   │   ├── 05_binary_search_recursive.py
    │   │   ├── 06_find_max.py
    │   │   └── 07_sum_array.py
    │   ├── 04.QuickSort
    │   │   ├── __pycache__
    │   │   │   ├── a_loop_sum.cpython-38.pyc
    │   │   │   ├── b_recursive_sum.cpython-38.pyc
    │   │   │   ├── c_recursive_count.cpython-38.pyc
    │   │   │   ├── d_recursive_max.cpython-38.pyc
    │   │   │   ├── e_quicksort.cpython-38-pytest-6.1.1.pyc
    │   │   │   ├── e_quicksort.cpython-38.pyc
    │   │   │   ├── test_DC_QuckSort.cpython-38-pytest-6.1.1.pyc
    │   │   │   └── test_DC_QuckSort.cpython-38.pyc
    │   │   ├── a_loop_sum.py
    │   │   ├── b_recursive_sum.py
    │   │   ├── c_recursive_count.py
    │   │   ├── d_recursive_max.py
    │   │   ├── e_quicksort.py
    │   │   └── test_DC_QuckSort.py
    │   ├── 05.HashTable
    │   │   ├── a_price_of_groceries.py
    │   │   └── b_check_voter.py
    │   ├── 06.BreadthFirstSearch
    │   │   └── breadth-first_search.py
    │   ├── 07.Dijkstras
    │   │   ├── BellmanFord_algorithm.py
    │   │   ├── __pycache__
    │   │   │   ├── dijkstras_algorithm.cpython-38.pyc
    │   │   │   └── test_A.cpython-38-pytest-6.1.1.pyc
    │   │   ├── dijkstras_algorithm.py
    │   │   ├── dijkstras_demo.png
    │   │   ├── dijkstras_exercise.png
    │   │   └── test_A.py
    │   ├── 08.Greedy
    │   │   └── set_covering.py
    │   ├── 09.Dynamic_Programming
    │   │   ├── longest-palindromic-substring.py
    │   │   └── longest_common_subsequence.py
    │   ├── 10.KNN
    │   ├── CaseStudy
    │   │   ├── Data Structures.ipynb
    │   │   ├── Dynamic Programming.ipynb
    │   │   ├── Graph Algorithm.ipynb
    │   │   └── Prime Numbers and Prime Factorization.ipynb
    │   ├── ReadMe.md
    │   ├── Searching_n_Sorting
    │   │   ├──  longest-substring-without-repeating-characters.py
    │   │   ├── 00 BinarySearch.ipynb
    │   │   ├── 01 QuickSort.ipynb
    │   │   ├── 02 MergeSort.ipynb
    │   │   ├── 03 TimSort.ipynb
    │   │   ├── 04 Order Statistics.ipynb
    │   │   ├── 05 KMP Algorithm for Pattern Searching.ipynb
    │   │   ├── 05 Rabin-Karp Algorithm for Pattern Searching.ipynb
    │   │   ├── 06 Z algorithm Linear time pattern searching Algorithm.ipynb
    │   │   ├── 07 Aho-Corasick Algorithm for Pattern Searching.ipynb
    │   │   ├── 08 Counting Sort.ipynb
    │   │   ├── Appendix_Master_Theorem.ipynb
    │   │   ├── BinarySearch.py
    │   │   ├── QuickSort.py
    │   │   ├── QuickSortFlameGraph.svg
    │   │   └── __pycache__
    │   │       └── BinarySearch.cpython-38.pyc
    │   └── playground.py
    ├── 10.Geospatial_Data
    │   ├── Analyze_Geospatial_Data.ipynb
    │   ├── CaseStudy_Source1854CholeraOutbreak.ipynb
    │   ├── Create_Aerial_Sample.ipynb
    │   ├── Create_RS_Sample.ipynb
    │   ├── Data
    │   │   ├── Aerial
    │   │   │   ├── GT.png
    │   │   │   └── RGB.png
    │   │   ├── AerialImageDataset
    │   │   │   ├── gt
    │   │   │   │   └── austin1.tif
    │   │   │   ├── hongkong
    │   │   │   │   ├── cm
    │   │   │   │   │   ├── cm.png
    │   │   │   │   │   └── hongkong-cm.tif
    │   │   │   │   ├── dates.txt
    │   │   │   │   ├── hongkong.geojson
    │   │   │   │   ├── imgs_1
    │   │   │   │   │   ├── S2A_OPER_MSI_L1C_TL_SGS__20160927T081713_A006607_T49QHE_B01.tif
    │   │   │   │   │   ├── S2A_OPER_MSI_L1C_TL_SGS__20160927T081713_A006607_T49QHE_B02.tif
    │   │   │   │   │   ├── S2A_OPER_MSI_L1C_TL_SGS__20160927T081713_A006607_T49QHE_B03.tif
    │   │   │   │   │   ├── S2A_OPER_MSI_L1C_TL_SGS__20160927T081713_A006607_T49QHE_B04.tif
    │   │   │   │   │   ├── S2A_OPER_MSI_L1C_TL_SGS__20160927T081713_A006607_T49QHE_B05.tif
    │   │   │   │   │   ├── S2A_OPER_MSI_L1C_TL_SGS__20160927T081713_A006607_T49QHE_B06.tif
    │   │   │   │   │   ├── S2A_OPER_MSI_L1C_TL_SGS__20160927T081713_A006607_T49QHE_B07.tif
    │   │   │   │   │   ├── S2A_OPER_MSI_L1C_TL_SGS__20160927T081713_A006607_T49QHE_B08.tif
    │   │   │   │   │   ├── S2A_OPER_MSI_L1C_TL_SGS__20160927T081713_A006607_T49QHE_B09.tif
    │   │   │   │   │   ├── S2A_OPER_MSI_L1C_TL_SGS__20160927T081713_A006607_T49QHE_B10.tif
    │   │   │   │   │   ├── S2A_OPER_MSI_L1C_TL_SGS__20160927T081713_A006607_T49QHE_B11.tif
    │   │   │   │   │   ├── S2A_OPER_MSI_L1C_TL_SGS__20160927T081713_A006607_T49QHE_B12.tif
    │   │   │   │   │   └── S2A_OPER_MSI_L1C_TL_SGS__20160927T081713_A006607_T49QHE_B8A.tif
    │   │   │   │   ├── imgs_1_rect
    │   │   │   │   │   ├── B01.tif
    │   │   │   │   │   ├── B02.tif
    │   │   │   │   │   ├── B03.tif
    │   │   │   │   │   ├── B04.tif
    │   │   │   │   │   ├── B05.tif
    │   │   │   │   │   ├── B06.tif
    │   │   │   │   │   ├── B07.tif
    │   │   │   │   │   ├── B08.tif
    │   │   │   │   │   ├── B09.tif
    │   │   │   │   │   ├── B10.tif
    │   │   │   │   │   ├── B11.tif
    │   │   │   │   │   ├── B12.tif
    │   │   │   │   │   └── B8A.tif
    │   │   │   │   ├── imgs_2
    │   │   │   │   │   ├── T50QKK_20180323T024539_B01.tif
    │   │   │   │   │   ├── T50QKK_20180323T024539_B02.tif
    │   │   │   │   │   ├── T50QKK_20180323T024539_B03.tif
    │   │   │   │   │   ├── T50QKK_20180323T024539_B04.tif
    │   │   │   │   │   ├── T50QKK_20180323T024539_B05.tif
    │   │   │   │   │   ├── T50QKK_20180323T024539_B06.tif
    │   │   │   │   │   ├── T50QKK_20180323T024539_B07.tif
    │   │   │   │   │   ├── T50QKK_20180323T024539_B08.tif
    │   │   │   │   │   ├── T50QKK_20180323T024539_B09.tif
    │   │   │   │   │   ├── T50QKK_20180323T024539_B10.tif
    │   │   │   │   │   ├── T50QKK_20180323T024539_B11.tif
    │   │   │   │   │   ├── T50QKK_20180323T024539_B12.tif
    │   │   │   │   │   └── T50QKK_20180323T024539_B8A.tif
    │   │   │   │   ├── imgs_2_rect
    │   │   │   │   │   ├── B01.tif
    │   │   │   │   │   ├── B02.tif
    │   │   │   │   │   ├── B03.tif
    │   │   │   │   │   ├── B04.tif
    │   │   │   │   │   ├── B05.tif
    │   │   │   │   │   ├── B06.tif
    │   │   │   │   │   ├── B07.tif
    │   │   │   │   │   ├── B08.tif
    │   │   │   │   │   ├── B09.tif
    │   │   │   │   │   ├── B10.tif
    │   │   │   │   │   ├── B11.tif
    │   │   │   │   │   ├── B12.tif
    │   │   │   │   │   └── B8A.tif
    │   │   │   │   └── pair
    │   │   │   │       ├── img1.png
    │   │   │   │       └── img2.png
    │   │   │   └── train
    │   │   │       └── austin1.tif
    │   │   ├── Shape_NYC
    │   │   │   ├── taxi_zone_table.csv
    │   │   │   ├── taxi_zones.dbf
    │   │   │   ├── taxi_zones.prj
    │   │   │   ├── taxi_zones.qgz
    │   │   │   ├── taxi_zones.sbn
    │   │   │   ├── taxi_zones.sbx
    │   │   │   ├── taxi_zones.shp
    │   │   │   ├── taxi_zones.shp.xml
    │   │   │   └── taxi_zones.shx
    │   │   ├── copy.tif
    │   │   ├── copy2.tif
    │   │   └── new.tif
    │   ├── SnowGIS
    │   │   ├── Cholera_Deaths.dbf
    │   │   ├── Cholera_Deaths.prj
    │   │   ├── Cholera_Deaths.sbn
    │   │   ├── Cholera_Deaths.sbx
    │   │   ├── Cholera_Deaths.shp
    │   │   ├── Cholera_Deaths.shx
    │   │   ├── OSMap.tfw
    │   │   ├── OSMap.tif
    │   │   ├── OSMap_Grayscale.tfw
    │   │   ├── OSMap_Grayscale.tif
    │   │   ├── OSMap_Grayscale.tif.aux.xml
    │   │   ├── OSMap_Grayscale.tif.ovr
    │   │   ├── Pumps.dbf
    │   │   ├── Pumps.prj
    │   │   ├── Pumps.sbx
    │   │   ├── Pumps.shp
    │   │   ├── Pumps.shx
    │   │   ├── README.txt
    │   │   ├── SnowMap.tfw
    │   │   ├── SnowMap.tif
    │   │   ├── SnowMap.tif.aux.xml
    │   │   └── SnowMap.tif.ovr
    │   ├── SnowGIS_v2.zip
    │   ├── data.py
    │   ├── explore shape file.ipynb
    │   ├── gdal_raster_API_explore.ipynb
    │   └── images
    │       ├── longitude-and-latitude-simple.width-1200.jpg
    │       └── raster_vs_vector.jpg
    ├── 11.AutoML
    │   ├── Iris-data-profiling
    │   │   ├── Iris-data-profiling-full-user-design.html
    │   │   ├── Iris-data-profiling-full.html
    │   │   ├── Iris.csv
    │   │   ├── config_default.yaml
    │   │   ├── config_user.yaml
    │   │   ├── data_profiling.ipynb
    │   │   ├── database.sqlite
    │   │   └── iris.zip
    │   ├── Tune_PyTorch_Model_on_MNIST.ipynb
    │   └── data
    │       └── MNIST
    │           └── processed
    │               ├── test.pt
    │               └── training.pt
    ├── 12.Jupyter
    │   ├── HelloWorld.ipynb
    │   └── Jupyter_widgets.ipynb
    ├── 13.GUI_Design
    │   └── PySimpleGUI
    │       ├── A01_PySimpleGUI.py
    │       ├── A02_PySimpleGUI.py
    │       ├── B01_FileBrowse.py
    │       ├── Demo_Dashboard.py
    │       ├── GUI_Matplotlib.py
    │       ├── GUI_OpenCV.py
    │       └── ImageViewer.py
    ├── MIT License.txt
    ├── README.md
    └── images
        ├── Logo.png
        ├── autoencoders
        │   ├── linear_autoencoder_pca_plot.png
        │   ├── reconstruction_plot.png
        │   └── sparsity_loss_plot.png
        ├── dancing.jpg
        ├── fashion-mnist-sprite.png
        ├── nltk_download_gui.png
        ├── output_images.jpg
        └── picasso.jpg
```


# TensorFlow.js — Making Predictions from 2D Data

Source: https://codelabs.developers.google.com/codelabs/tfjs-training-regression/index.html#0 

## 1. Introduction

In this codelab you will train a model to make predictions from numerical data describing a set of cars.

This exercise will demonstrate steps common to training many different kinds of models, but will use a small dataset and a simple (shallow) model. The primary aim is to help you get familiar with the basic terminology, concepts and syntax around training models with TensorFlow.js and provide a stepping stone for further exploration and learning.

Because we are training a model to predict continuous numbers, this task is sometimes referred to as a regression task. We will train the model by showing it many examples of inputs along with the correct output. This is referred to as supervised learning.

###  What you will build
You will make a webpage that uses TensorFlow.js to train a model in the browser. Given "Horsepower" for a car, the model will learn to predict "Miles per Gallon" (MPG).

To do this you will:

- Load the data and prepare it for training.
Define the architecture of the model.
- Train the model and monitor its performance as it trains.
- Evaluate the trained model by making some predictions.

### What you'll learn
- Best practices for data preparation for machine learning, including shuffling and normalization.
- TensorFlow.js syntax for creating models using the tf.layers API.
- How to monitor in-browser training using the tfjs-vis library.

### What you'll need
- A recent version of Chrome or another modern browser.
- A text editor, either running locally on your machine or on the web via something like Codepen or Glitch.
- Knowledge of HTML, CSS, JavaScript, and Chrome DevTools (or your preferred browsers devtools).
- A high-level conceptual understanding of Neural Networks. If you need an introduction or refresher, consider watching this video by 3blue1brown or this video on Deep Learning in Javascript by Ashi Krishnan.


## 2. Get set up
Create an HTML page and include the JavaScript

https://codelabs.developers.google.com/codelabs/tfjs-training-regression/index.html#1 
Create an HTML page and include the JavaScript

Create the JavaScript file for the code
 
### Test it out
Now that you've got the HTML and JavaScript files created, test them out. Open up the index.html file in your browser and open up the devtools console.

If everything is working, there should be two global variables created and available in the devtools console.:

- `tf` is a reference to the `TensorFlow.js` library
- `tfvis` is a reference to the `tfjs-vis` library
Open your browser's developer tools, You should see a message that says Hello TensorFlow in the console output. If so, you are ready to move on to the next step.


## 3. Load, format and visualize the input data

As a first step let us load, format and visualize the data we want to train the model on.

We will load the "cars" dataset from a JSON file that we have hosted for you. It contains many different features about each given car. For this tutorial, we want to only extract data about Horsepower and Miles Per Gallon.

https://codelabs.developers.google.com/codelabs/tfjs-training-regression/index.html#2

###  Conceptualize our task
Our goal is to train a model that will take one number, Horsepower and learn to predict one number, Miles per Gallon. Remember that one-to-one mapping, as it will be important for the next section.

We are going to feed these examples, the horsepower and the MPG, to a neural network that will learn from these examples a formula (or function) to predict MPG given horsepower. This learning from examples for which we have the correct answers is called Supervised Learning.

## 4. Define the model architecture

https://codelabs.developers.google.com/codelabs/tfjs-training-regression/index.html#3 

In this section we will write code to describe the model architecture. Model architecture is just a fancy way of saying "which functions will the model run when it is executing", or alternatively "what algorithm will our model use to compute its answers".

ML models are algorithms that take an input and produce an output. When using neural networks, the algorithm is a set of layers of neurons with ‘weights' (numbers) governing their output. The training process learns the ideal values for those weights.


## 5. Prepare the data for training

https://codelabs.developers.google.com/codelabs/tfjs-training-regression/index.html#4 

To get the performance benefits of TensorFlow.js that make training machine learning models practical, we need to convert our data to tensors. We will also perform a number of transformations on our data that are best practices, namely shuffling and normalization.

### Shuffle the data
Here we randomize the order of the examples we will feed to the training algorithm. Shuffling is important because typically during training the dataset is broken up into smaller subsets, called batches, that the model is trained on. Shuffling helps each batch have a variety of data from across the data distribution. By doing so we help the model:

- Not learn things that are purely dependent on the order the data was fed in
- Not be sensitive to the structure in subgroups (e.g. if it only see high horsepower cars for the first half of its training it may learn a relationship that does not apply across the rest of the dataset).

### Convert to tensors
Here we make two arrays, one for our input examples (the horsepower entries), and another for the true output values (which are known as labels in machine learning). We then convert each array data to a 2d tensor. The tensor will have a shape of [num_examples, num_features_per_example]. Here we have inputs.length examples and each example has 1 input feature (the horsepower).
 

**Best Practice 1: You should always shuffle your data before handing it to the training algorithms in TensorFlow.js**

### Normalize the data

Next we do another best practice for machine learning training. We normalize the data. Here we normalize the data into the numerical range 0-1 using min-max scaling. Normalization is important because the internals of many machine learning models you will build with tensorflow.js are designed to work with numbers that are not too big. Common ranges to normalize data to include 0 to 1 or -1 to 1. You will have more success training your models if you get into the habit of normalizing your data to some reasonable range.


**Best Practice 2: You should always consider normalizing your data before training. Some datasets can be learned without normalization, but normalizing your data will often eliminate a whole class of problems that would prevent effective learning.**

**You can normalize your data before turning it into tensors. We do it afterwards because we can take advantage of vectorization in TensorFlow.js to do the min-max scaling operations without writing any explicit for loops.**

### Return the data and the normalization bounds
We want to keep the values we used for normalization during training so that we can un-normalize the outputs to get them back into our original scale and to allow us to normalize future input data the same way.


## 6. Train the model
https://codelabs.developers.google.com/codelabs/tfjs-training-regression/index.html#5 

With our model instance created and our data represented as tensors we have everything in place to start the training process.

### Prepare for training
We have to ‘compile' the model before we train it. To do so, we have to specify a number of very important things:

- optimizer: This is the algorithm that is going to govern the updates to the model as it sees examples. There are many optimizers available in TensorFlow.js. Here we have picked the adam optimizer as it is quite effective in practice and requires no configuration.

- loss: this is a function that will tell the model how well it is doing on learning each of the batches (data subsets) that it is shown. Here we use meanSquaredError to compare the predictions made by the model with the true values.


Next we pick a batchSize and a number of epochs:

- batchSize refers to the size of the data subsets that the model will see on each iteration of training. Common batch sizes tend to be in the range 32-512. There isn't really an ideal batch size for all problems and it is beyond the scope of this tutorial to describe the mathematical motivations for various batch sizes.

- epochs refers to the number of times the model is going to look at the entire dataset that you provide it. Here we will take 50 iterations through the dataset.

### Start the train loop
model.fit is the function we call to start the training loop. It is an asynchronous function so we return the promise it gives us so that the caller can determine when training is complete.

To monitor training progress we pass some callbacks to model.fit. We use tfvis.show.fitCallbacks to generate functions that plot charts for the ‘loss' and ‘mse' metric we specified earlier.

### Put it all together
Now we have to call the functions we have defined from our run function.
When you refresh the page, after a few seconds you should see the following graphs updating.

These are created by the callbacks we created earlier. They display the loss and mse, averaged over the whole dataset, at the end of each epoch.

When training a model we want to see the loss go down. In this case, because our metric is a measure of error, we want to see it go down as well.

## 7. Make Predictions

https://codelabs.developers.google.com/codelabs/tfjs-training-regression/index.html#6 

Now that our model is trained, we want to make some predictions. Let's evaluate the model by seeing what it predicts for a uniform range of numbers of low to high horsepowers.


Congratulations! You have just trained a simple machine learning model. It currently performs what is known as linear regression which tries to fit a line to the trend present in input data.

## 8. Main takeaways

The steps in training a machine learning model include:

Formulate your task:

- Is it a regression problem or a classification one?
- Can this be done with supervised learning or unsupervised learning?
- What is the shape of the input data? What should the output data look like?


Prepare your data:
- Clean your data and manually inspect it for patterns when possible
- Shuffle your data before using it for training
- Normalize your data into a reasonable range for the neural network. Usually 0-1 or -1-1 are good ranges for numerical data.
- Convert your data into tensors

Build and run your model:
- Define your model using tf.sequential or tf.model then add layers to it using tf.layers.*
- Choose an optimizer ( adam is usually a good one), and parameters like batch size and number of epochs.
- Choose an appropriate loss function for your problem, and an accuracy metric to help your evaluate progress. meanSquaredError is a common loss function for regression problems.
- Monitor training to see whether the loss is going down

Evaluate your model

- Choose an evaluation metric for your model that you can monitor while training. Once it's trained, try making some test predictions to get a sense of prediction quality.
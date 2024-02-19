# Neural network training

This week, you'll learn how to train your model in TensorFlow, and also learn about other important activation functions (besides the sigmoid function), and where to use each type in a neural network. You'll also learn how to go beyond binary classification to multiclass classification (3 or more categories). Multiclass classification will introduce you to a new activation function and a new loss function. Optionally, you can also learn about the difference between multiclass classification and multi-label classification. You'll learn about the Adam optimizer, and why it's an improvement upon regular gradient descent for neural network training. Finally, you will get a brief introduction to other layer types besides the one you've seen thus far.

## Learning Objectives

- Train a neural network on data using TensorFlow
- Understand the difference between various activation functions (sigmoid, ReLU, and linear)
- Understand which activation functions to use for which type of layer
- Understand why we need non-linear activation functions
- Understand multiclass classification
- Calculate the softmax activation for implementing multiclass classification
- Use the categorical cross entropy loss function for multiclass classification
- Use the recommended method for implementing multiclass classification in code
- (Optional): Explain the difference between multi-label and multiclass classification

## Neural Network Training

- [Reading - Slides](./Readings/C2_W2.pdf)

- [Video - TensorFlow implementation](https://www.coursera.org/learn/advanced-learning-algorithms/lecture/oL0HT/tensorflow-implementation)

- [Video - Training Details](https://www.coursera.org/learn/advanced-learning-algorithms/lecture/35RQ3/training-details)

## Activation Functions

- [Video - Alternatives to the sigmoid activation](https://www.coursera.org/learn/advanced-learning-algorithms/lecture/04Maa/alternatives-to-the-sigmoid-activation)

- [Video - Choosing activation functions](https://www.coursera.org/learn/advanced-learning-algorithms/lecture/aWivF/choosing-activation-functions)

- [Video - Why do we need activation functions?](https://www.coursera.org/learn/advanced-learning-algorithms/lecture/mb9bw/why-do-we-need-activation-functions)

- [Lab - ReLU activation](./Labs/C2_W2_Relu.ipynb)

## Multiclass Classification

- [Video - Multiclass](https://www.coursera.org/learn/advanced-learning-algorithms/lecture/4u2wC/multiclass)

- [Video - Softmax](https://www.coursera.org/learn/advanced-learning-algorithms/lecture/mzLuU/softmax)

- [Video - Neural Network with Softmax output](https://www.coursera.org/learn/advanced-learning-algorithms/lecture/ZQPG3/neural-network-with-softmax-output)

- [Video - Improved implementation of softmax](https://www.coursera.org/learn/advanced-learning-algorithms/lecture/Tyil1/improved-implementation-of-softmax)

- [Video - Classification with multiple outputs (Optional)](https://www.coursera.org/learn/advanced-learning-algorithms/lecture/pjIk0/classification-with-multiple-outputs-optional)

- [Lab - Softmax](./Labs/C2_W2_SoftMax.ipynb)

- [Lab - Multiclass](./Labs/C2_W2_Multiclass_TF.ipynb)

## Additional Neural Network Concepts

- [Video - Advanced Optimization](https://www.coursera.org/learn/advanced-learning-algorithms/lecture/5Qt9E/advanced-optimization)

- [Video - Additional Layer Types](https://www.coursera.org/learn/advanced-learning-algorithms/lecture/L0aFK/additional-layer-types)

## Back Propagation (Optional)

- [Video - What is a derivative? (Optional)](https://www.coursera.org/learn/advanced-learning-algorithms/lecture/i9Dqr/what-is-a-derivative-optional)

- [Video - Computation graph (Optional)](https://www.coursera.org/learn/advanced-learning-algorithms/lecture/rhcTZ/computation-graph-optional)

- [Video - Larger neural network example (Optional)](https://www.coursera.org/learn/advanced-learning-algorithms/lecture/qqczh/larger-neural-network-example-optional)

- [Lab - Derivatives](./Labs/C2_W2_Derivatives.ipynb)

- [Lab - Back propagation](./Labs/C2_W2_Backprop.ipynb)

## Practice Lab: Neural network training

- [Lab - Neural Networks for Multiclass classification](./Labs/C2_W2_Assignment.ipynb)
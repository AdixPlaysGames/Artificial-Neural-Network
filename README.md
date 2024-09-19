# Neural Network Implementation

This repository contains a fully customizable and modular implementation of an artificial neural network (ANN) built from scratch in Python. This implementation provides flexibility in architecture design, activation functions, and optimization techniques, making it suitable for both regression and classification tasks.

## Features

- **Customizable Architecture**: Define the number of layers and neurons per layer.
- **Multiple Activation Functions**: Choose from Sigmoid, ReLU, Tanh, Softmax, and Linear activation functions.
- **Weight Initialization**: Supports Xavier, He, and Uniform initialization methods.
- **Backpropagation**: Implements backpropagation with options for Momentum and RMSProp optimization.
- **Batch Training**: Supports mini-batch gradient descent for efficient training.
- **Monitoring**: Tracks training progress and visualizes training performance.

## How it Works

### 1. Forward Propagation

Forward propagation computes the activations for each layer using the formula:

$$
a^{(l)} = f(W^{(l)} a^{(l-1)} + b^{(l)})
$$

Where:
- $$a^{(l)}$$ is the activation of the $$l$$-th layer
- $$W^{(l)}$$ and $$b^{(l)}$$ are the weights and biases of the $$l$$-th layer
- $$f$$ is the activation function chosen for that layer.

### 2. Activation Functions

This implementation supports various activation functions:
- **Sigmoid**: $$\sigma(z) = \frac{1}{1 + e^{-z}}$$
- **ReLU**: $$\text{ReLU}(z) = \max(0, z)$$
- **Tanh**: $$\tanh(z) = \frac{e^z - e^{-z}}{e^z + e^{-z}}$$
- **Softmax**: $$\text{Softmax}(z) = \frac{e^z}{\sum e^z}$$
- **Linear**: $$f(z) = z$$

### 3. Weight Initialization

The network provides the following weight initialization methods:
- **Xavier**: 
$$W^{(l)} \sim N(0, \sqrt{\frac{1}{n_{\text{in}}}})$$
- **He**: 
$$W^{(l)} \sim N(0, \sqrt{\frac{2}{n_{\text{in}}}})$$
- **Uniform**: 
$$W^{(l)} \sim U(0, 1)$$

Where $$n_{\text{in}}$$ is the number of input neurons.

### 4. Loss Functions

- **Mean Squared Error (MSE)**:
$$\text{MSE} = \frac{1}{m} \sum_{i=1}^{m} (y^{(i)} - \hat{y}^{(i)})^2$$

Where:
- $$y^{(i)}$$ is the actual output
- $$\hat{y}^{(i)}$$ is the predicted output
- $$m$$ is the number of samples.

- **F-Measure**:
$$F1 = 2 \cdot \frac{ \text{precision} \cdot \text{recall} }{ \text{precision} + \text{recall} }$$

This measures the performance of a classification model based on precision and recall.

### 5. Backward Propagation

The backpropagation algorithm updates the weights and biases based on the gradient of the loss function with respect to the weights:

$$W^{(l)} = W^{(l)} - \frac{\alpha}{m} \sum_{i=1}^{m} \frac{\partial L}{\partial W^{(l)}}$$

Where:
- $$L$$ is the loss function
- $$\alpha$$ is the learning rate
- $$m$$ is the number of samples.

### 6. Optimizers

- **Momentum**: Helps accelerate gradients vectors in the right directions, leading to faster converging:
$$v_t = \beta v_{t-1} + (1 - \beta) \nabla J(\theta)$$
$$\theta = \theta - \alpha v_t$$

- **RMSProp**: Uses a decaying average of squared gradients to adjust the learning rate:
$$s_t = \beta s_{t-1} + (1 - \beta) \nabla J(\theta)^2$$
$$\theta = \theta - \alpha \frac{\nabla J(\theta)}{\sqrt{s_t + \epsilon}}$$

Where:
- $$\beta$$ is the decay rate
- $$\epsilon$$ is a small constant to prevent division by zero.

### 7. Batch Training

This implementation supports mini-batch gradient descent, where the dataset is split into smaller batches:

$$\text{Batch Loss} = \frac{1}{n_{\text{batch}}} \sum_{i=1}^{n_{\text{batch}}} L(y^{(i)}, \hat{y}^{(i)})$$

Where $$n_{\text{batch}}$$ is the number of samples in the batch.

## Usage

1. **Initialize the Network**: Define the architecture and choose activation functions.
2. **Train the Network**: Call the `train` method with your data, specifying learning rate, epochs, and batch size.
3. **Monitor Training**: Use the built-in methods to visualize training progress and performance.

## Example

```python
architecture = [10, 5, 1]  # 10 input neurons, 5 in hidden layer, 1 output
nn = NN(architecture, activations='relu', initialization='he', model_type='regression')
nn.train(X_train, y_train, X_test, y_test, learning_rate=0.001, epochs=1000)
```

# Implementation of multilayer perceptron in pure C++

> Based on http://code-spot.co.za/2009/10/08/15-steps-to-implemented-a-neural-net/ and http://neuralnetworksanddeeplearning.com/chap1.html 


## Simple architecture for predicting **Iris** species (one or more hidden layers is possible)

<img src="neural_net.png" width="50%" height="50%" align="middle">

## Example

```cpp
// before load data and encode labels into onehotEncoder...
auto trainTest = trainTestSplit(data, 0.2);
auto trainData = std::get<0>(trainTest);
auto testData = std::get<1>(trainTest);

std::vector<int> sizes = {4, 8, 3};
NeuralNet nn(sizes);

int epochs = 10
int miniBatchSize = 4
double eta = 0.1 // learning rate
nn.SGD(trainData, epochs, miniBatchSize, eta, testData);

double accuracy = nn.accuracy(testData);
std::cout << "Accuracy: " << accuracy << std::endl;
```

Output:
```
Epoch: 1/10 complete    Test loss: 0.123425
Epoch: 2/10 complete    Test loss: 0.112543
Epoch: 3/10 complete    Test loss: 0.107681
Epoch: 4/10 complete    Test loss: 0.104236
Epoch: 5/10 complete    Test loss: 0.10241
Epoch: 6/10 complete    Test loss: 0.10078
Epoch: 7/10 complete    Test loss: 0.0988746
Epoch: 8/10 complete    Test loss: 0.097413
Epoch: 9/10 complete    Test loss: 0.0962183
Epoch: 10/10 complete   Test loss: 0.0950714
Accuracy: 0.8
```


## Setup and run

tested on Ubuntu 18.04.2 LTS 
```console
cmake --build .
bin/neuralnet <epochs> <minibatchSize> <learningRate>
```

## Comparision to `PyTorch` and `Keras`

Look at **jupyter_notebooks** directory



**TODO**
- extend file loader to be able to load any dataset

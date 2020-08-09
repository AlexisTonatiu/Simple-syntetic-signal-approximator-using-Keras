# Simple synthetic/periodic signal approximator using Keras

A simple feed forward neural network architecture implemented using Tensorflow and Keras to approximate sine, square, triangular and saw-tooth functions from a 0-N vector (This was a Digital Signal Processing class assignment). 

<br>

---
The architecture that seemed to work the best was using 4 hidden dense layers, three with 128 neurons and reLU activation and the fourth one with 32 neurons.
I tried before with a less deep network and it worked as good as this model, so it can be reduced. Something like two hidden layers with 128 neurons might work.

The input layer is just one neuron and a vector from 0 to 2pi with N data points feeds the NN.
The output layer are 4 neurons since what the NN is expected to do is to approximate 4 functions from the input vector.


## Results
---

![alt text](https://github.com/AlexisTonatiu/Simple-syntetic-signal-approximator-using-Keras/blob/master/gensenales.png?raw=true)

This can be improved tho.

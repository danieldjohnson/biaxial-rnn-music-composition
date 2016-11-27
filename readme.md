# Biaxial Recurrent Neural Network for Music Composition

This code implements a recurrent neural network trained to generate classical music. The model, which uses LSTM layers and draws inspiration from convolutional neural networks, learns to predict which notes will be played at each time step of a musical piece.

You can read about its design and hear examples on [this blog post](http://www.hexahedria.com/2015/08/03/composing-music-with-recurrent-neural-networks/). 

## Requirements

This code is written in Python, and depends on having Theano and theano-lstm (which can be installed with pip) installed. The bare minimum you should need to do to get everything running, assuming you have Python, is
```
sudo pip install --upgrade theano
sudo pip install numpy scipy theano-lstm python-midi
```

In addition, the included setup scripts should set up the environment exactly as it was when I trained the network on an Amazon EC2 g2.2xlarge instance with an external EBS volume. Installing it with other setups will likely be slightly different.

## Using it

First, you will need to obtain a large selection of midi music, preferably in 4/4 time, with notes correctly aligned to beats. These can be placed in a directory "music".

To use the model, you need to first create an instance of the Model class:
```python
import model
m = model.Model([300,300],[100,50], dropout=0.5)
```
where the numbers are the sizes of the hidden layers in the two parts of the network architecture. This will take a while, as this is where Theano will compile its optimized functions.

Next, you need to load in the data:
```python
import multi_training
pcs = multi_training.loadPieces("music")
```

Then, after creating an "output" directory for trained samples, you can start training:
```python
multi_training.trainPiece(m, pcs, 10000)
```

This will train using 10000 batches of 10 eight-measure segments at a time, and output a sampled output and the learned parameters every 500 iterations.

Finally, you can generate a full composition after training is complete. The function `gen_adaptive` in main.py will generate a piece and also prevent long empty gaps by increasing note probabilities if the network stops playing for too long.
```python
gen_adaptive(m,pcs,10,name="composition")
```

There are also mechanisms to observe the hidden activations and memory cells of the network, but these are still a work in progress at the moment.

Right now, there is no separate validation step, because my initial goal was to produce interesting music, not to assess the accuracy of this method. It does, however, print out the cost on the training set after every 100 iterations during training.

# Custom LIF Single Layer

This directory (and its subdirectories) contain code for the implementation of the Leaky-Integrate-and-Fire (LIF layer as a poplar custom op.

It is structured as follows:

* `custom_lif_layer` contains the custom op code for a single LIF layer.
* `sparse2dense` contains custom op code that implements a transformation from a dense tensor (seq_len, batchsize, num_neurons) to the sparse tensor construct chosen in this project.
* `util_and_experiments` contains several files that impelment utility functions and different experiments using the custom ops, like a training rountine on the randman task.

Files:

* `tf_code.py` implements a basic (multi-layer) SNN and compares both the output as well as the gradient to a reference pure tensorflow and jax implementation.
* `profiling.py` simple script that implements a forwad+backward pass for profiling purposes

In order to run code first you have to compile both the custom lif layer as well as the `sparse2dense` operation. For that run

```make all```

in both subdirectories called `custom_lif_layer` as well as `sparse2dense`.

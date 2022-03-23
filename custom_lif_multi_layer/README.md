# Custom LIF Single Layer

This directory (and its subdirectories) contain code for the implementation of muliple Leaky-Integrate-and-Fire (LIF) layers as a single poplar custom op.

It is structured as follows:

* `custom_lif_layer_vectorize` contains the custom op code for a single LIF layer.
* `sparse2dense` contains custom op code that implements a transformation from a dense tensor (seq_len, batchsize, num_neurons) to the sparse tensor construct chosen in this project.

Files:

* `keras_timing.py` implements the functions `train_ipu` and `train_gpu`, that train a basic (multi-layer) SNN. For an example an see the `main()` function of the file, where it is executed on randomly generated data. If you don't use a tensorflow version that supports the ipu, uncomment the line `from tensorflow.python import ipu` in order to run the (non-ipu) code.

In order to run code that uses the IPU, first you have to compile both the custom lif layer as well as the `sparse2dense` operation. For that source the poplar-env and run

```make all```

in both subdirectories called `custom_lif_layer_vectorize` as well as `sparse2dense`.

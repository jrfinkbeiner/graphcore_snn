# Dense SNNs

This folder contains work to demonstatre the features required for the implementation of dense Spiking Neural Networks (SNNs) using tensorflow2.

### Files:

* `tf_dense_neuron_cells.py` implements the `LIFNeuron`, a recurrent leaky integrate and fire (LIF) neuron cell with a 2 dimensional state and dense output. 
* `snn_dense_layerwise.py` implements a multilayer spiking neural network in a sequential feed-forward fashion, consisting of `tensorflow.keras.layers.Dense`+`LIFNeuron` blocks, subsequently referenced as SNN block.
    This means every SNN block feeds it's ouput strictly forward to the next SNN-block. Two alternatives are implemented with the respective functions:
    1. `model_fn_sequential_multilayer` wraps every SNN block in a `tensorflow.keras.layers.RNN` layer and multiple layers are sequentially stacked.  
    2. `model_fn_sequential_multicell` wraps every SNN block block in a SNN-cell, which are then sequentially stacked in a single `tensorflow.keras.layers.RNN` layer.  
        Exclaimer: Can not be parallized onto multiple IPUs, as the whole forward loop is wrapped in a single `tensorflow.keras.layers.RNN` layer.
        
    Note: The model is trained on random data, therefore do not expect a decrease in the error after training.

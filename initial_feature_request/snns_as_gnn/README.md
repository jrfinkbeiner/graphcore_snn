# SNN as GNN

This folder contains work to illustrate the similarities of spiking neural networks (SNNs) and a graph neural network (GNN).

This means the SNN-network architecture is initilized from (directed) adjacency information of neighbouring neuron populations (snn-blocks/layers) and the computations within each timestep are independent from each other. Therefore the node-update computaions are (theoretically) executable fully in parallel with (sparse) communication between the nodes before/after each timestep. 

Goining forward, such an architecture should ideally be parallized to multiple IPUs, which is not possible in the current form.

### Files:

* `tf_dense_neuron_cells.py` implements the `LIFNeuron`, a recurrent leaky integrate and fire (LIF) neuron cell with a 2 dimensional state and sparse output. 
* `snn_as_gnn.py` implements a multilayer spiking neural network in a GNN framework (still, however, implemented as a `tensorflow.keras.layers.RNN`).  
    Note: The model is trained on random data, therefore do not expect a decrease in the error after training.

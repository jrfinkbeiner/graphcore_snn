import os
import sys

import numpy as np
import tensorflow as tf
import tensorflow.keras as keras

from tensorflow.python.training.tracking.data_structures import NoDependency
from tensorflow.python.framework.tensor_shape import TensorShape

# from tensorflow.python import ipu

from tf_dense_neuron_cells import LIFNeuron

root_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(root_path)
from data import get_data


class SNNBlock(tf.keras.layers.Layer):
    def __init__(self, out_dim, alpha, beta, gamma, u_thresh, refac_val):
        super().__init__()
        self.out_dim = out_dim

        self.linear = tf.keras.layers.Dense(self.out_dim)
        self.lif = LIFNeuron((self.out_dim,), alpha, beta, gamma, u_thresh, refac_val)

        # TODO jan: how to specify sparse TensorShape ?
        self.state_size = (self.lif.state_size, TensorShape(out_dim))
        self.output_size = self.lif.output_size

    def call(self, inp, state):
        x = self.linear(inp)
        x, lif_state_new = self.lif(x, state[0]) 
        state_new = (lif_state_new, x)
        return x, state_new



def build_snn_block(node_id, inp_conn, dims, *args, **kwargs):
    inp_conn_nodes = inp_conn[node_id]
    inp_dims = [dims[inp_id] for inp_id in inp_conn_nodes]
    inp_dim = sum(inp_dims)
    out_dim = dims[node_id]
    return SNNBlock(inp_dim, out_dim, *args, **kwargs)



class SNNasGNNCellModel(tf.keras.layers.Layer):
    def __init__(self, num_nodes, inp_connectivity, dims, inp_nodes, final_ids, alpha, beta, gamma, u_thresh, refac_val):
        super().__init__()

        assert len(inp_connectivity) == num_nodes
        if isinstance(dims, int):
            dims = [dims]*num_nodes
        assert len(dims) == num_nodes

        self.inp_nodes = inp_nodes
        self.num_nodes = num_nodes
        self.final_ids = final_ids
        self.inp_conn = inp_connectivity
        out_conn = []
        for i in range(num_nodes):
            out_conn_nodei = []
            for j,inp_conn_nodej in enumerate(self.inp_conn):
                if i in inp_conn_nodej:
                    out_conn_nodei.append(j)
            out_conn.append(out_conn_nodei)
        self.out_conn = out_conn

        self.snn_blocks = [SNNBlock(dims[inode], alpha, beta, gamma, u_thresh, refac_val) for inode in range(self.num_nodes)]

        self.state_size = [(snn_block.state_size, TensorShape(dims[inode])) for inode,snn_block in enumerate(self.snn_blocks)]
        self.output_size = [self.snn_blocks[fin_idx].output_size for fin_idx in self.final_ids]

    def call(self, inputs, state):

        state_new = []
        final_outs = []
        # Note: This loop is executable fully in parallel
        for inode in range(self.num_nodes):
            inps = [state[inp_idx][1] for inp_idx in self.inp_conn[inode]]
            if inode in self.inp_nodes:
                inps.append(inputs)
            inp = tf.concat(inps, axis=-1)
            out, stat = self.snn_blocks[inode](inp, state[inode][0]) 
            state_new.append((stat, out))
        
        for fin_id in self.final_ids:
            final_outs.append(state_new[fin_id][1])

        return final_outs, state_new

def model_fn_snn_as_gnn(dim, alpha, beta, gamma, u_thresh, reset_val):

    num_nodes = 5
    # set graph structure via inp connectivity
    # layer 0: [inp, 1]
    # layer 1: [0, 2, 3]
    # layer 2: [1, 2] # includes self recurrence
    # layer 3: [1, 2, 3]
    # layer 4: [3, 4] # includes self recurrence
    inp_connectivity = [[1], [0, 2, 3], [1, 2], [1, 2, 3], [3, 4]]
    inp_nodes = [0]
    final_ids = [4] # layer 4 is output layer, could be multiple layers
    dims = [dim]*num_nodes # could also be different for every layer
    
    input_layer = keras.Input(shape=(None, dim))
    x = tf.keras.layers.RNN(SNNasGNNCellModel(num_nodes, inp_connectivity, dims, inp_nodes, final_ids, alpha, beta, gamma, u_thresh, reset_val))(input_layer)
    return input_layer, x[0]

def main(args):
    # Variables for model hyperparameters

    batch_size = args.batch_size
    seq_len = 32

    dim = args.layer_dim  
    alpha = 0.9
    beta = 0.9
    gamma = 0.9
    u_thresh = 0.5
    reset_val = u_thresh
    
    # gen random data
    (x_train, y_train), (x_test, y_test), train_steps_per_execution, test_steps_per_execution = get_data(batch_size, seq_len, dim)

    # init model
    model = keras.Model(*model_fn_snn_as_gnn(dim, alpha, beta, gamma, u_thresh, reset_val))

    # Compile our model with Stochastic Gradient Descent as an optimizer
    # and Categorical Cross Entropy as a loss.
    model.compile('sgd', 'mse',
                metrics=["mse"],
                steps_per_execution=train_steps_per_execution)
    model.summary()
    

    print('\nTraining')
    model.fit(x_train, y_train, epochs=5, batch_size=batch_size)
    model.compile('sgd', 'categorical_crossentropy',
                metrics=["mse"],
                steps_per_execution=test_steps_per_execution)
    print('\nEvaluation')
    model.evaluate(x_test, y_test, batch_size=batch_size)

    print("Program ran successfully")

if __name__ == "__main__":
    
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--layer_dim", default=512, type=int, help="Number of nodes of each layer.")
    parser.add_argument("--batch_size", default=16, type=int, help="Batchsize to use.")
    args = parser.parse_args()
    
    main(args)
# POPLAR_ENGINE_OPTIONS='{"autoReport.all":"true", "autoReport.directory":"./report_multi_ipu_rnn"}' python3 multi_ipu_rnn.py

import numpy as np
import tensorflow as tf
import tensorflow.keras as keras

from tensorflow.python.training.tracking.data_structures import NoDependency
from tensorflow.python.framework.tensor_shape import TensorShape

from tensorflow.python import ipu

from tf_sparse_neuron_cells import LIFNeuron
from nn_sparse import SparseLinear



@tf.custom_gradient
def heaviside_with_super_spike_surrogate(x):
  spikes = tf.experimental.numpy.heaviside(x, 1)
  beta = 10.0
  
  def grad(upstream):
    return upstream * 1/(beta*tf.math.abs(x)+1)
  return spikes, grad



class SmallSNNCellModel(tf.keras.layers.Layer):
    def __init__(self, dim, exp_decay_constant, u_thresh, num_ipus, layers_per_ipu):
        super().__init__()

        self.num_ipus = num_ipus
        self.layers_per_ipu = layers_per_ipu
        hidden_dim = dim

        for i in range(self.num_ipus):
            # with ipu.keras.PipelineStage(i):
            for j in range(layers_per_ipu):
                setattr(self, f"dense_{i}_{j}", SparseLinear(hidden_dim, hidden_dim))
                setattr(self, f"neuroncell_{i}_{j}", LIFNeuron((hidden_dim,), exp_decay_constant, u_thresh, refac_val=1.0))

        self.state_size = [TensorShape((hidden_dim,)) for _ in range(layers_per_ipu*num_ipus)]
        self.output_size = dim

    def call(self, inputs, state):

        print(inputs)
        inputs = tf.sparse.from_dense(inputs)

        # states = state[1::2]
        # outs = state[0::2]
        states = state
        # outs = state[0::2]
        state_new = []

        x = inputs

        for i in range(self.num_ipus):
            with ipu.keras.PipelineStage(i):
                for j in range(self.layers_per_ipu):
                    print(i, j)
                    # inp = inputs if i==0 and j==0 else outs[i*self.layers_per_ipu+j-1]
                    # print(inp)
                    x = getattr(self, f"dense_{i}_{j}")(x)
                    print(x)

                    x, stat = getattr(self, f"neuroncell_{i}_{j}")(x, states[i*self.layers_per_ipu+j]) 
                    # state_new.extend((out, stat))
                    state_new.append(stat)


        return tf.sparse.to_dense(x), state_new


def SNNModel(*args, **kwargs):
    return tf.keras.layers.RNN(SmallSNNCellModel(*args, **kwargs))


def make_divisible(number, divisor):
    return number - number % divisor

def gen_data(zero_probability, train_data_len, seq_len, hidden_dim):
    data = np.random.random((train_data_len, seq_len, hidden_dim))
    data = (data > zero_probability).astype(np.float32)
    return data

def get_data(batch_size, seq_len, hidden_dim):

    train_data_len = 2**12 
    test_data_len = 2**12
    zero_probability = 0.8

    x_train = gen_data(zero_probability, train_data_len, seq_len, hidden_dim)
    x_test  = gen_data(zero_probability, test_data_len,  seq_len, hidden_dim)
    y_train = gen_data(zero_probability, train_data_len, seq_len, hidden_dim)[:, 0, ...]
    y_test  = gen_data(zero_probability, test_data_len,  seq_len, hidden_dim)[:, 0, ...]

    # Adjust dataset lengths to be divisible by the batch size
    train_steps_per_execution = train_data_len // batch_size
    train_steps_per_execution = make_divisible(train_steps_per_execution, 4) # For multi IPU
    train_data_len = make_divisible(train_data_len, train_steps_per_execution * batch_size)
    x_train, y_train = x_train[:train_data_len], y_train[:train_data_len]

    test_steps_per_execution = test_data_len // batch_size
    test_steps_per_execution = make_divisible(test_steps_per_execution, 4) # For multi IPU
    test_data_len = make_divisible(test_data_len, test_steps_per_execution * batch_size)
    x_test, y_test = x_test[:test_data_len], y_test[:test_data_len]

    return (x_train, y_train), (x_test, y_test), train_steps_per_execution, test_steps_per_execution


def model_fn(dim, exp_decay_constant, u_thresh, num_ipus, layers_per_ipu):

    exp_decay_constant = 0.9
    u_thresh = 0.5

    input_layer = keras.Input(shape=(None, dim))
    # x = RNNModel(shapes, num_classes, exp_decay_constant)(input_layer)
    x = SNNModel(dim, exp_decay_constant, u_thresh, num_ipus, layers_per_ipu)(input_layer)

    return input_layer, x
    

def main(args):
    # Variables for model hyperparameters
    batch_size = args.batch_size
    seq_len = 32
    
    dim = args.layer_dim  
    exp_decay_constant = 0.9
    u_thresh = 0.5
    layers_per_ipu = args.layers_per_ipu
    num_ipus = args.num_ipus
    
    # Load the MNIST dataset from keras.datasets
    (x_train, y_train), (x_test, y_test), train_steps_per_execution, test_steps_per_execution = get_data(batch_size, seq_len, dim)

    # set ipu config and strategy 
    ipu_config = ipu.config.IPUConfig()
    ipu_config.auto_select_ipus = num_ipus
    ipu_config.configure_ipu_system()

    strategy = ipu.ipu_strategy.IPUStrategy()

    with strategy.scope():
        # init model
        model = keras.Model(*model_fn(dim, exp_decay_constant, u_thresh, num_ipus, layers_per_ipu))

        # model.set_pipelining_options(gradient_accumulation_steps_per_replica=32) #2*num_ipus)
        # model.print_pipeline_stage_assignment_summary()

        # sys.exit()

        # Compile our model with Stochastic Gradient Descent as an optimizer
        # and Categorical Cross Entropy as a loss.
        model.compile('sgd', 'mse',
                    metrics=["mse"],
                    steps_per_execution=train_steps_per_execution)
        model.summary()
        

        print('\nTraining')
        model.fit(x_train, y_train, epochs=10, batch_size=batch_size)
        # model.compile('sgd', 'categorical_crossentropy',
        #             metrics=["mse"],
        #             steps_per_execution=test_steps_per_execution)
        # print('\nEvaluation')
        # model.evaluate(x_test, y_test, batch_size=batch_size)

    print("Program ran successfully")

if __name__ == "__main__":
    
    import argparse
    parser = argparse.ArgumentParser()

    parser.add_argument("--num_ipus", default=1, type=int, help="Number of ipus to be used.")
    parser.add_argument("--layers_per_ipu", default=2, type=int, help="Number of dense and SNN layers per ipu.")
    parser.add_argument("--layer_dim", default=1024, type=int, help="Number of nudoes of each layer.")
    parser.add_argument("--batch_size", default=2, type=int, help="Number of nudoes of each layer.")
    args = parser.parse_args()
    
    main(args)

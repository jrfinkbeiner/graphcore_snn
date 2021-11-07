import numpy as np
import tensorflow as tf
import tensorflow.keras as keras

from tensorflow.python.training.tracking.data_structures import NoDependency
from tensorflow.python.framework.tensor_shape import TensorShape

# from tensorflow.python import ipu

from tf_sparse_neuron_cells import LIFNeuron
from nn_sparse import SparseLinear


# TODO jan: temporary helper calss until proper sparse dataloader is set-up
# maybe not so temporary as keras.layers.RNN apparently can't handle tf.SparseTensors
class ToSparseCell(tf.keras.layers.Layer):
    def __init__(self, dim):
        super().__init__()
        self.state_size = 0
        self.output_size = dim
    
    def __call__(self, inp, stat):
        return tf.sparse.from_dense(inp), stat
    
# TODO jan: temporary helper calss until proper sparse dataloader is set-up
# maybe not so temporary as keras.layers.RNN apparently can't handle tf.SparseTensors
class ToDenseCell(tf.keras.layers.Layer):
    def __init__(self, dim):
        super().__init__()
        self.state_size = 0
        self.output_size = dim
    
    def __call__(self, inp, stat):
        return tf.sparse.to_dense(inp), stat

class SNNCell(tf.keras.layers.Layer):
    def __init__(self, inp_dim, out_dim, alpha, beta, gamma, u_thresh, refac_val, self_recurrent):
        super().__init__()
        self.inp_dim = inp_dim+out_dim if self_recurrent else inp_dim
        self.out_dim = out_dim
        self.self_recurrent = self_recurrent

        self.linear = SparseLinear(self.inp_dim, self.out_dim)
        self.lif = LIFNeuron((self.out_dim,), alpha, beta, gamma, u_thresh, refac_val)

        # TODO jan: how to specify sparse TensorShape ?
        self.state_size = (self.lif.state_size, TensorShape(out_dim)) if self.self_recurrent else self.lif.state_size
        self.output_size = self.lif.output_size

    def call(self, inp, state):

        if self.self_recurrent:
            rec_inp = tf.sparse.from_dense(state[1]) # TODO jan: necessary until solution for sparse TensorShape found
            inp = tf.sparse.concat(-1, [inp, rec_inp])
            lif_state = state[0]
        else: 
            lif_state = state
        
        x = self.linear(inp)
        x, lif_state_new = self.lif(x, lif_state) 

        # TODO jan: for now, convert spare spikes to dense until clear how to deal with TensorShape for sparse initial state
        state_new = (lif_state_new, tf.sparse.to_dense(x)) if self.self_recurrent else lif_state_new
        return x, state_new


def make_divisible(number, divisor):
    return number - number % divisor

def gen_data(zero_probability, train_data_len, seq_len, hidden_dim):
    data = np.random.random((train_data_len, seq_len, hidden_dim))
    data = (data > zero_probability).astype(np.float32)
    return data

def get_data(batch_size, seq_len, hidden_dim):

    train_data_len = 2**8
    test_data_len = 2**8
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


def model_fn_sequential_multilayer(num_layers, dim, alpha, beta, gamma, u_thresh, reset_val, self_recurrent):

    input_layer = keras.Input(shape=(None, dim))
    x = input_layer
    x = tf.keras.layers.RNN(ToSparseCell(dim), return_sequences=True)(x) # TODO jan: temporary fix
    for _ in range(num_layers):
        x = tf.keras.layers.RNN(SNNCell(dim, dim, alpha, beta, gamma, u_thresh, reset_val, self_recurrent), return_sequences=True)(x)
    x = tf.keras.layers.RNN(ToDenseCell(dim), return_sequences=False)(x) # TODO jan: temporary fix
    return input_layer, x

def model_fn_sequential_multicell(num_layers, dim, alpha, beta, gamma, u_thresh, reset_val, self_recurrent):

    snn_cells = [
        SNNCell(dim, dim, alpha, beta, gamma, u_thresh, reset_val, self_recurrent) for _ in range(num_layers)
    ]
    cells = [ToSparseCell(dim), *snn_cells, ToDenseCell(dim)] # TODO jan: temporary fix to enable handling of sparse tensors in RNN

    input_layer = keras.Input(shape=(None, dim))
    x = tf.keras.layers.RNN(cells)(input_layer)
    return input_layer, x


def main(args):
    # Variables for model hyperparameters

    num_layers = 4    
    batch_size = args.batch_size
    seq_len = 32
    self_recurrent = bool(args.self_recurrent)

    dim = args.layer_dim  
    alpha = 0.9
    beta = 0.9
    gamma = 0.9
    u_thresh = 0.5
    reset_val = u_thresh
    
    mode_to_model_fn = {
        "multicell": model_fn_sequential_multicell,
        "multilayer": model_fn_sequential_multilayer,
    }

    # gen random data
    (x_train, y_train), (x_test, y_test), train_steps_per_execution, test_steps_per_execution = get_data(batch_size, seq_len, dim)

    # # set ipu config and strategy 
    # ipu_config = ipu.config.IPUConfig()
    # ipu_config.auto_select_ipus = num_ipus
    # ipu_config.configure_ipu_system()

    # strategy = ipu.ipu_strategy.IPUStrategy()

    # with strategy.scope():
    # init model
    model = keras.Model(*mode_to_model_fn[args.mode](num_layers, dim, alpha, beta, gamma, u_thresh, reset_val, self_recurrent))
    # model.set_pipelining_options(gradient_accumulation_steps_per_replica=32) #2*num_ipus)

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
    parser.add_argument("--layer_dim", default=512, type=int, help="Number of nodes of each layer.")
    parser.add_argument("--batch_size", default=16, type=int, help="Batchsize to use.")
    parser.add_argument("--self_recurrent", default=0, type=int, help="Whether to use self recurrence or not. `int` will be cast to `bool` (default `0`, therefore `False`).")
    parser.add_argument("--mode", default="multicell", type=str, help="Whether to use `multilayer` or `multicell` approach to stack SNN-blocks.")
    args = parser.parse_args()
    
    assert args.mode == "multilayer" or args.mode == "multicell", f"Unknown mode, got '{args.mode}'."
    if args.mode == "multilayer":
        raise ValueError("`multilayer` approach currently not supported due to issues with handling of sparse tensors in `tensorflow.keras.layers.RNN`")

    main(args)
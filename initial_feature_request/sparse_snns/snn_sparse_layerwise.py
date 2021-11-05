import numpy as np
import tensorflow as tf
import tensorflow.keras as keras

from tensorflow.python.training.tracking.data_structures import NoDependency
from tensorflow.python.framework.tensor_shape import TensorShape

# from tensorflow.python import ipu

from tf_sparse_neuron_cells import LIFNeuron
from nn_sparse import SparseLinear


# TODO jan: temporary helper calss until proper sparse dataloader is set-up
class ToSparseCell(tf.keras.layers.Layer):
    def __init__(self, dim):
        super().__init__()
        self.state_size = 0
        self.output_size = dim
    
    def __call__(self, inp, stat):
        return tf.sparse.from_dense(inp), stat
    

class SNNCell(tf.keras.layers.Layer):
    def __init__(self, inp_dim, out_dim, alpha, beta, gamma, u_thresh, refac_val, self_recurrent):
        super().__init__()
        self.inp_dim = inp_dim+out_dim if self_recurrent else inp_dim
        self.out_dim = out_dim
        self.self_recurrent = self_recurrent

        self.linear = SparseLinear(self.inp_dim, self.out_dim)
        self.lif = LIFNeuron((self.out_dim,), alpha, beta, gamma, u_thresh, refac_val)

        self.state_size = (*self.lif.state_size, TensorShape(out_dim)) if self.self_recurrent else self.lif.state_size # TODO jan: SparseTensorShape ?!
        self.output_size = self.lif.output_size

    def call(self, inp, state):

        if self.self_recurrent:
            print(inp.shape)
            print(state[-1].shape)
            inp = tf.sparse.concat(-1, [inp, state[-1]])
            lif_state = state[:-1]
        else: 
            lif_state = state
        
        x = self.linear(inp)
        x, lif_state_new = self.lif(x, state) 

        state_new = (*lif_state_new, x) if self.self_recurrent else lif_state_new
        return x, state_new


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


def SNNLayer(*args, **kwargs):
    return tf.keras.layers.RNN(SNNCell(*args, **kwargs))

def model_fn_sequential_multilayer(num_layers, dim, alpha, beta, gamma, u_thresh, reset_val, self_recurrent):

    input_layer = keras.Input(shape=(None, dim))
    x = input_layer
    x = tf.keras.layers.RNN(ToSparseCell(dim), return_sequences=True)(x) # TODO jan: temporary fix until sparse dataloader is set up
    for _ in range(num_layers):
        x = tf.keras.layers.RNN(SNNCell(dim, dim, alpha, beta, gamma, u_thresh, reset_val, self_recurrent), return_sequences=True)(x)

    # only last item of sequence
    x = tf.sparse.to_dense(x)[:, -1, :] # TODO jan: temporary fix until sparse dataloader is set up
    return input_layer, x


def model_fn_sequential_multicell(num_layers, dim, alpha, beta, gamma, u_thresh, reset_val, self_recurrent):

    snn_cells = [
        SNNCell(dim, dim, alpha, beta, gamma, u_thresh, reset_val, self_recurrent) for _ in range(num_layers)
    ]
    cells = [ToSparseCell(dim), *snn_cells] # TODO jan: temporary fix until sparse dataloader is set up

    input_layer = keras.Input(shape=(None, dim))
    x = tf.keras.layers.RNN(cells)(input_layer)
    x = tf.sparse.to_dense(x)
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
    
    # gen random data
    (x_train, y_train), (x_test, y_test), train_steps_per_execution, test_steps_per_execution = get_data(batch_size, seq_len, dim)

    # # set ipu config and strategy 
    # ipu_config = ipu.config.IPUConfig()
    # ipu_config.auto_select_ipus = num_ipus
    # ipu_config.configure_ipu_system()

    # strategy = ipu.ipu_strategy.IPUStrategy()

    # with strategy.scope():
    # init model
    model = keras.Model(*model_fn_sequential_multilayer(num_layers, dim, alpha, beta, gamma, u_thresh, reset_val, self_recurrent))
    # model = keras.Model(*model_fn_sequential_multicell(num_layers, dim, alpha, beta, gamma, u_thresh, reset_val, self_recurrent))
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
    parser.add_argument("--self_recurrent", default=0, type=int, help="Whether to use self recurrence or not. Int will be cast to bool (default 0, therefore False).")
    args = parser.parse_args()
    
    assert not bool(args.self_recurrent), "Currently self recurrence is not supported due to issues in the handling of TensorShapes for initial internal state."

    main(args)
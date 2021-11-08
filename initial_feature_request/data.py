import numpy as np

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
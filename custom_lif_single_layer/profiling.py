import os
import numpy as np
import tensorflow.compat.v1 as tf
import functools as ft

# import tensorflow.keras as keras
from tensorflow.python import ipu
tf.disable_v2_behavior()

from util_and_experiments.custom_snn import init_func_tf, init_func_np, snn_init_weights_func, get_multi_layer_snn, get_sgd, get_calc_loss_and_grad, get_update_func


def calc_loss(out_spikes_dense, targets_one_hot):
    sum_spikes = tf.math.reduce_sum(out_spikes_dense, axis=0)
    norm_sum_spikes = tf.nn.softmax(sum_spikes, axis=1)
    return tf.math.reduce_sum((norm_sum_spikes-targets_one_hot)**2) / out_spikes_dense.shape[0].value

def gen_sparse_spikes(rng, seq_len, batchsize, size_dense, size_sparse):
    sparse_spike_ids = np.empty((seq_len, batchsize, size_sparse))
    for ibatch in range(batchsize):
        for iseq in range(seq_len):
            sparse_spike_ids[iseq, ibatch, :] = rng.choice(size_dense, size_sparse, replace=False)
    num_sparse_spikes = rng.choice(size_sparse, (seq_len, batchsize, 1), replace=True).astype(np.int32)
    return sparse_spike_ids, num_sparse_spikes


def main(args):
    use_ipu_model = bool(args.use_ipu_model)
    seq_len = args.seq_len
    batchsize = args.batchsize
    size_dense = args.size_dense
    size_sparse = args.size_sparse
    num_hidden_layers = args.num_hidden_layers

    decay_constant = 0.95
    threshold = 1.0

    learning_rate = 1e-1
    momentum_alpha = None

    assert size_dense >= size_sparse, f"`size_dense` must be greater or equal to `size_sparse`, got {size_dense} and {size_sparse}."

    if use_ipu_model:
        os.environ["TF_POPLAR_FLAGS"] = "--use_ipu_model"

    cfg = ipu.config.IPUConfig()
    cfg.auto_select_ipus = 1
    cfg.configure_ipu_system()

    sizes_dense = [size_dense]*(num_hidden_layers+1)
    sizes_sparse = [size_sparse]*(num_hidden_layers+1)

    with tf.device("cpu"):
        weights_pl, init_states_pl, inp_spike_ids_pl, num_inp_spikes_pl, decay_constants_pl, thresholds_pl = init_func_tf(sizes_dense, batchsize, seq_len, sizes_sparse[0])
        targets_one_hot_pl = tf.placeholder(np.float32, [batchsize, sizes_dense[-1]])
        opt_state_pl = tf.placeholder(np.float32, [1]) # Not used as momentum is None/zero

    rng = np.random.default_rng(1)
    init_weight_func = ft.partial(snn_init_weights_func, rng, threshold, size_sparse/(2*seq_len))
    weights, init_states, decay_constants, thresholds = init_func_np(sizes_dense, batchsize, decay_constant, threshold, init_weight_func)
    inp_spike_ids, num_inp_spikes = gen_sparse_spikes(rng, seq_len, batchsize, size_dense, size_sparse)
    targets_one_hot_batch = np.zeros((batchsize, sizes_dense[-1]), dtype=np.float32)
    opt_state = np.zeros(1, dtype=np.float32) # Not used as momentum is None/zero

    snn_func = get_multi_layer_snn(sizes_sparse[1:])
    reg_fn = None
    _, opt = get_sgd(weights, learning_rate, momentum_alpha)

    update_func = get_update_func(
        loss_and_grad_func=get_calc_loss_and_grad(snn_func, calc_loss, reg_fn),
        optimizer=opt,
    )

    with ipu.scopes.ipu_scope("/device:IPU:0"):
        xla_result = ipu.ipu_compiler.compile(update_func, [weights_pl, init_states_pl, inp_spike_ids_pl, num_inp_spikes_pl, decay_constants_pl, thresholds_pl, targets_one_hot_pl, opt_state_pl])

    with tf.Session() as sess:
        (loss, aux), weights, opt_state = sess.run(xla_result, {
            weights_pl:weights, 
            init_states_pl:init_states, 
            inp_spike_ids_pl:inp_spike_ids, 
            num_inp_spikes_pl:num_inp_spikes, 
            decay_constants_pl:decay_constants, 
            thresholds_pl:thresholds,
            targets_one_hot_pl:targets_one_hot_batch,
            opt_state_pl:opt_state,
        })

    print("Succesful run!")


if __name__ == "__main__":
    
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--use_ipu_model", default=0, type=int, help="Whether to use an IPUModel (CPU) or an actual IPU.")
    parser.add_argument("--seq_len", default=32, type=int, help="Sequence length")
    parser.add_argument("--batchsize", default=8, type=int, help="Batchsize to be used.")
    parser.add_argument("--size_dense", default=512, type=int, help="Size (number of neurons) that each layer contains.")
    parser.add_argument("--size_sparse", default=32, type=int, help="Size of the sparse input and output spike tensors.")
    parser.add_argument("--num_hidden_layers", default=1, type=int, help="Number of hidden layers.")
    args = parser.parse_args()
    
    main(args)
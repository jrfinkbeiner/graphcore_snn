import numpy as np
import tensorflow as tf

@tf.custom_gradient
def heaviside_with_super_spike_surrogate(x):
  spikes = tf.experimental.numpy.heaviside(x, 1)
  beta = 10.0
  
  def grad(upstream):
    return upstream * 1/(beta*tf.math.abs(x)+1)
  return spikes, grad

def pure_tf_lif_step(weights, state, inp_, decay_constants, thresholds):
    # syn_inp = tf.einsum("ij,kj->ki", weights, inp_)
    syn_inp = tf.matmul(inp_, weights, transpose_b=True)
    state = state - tf.stop_gradient(state * tf.experimental.numpy.heaviside(state-thresholds, 0))
    new_state = state * decay_constants + (1 - decay_constants) * syn_inp
    # new_state = decay_constants*state * tf.experimental.numpy.heaviside(thresholds-state, 1) + (1-decay_constants)*syn_inp
    spikes_out = heaviside_with_super_spike_surrogate(new_state-thresholds)
    return spikes_out, new_state

def pure_tf_lif_layer(weights, init_state, inp_spikes, decay_constants, thresholds):
    seq_len = inp_spikes.shape[0]

    states, outs = [], []
    state = init_state
    for iseq in range(seq_len):
        out, state = pure_tf_lif_step(weights, state, inp_spikes[iseq], decay_constants, thresholds)
        states.append(state)
        outs.append(out)
    states = tf.stack(states, axis=0)
    outs = tf.stack(outs, axis=0)

    return outs, states # only last item of sequence

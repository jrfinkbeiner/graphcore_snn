from typing import Sequence, Union, Tuple, Any, Callable, Optional

import tensorflow as tf
from tensorflow.python.training.tracking.data_structures import NoDependency
from tensorflow.python.framework.tensor_shape import TensorShape

MoreArrays = Union[tf.Tensor, tf.TensorArray]
SparseArrays = tf.SparseTensor

@tf.custom_gradient
def heaviside_with_super_spike_surrogate(x):
  spikes = tf.experimental.numpy.heaviside(x, 1)
  beta = 10.0
  
  def grad(upstream):
    return upstream * 1/(beta*tf.math.abs(x)+1)
  return spikes, grad


class LIFNeuron(tf.keras.layers.Layer):

    def __init__(self,
            shape: Union[int, Sequence[int]],
            alpha: Union[float, MoreArrays], # TODO decide whether neuron specific decay constant, or not
            beta: Union[float, MoreArrays],
            gamma: Union[float, MoreArrays],
            u_thresh: Union[float, MoreArrays],
            reset_val: Union[float, MoreArrays],
            stop_reset_grad: Optional[bool] = True,
            spike_func: Callable[[MoreArrays], MoreArrays] = heaviside_with_super_spike_surrogate,
            state2_func: Optional[Callable[[MoreArrays], MoreArrays]] = lambda x: x,
        ) -> None:

        super().__init__()
        assert tf.math.reduce_all(alpha > 0) and tf.math.reduce_all(alpha < 1)
        assert tf.math.reduce_all(beta > 0) and tf.math.reduce_all(beta < 1)
        assert tf.math.reduce_all(gamma > 0) and tf.math.reduce_all(gamma < 1)
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.u_thresh = u_thresh
        self.spike_func = spike_func
        self.state2_func = state2_func
        self.reset_val = reset_val
        self.stop_reset_grad = stop_reset_grad

        self.state_size = (TensorShape(shape), TensorShape(shape))
        self.output_size = TensorShape(shape)

    def __call__(self, synaptic_input: MoreArrays, state: Tuple[MoreArrays, MoreArrays]) -> Tuple[Any, Any]:

        state1, state2 = state

        state1_new = self.alpha*state1 + (1-self.alpha) * (synaptic_input + self.state2_func(state2))
        state2_new = self.beta * state2 + (1-self.beta) * self.gamma * state1
        # TODO jan: use u_thresh or state2_new here? if not u_thresh, u_thresh is an unneseccary paramter
        spikes_out = self.spike_func(state1_new-state2_new)

        # TODO jan: don't know how else/less hacky to do it...
        reset_pot = -spikes_out*self.reset_val
        
        # optionally stop gradient propagation through reset potential       
        reset_pot = tf.stop_gradient(reset_pot) if self.stop_reset_grad else reset_pot
        state1_new = state1_new + tf.stop_gradient(reset_pot)

        output = spikes_out
        state_new = (state1_new, state2_new)
        return output, state_new

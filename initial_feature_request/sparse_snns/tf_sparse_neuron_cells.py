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
            v_func: Optional[Callable[[MoreArrays], MoreArrays]] = lambda x: x,
        ) -> None:

        super().__init__()

        assert tf.math.reduce_all(alpha > 0) and tf.math.reduce_all(alpha < 1) # TODO test assertion
        assert tf.math.reduce_all(beta > 0) and tf.math.reduce_all(beta < 1) # TODO test assertion
        assert tf.math.reduce_all(gamma > 0) and tf.math.reduce_all(gamma < 1) # TODO test assertion
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.u_thresh = u_thresh
        self.spike_func = spike_func
        self.v_func = v_func
        self.reset_val = reset_val
        self.stop_reset_grad = stop_reset_grad

        # self.shape = (shape,) if isinstance(shape, int) else shape
        self.state_size = (TensorShape(shape), TensorShape(shape))
        self.output_size = TensorShape(shape)


    def __call__(self, synaptic_input: MoreArrays, state: Tuple[MoreArrays, MoreArrays]) -> Tuple[Any, Any]:

        membrane_potential, V = state

        membrane_potential_new = self.alpha*membrane_potential + (1-self.alpha) * synaptic_input + self.v_func(V)
        V_new = self.beta * V + self.gamma * membrane_potential
        # TODO jan: use u_thresh or V_new here? if not u_thresh, u_thresh is an unneseccary paramter
        spikes_out = tf.sparse.from_dense(self.spike_func(membrane_potential_new-V_new)) 

        # TODO jan: don't know how else/less hacky to do it...
        reset_pot = tf.sparse.map_values(tf.math.multiply, spikes_out, -self.reset_val)
        
        # optionally stop gradient propagation through reset potential       
        # TODO jan: tf.stop_gradient doesn't seem to support sparse tensors
        if self.stop_reset_grad:
            reset_pot = tf.sparse.to_dense(reset_pot)
            membrane_potential = membrane_potential + tf.stop_gradient(reset_pot)
        else:
            membrane_potential = tf.sparse.add(membrane_potential, reset_pot)

        output = spikes_out
        state_new = (membrane_potential_new, V_new)
        return output, state_new

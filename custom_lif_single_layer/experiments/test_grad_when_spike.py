import functools as ft
import numpy as np
import jax
import jax.numpy as jnp
from jax import custom_jvp
from jax import vmap, value_and_grad, jacrev


def get_heaviside_with_super_spike_surrogate(beta=10.):

    @custom_jvp
    def heaviside_with_super_spike_surrogate(x):
        return jnp.heaviside(x, 0)

    @heaviside_with_super_spike_surrogate.defjvp
    def f_jvp(primals, tangents):
        x, = primals
        x_dot, = tangents
        primal_out = heaviside_with_super_spike_surrogate(x)
        tangent_out = 1/(beta*jnp.abs(x)+1) * x_dot
        return primal_out, tangent_out

    return heaviside_with_super_spike_surrogate

def lif_step_jax(weights, decay_constants, thresholds, state, inp_spikes):
    syn_inp = weights @ inp_spikes
    state = state - jax.lax.stop_gradient(state * jnp.heaviside(state-thresholds, 0))
    new_state = state * decay_constants + (1 - decay_constants) * syn_inp
    # new_state = state * decay_constants * jax.lax.stop_gradient(jnp.heaviside(thresholds-state, 1)) + (1 - decay_constants) * syn_inp
    out_spikes = get_heaviside_with_super_spike_surrogate()(new_state-thresholds)
    return new_state, (new_state, out_spikes)


size_in = 4
size_out = 3

weights = np.ones((size_out, size_in))
state  = np.array([0.5, 0.8, 1.1])
inp_spikes = np.zeros(size_in)
decay_constants = 0.95*np.ones(size_out)
thresholds = np.ones(size_out)


def update_state(weights, decay_constants, thresholds, state, inp_spikes):
    new_state, _ = lif_step_jax(weights, decay_constants, thresholds, state, inp_spikes)
    return new_state

# new_state = update_state(weights, decay_constants, thresholds, state, inp_spikes)

jac = jacrev(update_state, argnums=3)(weights, decay_constants, thresholds, state, inp_spikes)

print(jac)

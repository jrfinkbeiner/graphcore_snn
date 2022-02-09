import functools as ft
import jax
import jax.numpy as jnp
from jax import custom_jvp
from jax import vmap, value_and_grad


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


def lif_layer_jax(weights, init_state, inp_spikes, decay_constants, thresholds):
    scan_fn = ft.partial(lif_step_jax, weights, decay_constants, thresholds)
    fin_state, outs = jax.lax.scan(scan_fn, init_state, inp_spikes)
    states, out_spikes = outs
    return states, out_spikes


def calc_loss(weights, init_states, inp_spikes, decay_constants, thresholds):
    spikes = inp_spikes
    for ws, decay_consts, threshs, init_stat in zip(weights, decay_constants, thresholds, init_states):
        states, spikes = vmap(lif_layer_jax, in_axes=(None, 0, 1, None, None), out_axes=(1, 1))(ws, init_stat, spikes, decay_consts, threshs)
    return jnp.sum((spikes-1.0)**2), (states, spikes)


@jax.jit
def calc_result_and_gradient_jax(weights, init_state, inp_spikes, decay_constants, thresholds):
    (value, (states, out_spikes)), grads = value_and_grad(calc_loss, argnums=[0, 1, 2], has_aux=True)(weights, init_state, inp_spikes, decay_constants, thresholds)
    return (out_spikes, states), grads
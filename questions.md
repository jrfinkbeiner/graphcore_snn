# Questions

* How to make sure manual alloc is used?
* Could the weird alloc be the reason that popops::zero doesn't work as expected?

* when using poplar 2.6 I get the error:

```bash
Traceback (most recent call last):
  File "keras_train_util_ipu.py", line 715, in <module>
    test_sparse_vs_dense()
  File "keras_train_util_ipu.py", line 583, in test_sparse_vs_dense
    out_sparse_ops, grad_sparse_ops =  strategy.run(value_and_grad_on_batch, args=[model_sparse_ops, *data_sparse, True])
  File "/opt/poplar/lib/python/libpvti_py3.py", line 537, in inner
    result = func(*args, **kwargs)
  File "/usr/local/lib/python3.6/dist-packages/tensorflow/python/ipu/ipu_strategy.py", line 107, in run
    return super().run(fn, args, kwargs, options)
  File "/usr/local/lib/python3.6/dist-packages/tensorflow/python/distribute/distribute_lib.py", line 1286, in run
    return self._extended.call_for_each_replica(fn, args=args, kwargs=kwargs)
  File "/usr/local/lib/python3.6/dist-packages/tensorflow/python/distribute/distribute_lib.py", line 2849, in call_for_each_replica
    return self._call_for_each_replica(fn, args, kwargs)
  File "/opt/poplar/lib/python/libpvti_py3.py", line 537, in inner
    result = func(*args, **kwargs)
  File "/usr/local/lib/python3.6/dist-packages/tensorflow/python/ipu/ipu_strategy.py", line 207, in _call_for_each_replica
    _validate_function_for_arguments(fn, args, kwargs)
  File "/opt/poplar/lib/python/libpvti_py3.py", line 537, in inner
    result = func(*args, **kwargs)
  File "/usr/local/lib/python3.6/dist-packages/tensorflow/python/ipu/ipu_strategy.py", line 166, in _validate_function_for_arguments
    concrete_fn = fn.get_concrete_function(*args, **kwargs)
  File "/usr/local/lib/python3.6/dist-packages/tensorflow/python/eager/def_function.py", line 1233, in get_concrete_function
    concrete = self._get_concrete_function_garbage_collected(*args, **kwargs)
  File "/usr/local/lib/python3.6/dist-packages/tensorflow/python/eager/def_function.py", line 1213, in _get_concrete_function_garbage_collected
    self._initialize(args, kwargs, add_initializers_to=initializers)
  File "/usr/local/lib/python3.6/dist-packages/tensorflow/python/eager/def_function.py", line 760, in _initialize
    *args, **kwds))
  File "/usr/local/lib/python3.6/dist-packages/tensorflow/python/eager/function.py", line 3066, in _get_concrete_function_internal_garbage_collected
    graph_function, _ = self._maybe_define_function(args, kwargs)
  File "/usr/local/lib/python3.6/dist-packages/tensorflow/python/eager/function.py", line 3463, in _maybe_define_function
    graph_function = self._create_graph_function(args, kwargs)
  File "/usr/local/lib/python3.6/dist-packages/tensorflow/python/eager/function.py", line 3308, in _create_graph_function
    capture_by_value=self._capture_by_value),
  File "/usr/local/lib/python3.6/dist-packages/tensorflow/python/framework/func_graph.py", line 1010, in func_graph_from_py_func
    func_outputs = python_func(*func_args, **func_kwargs)
  File "/usr/local/lib/python3.6/dist-packages/tensorflow/python/eager/def_function.py", line 668, in wrapped_fn
    out = weak_wrapped_fn().__wrapped__(*args, **kwds)
  File "/usr/local/lib/python3.6/dist-packages/tensorflow/python/framework/func_graph.py", line 997, in wrapper
    raise e.ag_error_metadata.to_exception(e)
TypeError: in user code:

    keras_train_util_ipu.py:516 value_and_grad_on_batch  *
        gradients = tape.gradient(loss, model.trainable_weights)
    /usr/local/lib/python3.6/dist-packages/tensorflow/python/eager/backprop.py:1090 gradient  **
        unconnected_gradients=unconnected_gradients)
    /usr/local/lib/python3.6/dist-packages/tensorflow/python/eager/imperative_grad.py:77 imperative_grad
        compat.as_str(unconnected_gradients.value))
    /usr/local/lib/python3.6/dist-packages/tensorflow/python/ipu/sharding.py:166 _sharded_backprop_gradient_function
        return backprop._gradient_function(op_name, attr_tuple, *args, **kwargs)
    /usr/local/lib/python3.6/dist-packages/tensorflow/python/eager/backprop.py:159 _gradient_function
        return grad_fn(mock_op, *out_grads)
    /usr/local/lib/python3.6/dist-packages/tensorflow/python/ops/while_v2.py:384 _WhileGrad
        stateful_parallelism)
    /usr/local/lib/python3.6/dist-packages/tensorflow/python/ops/while_v2.py:687 _create_grad_func
        acd_record_initial_resource_uses=stateful_parallelism)
    /usr/local/lib/python3.6/dist-packages/tensorflow/python/framework/func_graph.py:1010 func_graph_from_py_func
        func_outputs = python_func(*func_args, **func_kwargs)
    /usr/local/lib/python3.6/dist-packages/tensorflow/python/ops/while_v2.py:682 <lambda>
        lambda *args: _grad_fn(ys, xs, args, body_graph),
    /usr/local/lib/python3.6/dist-packages/tensorflow/python/ops/while_v2.py:740 _grad_fn
        unconnected_gradients="zero")
    /usr/local/lib/python3.6/dist-packages/tensorflow/python/ops/gradients_util.py:682 _GradientsHelper
        lambda: grad_fn(op, *out_grads))
    /usr/local/lib/python3.6/dist-packages/tensorflow/python/ops/gradients_util.py:338 _MaybeCompile
        return grad_fn()  # Exit early
    /usr/local/lib/python3.6/dist-packages/tensorflow/python/ops/gradients_util.py:682 <lambda>
        lambda: grad_fn(op, *out_grads))
    /usr/local/lib/python3.6/dist-packages/tensorflow/python/ipu/ops/internal_ops_grad.py:143 _poputil_precompiled_user_op_layer_backward
        return _poputil_op_layer_backward(op, grads, add_op)
    /usr/local/lib/python3.6/dist-packages/tensorflow/python/ipu/ops/internal_ops_grad.py:93 _poputil_op_layer_backward
        [t.shape for t in inputs if t is not None])
    /usr/local/lib/python3.6/dist-packages/tensorflow/python/ipu/ops/internal_ops_grad.py:141 add_op
        output_shapes=output_shapes)
    /usr/local/lib/python3.6/dist-packages/tensorflow/compiler/plugin/poplar/ops/gen_poputil_ops.py:1681 ipu_user_op
        gradient_attributes=gradient_attributes, name=name)
    /usr/local/lib/python3.6/dist-packages/tensorflow/python/framework/op_def_library.py:469 _apply_op_helper
        "%s that are invalid. Tensors: %s" % (prefix, values))

    TypeError: Tensors in list passed to 'input' of 'IpuUserOp' Op have types [float32, <NOT CONVERTIBLE TO TENSOR>, float32, int32, float32, float32] that are invalid. Tensors: [<tf.Tensor 'gradient_tape/model/rnn/while/gradients/AddN:0' shape=(24, 256) dtype=float32>, None, <tf.Tensor 'model/rnn/while/keras_multi_lif_layer_sparse_cell/compute_sparse_spikes_op:0' shape=(24, 256) dtype=float32>, <tf.Tensor 'model/rnn/while/keras_multi_lif_layer_sparse_cell/compute_sparse_spikes_op:1' shape=(24, 1) dtype=int32>, <tf.Tensor 'model/rnn/while/keras_multi_lif_layer_sparse_cell/add:0' shape=(24, 256) dtype=float32>, <tf.Tensor 'model/rnn/while/keras_multi_lif_layer_sparse_cell/compute_sparse_spikes_op/ReadVariableOp:0' shape=(256,) dtype=float32>]
```
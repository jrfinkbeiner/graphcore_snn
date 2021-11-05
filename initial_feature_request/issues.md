## Remaining issues

### Sparsity
* keras.layers.RNN does not seem to support sparse tensors
* SparseTensorShape (?) in state_size in order to implement self-recurrence
* tf.stop_gradient doesn't work for sparse matrices
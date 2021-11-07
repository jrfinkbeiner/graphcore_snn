## Remaining issues

### Sparsity
* keras.layers.RNN does not seem to support sparse tensors as output
* SparseTensorShape (?) in state_size in order to implement self-recurrence without having to transform to dense
* tf.stop_gradient doesn't work for sparse tensors
## Open Questions and Notes

### RNN-Implementation
* are multilayer LSTMs/GRUs supported and if, how?

### Sparsity
* can gradients flow through sparse matrices?
* do convolutions on sparse matrices work in tf?
* We don't need the values in tf.SpareTensor (only indices (and shape))


### BPTT memory issues
* maybe create inference only version with true dynamic unrolling?
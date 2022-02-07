# Custom LIF - Single Layer

## TODO

* write batched sparse matmul (and it's gradients) and how to best allocate the tensors
* write custom op that takes, weights, internal state and inputs and outputs spiking activity
* write the full rnn code, like in poplibs (no more keras rnn or dynamic unroll layer around it)
* only write case with gradient computation (use inplace update operation for weights derivative tensor)
* write tensor allocation code for weights and inputs (see PopGRU for inspiration how allocation can be handled)

## Questions

* Are multiple threads automatically used if multiple vertices are put on the same tile for the same compute set?
* What is the `layerSizes` parameter in `rnn::RnnParams` used for? Should I set it to the dense, or the sparse sizes?
* hwo to define good cycle count estimate?
* does a vertex Input<poplar::Tensor> or sth similar with two d shape exist?

## Issues

* only int32 supported on ipu?
* in rnn outputs are created with respect to RnnParams.dataType and not with the respect to the datatype of the corresponding tensor. This prevents ouputs of different data types... -> solve by just using the input field!
* weird values appearing in arrays when used in vertex. see issue with dLdweights, (preliminary) solved by creating temporary tensor and copying values...
* `+=` not save to use in vertex code? even if same address is not written to multiple times? -> How to handle inplace add in vertex code ? Use it as both input and output? Possible to do multiple in place add on the same element in single vertex call?
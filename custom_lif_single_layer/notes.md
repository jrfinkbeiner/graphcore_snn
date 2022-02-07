# Custom LIF - Single Layer

## TODO

* write tensor allocation code for weights and inputs (see PopGRU for inspiration how allocation can be handled?)
* hello

## Questions

* Are multiple threads automatically used if multiple vertices are put on the same tile for the same compute set?
-> Yes
* What is the `layerSizes` parameter in `rnn::RnnParams` used for? Should I set it to the dense, or the sparse sizes?
-> as far as I can see only used for state creation. For now sufficient to set to sparse? 
* how to define good cycle count estimate?
-> will be public info soon, but not relevant for performance, just 
* does a vertex Input<poplar::Tensor> or sth similar with two d shape exist?
-> `Vector<Vector<>>`
* how to turn poplar 2d tensor to `Vector<Vector<>>`for vertex input/output


## Remaning Issues

* in rnn outputs are created with respect to RnnParams.dataType and not with the respect to the datatype of the corresponding tensor. This prevents ouputs of different data types... -> solve by just using the input field!
* how to parallize different batches of the weight tensor update ? access to identical tensor elements might happen... or just change structure of update in general (from row parallel to column parallel)?

## Solved Issues

* only int32 supported on ipu?
-> reinterpret
* weird values appearing in arrays when used in vertex. see issue with dLdweights, (preliminary) solved by creating temporary tensor and copying values...
-> InOut<> type
* `+=` not save to use in vertex code? even if same address is not written to multiple times? -> How to handle inplace add in vertex code ? Use it as both input and output? Possible to do multiple in place add on the same element in single vertex call?
-> InOut<> type

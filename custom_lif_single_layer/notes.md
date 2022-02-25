# Custom LIF - Single Layer

## TODO

* improve tile mapping (also for sparse2dense operation)
* implement tf + custom sparse matmul snn for comparison

## Questions

* Are multiple threads automatically used if multiple vertices are put on the same tile for the same compute set?
-> Yes
* What is the `layerSizes` parameter in `rnn::RnnParams` used for? Should I set it to the dense, or the sparse sizes?
-> as far as I can see only used for state creation. For now sufficient to set to sparse?
* how to define good cycle count estimate?
-> will be public info soon, but not relevant for performance or compiler, just debugging/analysis
* does a vertex Input<poplar::Tensor> or sth similar with two d shape exist?
-> `Vector<Vector<>>`
* how to turn poplar 2d tensor to `Vector<Vector<>>`for vertex input/output
* custom op with tf 2 ?
-> yes, see functional keras models
* tradeoff/sweetspot between compute and communication? rule of thumb for parallelization?
-> popvision/profiling will be helpful here, compare cycles spent for compute and communication
* how does parallization to multiple IPUs work for GNNs?
* how to execute compute-sets in parallel ? (instead of program::Sequence, program::Parallel or sth)

## Remaning Issues

* in rnn outputs are created with respect to RnnParams.dataType and not with the respect to the datatype of the corresponding tensor. This prevents ouputs of different data types... -> solve by just using the input field?
* how to parallize different batches of the weight tensor update ? access to identical tensor elements might happen... or just change structure of update in general (from row parallel to column parallel)?
-> there is probably no way of achieving this

## Solved Issues

* only int32 supported on ipu?
-> no, xla interface issue, possible solution: reinterpret
* weird values appearing in arrays when used in vertex. see issue with dLdweights, (preliminary) solved by creating temporary tensor and copying values...
-> InOut<> type
* `+=` not save to use in vertex code? even if same address is not written to multiple times? -> How to handle inplace add in vertex code ? Use it as both input and output? Possible to do multiple in place add on the same element in single vertex call?
-> InOut<> type


## Profiling

### General


* is there a way to choose whether to send spikes or not based on data? control communication at a lower level? or simply poplar::program::If ?
* how does communication work? can comminication to and from tile happen at the same time? or do they block eaach other ?
* how to reduce communication time? Is bandwidth or latency the issue? Could it actually be faster to just send the dense tensors to every tile?
* how to prevent code-copy for man loop? or alternitavely: how to implement my own loop/repeat function?
* why does choosing different tiles impact CopyCopy ? (changing tiles from {0, 1} to {32, 64} for `genBatchedLIFOutSpikes` in case of batchsize=2)


### Forward

* 

### Backward

* why does `popops::mulInPlace` take so many cycles? elements should already be on the correct tiles.



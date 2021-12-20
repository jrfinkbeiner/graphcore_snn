# Custom Dynamic Dense-Matrix-Sparse-Vector Multiplication

This repository features a basic (not yet optimized) implementeation of a dense matrix times sparse vector matrix multiplication, where the sparse vector can have a dynamically changing number of non-zero elements. It is implemented using the tensorflow-poplar interface with custom codelets.

Files:

* `custom_codelet.cpp` constains the vertex code for the forward matmul and for the gradient computation of the loss with respect to the dense matrix and sparse vector during the backward phase.
* `poplar_code.cpp` contains the poplar code to perform the forward matmul and gradient computations.
* `tf_code.py` contains tensorflow code that performs a dense matrix times sparse vector matmul and the respective gradient calculation and compares the reults to a manuel numpy implementation

In order to run the code execute the following steps:

1. source poplar
2. install poplar-tensorflow (from whl) or source poplar-tensorflow env with installed poplar-tensorflow
3. change directory to this directory
4. run ```make all```
5. run ```python tf_code.py```

## Implementation Details

The dynamic nature of the sparse vector is realized using static sizes as necessary for the poplar compiler. This is achieved by setting a maximum number of non-zero elements which determines the size of the sparse vector. Insead of the sparse tensor, the function takes two dense tensors: One tensor of size max-number-of-spikes which holds the indices of the non-zero elements and an aditinal tensor that contains one element, which is the number of non-zero elements of the vector. Within a vertex then only the actual non-zero elements of the vector are used instead of the whole allocated tensor/vector.

## Caveats

In the following we summarize a few issues we encountered or caveats that should be mentioned regarding the implementation. On the one hand these include the restrictions that we impose on the algorithm due to our choice of implementation (given the software constraints) and on the other IPU specific implementation details that we couldn't resolve for now (on our own). Generally, it should be noted, that the chosen tile-mapping is preliminary and is only supposed to demonstrate where we currently see a clear and easy way to parallalization and where not. Further work to improve this is clearly necessary.

### Restrictive Choices

1. We restrict ourselves to a maximum number of non-zero elemtents. This number determines the amount of memory that has to be allocated (irrespective of the actual number of non-zero elements) for the sparse vector.
2. Only non-zero elements in the sparse vector and not all elements of the underlying dense tensor have non-zero gradients: The gradient tensor of the loss with respect to the sparse vector has the same size as the sparse vector itself.

### Memory Allocation and other Optimizations

3. In order for the forward pass to run efficiently, it requires a specific allocation of the matrix. Contrary to dense matrix multiplication it does not make sense to spilt a row onto multiple tiles. Rows should rather be allocated as whole on a tile with the corresponding vertex. Different rows, however, can nicely be spilt onto multiple tiles.
4. The calculation of gradients (loss with respect to inputs) requires the transpose of the weight matrix. If done naively, this might results in lots of (unnecessary) communication. For now it is implemented to run on a single tile due to simplicity. Alternatively, row-wise splitting with an additional communication and summation of the results could be a good solution. Nonetheless, a good/better solution that co-optimizes the forward and backward pass will be necessary.
5. Tensorflow expects the derivative of the loss with respect to the weight matrix to be a dense tensor, which requires allocation of the whole matrix. Due to the sparsity of the input, the gradient matrix is, however, largely zero.
6. The current implementation does not make use of multiple workers yet.

Issues 3., 4. and 5. could potentially be solved by implementing not only the dense-sparse matrix-vector multiplication as a custom op, but a spiking nural network layer as a custom layer as a whole. There the allcocation of the weight matrix could be manually defined (solving 3.and co-optimizing with 4.). Additionally, gradients of the weights could be applied inplace, or other types of objects could be handled intermediately instead of the same-size-dense-tensor expected by the tensorflow interface (resolving 5.).

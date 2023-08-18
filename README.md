# graphcore_snn

Projects related to the development of an efficient spiking neural network (SNN) implementation on graphcore's IPU.

## Running Code with and without Access to IPUs

If you have access to an IPU, great, if not, you can also compile the code and run it in a simulation mode, the `IPUModel` on an CPU. 

If you are working on your own machine, go to https://www.graphcore.ai/downloads and download the latest polar sdk. 
Unpack it on your machine. In there you will find a tensorflow wheel, that you can install in a new environment. Additionally, you will have to source the "poplar<...sth...>/enable.sh" file in there and then you can run the same code but only using the IPUModel configuration.

## Installation Guide

Source the poplar as mentioned in the section above and install the provided tensorflow 2 wheel:

```console
source poplar<...sth...>/enable.sh
pip install --upgrade pip
pip install tensorflow-<... smth...>.whl
```

Then, to compile run:

```console
mkdir build
cd build
cmake ..
make
```

For now only compiling code via cmake is supported for sparse ops. For other code (sparse layer, ...) please go to the respective folders and run `make` manually.

```console
cd custom_lif_multi_layer/custom_lif_layer_vectorize_transpose/
make
```

Lastly, pip install necessary requirements for the benchmarking code:

```console
pip install -r requirements.txt
```

Test your installation by running the following code if executing with an actual IPU:

```console
cd benchmarking
./runfile_test.sh
```

Alternatively, if running on a cpu in simulation mode using the `IPUModel`, run:

```console
cd benchmarking
TF_POPLAR_FLAGS=--use_ipu_model ./runfile_test.sh
```


## First Steps

Everytime, you want to run anything, using either the `IPUModel` on CPU or on an actual IPU, amke sure you sourced the poplar sdk first. 

With this and the installations you will be able to use the sparse implementations as defined in `benchmarking/keras_train_util_ipu.py`.
If you are only interested in using select operations, just import them from this file in your python file:

```python
from keras_train_util_ipu import compute_sparse_spikes, dyn_dense_binary_sparse_matmul_op
```

For now there is no more sophisticated integration into python packages, so just place your files in the benchmarking folder and execute from there.

It follows a list of interesting files in the `benchmarking` folder:
- `keras_train_util.py`: Keras implementation of base SNN class and the dense version for the GPU
- `keras_train_util_ipu.py`: Keras SNN-implementation for the IPU, including dense and sparse implementations, both for the sparse ops as well as for the fully custom sparse multi-layer SNN. Executing this file directly will run and compare results for dense, sparse ops, and sparse layer implementation (essentially running a poor man's test case).
- `nmnist_util.py`: Utility file for dataloaders
- `multi_proc_helper.py`: Helper file for multiprocessing for dataloading
- `benchmarking_script.py`: File to execute training for benchmarking puproses
- `performance_jobscript.sh`: Sbatch file that executes a runscript.
- `runfile_nmnist_multi_layer_benchmark.sh`: Runfile that in it's current form just executes example runs for the `sprase_layer` and `sparse_ops` implementation.

In order to test your setup either run:

```console
srun singularity run <docker_image_path> ./runfile_test.sh
```

```console
sbatch performance_jobscript
```
(NOTE you have to adjust the file to use the correct docker image file. As well as the data-path in `benchmarking_script.py`)






## Relevant Resources
- Zenke, Friedemann, and Emre O. Neftci. "Brain-inspired learning on neuromorphic substrates." Proceedings of the IEEE 109.5 (2021): 935-950. (https://ieeexplore.ieee.org/abstract/document/9317744/)
- Menick, Jacob, et al. "A practical sparse approximation for real time recurrent learning." arXiv preprint arXiv:2006.07232 (2020). (https://arxiv.org/abs/2006.07232)
- Bellec, Guillaume, et al. "A solution to the learning dilemma for recurrent networks of spiking neurons." Nature communications 11.1 (2020): 1-15. (https://www.nature.com/articles/s41467-020-17236-y)
- Kaiser, Jacques, Hesham Mostafa, and Emre Neftci. "Synaptic plasticity dynamics for deep continuous local learning (DECOLLE)." Frontiers in Neuroscience 14 (2020): 424 (https://www.frontiersin.org/articles/10.3389/fnins.2020.00424/full?report=reader) code: (https://github.com/nmi-lab/decolle-public)

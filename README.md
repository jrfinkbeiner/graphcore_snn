# graphcore_snn

Projects related to the development of an efficient spiking neural network (SNN) implementation on graphcore's IPU.

To compile run:

```console
mkdir build
cd build
cmake ..
make
```

For now only omiling code for sparse ops is supported. For other code (sparse layer, ...) please go to the respective folders and run `make` manually.


## Relevant Resources
- Zenke, Friedemann, and Emre O. Neftci. "Brain-inspired learning on neuromorphic substrates." Proceedings of the IEEE 109.5 (2021): 935-950. (https://ieeexplore.ieee.org/abstract/document/9317744/)
- Menick, Jacob, et al. "A practical sparse approximation for real time recurrent learning." arXiv preprint arXiv:2006.07232 (2020). (https://arxiv.org/abs/2006.07232)
- Bellec, Guillaume, et al. "A solution to the learning dilemma for recurrent networks of spiking neurons." Nature communications 11.1 (2020): 1-15. (https://www.nature.com/articles/s41467-020-17236-y)
- Kaiser, Jacques, Hesham Mostafa, and Emre Neftci. "Synaptic plasticity dynamics for deep continuous local learning (DECOLLE)." Frontiers in Neuroscience 14 (2020): 424 (https://www.frontiersin.org/articles/10.3389/fnins.2020.00424/full?report=reader) code: (https://github.com/nmi-lab/decolle-public)

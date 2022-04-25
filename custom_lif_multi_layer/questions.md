# Questions

## Important

* Stream Copy 
    - why does stream copy only act on so few tiles ? Where is the bottleneck ? in writeToTile or in bandwidth betweeen IPU and host ?
    - why is there stream copy before the backkward pass in the first place ?
    - is the stream copy that the popvison tool shows applicable to all epochs? different to later epochs (because no more code and parameter transfer necessary) ? 

* parallelize poplar ops for multiple layers:\\
    - dynamic slice op
    - reduce op
    - there is no way to execute them as compute sets in parallel. Just using sequence.add(Execute(computeSet))

* How to make vertices run faster?\\
    - is there a way to have vectorization in our case, where we don't have 8 byte aligned memory ?
    - any other way to spped up the look up and write process ? (parallel read and execute mentioned in doc when (when writing assembler?))

* Multivertex\\
    - what happens if there are multiple mutlivertices on one tile? are the mutlivertices executed sequentially with 6 workers each, or are the 6 workers split between the multivertices?
    - how many cycles does 'read-add-write' take ? (If multiple workers access the same output memory, I'd like to estimate, how often I would overrwrite other computations. )



## Multi Ipu

* possible to write multi-ipu code within a cusom op ?
* can communication between ipus happen in parallel to processing? (While one time step is beeing processed data from one IPU is sent to another to be processed in the next timestep. (Accounting for axonal delays and improving parallelization))

## Vectorization

* __is there a way to have vectorization in our case, where we don't have 8 byte aligned memory ?__

## Multi layer

* could it be beneficial to combine states into one tensor ? How big is the overhead of looping over all the tensors ? Is it suboptimal memory wise to have a vector of tensors compared to one large tensor? (one large tensor couldn't even be realized for the weights, so worth the effort only for states ? not even there if different neuron types are introduced with a different number of internal states.)

## Random

* __is there a `shuffle` or random choice function in poplar that let's you randomly rearrange tensors or generate random indices for that?__

## Data types

* __is there a way to have different data types for forward tensor and gradient tensor?__ (relevant for all spike ids operations, both in multilayer, as well as single custom ops.) Would love to have unisgned int16 both for spikes and num_inputs, possible with tf ipu interface ?
* is `int` just as good as `unsigned` for indexing (or does there happen a cast, and if is it expensive/how expensive compared to float to unsigned?)

## Tile mapping and tensor splitting

* __if the input spikes are called for mulitiple vertices on one tile, does that move the tensors for every vertex operation or once for all vertices on the tile?__
* does the `.slice()` operation create new tensors or use a view of the parent tensor ?
* is there a better way to perform the tile ampping ?
* is there an advantage in having one vertex call for all neurons per thread on a tile (and spiltting the workers manually beforehand into batches per thread per tile) or is it just as good to have one vertex per neuron and let the compiler determine the splitting between workers? Does is increase code memory or add overheads in spawning all the vertices?

* how does the dynamicSlice work? is the initialized target slice tensor allocated on the same tiles? Is the data copied somewhere or is it just a view?

## Multivertex

* if using Multivertex, should ideally everything that happens on one tile be handeled by one Multivertex? (at the moment there is a multivertex per neuron on the tile, therefore not efficiently making use of multiple neurons on one tile?)

## TODO Jan

* fix sparse ops (even forward pass seems broken)

* can the loop in `LIFWeightsGrad` really be used with MultiVertex? might touch same elements because of different batches -> can be done if batches are handled sequentially in sqquential compute sets! worth it/ is it really faster, especially if there are multiple/6 neurons on a tile...? (might help to balance workers, e.g. when there are 1 or 7 neurons per tile.)
* think about reimplementing calcLIFStateGrad to not only make use of few tiles and copy over states.
* use parallel Reduce Operation with compute set cevor for multi layer lif calcLICInpSpikeGrads
* appply the (1-decay_constant) in the gradient calculation once to all state derivatives (not as currently both in inp spike grad and weights grad calculation.) a little bit more memory necessary, but should be negligable compared to state sequence

* impelment checkpointing/recomputation ?

* make tile mapping dependent of input spikes, output spikes, or both ? or independant ?

* perform slicing and reduce operation in parallel for all layers

## TODO less important Jan

* implement loss for all layers
* implement sparse loss
* implement second threshold as variable and dynamically adjust it

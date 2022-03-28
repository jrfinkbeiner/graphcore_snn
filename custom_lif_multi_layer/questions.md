# Questions

## Vectorization

* is there a way to have vectorization in our case, where we don't have 8 byte aligned memory ?

## Multi layer

* could it be beneficial to combine states into one tensor ? How big is the overhead of looping over all the tensors ?

## Tile mapping and tensor splitting

* if the input spikes are called for mulitiple vertices on one tile, does that move the tensors for every vertex operation or once for all vertices on the tile?
* does the `.slice()` operation create new tensors or use a view of the parent tensor ?
* is there a better way to perform the tile ampping ?
* is there an advantage in having one vertex call for all neurons per thread on a tile (and spiltting the workers manually beforehand into batches per thread per tile) or is it just as good to have one vertex per neuron and let the compiler determine the splitting between workers? Does is increase code memory or add overheads in spawning all the vertices?

* how does the dynamicSlice work? should is the initialized target slice tensor allocated on the same tiles? Is the data copied somewhere or is it just a view?

## Multivertex

* if using Multivertex, should ideally everything that happens on one tile be handeled by one Multivertex? (at the moment there is a multivertex per neuron on the tile, therefore not efficiently making use of multiple neurons on one tile?)

## TODO Jan

* can the loop in `LIFWeightsGrad` really be used with MultiVertex? might touch same elements because of different batches
* think about reimplementing calcLIFStateGrad to not only make use of few tiles and copy over states.

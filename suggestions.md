# Suggestions

In general, make some performance relevant things more obvious
* include `-O3` in custom op build scripts
* for keras api include `prefetch_depth` in your examples, or make it easier, more obvious to find. To be honest, if one uses, `.prefetch()` on a tf dataset, I assumed it would already do the trick, even when I read in the documentation that one should use prefetching. (Explicitly link to the keras API prefetch_depth)

## Minor things

* using the variable name `attributes` for `Build` and `Build_grad` makes you think the attribute filed is used for both. Even if it states in the docu, that there is a separate variable called `gradient_attributes`
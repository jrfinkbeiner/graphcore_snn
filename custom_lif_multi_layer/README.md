# Custom LIF Single Layer

This directory (and its subdirectories) contain code for the implementation of muliple Leaky-Integrate-and-Fire (LIF) layers as a single poplar custom op.

It is structured as follows:

* `custom_lif_layer_vectorize` contains the custom op code for a single LIF layer.
* `sparse2dense` contains custom op code that implements a transformation from a dense tensor (`seq_len`, `batchsize`, `num_neurons`) to the sparse tensor construct chosen in this project.

Files:

* `keras_timing.py` implements the functions `train_ipu` and `train_gpu`, that train a basic (multi-layer) SNN. For an example an see the `main()` function of the file, where it is executed on randomly generated data. If you don't use a tensorflow version that supports the ipu, uncomment the line `from tensorflow.python import ipu` in order to run the (non-ipu) code.

## Compiling and running code on the IPU/IPUModel

The steps to enable running the code on the IPU or the IPUModel are the following:

1. Download Poplar SDK including IPU-tensorflow wheel
2. Source Poplar SDK and compile the custom ops
3. Install the tensorflow version from the IPU-tensorflow wheel to a python virtual environment
4. Source IPU-tensorflow environment and run the code

For more details regarding the Poplar SDK and its integration with tensorflow (and other frameworks) see:
https://docs.graphcore.ai/projects/sdk-overview/en/latest/overview.html
or specifically for tensorflow on the IPU https://docs.graphcore.ai/projects/tensorflow-user-guide/en/latest/.

### 1. Download

Download the Poplar SDK from https://downloads.graphcore.ai/ for your operationg system and extract/unzip the file structure. This contains the Poplar SDK and pre-built wheels that allow the integration of various machine learning frameworks (including tensorflow) with the IPU.

Note that for the download a Graphcore account is required.

Alternatively, Graphcore provides ready to use Docker containers with the machine-learning framework of your choice that can be dowloaded from the same link above. For instructions how to use these, please refer to the official documentation.

### 2. Compiling custom ops

In order to run code that uses the IPU, first you have to compile both the custom lif layer as well as the `sparse2dense` operation. 

First, source the Poplar SDK by running

```bash
source /<storage_path>/poplar_sdk-<sdk_version>/poplar-<poplar_version>/enable.sh
```

Then compile the custom codelets in both subdirectories called `sparse2dense` as well as `custom_lif_layer_vectorize` by running:

```bash
cd sparse2dense
make all
cd ../custom_lif_layer_vectorize
make all
cd ..
```

### 3. Install IPU-tensorflow version

Best practice is to insall the specific IPU-tensorflow version in a python virtual environment. Here we will be using python-venv and pip. For that choose a desired root path for your environment `<envs_root_path>` and a name <env_name>.

First, a new env is created, then the env is activated and the IPU-tensorflow version for your system requirements installed in it.

```bash
python3 -m venv /<envs_root_path>/<env_name>
source /<envs_root_path>/<env_name>/bin/activate
pip install /<storage_path>/poplar_sdk-<sdk_version>/tensorflow-<tf_version>+gc<gc_version>+<sys_specific>.whl
```

### 4. Run code in python venv

Source the IPU-tensorflow env if not already done.

```bash
source /<envs_root_path>/<env_name>/bin/activate
```

and run the ipu python application of your choice. If you want to use the `IPUModel`, which simulates the IPU using CPUs, set the environemnt variable `TF_POPLAR_FLAGS` with `--use_ipu_model` before running your script, or just add the line

```python
os.environ["TF_POPLAR_FLAGS"] = "--use_ipu_model"
```

to your python script. For additional information on how to target the poplar xla (tensorflow) device see https://docs.graphcore.ai/projects/tensorflow1-user-guide/en/latest/tensorflow/device_selection.html or for mor einformation on the IPU Model see https://docs.graphcore.ai/projects/poplar-user-guide/en/latest/poplar_programs.html.


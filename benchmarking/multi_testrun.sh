#!/bin/bash

# python3 multi_ipu.py
POPLAR_ENGINE_OPTIONS='{"autoReport.all":"true","autoReport.directory":"./reports_multiIpu/pipelineAnalysis_2ipu_1_noOffloading"}' python3 multi_ipu.py
# POPLAR_ENGINE_OPTIONS='{"autoReport.all":"true","autoReport.directory":"./reports_multiIpu/analyseMem_singleIpu_01_noShard"}' python3 keras_train_util_ipu.py

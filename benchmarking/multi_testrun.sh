#!/bin/bash

python3 multi_ipu.py
# POPLAR_ENGINE_OPTIONS='{"autoReport.all":"true","autoReport.directory":"./reports_multiIpu/analyseMem_02ipu_7_firstTileFix_reAlloc"}' python3 multi_ipu.py
# POPLAR_ENGINE_OPTIONS='{"autoReport.all":"true","autoReport.directory":"./reports_multiIpu/analyseMem_singleIpu_01_noShard"}' python3 keras_train_util_ipu.py

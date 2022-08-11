#!/bin/bash

POPLAR_ENGINE_OPTIONS='{"autoReport.all":"true","autoReport.directory":"./reports_multiIpu/analyseMem_02ipu_4_firstTileFix_twoLayersOnly"}' python3 multi_ipu.py
# POPLAR_ENGINE_OPTIONS='{"autoReport.all":"true","autoReport.directory":"./reports_multiIpu/analyseMem_singleIpu_01_noShard"}' python3 keras_train_util_ipu.py
